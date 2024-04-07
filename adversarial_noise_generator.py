from torch.optim import Adam, SGD, LBFGS
from torch.nn import CrossEntropyLoss
import torch
import requests
import PIL.Image as Image
from io import BytesIO
import numpy as np
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
class TargetedAdvNoiseGenerator:
    def __init__(self, model, weights):
        self.model = model 
        self.weights = weights

    def generate(self, image, original_class, target_class, max_iter=500):
        for i in range(max_iter):
            def closure():
                self.optimizer.zero_grad()
                # self.adversarial_noise = torch.clip(self.adversarial_noise, -0.001, 0.001)
                perturbed_image = image + self.adversarial_noise
                output = self.model(perturbed_image)
                loss = CrossEntropyLoss()
                loss_origin = -loss(output, torch.tensor([original_class]))
                loss_target = loss(output, torch.tensor([target_class]))

                loss = loss_target
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                # showing progress
                if i % 100 == 0:
                    print(f"Iteration {i}")
                    if i!=0:
                        print("loss_origin: {}, loss_target: {}".format(loss_origin, loss_target))
                    print(self.adversarial_noise.max(), self.adversarial_noise.min())
                    print(self.adversarial_noise)
                return loss
            

            self.optimizer.step(closure)
        return self.adversarial_noise

    def apply(self, image_url, target_class, max_iter=500, opt_lr = 5e-5):        
        # image processing
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        # these steps are based on the resnet50 model, need to generalise for other models in the future
        resize_ratio = 232/min(image.size)
        image = image.resize((int(image.size[0]*resize_ratio),int(image.size[1]*resize_ratio) ), Image.BILINEAR)
        image = transforms.CenterCrop(224)(image)
        # save the image for debugging
        image.save("output_img/{}".format(image_url.split("/")[-1]))

        #  -----------------  #

        preprocess = self.weights.transforms()
        preprocessed_image = preprocess(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        print("preprocessed image size: {}".format(preprocessed_image.size()))
        # get original class

        output = self.model(preprocessed_image).squeeze().softmax(dim=0)
        original_class = output.argmax().item()
        print("original class: {}".format(original_class))
        # initiale adversarial noise corresponding to the image size
        self.adversarial_noise = torch.zeros((preprocessed_image.size(dim=0), 
                                              preprocessed_image.size(dim=1), 
                                              preprocessed_image.size(dim=2), 
                                              preprocessed_image.size(dim=3)), requires_grad=True)
        self.optimizer = Adam([self.adversarial_noise], lr=opt_lr)
        # generate adversarial noise
        noise = self.generate(preprocessed_image, original_class, target_class, max_iter)
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

        noise = invTrans(noise)
        noise = noise.squeeze(0).detach().numpy()
        print("noise size: {}".format(noise.shape))
        noise = noise.reshape(image.size[0], image.size[1], 3)
        output = np.array(image) + noise*255
        # turn output int into uint8
        output = np.array(output, dtype=np.uint8)
        output = Image.fromarray(output)
        # save the image for debugging
        output.save("output_img/{}".format(image_url.split("/")[-1].split(".")[0]+"_singleloss_altered."+image_url.split("/")[-1].split(".")[1])   )
        return output
    

