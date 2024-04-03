from torch.optim import Adam, SGD
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
            # self.adversarial_noise = self.adversarial_noise.clamp(-0.001, 0.001)
            perturbed_image = image + self.adversarial_noise
            output = self.model(perturbed_image)
            loss = CrossEntropyLoss()
            loss_origin = -loss(output, torch.tensor([original_class]))
            loss_target = loss(output, torch.tensor([target_class]))
            
            # showing progress
            if i % 100 == 0:
                print(f"Iteration {i}")
                print("loss_origin: {}, loss_target: {}".format(loss_origin, loss_target))

            loss = loss_origin+loss_target
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.adversarial_noise

    def apply(self, image_url, target_class, max_iter=500, opt_lr = 5e-5):        
        # image processing
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        # these steps are based on the resnet50 model, need to generalize for other models in the future
        resize_ratio = 232/min(image.size)
        image = image.resize((int(image.size[0]*resize_ratio),int(image.size[1]*resize_ratio) ), Image.BILINEAR)
        image = transforms.CenterCrop(224)(image)
        #  -----------------  #

        preprocess = self.weights.transforms()
        preprocessed_image = preprocess(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        print("preprocessed image size: {}".format(preprocessed_image.size()))
        # get original class

        output = self.model(preprocessed_image).squeeze().softmax(dim=0)
        original_class = output.argmax().item()
        # initiale adversarial noise corresponding to the image size
        self.adversarial_noise = torch.zeros((preprocessed_image.size(dim=0), 
                                              preprocessed_image.size(dim=1), 
                                              preprocessed_image.size(dim=2), 
                                              preprocessed_image.size(dim=3)), requires_grad=True)
        self.optimizer = SGD([self.adversarial_noise], lr=opt_lr)
        # generate adversarial noise
        noise = self.generate(preprocessed_image, original_class, target_class, max_iter).squeeze(0).detach().numpy()
        noise = noise.reshape(image.size[1], image.size[0], 3)
        output = np.array(image) + noise*255
        # turn output int into uint8
        output = np.array(output, dtype=np.uint8)
        output = Image.fromarray(output)
        return output
    

