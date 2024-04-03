from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
import requests
import PIL.Image as Image
from io import BytesIO
import numpy as np
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
class AdversarialNoiseGenerator:
    def __init__(self, model):
        self.model = model 

    def generate(self, image, target_class, max_iter=1000):
        for i in range(max_iter):
            self.optimizer.zero_grad()
            perturbed_image = image + self.adversarial_noise
            output = self.model(perturbed_image)
            loss = CrossEntropyLoss()
            loss_origin = -loss(output, torch.tensor([388]))
            loss_target = loss(output, torch.tensor([target_class]))
            
            # showing progress
            if i % 100 == 0:
                print(f"Iteration {i}")
                print("loss_origin: {}, loss_target: {}".format(loss_origin, loss_target))

            loss = loss_origin+loss_target
            loss.backward()
            self.optimizer.step()
        return self.adversarial_noise

    def apply(self, image_url, target_class):        
        # image processing
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        resize_ratio = 232 / min(image.size)
        image = image.resize((int(image.size[0]*resize_ratio),int(image.size[1]*resize_ratio) ), Image.BILINEAR)
        image = transforms.CenterCrop(224)(image)
        # print("image size: {}".format(image.size))
        # plt.imshow(image)
        # plt.show()
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        preprocessed_image = preprocess(image)
        # print("image: {}".format(np.array(preprocessed_image)[0]))
        
        preprocessed_image = preprocessed_image.unsqueeze(0)
        print("preprocessed image size: {}".format(preprocessed_image.size()))

        # initiale adversarial noise corresponding to the image size
        self.adversarial_noise = torch.zeros((preprocessed_image.size(dim=0), 
                                              preprocessed_image.size(dim=1), 
                                              preprocessed_image.size(dim=2), 
                                              preprocessed_image.size(dim=3)), requires_grad=True)
        self.optimizer = Adam([self.adversarial_noise], lr=1e-5)
        # generate adversarial noise
        noise = self.generate(preprocessed_image, target_class).squeeze(0).detach().numpy()
        noise = noise.reshape((224,224, 3))
        output = np.array(image) + noise*255
        # turn output int into uint8
        output = np.array(output, dtype=np.uint8)
        output = Image.fromarray(output)
        # plt.imshow(output)
        # plt.show()
        return output
    

