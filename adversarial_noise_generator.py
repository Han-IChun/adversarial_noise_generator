from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
import requests
import PIL.Image as Image
from io import BytesIO
import numpy as np
import torchvision.transforms as transforms 

class AdversarialNoiseGenerator:
    def __init__(self, model):
        self.model = model 

    def generate(self, image, target_class, max_iter=5):
        for i in range(max_iter):
            # showing progress
            if i % 100 == 0:
                print(f"Iteration {i}")

            self.optimizer.zero_grad()
            perturbed_image = image + self.adversarial_noise
            output = self.model(perturbed_image)
            print(output)
            loss = -output[0, target_class]
            loss.backward()
            self.optimizer.step()
        return self.adversarial_noise

    def apply(self, image_url, target_class):        
        # image processing
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        preprocessed_image = preprocess(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        print("preprocessed image size: {}".format(preprocessed_image.size()))

        # initiale adversarial noise corresponding to the image size
        self.adversarial_noise = torch.zeros((preprocessed_image.size(dim=0), 
                                              preprocessed_image.size(dim=1), 
                                              preprocessed_image.size(dim=2), 
                                              preprocessed_image.size(dim=3)), requires_grad=True)
        print("noise size: {}".format(self.adversarial_noise.size()))
        self.optimizer = Adam([self.adversarial_noise], lr=0.01)

        # generate adversarial noise
        output = preprocessed_image + self.generate(preprocessed_image, target_class)
        output = np.squeeze(output)
        output = transforms.ToPILImage()(output)
        return output
    

