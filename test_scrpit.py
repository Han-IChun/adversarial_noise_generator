from adversarial_noise_generator import AdversarialNoiseGenerator
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

model = resnet50(weights=ResNet50_Weights)
model.eval()
adv_generator = AdversarialNoiseGenerator(model)
# test with a panda
# target class lion
image_url = "https://i.natgeofe.com/n/36daf2b7-a4f5-4f43-8f5d-4d4eb32871aa/naturepl_01679242_3x2.jpg?w=1436&h=958" 
adv_generator = adv_generator.apply(image_url, 291)
plt.imshow(adv_generator)
plt.show()