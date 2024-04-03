from adversarial_noise_generator import AdversarialNoiseGenerator
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
adv_generator = AdversarialNoiseGenerator(model)
# test with a panda, 388
# target class lion, 291
image_url = "https://i.natgeofe.com/n/36daf2b7-a4f5-4f43-8f5d-4d4eb32871aa/naturepl_01679242_3x2.jpg?w=1436&h=958" 
altered_image = adv_generator.apply(image_url, 291)

# Varify if noise is effective
preprocess = weights.transforms()
preprocessed_image = preprocess(altered_image)
preprocessed_image = preprocessed_image.unsqueeze(0)
output = model(preprocessed_image).squeeze().softmax(dim=0)
print(output.argmax().item()) # should be 291


