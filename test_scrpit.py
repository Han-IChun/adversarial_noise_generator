from adversarial_noise_generator import TargetedAdvNoiseGenerator
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
adv_generator = TargetedAdvNoiseGenerator(model, weights)
# test with a panda, 388
# test with a tabby, tabby cat, 281
# target class lion, 291
# target class Samoyede, 258
image_url = "https://i.natgeofe.com/n/36daf2b7-a4f5-4f43-8f5d-4d4eb32871aa/naturepl_01679242_3x2.jpg" # panda ?w=1436&h=958
# image_url = "https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg" # cat
altered_image = adv_generator.apply(image_url, 258, max_iter=500, opt_lr = 5e-4)

# Varify if noise is effective
preprocess = weights.transforms()
preprocessed_image = preprocess(altered_image)
preprocessed_image = preprocessed_image.unsqueeze(0)
output = model(preprocessed_image).squeeze().softmax(dim=0)
print("Altered class:{}".format(output.argmax().item())) # should the same as target class index
plt.imshow(altered_image)
plt.show()


