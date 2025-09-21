import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os


class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        return decoded


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#trained model
model = DiffusionModel()
model_path = r"    "# PATH TO MODEL
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

#single grayscale image
def load_single_grayscale_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(Image.open(image_path).convert('L')).unsqueeze(0)
    return image

test_image_path = r" "#PATH TO THE INPUT SAR IMAGE
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found at: {test_image_path}")

test_image = load_single_grayscale_image(test_image_path)

#To save 
output_dir = r"  "#PATH OF THE DIRECTORY TO SAVE COLOURISED IMAGES
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    test_image = test_image.to(device)
    output = model(test_image)
    output = output.squeeze(0)
    output_image_path = os.path.join(output_dir, os.path.basename(test_image_path).replace('.jpg', '_colorized.jpg'))
    save_image(output, output_image_path)
    print(f"Colorized image saved to {output_image_path}")

print("Inference completed and colorized image saved.")

