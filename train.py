import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image


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


def get_image_paths(root_dir):
    image_paths = []
    if not os.path.exists(root_dir):
        print(f"Warning: Directory does not exist - {root_dir}")
        return image_paths
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, file))
    return image_paths

#Dataset
class SARColorizationDataset(Dataset):
    def __init__(self, grayscale_dir, color_dir, transform=None, subset_size=None):
        self.grayscale_paths = sorted(get_image_paths(grayscale_dir))
        self.color_paths = sorted(get_image_paths(color_dir))
        self.transform = transform
        print(f"Grayscale images found: {len(self.grayscale_paths)}")
        print(f"Color images found: {len(self.color_paths)}")
        if len(self.grayscale_paths) == 0 or len(self.color_paths) == 0:
            raise ValueError("No images found in the specified directories!")
        if len(self.grayscale_paths) != len(self.color_paths):
            min_len = min(len(self.grayscale_paths), len(self.color_paths))
            self.grayscale_paths = self.grayscale_paths[:min_len]
            self.color_paths = self.color_paths[:min_len]
            print(f"Using {min_len} image pairs")
        if subset_size:
            subset_size = min(subset_size, len(self.grayscale_paths))
            self.grayscale_paths = self.grayscale_paths[:subset_size]
            self.color_paths = self.color_paths[:subset_size]
            print(f"Using subset of {subset_size} images")

    def __len__(self):
        return len(self.grayscale_paths)

    def __getitem__(self, idx):
        grayscale_image = Image.open(self.grayscale_paths[idx]).convert('L')
        color_image = Image.open(self.color_paths[idx]).convert('RGB')
        if self.transform:
            grayscale_image = self.transform(grayscale_image)
            color_image = self.transform(color_image)
        return grayscale_image, color_image

# Training
def train_model(model, dataloader, num_epochs=10, save_interval=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % save_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                save_image(outputs[:4], f'output_epoch{epoch+1}_step{i+1}.png')
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), r'   ')#PATH TO MODEL
    print("Model saved as 'NEW__sar_colorization_model.pth'")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#Path to inputs
grayscale_dir = r"   "#PATH TO DIRECTORY(SAR IMAGES) 
color_dir = r"  "#PATH TO DIRECTORY(CORRESPONDING COLOURISED IMAGES)
print(f"Grayscale directory: {grayscale_dir}")
print(f"Color directory: {color_dir}")

try:
    subset_size = 3000
    dataset = SARColorizationDataset(grayscale_dir, color_dir, transform=transform, subset_size=subset_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    print(f"Dataset created with {len(dataset)} image pairs")
    model = DiffusionModel().to(device)
    print("Model created successfully")
    train_model(model, dataloader, num_epochs=10)
except Exception as e:
    print(f"Error: {e}")
    print("Please check your directory paths and ensure both folders contain images!")


