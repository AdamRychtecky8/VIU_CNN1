import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Constants
image_size = 400
num_gaussian_blobs = 30
sigma_values = [0.226 * (image_size / 15.04), 0.301 * (image_size / 15.04)]
contrast_values = [-0.109, -0.141, -0.172, 0.109, 0.141, 0.172]
batch_size = 32
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gaussian Functions
def gaussian_2d(x, y, sigma, amplitude=1):
    return amplitude * np.exp(-0.5 * ((x**2 / sigma**2) + (y**2 / sigma**2)))

def generate_gaussian_blob_positions(num_blobs, image_size, min_sigma, max_sigma, min_contrast, max_contrast):
    return [(np.random.randint(0, image_size),
             np.random.randint(0, image_size),
             np.random.uniform(min_sigma, max_sigma),
             np.random.uniform(min_contrast, max_contrast)) for _ in range(num_blobs)]

def generate_background(image_size, blobs):
    background = np.zeros((image_size, image_size))
    for x0, y0, sigma, contrast in blobs:
        x, y = np.meshgrid(np.arange(image_size) - x0, np.arange(image_size) - y0)
        background += gaussian_2d(x, y, sigma, amplitude=contrast)
    return background

def shift_blobs(blobs, shift_x, shift_y, image_size):
    return [((x0 + shift_x) % image_size, (y0 + shift_y) % image_size, sigma, contrast) for x0, y0, sigma, contrast in blobs]

def place_gaussian_target(image, sigma, contrast):
    x0, y0 = np.random.randint(0, image_size), np.random.randint(0, image_size)
    x, y = np.meshgrid(np.arange(image_size) - x0, np.arange(image_size) - y0)
    image += gaussian_2d(x, y, sigma, amplitude=contrast)
    return image, (x0 / image_size, y0 / image_size)

# PyTorch Dataset Class
class GaussianImageDataset(Dataset):
    def __init__(self, num_images, blobs):
        images, labels = [], []
        for _ in range(num_images):
            shift_x, shift_y = np.random.randint(-image_size, image_size, 2)
            background = generate_background(image_size, shift_blobs(blobs, shift_x, shift_y, image_size))
            image_with_target, target_position = place_gaussian_target(background, np.random.choice(sigma_values), np.random.choice(contrast_values))
            images.append(image_with_target)
            labels.append(target_position)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Standardize images
        self.images = (images - np.mean(images)) / np.std(images)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]).unsqueeze(0), torch.tensor(self.labels[idx])

# Generate dataset
print("Generating dataset...")
common_blobs = generate_gaussian_blob_positions(num_gaussian_blobs, image_size, 0.188 * (image_size / 15.04), 0.564 * (image_size / 15.04), -0.172, 0.172)
dataset = GaussianImageDataset(1000, common_blobs)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Initialize model
model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Training loop
print("Training model...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()

    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}, Training Loss: {epoch_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")

# Save model
torch.save(model.state_dict(), "cnn_model.pth")

# Prediction Visualization
def visualize_predictions(model, dataset, num_images=10):
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
    with torch.no_grad():
        for i in range(num_images):
            img, true_pos = dataset[i]
            img = img.unsqueeze(0).to(device)
            pred_pos = model(img).cpu().numpy()[0]

            axes[i].imshow(img.cpu().numpy().squeeze(), cmap='gray')
            axes[i].scatter(pred_pos[0] * 400, pred_pos[1] * 400, color='red', marker='x', label="Predicted")
            axes[i].scatter(true_pos[0] * 400, true_pos[1] * 400, color='blue', marker='o', label="Actual")
            axes[i].legend()
            axes[i].set_title(f"Predicted: {pred_pos}\nActual: {true_pos.numpy()}")
    plt.show()

# Show predictions
visualize_predictions(model, val_dataset, num_images=10)
