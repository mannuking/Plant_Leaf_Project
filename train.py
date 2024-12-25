import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image

# Explicitly set the start method before defining anything else
mp.set_start_method('spawn', force=True)


# Define the paths of the dataset
DATA_DIR = 'dataset/Plant_split' # Path to the data set
MODEL_PATH = 'models/custom_vision_model.pth' # Path to save the trained model


# Check if models directory exists and create it if not
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
# Set the device as soon as possible and remove it from the dataset load procedure.

# Custom Dataset class to handle label mapping explicitly
class AppleLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None, split = "train"):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir)) # Get the order by listing directories and sort them
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(self.class_to_idx[class_name])

        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor) # convert to tensor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx] # load already created labels
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



if __name__ == '__main__':
    mp.freeze_support()
    # Setup GPU utilization if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Define transforms for data augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=30), # Remove this for now to simplify
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)), # Remove this for now to simplify
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Remove this for now to simplify
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Load your dataset from the local directory using the custom class
    train_data = AppleLeafDataset(DATA_DIR, transform=transform_train, split = 'train')
    val_data = AppleLeafDataset(DATA_DIR, transform=transform_val, split='val')

    # Create DataLoaders
    batch_size = 64 # Experiment with different batch sizes, 32, 64 or higher if GPU allows
    num_workers = 4 # Experiment with different values, based on the number of CPU cores.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Model Definition (Custom CNN)
    num_classes = len(train_data.classes)
    class CustomCNN(nn.Module):
        def __init__(self, num_classes):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.relu4 = nn.ReLU()
            self.pool4 = nn.MaxPool2d(kernel_size=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256 * 14 * 14, 256)
            self.relu5 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
            x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
            x = self.flatten(x)
            x = self.relu5(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = CustomCNN(num_classes).to(device)

    # Optimizer and Loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)

    # Training Loop
    epochs = 20
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validation loop after each training epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")

    print("Training Complete")

    # Save the trained model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model weights saved to {MODEL_PATH}")
    # Generate a plot of traning and validation losses for each epoch
    epochs_list = np.linspace(1, epochs, epochs)  # Create range of epochs for the x-axis.

    # Plot training and validation loss for every epoch
    plt.figure(figsize=(10,5))
    plt.plot(epochs_list, train_losses, label="Training Loss", color="blue")
    plt.plot(epochs_list, val_losses, label='Validation Loss', color="red", linestyle='--')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
