# File: train_pacs_wideresnet.py

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import wide_resnet50_2
from sklearn.metrics import accuracy_score
from albumentations import Compose, RandomBrightnessContrast, GaussianBlur, MotionBlur, CoarseDropout, ToGray, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

# Dataset directory
PACS_DIR = "./PACS"

# AugMax Transformations
class AugMaxTransform:
    def __init__(self):
        self.augmentations = Compose([
            Resize(224, 224),
            RandomBrightnessContrast(p=0.5),
            GaussianBlur(blur_limit=7, p=0.3),
            MotionBlur(p=0.3),
            CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
            ToGray(p=0.2),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __call__(self, image):
        if isinstance(image, torch.Tensor):  # Convert Tensor to NumPy
            image = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] < 10:  # Likely CHW format
                image = np.transpose(image, (1, 2, 0))  # Convert CHW -> HWC
        elif hasattr(image, "convert"):  # PIL image
            image = np.array(image)
        return self.augmentations(image=image)["image"]

# Dataset Loader
def load_pacs_data(domain, batch_size=64, is_train=True, use_augmax=False, corruption=None):
    transforms_list = [Resize(224, 224)]

    if corruption == "blur":
        transforms_list.append(GaussianBlur(blur_limit=(15, 25), p=1.0))  # Extreme blur
    elif corruption == "contrast":
        transforms_list.append(RandomBrightnessContrast(brightness_limit=1.0, contrast_limit=1.0, p=1.0))  # Extreme contrast
    elif corruption == "noise":
        transforms_list.append(CoarseDropout(max_holes=32, max_height=16, max_width=16, p=1.0))  # Heavy dropout (noise)

    transforms_list.extend([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    transform = Compose(transforms_list)

    def transform_wrapper(image):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        elif not isinstance(image, np.ndarray):
            image = np.array(image)
        return transform(image=image)["image"]

    dataset = torchvision.datasets.ImageFolder(
        os.path.join(PACS_DIR, domain),
        transform=lambda img: transform_wrapper(img)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing function
def test_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# Robustness Evaluation
def evaluate_robustness(model, corrupted_loaders, device):
    accs = []
    for corruption, loader in corrupted_loaders.items():
        print(f"Evaluating corruption: {corruption}")
        acc = test_model(model, loader, device)
        accs.append(acc)
        print(f"Corruption: {corruption}, Accuracy: {acc:.4f}")
    mean_corruption_error = 1 - np.mean(accs)
    return np.mean(accs), mean_corruption_error

# Main execution
if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0001

    # Load PACS Photo domain
    photo_loader = load_pacs_data("photo", batch_size, is_train=True, use_augmax=True)

    # Test Loaders for Standard Accuracy
    other_domains = ["art_painting", "cartoon", "sketch"]
    test_loaders = {domain: load_pacs_data(domain, batch_size, is_train=False) for domain in other_domains}

    # Corrupted Loaders for Robustness Evaluation
    corrupted_loaders = {
        "gaussian_noise": load_pacs_data("photo", batch_size, is_train=False, corruption="noise"),
        "motion_blur": load_pacs_data("photo", batch_size, is_train=False, corruption="blur"),
        "low_contrast": load_pacs_data("photo", batch_size, is_train=False, corruption="contrast"),
    }

    # Model setup (WideResNet)
    model = wide_resnet50_2(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)  # Adjust for 7 classes in PACS
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, photo_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

    # Robustness Evaluation
    robustness_acc, mCE = evaluate_robustness(model, corrupted_loaders, device)

    # Standard Accuracy
    standard_accs = []
    for domain, loader in test_loaders.items():
        acc = test_model(model, loader, device)
        standard_accs.append(acc)
        print(f"Domain: {domain}, Standard Accuracy: {acc:.4f}")

    # Results
    print(f"Mean Standard Accuracy: {np.mean(standard_accs):.4f}")
    print(f"Robustness Accuracy: {robustness_acc:.4f}")
    print(f"Mean Corruption Error: {mCE:.4f}")
