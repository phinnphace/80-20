# utils.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def get_data_loaders(config):
    """Create train, validation, and test data loaders with augmentation."""
    
    # Training transformations with augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=config.rotation_degrees,
            translate=(config.translation_pixels/28, config.translation_pixels/28)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Validation and test transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full MNIST dataset
    full_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split into train (50k) and validation (10k)
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Note: We need to apply the non-augmented transform to validation
    # But random_split creates subsets, so we need to reassign transform
    val_dataset.dataset.transform = test_transform
    
    # Test dataset
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Learning curves saved to {save_path}")

def plot_misclassified(model, test_loader, device, num_samples=25, save_path=None):
    """Plot misclassified examples."""
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Find misclassified
            incorrect = (predicted != labels)
            for i in range(len(images)):
                if incorrect[i]:
                    misclassified.append((images[i].cpu(), labels[i].cpu(), predicted[i].cpu()))
                    
            if len(misclassified) >= num_samples:
                break
    
    # Plot misclassified samples
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.ravel()
    
    for idx, (image, true_label, pred_label) in enumerate(misclassified[:num_samples]):
        # Denormalize for display
        image = image * 0.3081 + 0.1307
        axes[idx].imshow(image.squeeze(), cmap='gray')
        axes[idx].set_title(f'True: {true_label.item()}, Pred: {pred_label.item()}', fontsize=10)
        axes[idx].axis('off')
    
    # Hide any unused subplots
    for idx in range(len(misclassified[:num_samples]), 25):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✅ Misclassified examples saved to {save_path}")
    
    plt.show()