#!/usr/bin/env python3
"""
Phase 15: Pretrained ResNet-18 Baseline
Fine-tunes a ResNet-18 (ImageNet pretrained) on both conditions.
Adds a third model to the comparison: transfer learning vs. from-scratch CNNs.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from PIL import Image
import secrets
import json

BATCH_SIZE = 32
EPOCHS = 15
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_SEED = 42
TARGET_SIZE = (224, 224)  # ResNet requires 224x224

print(f"Device: {DEVICE}")

# Dataset
class CharacterDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for class_name, label in [("da", 1), ("other", 0)]:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in sorted(os.listdir(class_dir)):
                    if fname.endswith('.png'):
                        self.samples.append((os.path.join(class_dir, fname), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, label

# Model
def build_resnet():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    return model.to(DEVICE)

# Training
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                _, preds = torch.max(model(imgs), 1)
                total += lbls.size(0)
                correct += (preds == lbls).sum().item()
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'models/resnet_best.pth')
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: val_acc={acc:.4f}")
    
    return best_acc

# Main
print("Phase 15: ResNet-18 Baseline")
results = {}

for condition, path in [("condition_a", "data/condition_a"), ("condition_b", "data/condition_b")]:
    print(f"\n{'='*50}")
    print(f"  {condition}")
    
    train_ds = CharacterDataset(os.path.join(path, "train"))
    val_ds = CharacterDataset(os.path.join(path, "val"))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    torch.manual_seed(FIXED_SEED)
    model = build_resnet()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    acc = train_model(model, train_loader, val_loader)
    results[condition] = acc
    print(f"  Best val acc: {acc:.4f}")

# Comparison
print(f"\n{'='*50}")
print("  THREE-MODEL COMPARISON")
print(f"{'='*50}")
print(f"  {'Model':<25} {'Cond A':<12} {'Cond B':<12}")
print(f"  {'-'*49}")
print(f"  {'CNN (Isolated)':<25} {'0.6122':<12} {'-':<12}")
print(f"  {'CNN (Contextual)':<25} {'-':<12} {'0.9138':<12}")
print(f"  {'ResNet-18 (Pretrained)':<25} {results['condition_a']:<12.4f} {results['condition_b']:<12.4f}")

with open('models/resnet_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nPhase 15 complete.")
