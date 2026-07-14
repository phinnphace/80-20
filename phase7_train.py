#!/usr/bin/env python3
"""
Phase 7: Model Training
Identical CNN trained on Condition A (Isolated) and Condition B (Contextual).
Outputs: trained models, training histories, and embedding extracts for assay.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import secrets
from collections import defaultdict
import json

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
IMAGE_SIZE = (100, 100)
SEED = secrets.randbits(32)

torch.manual_seed(SEED)
np.random.seed(SEED % 2**32)

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
print(f"Seed: {SEED}")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
# ============================================================
# DATASET CLASS
# ============================================================
class CharacterDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for class_name, label in [("da", 1), ("other", 0)]:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in sorted(os.listdir(class_dir)):
                    if fname.endswith('.png'):
                        self.samples.append((os.path.join(class_dir, fname), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        image = image.unsqueeze(0)  # Add channel dimension: [1, 100, 100]
        return image, label

# ============================================================
# CNN MODEL
# ============================================================
class CharacterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [batch, 1, 100, 100]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # -> [32, 100, 100]
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> [32, 50, 50]
            
            nn.Conv2d(32, 64, 3, padding=1),   # -> [64, 50, 50]
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> [64, 25, 25]
            
            nn.Conv2d(64, 128, 3, padding=1),  # -> [128, 25, 25]
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> [128, 12, 12]
            
            nn.AdaptiveAvgPool2d((1, 1))       # -> [128, 1, 1]
        )
        self.fc = nn.Linear(128, 2)  # Binary classification
    
    def forward(self, x, return_embedding=False):
        x = self.conv(x)
        embedding = x.view(x.size(0), -1)  # [batch, 128]
        x = self.fc(embedding)
        if return_embedding:
            return x, embedding
        return x

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, condition_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = defaultdict(list)
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        # Track
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/{condition_name}_best.pth")
    
    # Save final model and history
    torch.save(model.state_dict(), f"models/{condition_name}_final.pth")
    with open(f"models/{condition_name}_history.json", 'w') as f:
        json.dump(history, f)
    
    print(f"  Best val acc: {best_val_acc:.4f}")
    return model, history

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    for condition, data_path in [
        ("condition_a", "data/condition_a"),
        ("condition_b", "data/condition_b")
    ]:
        print(f"\n{'='*60}")
        print(f"  Training: {condition}")
        print(f"{'='*60}")
        
        # Data
        train_dataset = CharacterDataset(os.path.join(data_path, "train"))
        val_dataset = CharacterDataset(os.path.join(data_path, "val"))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        
        # Model
        model = CharacterCNN()
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        model, history = train_model(model, train_loader, val_loader, condition)
    
    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"  Models saved to: models/")
    print(f"{'='*60}")
