#!/usr/bin/env python3
"""
Phase 5: DataLoader Classes
Creates PyTorch Dataset and DataLoader objects for both experimental conditions.
Run once to verify data loads correctly, then import in training script.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32
IMAGE_SIZE = (100, 100)  # Must match Phase 4 output

# Standard transforms for both conditions
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image [0,255] to tensor [0.0, 1.0]
])

# ============================================================
# DATASET CLASS
# ============================================================
class CharacterDataset(Dataset):
    """
    Reads images from a directory structure:
        root/
            da/     -> label 1 (contains 打)
            other/  -> label 0 (does not contain 打)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Load positive samples (da/)
        da_dir = os.path.join(root_dir, "da")
        if os.path.isdir(da_dir):
            for fname in sorted(os.listdir(da_dir)):
                if fname.endswith('.png'):
                    self.samples.append((os.path.join(da_dir, fname), 1))
        
        # Load negative samples (other/)
        other_dir = os.path.join(root_dir, "other")
        if os.path.isdir(other_dir):
            for fname in sorted(os.listdir(other_dir)):
                if fname.endswith('.png'):
                    self.samples.append((os.path.join(other_dir, fname), 0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================
# CREATE DATALOADERS
# ============================================================
def get_dataloaders(data_root, batch_size=BATCH_SIZE):
    """
    Returns train_loader, val_loader for a given condition directory.
    data_root should contain train/ and val/ subdirectories.
    """
    train_dataset = CharacterDataset(
        os.path.join(data_root, "train"),
        transform=transform
    )
    val_dataset = CharacterDataset(
        os.path.join(data_root, "val"),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for WSL compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader

# ============================================================
# VERIFICATION
# ============================================================
if __name__ == "__main__":
    print("=== Phase 5: DataLoader Verification ===\n")
    
    for condition, path in [("Condition A (Isolated)", "data/condition_a"),
                             ("Condition B (Contextual)", "data/condition_b")]:
        print(f"{condition}:")
        train_loader, val_loader = get_dataloaders(path)
        
        # Check a batch
        images, labels = next(iter(train_loader))
        print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
        print(f"  Batch shape: {images.shape}")  # Should be [32, 1, 100, 100]
        print(f"  Label distribution in first batch: {labels.sum().item()}/{len(labels)} positives")
        
        # Check class balance
        train_pos = sum(1 for _, l in train_loader.dataset if l == 1)
        train_neg = sum(1 for _, l in train_loader.dataset if l == 0)
        val_pos = sum(1 for _, l in val_loader.dataset if l == 1)
        val_neg = sum(1 for _, l in val_loader.dataset if l == 0)
        print(f"  Train balance: {train_pos} pos / {train_neg} neg")
        print(f"  Val balance:   {val_pos} pos / {val_neg} neg")
        print()
    
    print("Phase 5 complete. Ready for training.")