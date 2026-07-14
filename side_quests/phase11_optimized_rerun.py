#!/usr/bin/env python3
"""
Phase 11: Optimized Split Re-run
Condition A: 90/10 split (440 train, 49 val)
Condition B: 60/40 split (171 train, 115 val)
Re-runs: internal validation, confusion matrix assay, CASIA transfer, CalliBench transfer.
Tests whether optimized splits change the generalization trade-off.
"""
import os
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from codecs import decode
import secrets
import json

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
TARGET_SIZE = (100, 100)
DEVICE = torch.device("cpu")
FIXED_SEED = 42

SPLITS = {"condition_a": 0.9, "condition_b": 0.6}

# ============================================================
# MODEL
# ============================================================
class CharacterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 2)
    def forward(self, x, return_embedding=False):
        x = self.conv(x)
        embedding = x.view(x.size(0), -1)
        x = self.fc(embedding)
        if return_embedding:
            return x, embedding
        return x

class InMemoryDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

# ============================================================
# DATA LOADING
# ============================================================
def load_isolated_data():
    images, labels = [], []
    for f in sorted(os.listdir("data/isolated/target")):
        if f.endswith('.png'):
            img = Image.open(f"data/isolated/target/{f}").convert('L')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            images.append(t.unsqueeze(0)); labels.append(1)
    for ch in sorted(os.listdir("data/isolated/distractors")):
        ch_dir = f"data/isolated/distractors/{ch}"
        if os.path.isdir(ch_dir):
            for f in sorted(os.listdir(ch_dir)):
                if f.endswith('.png'):
                    img = Image.open(f"{ch_dir}/{f}").convert('L')
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)
                    t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                    images.append(t.unsqueeze(0)); labels.append(0)
    return images, labels

def load_contextual_data():
    images, labels = [], []
    for f in sorted(os.listdir("data/manual_fixes")):
        if f.startswith('line_') and f.endswith('.png'):
            img = Image.open(f"data/manual_fixes/{f}").convert('L')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            images.append(t.unsqueeze(0)); labels.append(1)
    for split in ['train', 'val']:
        other_dir = f"data/condition_b/{split}/other"
        if os.path.isdir(other_dir):
            for f in sorted(os.listdir(other_dir)):
                if f.endswith('.png'):
                    img = Image.open(f"{other_dir}/{f}").convert('L')
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)
                    t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                    images.append(t.unsqueeze(0)); labels.append(0)
    return images, labels

def load_casia_test():
    images, labels = [], []
    gnt_files = sorted([f for f in os.listdir("/mnt/c/Users/pmark/CASIA_data/HWDB1.1tst_gnt") if f.endswith('.gnt')])
    for gnt_file in gnt_files:
        filepath = f"/mnt/c/Users/pmark/CASIA_data/HWDB1.1tst_gnt/{gnt_file}"
        with open(filepath, 'rb') as f:
            while True:
                packed_length = f.read(4)
                if packed_length == b'': break
                length = struct.unpack("<I", packed_length)[0]
                raw_label = struct.unpack(">cc", f.read(2))
                width = struct.unpack("<H", f.read(2))[0]
                height = struct.unpack("<H", f.read(2))[0]
                raw_bytes = f.read(height * width)
                photo_bytes = struct.unpack("{}B".format(height * width), raw_bytes)
                label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
                if label == '打':
                    img_array = np.array(photo_bytes, dtype=np.uint8).reshape(height, width)
                    img = Image.fromarray(img_array, mode='L').resize(TARGET_SIZE, Image.LANCZOS)
                    t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                    images.append(t.unsqueeze(0)); labels.append(1)
    return images, labels

def load_callibench_test():
    images, labels = [], []
    if os.path.isdir("callibench_da"):
        for f in sorted(os.listdir("callibench_da")):
            if f.endswith('.png'):
                img = Image.open(f"callibench_da/{f}").convert('L')
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                images.append(t.unsqueeze(0)); labels.append(1)
    return images, labels

# ============================================================
# TRAINING
# ============================================================
def train_model(train_img, train_lbl, val_img, val_lbl):
    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    train_ds = InMemoryDataset(train_img, train_lbl)
    val_ds = InMemoryDataset(val_img, val_lbl)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = CharacterCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
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
        if acc > best_val_acc:
            best_val_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val_acc

def evaluate_model(model, images, labels):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch_imgs = torch.stack(images[i:i+BATCH_SIZE]).to(DEVICE)
            batch_lbls = torch.tensor(labels[i:i+BATCH_SIZE])
            outputs = model(batch_imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_lbls.tolist())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    return {'accuracy': acc, 'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'predictions': all_preds.tolist(), 'labels': all_labels.tolist()}

# ============================================================
# MAIN
# ============================================================
print("Phase 11: Optimized Split Re-run")
print(f"Condition A: 90/10 | Condition B: 60/40")
print(f"Fixed seed: {FIXED_SEED}\n")

# Load all data
print("Loading data...")
iso_images, iso_labels = load_isolated_data()
ctx_images, ctx_labels = load_contextual_data()
casia_images, casia_labels = load_casia_test()
calli_images, calli_labels = load_callibench_test()

print(f"  Isolated: {len(iso_images)} ({sum(iso_labels)} da)")
print(f"  Contextual: {len(ctx_images)} ({sum(ctx_labels)} da)")
print(f"  CASIA test: {len(casia_images)} da")
print(f"  CalliBench: {len(calli_images)} da\n")

all_results = {}

for condition, images, labels, split_ratio in [
    ("condition_a", iso_images, iso_labels, SPLITS["condition_a"]),
    ("condition_b", ctx_images, ctx_labels, SPLITS["condition_b"])
]:
    print(f"{'='*60}")
    print(f"  {condition} ({split_ratio:.0%}/{(1-split_ratio):.0%})")
    print(f"{'='*60}")
    
    # Split
    rng = secrets.SystemRandom()
    indices = list(range(len(images)))
    rng.shuffle(indices)
    split_idx = int(len(indices) * split_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_img = [images[i] for i in train_idx]
    train_lbl = [labels[i] for i in train_idx]
    val_img = [images[i] for i in val_idx]
    val_lbl = [labels[i] for i in val_idx]
    
    print(f"  Train: {len(train_img)}, Val: {len(val_img)}")
    
    # Train
    model, val_acc = train_model(train_img, train_lbl, val_img, val_lbl)
    print(f"  Best val accuracy: {val_acc:.4f}")
    
    # Evaluate all test sets
    val_results = evaluate_model(model, val_img, val_lbl)
    casia_results = evaluate_model(model, casia_images, casia_labels)
    calli_results = evaluate_model(model, calli_images, calli_labels)
    
    all_results[condition] = {
        'split': f"{split_ratio:.0%}/{(1-split_ratio):.0%}",
        'train_size': len(train_img),
        'val_size': len(val_img),
        'val_best_acc': val_acc,
        'val_results': val_results,
        'casia_transfer': casia_results,
        'callibench_transfer': calli_results
    }
    
    print(f"  Val:        acc={val_results['accuracy']:.4f}, TP={val_results['tp']}, TN={val_results['tn']}, FP={val_results['fp']}, FN={val_results['fn']}")
    print(f"  CASIA:      acc={casia_results['accuracy']:.4f}")
    print(f"  CalliBench: acc={calli_results['accuracy']:.4f}")

# ============================================================
# COMPARISON TO ORIGINAL 80/20 RESULTS
# ============================================================
print(f"\n{'='*60}")
print("  COMPARISON: Original 80/20 vs Optimized")
print(f"{'='*60}")

original = {
    'condition_a': {'val': 0.6122, 'casia': 0.3428, 'calli': 1.0000},
    'condition_b': {'val': 0.9138, 'casia': 0.2663, 'calli': 0.0526}
}

for condition in ['condition_a', 'condition_b']:
    print(f"\n{condition}:")
    print(f"  {'Metric':<20} {'Original 80/20':<16} {'Optimized':<16} {'Delta':<10}")
    for metric, label in [('val_best_acc', 'Val Accuracy'), 
                           ('casia_transfer', 'CASIA Transfer'),
                           ('callibench_transfer', 'CalliBench Transfer')]:
        orig = original[condition][metric.split('_')[0]] if metric != 'val_best_acc' else original[condition]['val']
        if metric == 'val_best_acc':
            new = all_results[condition]['val_best_acc']
        else:
            new = all_results[condition][metric]['accuracy']
        delta = new - orig
        print(f"  {label:<20} {orig:<16.4f} {new:<16.4f} {delta:+.4f}")

# Save
with open('models/optimized_rerun_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to models/optimized_rerun_results.json")
print("Phase 11 complete.")
