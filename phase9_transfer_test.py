#!/usr/bin/env python3
"""
Phase 9: Transfer Test
Tests both models on unseen isolated characters from the CASIA test set.
Compares whether contextual training transfers to better character recognition.
"""
import os
import struct
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from codecs import decode
import json

# ============================================================
# CONFIGURATION
# ============================================================
TEST_GNT_DIR = "/mnt/c/Users/pmark/CASIA_data/HWDB1.1tst_gnt"
TARGET_SIZE = (100, 100)
DEVICE = torch.device("cpu")
MODEL_DIR = "models"
BATCH_SIZE = 32

TARGET_CHAR = "打"
DISTRACTOR_CHARS = ["扔", "扛", "扫", "托", "扣"]

# ============================================================
# GNT PARSER
# ============================================================
def parse_gnt_file(filepath):
    results = []
    with open(filepath, 'rb') as f:
        while True:
            packed_length = f.read(4)
            if packed_length == b'':
                break
            length = struct.unpack("<I", packed_length)[0]
            raw_label = struct.unpack(">cc", f.read(2))
            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]
            raw_bytes = f.read(height * width)
            photo_bytes = struct.unpack("{}B".format(height * width), raw_bytes)
            label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
            img_array = np.array(photo_bytes, dtype=np.uint8).reshape(height, width)
            image = Image.fromarray(img_array, mode='L')
            results.append((image, label))
    return results

# ============================================================
# MODEL (same as training)
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

# ============================================================
# BUILD TEST DATASET
# ============================================================
print("Building transfer test set from CASIA test data...")

test_samples = []
gnt_files = sorted([f for f in os.listdir(TEST_GNT_DIR) if f.endswith('.gnt')])

for gnt_file in gnt_files:
    filepath = os.path.join(TEST_GNT_DIR, gnt_file)
    samples = parse_gnt_file(filepath)
    for img, label in samples:
        if label == TARGET_CHAR:
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            test_samples.append((img, 1, label))
        elif label in DISTRACTOR_CHARS:
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            test_samples.append((img, 0, label))

print(f"  Found {sum(1 for _, l, _ in test_samples if l == 1)} 打 samples")
print(f"  Found {sum(1 for _, l, _ in test_samples if l == 0)} distractor samples")
print(f"  Total transfer test set: {len(test_samples)}")

# ============================================================
# TEST DATASET CLASS
# ============================================================
class TransferTestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, label, char = self.samples[idx]
        tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        tensor = tensor.unsqueeze(0)
        return tensor, label, char

# ============================================================
# EVALUATE
# ============================================================
dataset = TransferTestDataset(test_samples)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

results = {}

for condition in ["condition_a", "condition_b"]:
    print(f"\nTesting: {condition}")
    
    model = CharacterCNN()
    model_path = os.path.join(MODEL_DIR, f"{condition}_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_chars = []
    errors = []
    
    with torch.no_grad():
        for images, labels, chars in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_chars.extend(chars)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class accuracy
    da_mask = np.array(all_labels) == 1
    other_mask = ~da_mask
    
    da_acc = (all_preds[da_mask] == 1).mean() if da_mask.sum() > 0 else 0
    other_acc = (all_preds[other_mask] == 0).mean() if other_mask.sum() > 0 else 0
    
    # Confusion matrix
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    
    # Collect errors by character
    error_by_char = {}
    for pred, label, char in zip(all_preds, all_labels, all_chars):
        if pred != label:
            key = f"{char} (true={label}, pred={pred})"
            error_by_char[key] = error_by_char.get(key, 0) + 1
    
    results[condition] = {
        'accuracy': float(accuracy),
        'da_accuracy': float(da_acc),
        'other_accuracy': float(other_acc),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'total_samples': len(all_labels),
        'total_errors': int((all_preds != all_labels).sum()),
        'error_by_char': error_by_char
    }
    
    print(f"  Overall accuracy: {accuracy:.4f}")
    print(f"  打 accuracy: {da_acc:.4f}")
    print(f"  Distractor accuracy: {other_acc:.4f}")
    print(f"  Errors: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ============================================================
# COMPARISON
# ============================================================
print(f"\n{'='*60}")
print("  TRANSFER TEST COMPARISON")
print(f"{'='*60}")

for metric in ['accuracy', 'da_accuracy', 'other_accuracy']:
    a_val = results['condition_a'][metric]
    b_val = results['condition_b'][metric]
    delta = b_val - a_val
    direction = "B better" if delta > 0 else "A better" if delta < 0 else "equal"
    print(f"  {metric}: A={a_val:.4f}, B={b_val:.4f}, delta={delta:+.4f} ({direction})")

print(f"\n  Condition A errors: {results['condition_a']['total_errors']}")
print(f"  Condition B errors: {results['condition_b']['total_errors']}")

if results['condition_b']['accuracy'] > results['condition_a']['accuracy']:
    print("\n  CONCLUSION: Contextual training transferred to better isolated character recognition.")
    print("  The model trained on bigrams learned more robust visual features.")
else:
    print("\n  CONCLUSION: Contextual training did NOT transfer to isolated character recognition.")
    print("  The benefit may be specific to the contextual task.")

# Save
with open('models/transfer_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to models/transfer_test_results.json")
print("Phase 9 complete.")
