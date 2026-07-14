#!/usr/bin/env python3
"""
Phase 13: Full Split × Transfer Matrix
Trains missing split/condition combinations and evaluates on CASIA and CalliBench.
Fills the complete matrix for the demo.
"""
import os, struct, json, secrets
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from codecs import decode

BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
TARGET_SIZE = (100, 100)
DEVICE = torch.device("cpu")
FIXED_SEED = 42

# Missing combinations: (condition, split, data_loader_fn)
MISSING = [
    ("condition_a", 0.5, "iso"),
    ("condition_a", 0.6, "iso"),
    ("condition_a", 0.7, "iso"),
    ("condition_b", 0.5, "ctx"),
    ("condition_b", 0.7, "ctx"),
    ("condition_b", 0.9, "ctx"),
]

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
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

class InMemoryDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

def load_iso():
    imgs, lbls = [], []
    for f in sorted(os.listdir("data/isolated/target")):
        if f.endswith('.png'):
            img = Image.open(f"data/isolated/target/{f}").convert('L').resize(TARGET_SIZE, Image.LANCZOS)
            imgs.append(torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)/255.0)
            lbls.append(1)
    for ch in sorted(os.listdir("data/isolated/distractors")):
        d = f"data/isolated/distractors/{ch}"
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.png'):
                    img = Image.open(f"{d}/{f}").convert('L').resize(TARGET_SIZE, Image.LANCZOS)
                    imgs.append(torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)/255.0)
                    lbls.append(0)
    return imgs, lbls

def load_ctx():
    imgs, lbls = [], []
    for f in sorted(os.listdir("data/manual_fixes")):
        if f.startswith('line_') and f.endswith('.png'):
            img = Image.open(f"data/manual_fixes/{f}").convert('L').resize(TARGET_SIZE, Image.LANCZOS)
            imgs.append(torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)/255.0)
            lbls.append(1)
    for split in ['train', 'val']:
        d = f"data/condition_b/{split}/other"
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.png'):
                    img = Image.open(f"{d}/{f}").convert('L').resize(TARGET_SIZE, Image.LANCZOS)
                    imgs.append(torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)/255.0)
                    lbls.append(0)
    return imgs, lbls

def load_casia():
    imgs, lbls = [], []
    d = "/mnt/c/Users/pmark/CASIA_data/HWDB1.1tst_gnt"
    for gf in sorted([f for f in os.listdir(d) if f.endswith('.gnt')]):
        with open(f"{d}/{gf}", 'rb') as f:
            while True:
                pl = f.read(4)
                if pl == b'': break
                length = struct.unpack("<I", pl)[0]
                rl = struct.unpack(">cc", f.read(2))
                w = struct.unpack("<H", f.read(2))[0]
                h = struct.unpack("<H", f.read(2))[0]
                rb = f.read(h*w)
                pb = struct.unpack(f"{h*w}B", rb)
                label = decode(rl[0]+rl[1], encoding="gb2312")
                if label == '打':
                    arr = np.array(pb, dtype=np.uint8).reshape(h, w)
                    img = Image.fromarray(arr, mode='L').resize(TARGET_SIZE, Image.LANCZOS)
                    imgs.append(torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)/255.0)
                    lbls.append(1)
    return imgs, lbls

def load_calli():
    imgs, lbls = [], []
    d = "callibench_da"
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.png'):
                img = Image.open(f"{d}/{f}").convert('L').resize(TARGET_SIZE, Image.LANCZOS)
                imgs.append(torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)/255.0)
                lbls.append(1)
    return imgs, lbls

def train_and_eval(train_img, train_lbl, val_img, val_lbl):
    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    train_ds = InMemoryDataset(train_img, train_lbl)
    val_ds = InMemoryDataset(val_img, val_lbl)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = CharacterCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    for _ in range(EPOCHS):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            crit(model(imgs), lbls).backward()
            opt.step()
        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                _, p = torch.max(model(imgs), 1)
                t += lbls.size(0)
                c += (p == lbls).sum().item()
        acc = c/t
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc

def test(model, images, labels):
    model.eval()
    c = 0
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch = torch.stack(images[i:i+BATCH_SIZE]).to(DEVICE)
            lbls = torch.tensor(labels[i:i+BATCH_SIZE])
            _, p = torch.max(model(batch), 1)
            c += (p == lbls).sum().item()
    return c/len(labels) if labels else 0

# Main
print("Phase 13: Full Split × Transfer Matrix")
print(f"Missing combinations: {len(MISSING)}")

iso_imgs, iso_lbls = load_iso()
ctx_imgs, ctx_lbls = load_ctx()
casia_imgs, casia_lbls = load_casia()
calli_imgs, calli_lbls = load_calli()

# Load existing results
with open('models/split_window.json') as f:
    split_a = json.load(f)
with open('models/split_window_contextual.json') as f:
    split_b = json.load(f)

results = {}
# Add existing transfer results
results['condition_a_0.8'] = {'val': split_a['train_80_val_19']['val_accuracy'], 'casia': 0.3428, 'calli': 1.0000}
results['condition_a_0.9'] = {'val': 0.6939, 'casia': 0.6949, 'calli': 0.9474}
results['condition_b_0.6'] = {'val': 0.9565, 'casia': 0.9831, 'calli': 0.0526}
results['condition_b_0.8'] = {'val': split_b['train_80_val_19']['val_accuracy'], 'casia': 0.2663, 'calli': 0.0526}

for cond, split, dtype in MISSING:
    name = f"{cond}_{split}"
    print(f"\nTraining: {name}")
    
    images = iso_imgs if dtype == "iso" else ctx_imgs
    labels = iso_lbls if dtype == "iso" else ctx_lbls
    
    rng = secrets.SystemRandom()
    idx = list(range(len(images)))
    rng.shuffle(idx)
    si = int(len(idx) * split)
    train_idx, val_idx = idx[:si], idx[si:]
    
    train_img = [images[i] for i in train_idx]
    train_lbl = [labels[i] for i in train_idx]
    val_img = [images[i] for i in val_idx]
    val_lbl = [labels[i] for i in val_idx]
    
    model, val_acc = train_and_eval(train_img, train_lbl, val_img, val_lbl)
    casia_acc = test(model, casia_imgs, casia_lbls)
    calli_acc = test(model, calli_imgs, calli_lbls)
    
    results[name] = {'val': val_acc, 'casia': casia_acc, 'calli': calli_acc}
    print(f"  Val: {val_acc:.4f}, CASIA: {casia_acc:.4f}, CalliBench: {calli_acc:.4f}")

# Add existing internal-only results
for split_name, data in split_a.items():
    s = float(split_name.split('_')[1]) / 100
    key = f"condition_a_{s}"
    if key not in results:
        results[key] = {'val': data['val_accuracy'], 'casia': None, 'calli': None}

for split_name, data in split_b.items():
    s = float(split_name.split('_')[1]) / 100
    key = f"condition_b_{s}"
    if key not in results:
        results[key] = {'val': data['val_accuracy'], 'casia': None, 'calli': None}

# Print full matrix
print(f"\n{'='*80}")
print("  FULL SPLIT × TRANSFER MATRIX")
print(f"{'='*80}")
splits = [0.5, 0.6, 0.7, 0.8, 0.9]
print(f"  {'Split':<10} {'A Val':<10} {'A CASIA':<10} {'A Calli':<10} {'B Val':<10} {'B CASIA':<10} {'B Calli':<10}")
print(f"  {'-'*70}")
for s in splits:
    ka = f"condition_a_{s}"
    kb = f"condition_b_{s}"
    ra = results.get(ka, {})
    rb = results.get(kb, {})
    print(f"  {f'{s:.0%}/{(1-s):.0%}':<10} "
          f"{ra.get('val', '-'):<10} "
          f"{ra.get('casia', '-'):<10} "
          f"{ra.get('calli', '-'):<10} "
          f"{rb.get('val', '-'):<10} "
          f"{rb.get('casia', '-'):<10} "
          f"{rb.get('calli', '-'):<10}")

with open('models/full_matrix.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to models/full_matrix.json")
