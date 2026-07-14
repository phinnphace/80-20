#!/usr/bin/env python3
"""
Phase 4: Standardize and Split
- Resize all images to 100x100
- Create train/val splits (80/20) for both conditions
- Extract random negative bigrams for Condition B using secrets.SystemRandom()
- Output: data/condition_a/ and data/condition_b/ with train/val subfolders
"""
import os
import struct
import shutil
import secrets
import numpy as np
from PIL import Image
from codecs import decode
from datasets import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_SIZE = (100, 100)
TRAIN_RATIO = 0.8
SEED = secrets.randbits(32)  # TRNG seed
NEGATIVE_COUNT = 135  # Match positive bigram count

# Paths
ISOLATED_TARGET_DIR = "data/isolated/target"
ISOLATED_DISTRACTOR_DIR = "data/isolated/distractors"
CONTEXTUAL_POSITIVE_DIR = "data/contextual/val"
OUTPUT_A = "data/condition_a"
OUTPUT_B = "data/condition_b"

# Create output directory structure
for condition in [OUTPUT_A, OUTPUT_B]:
    for split in ["train", "val"]:
        for label in ["da", "other"]:
            os.makedirs(os.path.join(condition, split, label), exist_ok=True)

print(f"Target size: {TARGET_SIZE}")
print(f"Train ratio: {TRAIN_RATIO}")
print(f"TRNG seed: {SEED}")
print(f"Negative bigram count: {NEGATIVE_COUNT}")

# ============================================================
# HELPER: Resize and save
# ============================================================
def resize_and_save(src_path, dst_path):
    """Resize a single image to TARGET_SIZE and save to dst_path."""
    img = Image.open(src_path)
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    img.save(dst_path)

# ============================================================
# CONDITION A: Isolated Characters
# ============================================================
print("\n=== CONDITION A: Isolated Characters ===")

# Gather all 打 (target) and distractor images
da_files = [os.path.join(ISOLATED_TARGET_DIR, f) for f in os.listdir(ISOLATED_TARGET_DIR) if f.endswith('.png')]
other_files = []
for ch_dir in os.listdir(ISOLATED_DISTRACTOR_DIR):
    ch_path = os.path.join(ISOLATED_DISTRACTOR_DIR, ch_dir)
    if os.path.isdir(ch_path):
        other_files.extend([os.path.join(ch_path, f) for f in os.listdir(ch_path) if f.endswith('.png')])

print(f"  Target (打): {len(da_files)}")
print(f"  Distractors: {len(other_files)}")

# Shuffle with TRNG
rng = secrets.SystemRandom()
rng.shuffle(da_files)
rng.shuffle(other_files)

# Split
da_split = int(len(da_files) * TRAIN_RATIO)
other_split = int(len(other_files) * TRAIN_RATIO)

da_train = da_files[:da_split]
da_val = da_files[da_split:]
other_train = other_files[:other_split]
other_val = other_files[other_split:]

print(f"  Train: {len(da_train)} da + {len(other_train)} other = {len(da_train) + len(other_train)}")
print(f"  Val:   {len(da_val)} da + {len(other_val)} other = {len(da_val) + len(other_val)}")

# Copy and resize
for i, src in enumerate(da_train):
    resize_and_save(src, os.path.join(OUTPUT_A, "train", "da", f"da_{i:04d}.png"))
for i, src in enumerate(da_val):
    resize_and_save(src, os.path.join(OUTPUT_A, "val", "da", f"da_{i:04d}.png"))
for i, src in enumerate(other_train):
    resize_and_save(src, os.path.join(OUTPUT_A, "train", "other", f"other_{i:04d}.png"))
for i, src in enumerate(other_val):
    resize_and_save(src, os.path.join(OUTPUT_A, "val", "other", f"other_{i:04d}.png"))

print("  Condition A done.")

# ============================================================
# CONDITION B: Contextual Bigrams
# ============================================================
print("\n=== CONDITION B: Contextual Bigrams ===")

# --- Positives: existing verified bigrams ---
positive_files = [os.path.join(CONTEXTUAL_POSITIVE_DIR, f) for f in os.listdir(CONTEXTUAL_POSITIVE_DIR) 
                  if f.endswith('.png') and not f.startswith('debug') and not f.startswith('full_line') and not f.startswith('fixed')]

# Include fixed crops if they exist
fixed_files = [os.path.join(CONTEXTUAL_POSITIVE_DIR, f) for f in os.listdir(CONTEXTUAL_POSITIVE_DIR) 
               if f.endswith('.png') and f.startswith('fixed')]
positive_files.extend(fixed_files)

# Remove duplicates just in case
positive_files = list(set(positive_files))
rng.shuffle(positive_files)

pos_split = int(len(positive_files) * TRAIN_RATIO)
pos_train = positive_files[:pos_split]
pos_val = positive_files[pos_split:]

print(f"  Positive bigrams: {len(positive_files)} total")
print(f"  Pos train: {len(pos_train)}, Pos val: {len(pos_val)}")

# Copy and resize positives
for i, src in enumerate(pos_train):
    resize_and_save(src, os.path.join(OUTPUT_B, "train", "da", f"da_{i:04d}.png"))
for i, src in enumerate(pos_val):
    resize_and_save(src, os.path.join(OUTPUT_B, "val", "da", f"da_{i:04d}.png"))

# --- Negatives: randomly sampled bigrams without 打 ---
print("\n  Extracting random negative bigrams...")

# Load the validation split
ds = load_dataset("Teklia/CASIA-HWDB2-line", split="validation")

# GNT parser isn't needed here; we extract from HuggingFace text lines
def extract_all_bigrams(dataset, exclude_char="打"):
    """Extract all possible 2-character windows that don't contain the excluded character."""
    all_bigrams = []
    for example in dataset:
        text = example["text"]
        img = example["image"]
        w, h = img.size
        text_len = len(text)
        
        if text_len < 2:
            continue
        
        char_width = w / text_len
        
        for i in range(text_len - 1):
            window = text[i:i+2]
            if exclude_char not in window:
                x_start = int(i * char_width)
                x_end = int((i + 2) * char_width)
                x_start = max(0, x_start)
                x_end = min(w, x_end)
                
                crop = img.crop((x_start, 0, x_end, h))
                all_bigrams.append((crop, window))
    
    return all_bigrams

print("  Scanning validation split for bigrams without 打...")
all_negative_bigrams = extract_all_bigrams(ds, exclude_char="打")
print(f"  Found {len(all_negative_bigrams)} candidate negative bigrams")

# Random sample using TRNG
sampled_negatives = rng.sample(all_negative_bigrams, min(NEGATIVE_COUNT, len(all_negative_bigrams)))

# Split negatives
neg_split = int(len(sampled_negatives) * TRAIN_RATIO)
neg_train = sampled_negatives[:neg_split]
neg_val = sampled_negatives[neg_split:]

print(f"  Neg train: {len(neg_train)}, Neg val: {len(neg_val)}")

# Save negatives
for i, (img, window) in enumerate(neg_train):
    img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
    img_resized.save(os.path.join(OUTPUT_B, "train", "other", f"other_{i:04d}.png"))

for i, (img, window) in enumerate(neg_val):
    img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
    img_resized.save(os.path.join(OUTPUT_B, "val", "other", f"other_{i:04d}.png"))

print("  Condition B done.")

# ============================================================
# VERIFICATION
# ============================================================
print("\n=== VERIFICATION ===")
for condition in [OUTPUT_A, OUTPUT_B]:
    print(f"\n{condition}/")
    for split in ["train", "val"]:
        da_count = len(os.listdir(os.path.join(condition, split, "da")))
        other_count = len(os.listdir(os.path.join(condition, split, "other")))
        print(f"  {split}/: da={da_count}, other={other_count}")

print("\nPhase 4 complete.")