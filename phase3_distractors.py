#!/usr/bin/env python3
"""
Phase 3: Extract and save distractor character samples from CASIA-HWDB1.1 training set.
Run directly from terminal: python3 phase3_distractors.py
"""
import struct
import os
import numpy as np
from PIL import Image
from codecs import decode

# Paths
TRAIN_GNT_DIR = "/mnt/c/Users/pmark/CASIA_data/HWDB1.1trn_gnt"
OUTPUT_DISTRACTORS = "data/isolated/distractors"
TARGET_SIZE = (64, 64)

# Characters to collect (visually similar to 打, all share 扌 radical)
SIMILAR_CHARS = ["扔", "扛", "扫", "托", "扣"]
MAX_PER = 50

# Create output directories
for ch in SIMILAR_CHARS:
    os.makedirs(os.path.join(OUTPUT_DISTRACTORS, ch), exist_ok=True)

# GNT parser
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

# Scan and save
gnt_files = sorted([f for f in os.listdir(TRAIN_GNT_DIR) if f.endswith('.gnt')])
counts = {ch: 0 for ch in SIMILAR_CHARS}

print(f"Scanning {len(gnt_files)} files...")
for i, gnt_file in enumerate(gnt_files):
    if i % 30 == 0:
        print(f"  File {i}/{len(gnt_files)}")
    
    filepath = os.path.join(TRAIN_GNT_DIR, gnt_file)
    samples = parse_gnt_file(filepath)
    
    for img, label in samples:
        if label in counts:
            counts[label] += 1
            if counts[label] <= MAX_PER:
                img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
                save_path = os.path.join(OUTPUT_DISTRACTORS, label, f"{label}_{counts[label]:04d}.png")
                img_resized.save(save_path)

# Report
print("\nDone!")
for ch in SIMILAR_CHARS:
    folder = os.path.join(OUTPUT_DISTRACTORS, ch)
    saved = len(os.listdir(folder)) if os.path.isdir(folder) else 0
    print(f"  {ch}: {counts[ch]} found, {saved} saved to {folder}")

total = sum(len(os.listdir(os.path.join(OUTPUT_DISTRACTORS, ch))) for ch in SIMILAR_CHARS)
print(f"\nTotal files on disk: {total}")
