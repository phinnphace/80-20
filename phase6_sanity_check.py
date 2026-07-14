#!/usr/bin/env python3
"""
Phase 6: Data Sanity Check (corrected thresholds)
"""
import os
import torch
from PIL import Image
import numpy as np
import secrets

rng = secrets.SystemRandom()

CONDITION_A = "data/condition_a"
CONDITION_B = "data/condition_b"
SAMPLES_PER_CLASS = 5

def image_to_ascii(img_tensor, width=40):
    img = img_tensor.squeeze().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    chars = " .:-=+*#%@"
    ascii_img = ""
    h, w = img.shape
    step_h = max(1, h // (width // 2))
    step_w = max(1, w // width)
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            pixel = img[y, x]
            char_idx = int(pixel * (len(chars) - 1))
            ascii_img += chars[char_idx]
        ascii_img += "\n"
    return ascii_img

def get_pixels(img):
    try:
        data = list(img.get_flattened_data())
    except AttributeError:
        data = list(img.getdata())
    if data and isinstance(data[0], tuple):
        data = [p[0] for p in data]
    return data

def check_condition(name, root_dir):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for split in ["train", "val"]:
        split_dir = os.path.join(root_dir, split)
        for label, class_name in [("da", "POSITIVE (contains 打)"), ("other", "NEGATIVE (no 打)")]:
            class_dir = os.path.join(split_dir, label)
            if not os.path.isdir(class_dir):
                continue
            files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            if not files:
                continue
            sample_files = rng.sample(files, min(SAMPLES_PER_CLASS, len(files)))
            print(f"\n  [{split}] {class_name}")
            print(f"  Total images: {len(files)}")
            for fname in sample_files:
                filepath = os.path.join(class_dir, fname)
                img = Image.open(filepath)
                assert img.size == (100, 100), f"Wrong size: {img.size}"
                assert img.mode == "L", f"Wrong mode: {img.mode}"
                pixels = get_pixels(img)
                dark_150 = sum(1 for p in pixels if p < 150)
                ratio_150 = dark_150 / len(pixels)
                dark_100 = sum(1 for p in pixels if p < 100)
                ratio_100 = dark_100 / len(pixels)
                
                if ratio_150 < 0.001 and max(pixels) == 255:
                    status = "BLANK"
                elif ratio_100 < 0.005:
                    status = "LIGHT"
                else:
                    status = "OK"
                
                print(f"\n  {fname} [{status}] ink<150={ratio_150:.4f} ink<100={ratio_100:.4f}")
                img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                print(image_to_ascii(img_tensor, width=40))
                print("-" * 40)

print("Phase 6: Data Sanity Check (corrected thresholds)\n")

check_condition("Condition A (Isolated)", CONDITION_A)
check_condition("Condition B (Contextual)", CONDITION_B)

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")

for name, path in [("Condition A", CONDITION_A), ("Condition B", CONDITION_B)]:
    total = 0
    blank = 0
    light = 0
    for split in ["train", "val"]:
        for label in ["da", "other"]:
            class_dir = os.path.join(path, split, label)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.endswith('.png'):
                        total += 1
                        img = Image.open(os.path.join(class_dir, fname))
                        pixels = get_pixels(img)
                        dark_150 = sum(1 for p in pixels if p < 150)
                        ratio_150 = dark_150 / len(pixels)
                        dark_100 = sum(1 for p in pixels if p < 100)
                        ratio_100 = dark_100 / len(pixels)
                        if ratio_150 < 0.001 and max(pixels) == 255:
                            blank += 1
                        elif ratio_100 < 0.005:
                            light += 1
    print(f"\n{name}:")
    print(f"  Total images: {total}")
    print(f"  Truly blank: {blank}")
    print(f"  Light but real: {light}")
    print(f"  OK: {total - blank - light}")

print("\nPhase 6 complete.")