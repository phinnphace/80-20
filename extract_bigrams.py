"""
Phase 2: Bigram Extraction
Extracts 2-character windows around 打 from CASIA-HWDB2 text lines.
"""
import os
from datasets import load_dataset
from PIL import Image
import numpy as np

# Configuration
TARGET_CHAR = "打"
TARGET_SIZE = (128, 128)  # square output for CNN
OUTPUT_DIR = "/home/pmark/character_context_project/data/contextual/val"
SPLIT = "validation"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print(f"Loading {SPLIT} split...")
ds = load_dataset("Teklia/CASIA-HWDB2-line", split=SPLIT)
target_lines = ds.filter(lambda x: TARGET_CHAR in x["text"])
print(f"Found {len(target_lines)} lines containing {TARGET_CHAR}")

# Process each line
extracted = 0
skipped = 0
edge_cases = 0

for idx, example in enumerate(target_lines):
    image = example["image"]
    text = example["text"]
    img_width, img_height = image.size
    text_len = len(text)
    
    # Find all occurrences of 打 (there might be multiple)
    positions = [i for i, c in enumerate(text) if c == TARGET_CHAR]
    
    for pos in positions:
        # Determine bigram window
        if pos == 0:
            # At start: take positions [0, 2]
            start_pos = 0
            end_pos = 2
            edge_cases += 1
        elif pos == text_len - 1:
            # At end: take positions [pos-1, pos+1]
            start_pos = pos - 1
            end_pos = pos + 1
            edge_cases += 1
        else:
            # Middle: take [pos, pos+2] (target + right neighbor)
            start_pos = pos
            end_pos = pos + 2
        
        # Calculate pixel boundaries
        char_width = img_width / text_len
        x_start = int(start_pos * char_width)
        x_end = int(end_pos * char_width)
        
        # Ensure bounds are within image
        x_start = max(0, x_start)
        x_end = min(img_width, x_end)
        
        # Crop the bigram region
        crop = image.crop((x_start, 0, x_end, img_height))
        
        # Resize to target dimensions
        crop_resized = crop.resize(TARGET_SIZE, Image.LANCZOS)
        
        # Save
        neighbor = text[end_pos - 1] if end_pos - 1 != pos else text[start_pos]
        context_label = text[start_pos:end_pos]
        filename = f"{TARGET_CHAR}_with_{context_label.replace(TARGET_CHAR, '')}_{idx}_{pos}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        crop_resized.save(filepath)
        
        extracted += 1
    
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(target_lines)} lines, {extracted} bigrams saved...")

print(f"\nDone! Extracted {extracted} bigram images.")
print(f"Edge cases (character at start/end): {edge_cases}")
print(f"Skipped: {skipped}")
print(f"Output: {OUTPUT_DIR}")
