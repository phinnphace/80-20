"""
Manual crop adjustment for bad bigram extractions.
For each bad file, specify new proportional boundaries.
"""
from datasets import load_dataset
from PIL import Image
import os

TARGET_CHAR = "打"
TARGET_SIZE = (128, 128)
OUTPUT_DIR = "data/contextual/val"
SPLIT = "validation"

# Load dataset once
ds = load_dataset("Teklia/CASIA-HWDB2-line", split=SPLIT)
target = ds.filter(lambda x: TARGET_CHAR in x["text"])

# Manual adjustments: (line_idx, position, start_offset, end_offset)
# start_offset: shift left (-) or right (+) in characters
# end_offset: extend (+) or shrink (-) window on right, in characters
# Example: (27, 0, 0, +0.5) means line 27, position 0, same start, extend right by 0.5 char

fixes = [
    # Format: (line_idx, position, start_offset, end_offset)
    (27, 0, 0, +0.5),      # needs cropping on right
    (56, 28, 0, +0.5),     # needs cropping right
    (80, 19, -0.3, 0),     # needs cropping on left
    (46, 16, 0, +0.3),     # less crop on top, needs cropping on right
    (68, 21, 0, +0.5),     # needs cropping on right
    (31, 0, 0, +0.5),      # needs cropping on right
    (50, 8, 0, +0.5),      # needs cropping on right and bottom (bottom fixed by resize)
    (78, 9, -0.2, +0.5),   # less crop on left, needs cropping on right and bottom
    (76, 25, 0, +0.5),     # needs cropping on right
    (41, 22, -0.5, -0.3),  # needs cropping all but bottom left (adjust both sides)
    (79, 22, 0, +0.5),     # needs cropping on right
    (51, 18, 0, +0.3),     # needs cropping on right and top (top fixed by resize)
]

for line_idx, position, start_off, end_off in fixes:
    example = target[line_idx]
    img = example["image"]
    text = example["text"]
    w, h = img.size
    char_w = w / len(text)
    
    # Apply offsets
    start_pos = position + start_off
    end_pos = position + 2 + end_off
    
    x1 = int(start_pos * char_w)
    x2 = int(end_pos * char_w)
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    
    crop = img.crop((x1, 0, x2, h))
    crop_resized = crop.resize(TARGET_SIZE, Image.LANCZOS)
    
    # Save with fixed_ prefix
    old_name = f"{TARGET_CHAR}_with_{text[position+1] if position+1 < len(text) else ''}_{line_idx}_{position}.png"
    new_name = f"fixed_{TARGET_CHAR}_with_{text[position+1] if position+1 < len(text) else ''}_{line_idx}_{position}.png"
    
    filepath = os.path.join(OUTPUT_DIR, new_name)
    crop_resized.save(filepath)
    print(f"Fixed: {new_name} (offset: {start_off}, {end_off})")

print("\nDone! Check the fixed_*.png files in data/contextual/val/")
