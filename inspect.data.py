# ============================================================
# CONDITION A: CASIA Isolated Characters
# ============================================================
print("\n\n=== CONDITION A: Isolated Characters ===")

import os
import struct
from codecs import decode
import numpy as np
from PIL import Image

gnt_dir = "/mnt/c/Users/pmark/CASIA_data/HWDB1.1tst_gnt/"

if not os.path.isdir(gnt_dir):
    print(f"  ERROR: Directory not found: {gnt_dir}")
else:
    gnt_files = sorted([f for f in os.listdir(gnt_dir) if f.endswith('.gnt')])
    print(f"  Found {len(gnt_files)} .gnt files")
    
    # Parse using the exact format from PyCasia source
    def parse_gnt_correct(filepath, max_samples=None):
        """Parse GNT files using the correct binary format."""
        results = []
        with open(filepath, 'rb') as f:
            count = 0
            while True:
                if max_samples and count >= max_samples:
                    break
                    
                packed_length = f.read(4)
                if packed_length == b'':
                    break
                
                length = struct.unpack("<I", packed_length)[0]
                raw_label = struct.unpack(">cc", f.read(2))
                width = struct.unpack("<H", f.read(2))[0]
                height = struct.unpack("<H", f.read(2))[0]
                raw_bytes = f.read(height * width)
                photo_bytes = struct.unpack("{}B".format(height * width), raw_bytes)
                
                # Decode GB2312 label
                label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
                
                # Create PIL Image
                img_array = np.array(photo_bytes, dtype=np.uint8).reshape(height, width)
                image = Image.fromarray(img_array, mode='L')
                
                results.append((image, label))
                count += 1
                
        return results
    
    # Parse first file
    first_file = os.path.join(gnt_dir, gnt_files[0])
    print(f"\n  Parsing: {gnt_files[0]}")
    samples = parse_gnt_correct(first_file, max_samples=5)
    
    print("  Sample label format (first 5):")
    for i, (img, label) in enumerate(samples):
        print(f"  [{i}] Image size: {img.size}, Label: '{label}'")
    
    # Count 打 occurrences across first 10 files
    print("\n  Counting 打 occurrences (first 10 files)...")
    da_count = 0
    total_count = 0
    sizes = set()
    
    for gnt_file in gnt_files[:10]:
        filepath = os.path.join(gnt_dir, gnt_file)
        samples = parse_gnt_correct(filepath)
        for img, label in samples:
            sizes.add(img.size)
            if label == "打":
                da_count += 1
            total_count += 1
    
    print(f"  Found {da_count} instances of 打 in {total_count} characters (10 files)")
    print(f"  Image sizes found: {sizes}")
