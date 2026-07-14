python3 -c "
import os
from PIL import Image

d = 'data/contextual/val'
# Show first 10 files and check for blanks
files = sorted(os.listdir(d))
for f in files[:15]:
    # Parse what we can from filename
    parts = f.replace('.png', '').split('_')
    print(f'File: {f}')
    print(f'  Parts: {parts}')
    
# Count files with blank neighbor
blank_count = sum(1 for f in files if '_with__' in f or f.endswith('_.png'))
print(f'\nFiles with potentially blank neighbor: {blank_count}/{len(files)}')
"
