import os

# Create directory structure
os.makedirs('data/contextual/val', exist_ok=True)
os.makedirs('data/isolated/target', exist_ok=True)
os.makedirs('data/isolated/distractors', exist_ok=True)

print("Directory structure created.")