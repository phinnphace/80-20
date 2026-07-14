#!/usr/bin/env python3
"""
Generate confusion matrix plots from assay results.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('models/assay_results.json', 'r') as f:
    data = json.load(f)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

conditions = [('condition_a', 'Condition A: Isolated'), 
              ('condition_b', 'Condition B: Contextual')]
for ax, (name, title) in zip(axes, conditions):
    cm = np.array(data[name]['confusion_matrix'])

    
    # Plot
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Other', 'Pred: da'])
    ax.set_yticklabels(['True: Other', 'True: da'])
    ax.set_title(f'{title}\nAccuracy: {data[name]["accuracy"]:.1%}')
    
    # Annotate cells
    for i in range(2):
        for j in range(2):
            text = f'{cm[i][j]}'
            if i == 1 and j == 1:
                text += '\n(TP)'
            elif i == 0 and j == 0:
                text += '\n(TN)'
            elif i == 0 and j == 1:
                text += '\n(FP)'
            elif i == 1 and j == 0:
                text += '\n(FN)'
            color = 'white' if cm[i][j] > cm.max() / 2 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=14, fontweight='bold')

# Add structural analysis text
fig.text(0.5, 0.01, 
         f'Structural Difference: {data["structural_analysis"]["structural_difference"]:.3f} | '
         f'{data["structural_analysis"]["interpretation"]}',
         ha='center', fontsize=12, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('models/confusion_matrices.png', dpi=150, bbox_inches='tight')
print('Saved: models/confusion_matrices.png')
