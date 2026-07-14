#!/usr/bin/env python3
"""
Phase 12: Comprehensive Results Table
Tracks every experimental condition, split, and test result across all phases.
Outputs a clean DataFrame and markdown table for the report.
"""
import json
import pandas as pd

# ============================================================
# COLLECT ALL RESULTS
# ============================================================

# Phase 7: Original 80/20 results
original = {
    'phase': 'Phase 7 (Original)',
    'condition_a_split': '80/20',
    'condition_a_train': 391,
    'condition_a_val': 98,
    'condition_a_val_acc': 0.6122,
    'condition_a_precision': 0.5595,
    'condition_a_recall': 0.9792,
    'condition_a_f1': 0.7121,
    'condition_a_tn': 13,
    'condition_a_fp': 37,
    'condition_a_fn': 1,
    'condition_a_tp': 47,
    
    'condition_b_split': '80/20',
    'condition_b_train': 228,
    'condition_b_val': 58,
    'condition_b_val_acc': 0.9138,
    'condition_b_precision': 0.9062,
    'condition_b_recall': 0.9355,
    'condition_b_f1': 0.9206,
    'condition_b_tn': 24,
    'condition_b_fp': 3,
    'condition_b_fn': 2,
    'condition_b_tp': 29,
    
    'structural_difference': 0.3363,
    'casia_transfer_a': 0.3428,
    'casia_transfer_b': 0.2663,
    'callibench_transfer_a': 1.0000,
    'callibench_transfer_b': 0.0526,
}

# Phase 10: Split ratio window - Condition A
with open('models/split_window.json') as f:
    split_a = json.load(f)

# Phase 10b: Split ratio window - Condition B
with open('models/split_window_contextual.json') as f:
    split_b = json.load(f)

# Phase 11: Optimized rerun
with open('models/optimized_rerun_results.json') as f:
    optimized = json.load(f)

# ============================================================
# BUILD TABLE 1: All Experimental Conditions
# ============================================================
rows = []

# Original 80/20
rows.append({
    'Phase': 'Original',
    'Condition': 'A (Isolated)',
    'Split': '80/20',
    'Train': original['condition_a_train'],
    'Val': original['condition_a_val'],
    'Val Acc': original['condition_a_val_acc'],
    'Precision': original['condition_a_precision'],
    'Recall': original['condition_a_recall'],
    'F1': original['condition_a_f1'],
    'TN/FP/FN/TP': f"{original['condition_a_tn']}/{original['condition_a_fp']}/{original['condition_a_fn']}/{original['condition_a_tp']}",
    'CASIA Transfer': original['casia_transfer_a'],
    'CalliBench Transfer': original['callibench_transfer_a'],
})

rows.append({
    'Phase': 'Original',
    'Condition': 'B (Contextual)',
    'Split': '80/20',
    'Train': original['condition_b_train'],
    'Val': original['condition_b_val'],
    'Val Acc': original['condition_b_val_acc'],
    'Precision': original['condition_b_precision'],
    'Recall': original['condition_b_recall'],
    'F1': original['condition_b_f1'],
    'TN/FP/FN/TP': f"{original['condition_b_tn']}/{original['condition_b_fp']}/{original['condition_b_fn']}/{original['condition_b_tp']}",
    'CASIA Transfer': original['casia_transfer_b'],
    'CalliBench Transfer': original['callibench_transfer_b'],
})

# Split window - Condition A
for split_name, data in split_a.items():
    rows.append({
        'Phase': 'Split Window',
        'Condition': 'A (Isolated)',
        'Split': split_name.replace('train_', '').replace('val_', '').replace('_', '/'),
        'Train': data['train_size'],
        'Val': data['val_size'],
        'Val Acc': data['val_accuracy'],
        'Precision': '-',
        'Recall': '-',
        'F1': '-',
        'TN/FP/FN/TP': '-',
        'CASIA Transfer': '-',
        'CalliBench Transfer': '-',
    })

# Split window - Condition B
for split_name, data in split_b.items():
    rows.append({
        'Phase': 'Split Window',
        'Condition': 'B (Contextual)',
        'Split': split_name.replace('train_', '').replace('val_', '').replace('_', '/'),
        'Train': data['train_size'],
        'Val': data['val_size'],
        'Val Acc': data['val_accuracy'],
        'Precision': '-',
        'Recall': '-',
        'F1': '-',
        'TN/FP/FN/TP': '-',
        'CASIA Transfer': '-',
        'CalliBench Transfer': '-',
    })

# Optimized rerun
for condition in ['condition_a', 'condition_b']:
    data = optimized[condition]
    cond_label = 'A (Isolated)' if condition == 'condition_a' else 'B (Contextual)'
    
    val_r = data['val_results']
    casia_r = data['casia_transfer']
    calli_r = data['callibench_transfer']
    
    # Calculate precision/recall/f1
    tp = val_r['tp']; tn = val_r['tn']; fp = val_r['fp']; fn = val_r['fn']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    rows.append({
        'Phase': 'Optimized',
        'Condition': cond_label,
        'Split': data['split'],
        'Train': data['train_size'],
        'Val': data['val_size'],
        'Val Acc': data['val_best_acc'],
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TN/FP/FN/TP': f"{tn}/{fp}/{fn}/{tp}",
        'CASIA Transfer': casia_r['accuracy'],
        'CalliBench Transfer': calli_r['accuracy'],
    })

df = pd.DataFrame(rows)

# ============================================================
# OUTPUT
# ============================================================
print("=" * 120)
print("  COMPREHENSIVE RESULTS TABLE")
print("=" * 120)
print()
print(df.to_string(index=False))
print()

# Save to CSV
df.to_csv('models/comprehensive_results.csv', index=False)
print("Saved to models/comprehensive_results.csv")

# ============================================================
# TABLE 2: Structural Analysis Summary
# ============================================================
print()
print("=" * 80)
print("  STRUCTURAL ANALYSIS")
print("=" * 80)
print(f"  Confusion matrix structural difference (80/20): {original['structural_difference']:.4f}")
print(f"  Interpretation: Error structures are substantially different.")
print(f"  Condition A error pattern: biased toward false positives (FP=37, FN=1)")
print(f"  Condition B error pattern: balanced errors (FP=3, FN=2)")
print(f"  Condition A positive prediction rate: 85.7%")
print(f"  Condition B positive prediction rate: 55.2%")

# ============================================================
# TABLE 3: Transfer Test Summary
# ============================================================
print()
print("=" * 80)
print("  TRANSFER TEST SUMMARY")
print("=" * 80)
print(f"  {'Test':<25} {'Cond A (80/20)':<16} {'Cond B (80/20)':<16} {'Cond A (Opt)':<16} {'Cond B (Opt)':<16}")
print(f"  {'-'*85}")
print(f"  {'CASIA Isolated':<25} {original['casia_transfer_a']:<16.4f} {original['casia_transfer_b']:<16.4f} {optimized['condition_a']['casia_transfer']['accuracy']:<16.4f} {optimized['condition_b']['casia_transfer']['accuracy']:<16.4f}")
print(f"  {'CalliBench Calligraphy':<25} {original['callibench_transfer_a']:<16.4f} {original['callibench_transfer_b']:<16.4f} {optimized['condition_a']['callibench_transfer']['accuracy']:<16.4f} {optimized['condition_b']['callibench_transfer']['accuracy']:<16.4f}")

# ============================================================
# TABLE 4: Split Ratio Impact
# ============================================================
print()
print("=" * 80)
print("  SPLIT RATIO WINDOW: INTERNAL VALIDATION ONLY")
print("=" * 80)
print(f"  {'Split':<12} {'Cond A':<10} {'Cond B':<10} {'Delta':<10}")
print(f"  {'-'*42}")
for split_name in split_a:
    a_acc = split_a[split_name]['val_accuracy']
    b_acc = split_b[split_name]['val_accuracy']
    label = split_name.replace('train_', '').replace('val_', '').replace('_', '/')
    print(f"  {label:<12} {a_acc:<10.4f} {b_acc:<10.4f} {b_acc-a_acc:+.4f}")

print()
print("Phase 12 complete.")
