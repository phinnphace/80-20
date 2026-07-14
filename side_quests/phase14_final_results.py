#!/usr/bin/env python3
"""
Phase 14: Final Results Compilation
Combines all experimental phases into a single comprehensive CSV and report.
"""
import json
import pandas as pd
import numpy as np

# ============================================================
# LOAD ALL DATA
# ============================================================
with open('models/full_matrix.json') as f:
    matrix = json.load(f)

with open('models/split_window.json') as f:
    split_a = json.load(f)

with open('models/split_window_contextual.json') as f:
    split_b = json.load(f)

with open('models/optimized_rerun_results.json') as f:
    optimized = json.load(f)

# ============================================================
# TABLE 1: COMPLETE EXPERIMENTAL RESULTS
# ============================================================
rows = []
splits = [0.5, 0.6, 0.7, 0.8, 0.9]
split_labels = ['50/50', '60/40', '70/30', '80/20', '90/10']

for s, sl in zip(splits, split_labels):
    ka = f"condition_a_{s}"
    kb = f"condition_b_{s}"
    
    ra = matrix.get(ka, {})
    rb = matrix.get(kb, {})
    
    rows.append({
        'Split': sl,
        'A_Train': int(s * 489),
        'A_Val': int((1-s) * 489),
        'A_Val_Acc': ra.get('val'),
        'A_CASIA': ra.get('casia'),
        'A_CalliBench': ra.get('calli'),
        'B_Train': int(s * 286),
        'B_Val': int((1-s) * 286),
        'B_Val_Acc': rb.get('val'),
        'B_CASIA': rb.get('casia'),
        'B_CalliBench': rb.get('calli'),
    })

df_main = pd.DataFrame(rows)

# ============================================================
# TABLE 2: CALLIBENCH 5×5 DIFFERENCE MATRIX
# ============================================================
diff_rows = []
for sa, sla in zip(splits, split_labels):
    row = {'A_Split': sla}
    for sb, slb in zip(splits, split_labels):
        a_val = matrix.get(f"condition_a_{sa}", {}).get('calli')
        b_val = matrix.get(f"condition_b_{sb}", {}).get('calli')
        if a_val is not None and b_val is not None:
            row[f"B={slb}"] = b_val - a_val
        else:
            row[f"B={slb}"] = None
    diff_rows.append(row)

df_diff = pd.DataFrame(diff_rows)

# ============================================================
# TABLE 3: SPLIT WINDOW COMPARISON
# ============================================================
window_rows = []
for split_name in split_a:
    s = split_name.replace('train_', '').replace('val_', '').replace('_', '/')
    a_acc = split_a[split_name]['val_accuracy']
    b_acc = split_b[split_name]['val_accuracy']
    window_rows.append({
        'Split': s,
        'Cond_A_Val': a_acc,
        'Cond_B_Val': b_acc,
        'Delta': b_acc - a_acc
    })

df_window = pd.DataFrame(window_rows)

# ============================================================
# TABLE 4: OPTIMIZED VS ORIGINAL COMPARISON
# ============================================================
original_data = {
    'condition_a': {'val': 0.6122, 'casia': 0.3428, 'calli': 1.0000},
    'condition_b': {'val': 0.9138, 'casia': 0.2663, 'calli': 0.0526}
}

comp_rows = []
for condition, label in [('condition_a', 'A (Isolated)'), ('condition_b', 'B (Contextual)')]:
    orig = original_data[condition]
    opt = optimized[condition]
    opt_casia = opt['casia_transfer']['accuracy']
    opt_calli = opt['callibench_transfer']['accuracy']
    
    comp_rows.append({
        'Condition': label,
        'Original_Split': '80/20',
        'Optimized_Split': opt['split'],
        'Orig_Val': orig['val'],
        'Opt_Val': opt['val_best_acc'],
        'Delta_Val': opt['val_best_acc'] - orig['val'],
        'Orig_CASIA': orig['casia'],
        'Opt_CASIA': opt_casia,
        'Delta_CASIA': opt_casia - orig['casia'],
        'Orig_CalliBench': orig['calli'],
        'Opt_CalliBench': opt_calli,
        'Delta_CalliBench': opt_calli - orig['calli'],
    })

df_comp = pd.DataFrame(comp_rows)

# ============================================================
# TABLE 5: STRUCTURAL ANALYSIS
# ============================================================
structural_rows = [{
    'Metric': 'Structural Difference',
    'Value': 0.3363,
    'Interpretation': 'Substantially different error structures (>0.2 threshold)'
}, {
    'Metric': 'Cond A Error Rate',
    'Value': 0.3878,
    'Interpretation': '38.8% of validation samples misclassified'
}, {
    'Metric': 'Cond B Error Rate',
    'Value': 0.0862,
    'Interpretation': '8.6% of validation samples misclassified'
}, {
    'Metric': 'Cond A FP/FN Ratio',
    'Value': 37.0,
    'Interpretation': 'Heavily biased toward false positives (37 FP, 1 FN)'
}, {
    'Metric': 'Cond B FP/FN Ratio',
    'Value': 1.5,
    'Interpretation': 'Balanced errors (3 FP, 2 FN)'
}, {
    'Metric': 'Cond A Positive Prediction Rate',
    'Value': 0.8571,
    'Interpretation': 'Model predicts 打 85.7% of the time'
}, {
    'Metric': 'Cond B Positive Prediction Rate',
    'Value': 0.5517,
    'Interpretation': 'Model predicts 打 55.2% of the time (near true rate of 53.4%)'
}]

df_structural = pd.DataFrame(structural_rows)

# ============================================================
# SAVE ALL TO CSV
# ============================================================
df_main.to_csv('models/table1_experimental_results.csv', index=False)
df_diff.to_csv('models/table2_callibench_matrix.csv', index=False)
df_window.to_csv('models/table3_split_window.csv', index=False)
df_comp.to_csv('models/table4_optimized_comparison.csv', index=False)
df_structural.to_csv('models/table5_structural_analysis.csv', index=False)

# ============================================================
# PRINT REPORT
# ============================================================
print("=" * 80)
print("  FINAL RESULTS REPORT")
print("=" * 80)

print("\nTABLE 1: Complete Experimental Results (All Splits, All Tests)")
print("-" * 80)
print(df_main.to_string(index=False))

print("\n\nTABLE 2: CalliBench Transfer Difference Matrix (B minus A)")
print("-" * 80)
print(df_diff.to_string(index=False))

print("\n\nTABLE 3: Split Ratio Window (Internal Validation Only)")
print("-" * 80)
print(df_window.to_string(index=False))

print("\n\nTABLE 4: Optimized vs Original 80/20 Split Comparison")
print("-" * 80)
print(df_comp.to_string(index=False))

print("\n\nTABLE 5: Structural Analysis (Confusion Matrix Assay)")
print("-" * 80)
print(df_structural.to_string(index=False))

print("\n\nKEY FINDINGS:")
print("-" * 80)
print("1. Split ratio changes internal validation by up to 21.4 points (Cond A: 53.7%-73.5%)")
print("2. Split ratio changes CASIA transfer by up to 54.2 points (Cond A: 15.3%-69.5%)")
print("3. Split ratio does NOT change the CalliBench story: Cond A dominates across all splits")
print("4. At 80/20, Cond B appears to fail CASIA transfer (26.6%). At 70/30, it's perfect (100%)")
print("5. The 80/20 default was the most misleading split for Cond B's CASIA transfer")
print("6. CalliBench is a persistent domain gap: Cond B never exceeds 15.8%")
print("7. Cond A's calligraphy performance is volatile: 0% at 70/30, 100% at 50/50 and 80/20")
print("8. Internal validation does not predict transfer performance")
print("9. The optimal split for internal validation is not the optimal split for transfer")
print("10. Structural difference of 0.336 confirms models fail in fundamentally different ways")

print("\n\nAll tables saved to models/")
print("  table1_experimental_results.csv")
print("  table2_callibench_matrix.csv")
print("  table3_split_window.csv")
print("  table4_optimized_comparison.csv")
print("  table5_structural_analysis.csv")
print("\nPhase 14 complete.")
