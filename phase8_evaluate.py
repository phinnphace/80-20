#!/usr/bin/env python3
"""
Phase 8: Evaluation & Confusion Matrix Assay
Runs both trained models on their validation sets and compares error patterns.
Tests whether contextual training produces structurally different errors.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import json

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32
DEVICE = torch.device("cpu")
MODEL_DIR = "models"
DATA_DIR = "data"

# ============================================================
# DATASET (same as training)
# ============================================================
class CharacterDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for class_name, label in [("da", 1), ("other", 0)]:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in sorted(os.listdir(class_dir)):
                    if fname.endswith('.png'):
                        self.samples.append((os.path.join(class_dir, fname), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        image = image.unsqueeze(0)
        return image, label, img_path  # Return path for error analysis

# ============================================================
# MODEL (same as training)
# ============================================================
class CharacterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 2)
    
    def forward(self, x, return_embedding=False):
        x = self.conv(x)
        embedding = x.view(x.size(0), -1)
        x = self.fc(embedding)
        if return_embedding:
            return x, embedding
        return x

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, loader, condition_name):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    all_embeddings = []
    
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            outputs, embeddings = model(images, return_embedding=True)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.vstack(all_embeddings)
    
    # Basic metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Confusion matrix: rows=true, cols=predicted
    # Index 0 = other (negative), Index 1 = da (positive)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1
    
    # Per-class metrics
    # True Positives: label=1, pred=1
    tp = cm[1][1]
    # True Negatives: label=0, pred=0
    tn = cm[0][0]
    # False Positives: label=0, pred=1
    fp = cm[0][1]
    # False Negatives: label=1, pred=0
    fn = cm[1][0]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Error analysis: which files were misclassified?
    errors = []
    for i, (pred, label, path) in enumerate(zip(all_preds, all_labels, all_paths)):
        if pred != label:
            errors.append({
                'path': path,
                'true': int(label),
                'pred': int(pred),
                'type': 'false_positive' if pred == 1 else 'false_negative'
            })
    
    results = {
        'condition': condition_name,
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'total_samples': len(all_labels),
        'total_errors': len(errors),
        'errors': errors,
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
    }
    
    return results, all_embeddings, all_labels

# ============================================================
# CONFUSION MATRIX STRUCTURAL ANALYSIS
# ============================================================
def analyze_error_structure(results_a, results_b):
    """
    Compare the structure of errors between the two conditions.
    Tests whether errors in Condition A cluster differently than Condition B.
    """
    analysis = {}
    
    cm_a = np.array(results_a['confusion_matrix'])
    cm_b = np.array(results_b['confusion_matrix'])
    
    # 1. Error rate comparison
    analysis['error_rate_a'] = (cm_a[0][1] + cm_a[1][0]) / cm_a.sum()
    analysis['error_rate_b'] = (cm_b[0][1] + cm_b[1][0]) / cm_b.sum()
    
    # 2. Error balance: ratio of false positives to false negatives
    fp_a, fn_a = cm_a[0][1], cm_a[1][0]
    fp_b, fn_b = cm_b[0][1], cm_b[1][0]
    
    analysis['fp_fn_ratio_a'] = fp_a / (fn_a + 1e-8)
    analysis['fp_fn_ratio_b'] = fp_b / (fn_b + 1e-8)
    
    # 3. Bias toward positive or negative predictions
    total_pred_pos_a = cm_a[0][1] + cm_a[1][1]
    total_pred_pos_b = cm_b[0][1] + cm_b[1][1]
    analysis['positive_prediction_rate_a'] = total_pred_pos_a / cm_a.sum()
    analysis['positive_prediction_rate_b'] = total_pred_pos_b / cm_b.sum()
    
    # 4. Normalized confusion matrices for comparison
    cm_a_norm = cm_a / cm_a.sum(axis=1, keepdims=True)
    cm_b_norm = cm_b / cm_b.sum(axis=1, keepdims=True)
    analysis['cm_normalized_a'] = cm_a_norm.tolist()
    analysis['cm_normalized_b'] = cm_b_norm.tolist()
    
    # 5. Structural difference metric
    # How different are the two confusion matrices?
    structural_diff = np.abs(cm_a_norm - cm_b_norm).sum() / 4
    analysis['structural_difference'] = float(structural_diff)
    
    # 6. Qualitative interpretation
    if structural_diff > 0.2:
        analysis['interpretation'] = "Error structures are substantially different. Contextual training changed how the model fails."
    elif structural_diff > 0.1:
        analysis['interpretation'] = "Error structures show moderate differences. Contextual training may be shifting error patterns."
    else:
        analysis['interpretation'] = "Error structures are similar. Both models fail in comparable ways despite accuracy differences."
    
    return analysis

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Phase 8: Confusion Matrix Assay")
    print("=" * 60)
    
    all_results = {}
    all_embeddings = {}
    all_labels_dict = {}
    
    for condition, data_path in [
        ("condition_a", "data/condition_a"),
        ("condition_b", "data/condition_b")
    ]:
        print(f"\nEvaluating: {condition}")
        
        # Load model
        model = CharacterCNN()
        model_path = os.path.join(MODEL_DIR, f"{condition}_best.pth")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        
        # Load validation data
        val_dataset = CharacterDataset(os.path.join(data_path, "val"))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Evaluate
        results, embeddings, labels = evaluate_model(model, val_loader, condition)
        all_results[condition] = results
        all_embeddings[condition] = embeddings
        all_labels_dict[condition] = labels
        
        # Print results
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1:        {results['f1']:.4f}")
        print(f"  Confusion Matrix (rows=true, cols=pred):")
        print(f"    TN={results['tn']:4d}  FP={results['fp']:4d}")
        print(f"    FN={results['fn']:4d}  TP={results['tp']:4d}")
        print(f"  Total errors: {results['total_errors']}")
    
    # Structural analysis
    print(f"\n{'='*60}")
    print("  CONFUSION MATRIX STRUCTURAL ANALYSIS")
    print(f"{'='*60}")
    
    analysis = analyze_error_structure(all_results['condition_a'], all_results['condition_b'])
    
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, str):
            print(f"\n  INTERPRETATION: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Save all results
    output = {
        'condition_a': all_results['condition_a'],
        'condition_b': all_results['condition_b'],
        'structural_analysis': analysis
    }
    
        # Save error details separately for manual inspection FIRST
    errors_a = all_results['condition_a']['errors']
    errors_b = all_results['condition_b']['errors']
    
    # Then remove from JSON output (file paths aren't serializable)
    output['condition_a'].pop('errors', None)
    output['condition_b'].pop('errors', None)
    
    with open('models/assay_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Condition A errors: {len(errors_a)}")
    print(f"    False positives (other predicted as da): {sum(1 for e in errors_a if e['type'] == 'false_positive')}")
    print(f"    False negatives (da predicted as other): {sum(1 for e in errors_a if e['type'] == 'false_negative')}")
    
    print(f"\n  Condition B errors: {len(errors_b)}")
    print(f"    False positives (other predicted as da): {sum(1 for e in errors_b if e['type'] == 'false_positive')}")
    print(f"    False negatives (da predicted as other): {sum(1 for e in errors_b if e['type'] == 'false_negative')}")
    
    print(f"\n  Full results saved to models/assay_results.json")
    print(f"\nPhase 8 complete.")
