#!/usr/bin/env python3
"""
Interactive Results Dashboard
Streamlit app for exploring the contextual training experiment.
"""
import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Contextual Training Effects", layout="wide")
st.title("Contextual Training Effects on Visual Feature Learning")
st.caption("Interactive exploration of experimental results")

# Load all data
@st.cache_data
def load_data():
    with open('models/assay_results.json') as f:
        assay = json.load(f)
    with open('models/transfer_test_results.json') as f:
        transfer = json.load(f)
    with open('models/resnet_results.json') as f:
        resnet = json.load(f)
    
    # Load side quest data if available
    full_matrix = None
    split_window_a = None
    split_window_b = None
    try:
        with open('side_quests/full_matrix.json') as f:
            full_matrix = json.load(f)
    except:
        pass
    try:
        with open('side_quests/split_window.json') as f:
            split_window_a = json.load(f)
    except:
        pass
    try:
        with open('side_quests/split_window_contextual.json') as f:
            split_window_b = json.load(f)
    except:
        pass
    
    return assay, transfer, resnet, full_matrix, split_window_a, split_window_b

assay, transfer, resnet, full_matrix, split_window_a, split_window_b = load_data()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Comparison", 
    "Confusion Matrices", 
    "Transfer Tests",
    "Split Ratio Explorer",
    "About"
])

# ============================================================
# TAB 1: MODEL COMPARISON
# ============================================================
with tab1:
    st.header("Three-Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CNN (Isolated)", "61.2%", delta="Baseline")
        st.caption("92,930 params | Trained from scratch")
    
    with col2:
        st.metric("CNN (Contextual)", "91.4%", delta="+30.2 pp")
        st.caption("92,930 params | Trained from scratch")
    
    with col3:
        st.metric("ResNet-18", "100%", delta="+38.8 pp vs CNN")
        st.caption("11.2M params | ImageNet pretrained")
    
    st.divider()
    
    st.subheader("Model Details")
    df_models = pd.DataFrame([
        {"Model": "CNN (Isolated)", "Parameters": "92,930", "Pretrained": "No", 
         "Cond A": "61.2%", "Cond B": "—", "Best For": "Cross-domain generalization"},
        {"Model": "CNN (Contextual)", "Parameters": "92,930", "Pretrained": "No", 
         "Cond A": "—", "Cond B": "91.4%", "Best For": "In-distribution accuracy"},
        {"Model": "ResNet-18", "Parameters": "11,177,538", "Pretrained": "ImageNet", 
         "Cond A": "100%", "Cond B": "100%", "Best For": "Everything (if you have the compute)"},
    ])
    st.dataframe(df_models, use_container_width=True, hide_index=True)
    
    st.subheader("Key Insight")
    st.info(
        "Contextual training provides a **30-point boost** when models are small and trained from scratch. "
        "But this benefit **completely vanishes** when large-scale pretraining is available. "
        "The value of context depends on what else the model already knows."
    )

# ============================================================
# TAB 2: CONFUSION MATRICES
# ============================================================
with tab2:
    st.header("Confusion Matrix Comparison")
    st.caption("How each model fails tells us more than how often it fails.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Condition A (Isolated)")
        cm_a = np.array([[13, 37], [1, 47]])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm_a, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred: Other', 'Pred: 打'])
        ax.set_yticklabels(['True: Other', 'True: 打'])
        for i in range(2):
            for j in range(2):
                text = f'{cm_a[i][j]}\n(TP)' if i==1 and j==1 else f'{cm_a[i][j]}\n(TN)' if i==0 and j==0 else f'{cm_a[i][j]}\n(FP)' if i==0 else f'{cm_a[i][j]}\n(FN)'
                ax.text(j, i, text, ha='center', va='center', color='white' if cm_a[i][j] > 20 else 'black', fontsize=14, fontweight='bold')
        ax.set_title("Isolated Training\nAccuracy: 61.2%")
        st.pyplot(fig)
        st.caption("Heavy bias toward predicting 打 (85.7% positive rate). 37 false positives.")
    
    with col2:
        st.subheader("Condition B (Contextual)")
        cm_b = np.array([[24, 3], [2, 29]])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm_b, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred: Other', 'Pred: 打'])
        ax.set_yticklabels(['True: Other', 'True: 打'])
        for i in range(2):
            for j in range(2):
                text = f'{cm_b[i][j]}\n(TP)' if i==1 and j==1 else f'{cm_b[i][j]}\n(TN)' if i==0 and j==0 else f'{cm_b[i][j]}\n(FP)' if i==0 else f'{cm_b[i][j]}\n(FN)'
                ax.text(j, i, text, ha='center', va='center', color='white' if cm_b[i][j] > 15 else 'black', fontsize=14, fontweight='bold')
        ax.set_title("Contextual Training\nAccuracy: 91.4%")
        st.pyplot(fig)
        st.caption("Balanced predictions (55.2% positive rate). Only 3 false positives.")
    
    st.divider()
    st.subheader("Structural Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Structural Difference", "0.336")
        st.caption(">0.2 = substantially different")
    with col2:
        st.metric("A FP/FN Ratio", "37.0")
        st.caption("Heavily biased toward FP")
    with col3:
        st.metric("B FP/FN Ratio", "1.5")
        st.caption("Balanced errors")

# ============================================================
# TAB 3: TRANSFER TESTS
# ============================================================
with tab3:
    st.header("Transfer Test Results")
    st.caption("Does training that works internally work externally?")
    
    st.subheader("CASIA Isolated Transfer (unseen handwriting)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CNN (Isolated)", f"{transfer['condition_a']['accuracy']:.1%}")
    with col2:
        st.metric("CNN (Contextual)", f"{transfer['condition_b']['accuracy']:.1%}")
    
    st.subheader("CalliBench Calligraphy Transfer (artistic calligraphy)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CNN (Isolated)", "100%", delta="Perfect")
    with col2:
        st.metric("CNN (Contextual)", "5.3%", delta="-94.7 pp", delta_color="inverse")
    
    st.divider()
    st.info(
        "The model trained on **isolated characters** built a visual representation of 打 "
        "that transferred across domains—even to artistic calligraphy it never saw during training. "
        "The model trained on **contextual bigrams** learned to rely on specific contextual cues "
        "that didn't exist in calligraphy, causing near-total failure."
    )

# ============================================================
# TAB 4: SPLIT RATIO EXPLORER (Side Quest)
# ============================================================
with tab4:
    st.header("Split Ratio Explorer")
    st.caption("What if we hadn't used 80/20? Drag the slider to find out.")
    
    if split_window_a and split_window_b:
        # Extract split data
        splits = [0.5, 0.6, 0.7, 0.8, 0.9]
        a_vals = [split_window_a[f'train_{int(s*100)}_val_{int((1-s)*100)}']['val_accuracy'] for s in splits]
        b_vals = [split_window_b[f'train_{int(s*100)}_val_{int((1-s)*100)}']['val_accuracy'] for s in splits]
        
        selected_split = st.select_slider(
            "Training split ratio",
            options=["50/50", "60/40", "70/30", "80/20", "90/10"],
            value="80/20"
        )
        
        idx = ["50/50", "60/40", "70/30", "80/20", "90/10"].index(selected_split)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cond A Accuracy", f"{a_vals[idx]:.1%}")
        with col2:
            st.metric("Cond B Accuracy", f"{b_vals[idx]:.1%}")
        with col3:
            delta = b_vals[idx] - a_vals[idx]
            st.metric("Gap (B − A)", f"{delta:+.1%}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        split_labels = ["50/50", "60/40", "70/30", "80/20", "90/10"]
        x = range(len(splits))
        ax.plot(x, a_vals, 'o-', label='Cond A (Isolated)', color='#1f77b4', linewidth=2)
        ax.plot(x, b_vals, 's-', label='Cond B (Contextual)', color='#ff7f0e', linewidth=2)
        ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5, label='Default 80/20')
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels)
        ax.set_ylabel('Validation Accuracy')
        ax.set_xlabel('Train/Val Split')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Internal Validation Accuracy by Split Ratio')
        st.pyplot(fig)
        
        st.caption(
            "The default 80/20 split (dashed line) was not optimal for either condition. "
            "Condition A improved with more training data. Condition B was stable across splits. "
            "The choice of split ratio changes the conclusion you'd draw about which model is better."
        )
    else:
        st.warning("Split ratio data not found. Run Phase 10 and 10b to generate this explorer.")

# ============================================================
# TAB 5: ABOUT
# ============================================================
with tab5:
    st.header("About This Project")
    st.markdown("""
    **Research Question:** Does training a vision model on Chinese characters in contextual 
    relationship (bigrams) produce fundamentally different internal representations than 
    training on isolated characters?
    
    **Key Finding:** Yes—but the benefit depends on model capacity and pretraining. Contextual 
    training helps small models trained from scratch (+30 points). With large-scale pretraining 
    (ResNet-18 on ImageNet), the gap disappears entirely.
    
    **Methodological Contribution:** The confusion matrix structural assay provides a diagnostic 
    for *how* models fail, not just how often. Combined with the split ratio explorer, this 
    project demonstrates that standard ML defaults (80/20 splits, accuracy-only evaluation) 
    can obscure important patterns in model behavior.
    
    **Data:** CASIA Chinese Handwriting Database (HWDB1.1 and HWDB2) via HuggingFace.
    打 (dǎ) as target character with 5 visually similar distractors.
    
    **Original Motivation:** Understanding failure patterns in vision models for sign language 
    recognition, where visual identity is inherently contextual.
    """)

st.divider()
st.caption("Built with Streamlit • Data from CASIA-HWDB via HuggingFace • Models trained with PyTorch")
