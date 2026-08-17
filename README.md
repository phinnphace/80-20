# 80/20: Contextual Training Effects on Visual Feature Learning

Experiment testing whether training context (isolated vs. bigram) changes how vision models learn Chinese characters. Demonstrates that the standard 80/20 train/test split is an experimental variable, not a neutral default.

## Quick Start
pip install -r requirements.txt
streamlit run app.py

## Data
CASIA-HWDB (Liu et al., 2011) via HuggingFace `Teklia/CASIA-HWDB2-line`
CalliBench (Luo et al., 2025) for transfer evaluation

## Key Finding
Split ratio changes internal validation by 21+ percentage points. The 80/20 test/train default is an imputed, methodological decision that should be assessed per use case. 
 The original slide deck dashboard/Vercel  can be found here:(http://go.dataacorns.com/80-20/splits) 
 
