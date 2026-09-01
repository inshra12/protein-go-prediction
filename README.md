cat > README.md << 'EOF'
# Protein GO Function Prediction

Multi-modal protein function prediction using protein language model embeddings (ProtBERT + ESM2), structural features, and dynamic features.

## Overview
This project predicts Gene Ontology (GO) terms for proteins by combining:
- Sequence embeddings from ProtBERT and ESM2
- Static structural features
- Dynamic features
- Homology-aware data splitting (CD-HIT)
- Multiple neural architectures (MLP, CNN, Residual FFN)
- Focal loss and bootstrap evaluation
- Explainability analysis (Integrated Gradients, permutation importance, leave-one-modality-out)

## Project Structure
protein-go-prediction/
├── configs/           # Configuration files
├── data/              # Raw and processed data
├── results/           # Figures and tables
├── src/               # Notebooks and (future) source code
├── requirements.txt
└── README.md

## Status
Currently research notebooks. Being refactored into a clean, reproducible software package.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
