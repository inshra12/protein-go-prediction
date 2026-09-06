# Protein GO Function Prediction with Protein Language Models and ATLAS Features

Multi-label Gene Ontology (GO) prediction using protein language model embeddings (ProtBERT + ESM2) and ATLAS molecular dynamics features.

## Publication Status

This project is associated with a manuscript currently **under revision**.  
The code and evaluation pipeline reproduce the main experiments described in the manuscript.

## Overview

Protein function prediction remains challenging because most sequenced proteins lack experimental GO annotations. This project investigates whether molecular dynamics (MD) features from the ATLAS database improve multi-label GO prediction beyond what modern protein language models already capture.

We perform a controlled ablation study across:

- 9 feature combinations (ProtBERT, ESM2, ATLAS Static, ATLAS Dynamic)
- 3 neural architectures (MLP, CNN1D, Residual FFN)
- Homology-aware data splits (CD-HIT at 40% identity)
- GO true-path propagation
- CAFA protein-centric Fmax evaluation

**Best configuration:** MLP trained on ProtBERT + ESM2 + ATLAS Static features (PES)

## Key Results

| Model              | Feature Set | Overall Fmax (mean ± SD) | Overall AUPRC |
|--------------------|-------------|---------------------------|---------------|
| Naïve Baseline     | —           | 0.358                     | 0.037         |
| **MLP**            | **PES**     | **0.468 ± 0.012**         | **0.297**     |
| ResFFN             | PES         | 0.465 ± 0.019             | 0.284         |
| CNN1D              | ESD         | 0.439 ± 0.023             | 0.220         |

**Main ablation findings:**
- Combining ProtBERT + ESM2 consistently outperformed either embedding alone.
- Adding ATLAS Dynamic features produced no statistically significant improvement in any of the nine matched comparisons.
- Integrated Gradients (supported by permutation importance and leave-one-modality-out) showed that ESM2 and ProtBERT together account for ~97.5% of model attribution, while ATLAS Static features contribute only ~2.4%.

## Method Summary

**Research question:**  
Do ATLAS molecular dynamics features provide additional signal for multi-label GO prediction beyond ProtBERT and ESM2 embeddings?

**Approach:**
- Constructed a dataset of 1,356 proteins at the intersection of ATLAS and UniProtKB.
- Extracted four feature types: ProtBERT, ESM2, ATLAS Static, and ATLAS Dynamic descriptors.
- Formed 9 feature combinations and trained 3 architectures (MLP, CNN1D, ResFFN), producing 27 configurations.
- Used homology-aware CD-HIT splits (40% identity) to prevent sequence-family leakage.
- Applied GO true-path propagation and evaluated with CAFA protein-centric Fmax and macro AUPRC.
- Quantified feature importance using Integrated Gradients, permutation importance, and leave-one-modality-out ablation.

## Repository Structure

```text
protein-go-prediction/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── src/
│   └── protein_go/
│       ├── config.py
│       ├── data.py
│       ├── models.py
│       └── evaluate.py
├── scripts/
│   └── evaluate_best_model.py      
└── README.md

```
## How to Run the Best Model Evaluation
Bash# Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run evaluation of the best MLP + PES model
python -m scripts.evaluate_best_model
Expected output (approximate):
textOverall Fmax:   ~0.45
Overall AUPRC:  ~0.27
## Main Findings

Protein language model embeddings (especially ESM2) already capture most of the functional signal relevant to GO prediction on this dataset.
Adding ATLAS Dynamic features does not yield consistent gains under the tested conditions.
A relatively simple MLP outperforms or matches more complex architectures when strong embeddings are used.

## Limitations

Dataset size is constrained by ATLAS coverage (1,356 proteins).
Only three random seeds were used for statistical comparisons.
Results are specific to the ATLAS–UniProt intersection and should not be directly compared with large-scale CAFA benchmarks.

## Future Work

Residue-level dynamics features instead of protein-level aggregates
Hierarchical or ontology-aware loss functions
Larger-scale evaluation beyond ATLAS coverage
Integration with structure-based representations (e.g. Foldseek / AlphaFold)

## Citation
If you use this code or findings, please cite the associated manuscript (currently under revision).