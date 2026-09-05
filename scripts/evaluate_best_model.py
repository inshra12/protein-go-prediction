"""
Evaluate the best MLP + PES model using the clean modules.
"""

from pathlib import Path
import numpy as np
import torch
import pandas as pd

from src.protein_go.config import load_config
from src.protein_go.data import load_feature_matrix, load_labels, load_splits
from src.protein_go.models import MLP
from src.protein_go.evaluate import (
    load_ancestor_indices,
    evaluate_predictions,
)


def main():
    # --------------------------------------------------
    # Load config and data
    # --------------------------------------------------
    cfg = load_config()
    processed_dir = Path(cfg["paths"]["data_processed"])

    X = load_feature_matrix("pes")
    Y = load_labels()
    _, _, test_idx = load_splits()

    go_map = pd.read_csv(processed_dir / "go_namespace_map.csv")
    go_ids = go_map["go_id"].tolist()

    ancestor_indices = load_ancestor_indices(processed_dir, go_ids)

    print(f"Feature matrix: {X.shape}")
    print(f"Labels:         {Y.shape}")
    print(f"Test proteins:  {len(test_idx)}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    input_dim = X.shape[1]
    num_classes = Y.shape[1]

    model = MLP(input_dim=input_dim, num_classes=num_classes)
    checkpoint_path = processed_dir / "models" / "best_homology_MLP_pes_seed42.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # --------------------------------------------------
    # Inference on test set
    # --------------------------------------------------
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.sigmoid(logits).numpy()

    print(f"Predictions shape: {probabilities.shape}")

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    results = evaluate_predictions(
        probabilities=probabilities,
        labels=Y[test_idx],
        go_ids=go_ids,
        ancestor_indices=ancestor_indices,
        apply_propagation=True,
    )

    print("\n===== Test Set Results (MLP + PES) =====")
    print(f"Overall Fmax:    {results['Overall_Fmax']:.4f}")
    print(f"Best threshold:  {results['best_threshold']:.2f}")
    print(f"Overall AUPRC:   {results['Overall_AUPRC']:.4f}")
    print("========================================")


if __name__ == "__main__":
    main()