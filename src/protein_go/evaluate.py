"""
Evaluation metrics for multi-label GO prediction.

Implements CAFA protein-centric Fmax and macro AUPRC,
with optional GO true-path propagation and root-term exclusion.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


from src.protein_go.config import load_config

# --------------------------------------------------
# Root terms (CAFA standard)
# --------------------------------------------------
ROOT_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}


def get_scored_term_mask(go_ids):
    """
    Return a boolean mask that excludes the three ontology root terms.
    """
    root_indexes = [i for i, go_id in enumerate(go_ids) if go_id in ROOT_TERMS]
    mask = np.ones(len(go_ids), dtype=bool)
    mask[root_indexes] = False
    return mask


# --------------------------------------------------
# GO true-path propagation
# --------------------------------------------------
def load_ancestor_indices(processed_dir, go_ids):
    """
    Load the retained GO ancestor map and convert it to index-based form.
    """
    map_path = Path(processed_dir) / "retained_go_ancestor_map.json"
    with open(map_path) as f:
        ancestor_go_map = json.load(f)

    go_to_index = {go_id: idx for idx, go_id in enumerate(go_ids)}
    ancestor_indices = {}

    for child_go, parent_gos in ancestor_go_map.items():
        if child_go not in go_to_index:
            continue
        child_idx = go_to_index[child_go]
        parent_idxs = [go_to_index[p] for p in parent_gos if p in go_to_index]
        ancestor_indices[child_idx] = parent_idxs

    return ancestor_indices


def propagate_prediction_scores(probabilities, ancestor_indices):
    """
    Apply the GO true-path rule to prediction scores.
    """
    propagated = probabilities.copy()
    for child_idx, parent_idxs in ancestor_indices.items():
        if parent_idxs:
            propagated[:, parent_idxs] = np.maximum(
                propagated[:, parent_idxs],
                propagated[:, [child_idx]],
            )
    return propagated


# --------------------------------------------------
# Core metrics
# --------------------------------------------------
def cafa_fmax(probabilities, labels, thresholds=None):
    """
    CAFA protein-centric Fmax.

    Precision is averaged only over proteins that receive at least one prediction.
    Recall is averaged over all proteins.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    best_fmax = 0.0
    best_threshold = 0.0
    labels_bool = labels.astype(bool)
    true_count = labels_bool.sum(axis=1)

    for t in thresholds:
        predicted = probabilities >= t
        pred_count = predicted.sum(axis=1)
        tp = (predicted & labels_bool).sum(axis=1)

        has_pred = pred_count > 0
        precision = (
            np.mean(tp[has_pred] / pred_count[has_pred]) if has_pred.any() else 0.0
        )
        recall = np.mean(tp / np.maximum(true_count, 1))

        if precision + recall > 0:
            f = 2 * precision * recall / (precision + recall)
        else:
            f = 0.0

        if f > best_fmax:
            best_fmax = float(f)
            best_threshold = float(t)

    return {
        "Fmax": best_fmax,
        "best_threshold": best_threshold,
    }


def macro_auprc(probabilities, labels):
    """
    Macro-averaged AUPRC over GO terms that have both positive and negative examples.
    """
    scores = []
    for i in range(labels.shape[1]):
        y = labels[:, i]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        scores.append(average_precision_score(y, probabilities[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


# --------------------------------------------------
# High-level evaluation helper
# --------------------------------------------------
def evaluate_predictions(
    probabilities,
    labels,
    go_ids,
    ancestor_indices=None,
    apply_propagation=True,
):
    """
    Compute Overall Fmax and AUPRC with optional propagation and root-term exclusion.
    """
    if apply_propagation and ancestor_indices is not None:
        probabilities = propagate_prediction_scores(probabilities, ancestor_indices)

    scored_mask = get_scored_term_mask(go_ids)

    fmax_result = cafa_fmax(
        probabilities[:, scored_mask],
        labels[:, scored_mask],
    )
    auprc = macro_auprc(
        probabilities[:, scored_mask],
        labels[:, scored_mask],
    )

    return {
        "Overall_Fmax": fmax_result["Fmax"],
        "best_threshold": fmax_result["best_threshold"],
        "Overall_AUPRC": auprc,
    }