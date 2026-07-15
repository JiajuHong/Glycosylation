from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true, y_prob, threshold: float = 0.5, include_threshold_agnostic: bool = True
) -> dict[str, float | str]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float | str] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist().__repr__(),
    }
    if include_threshold_agnostic:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["auprc"] = float("nan")
    return metrics


def search_threshold(
    y_true, y_prob, start: float = 0.05, end: float = 0.95, num: int = 181
) -> tuple[float, dict]:
    """Search for the threshold that maximizes balanced accuracy on validation set."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict = {}

    for threshold in np.linspace(start, end, num):
        y_pred = (y_prob >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = compute_binary_metrics(
                y_true, y_prob, threshold=float(threshold), include_threshold_agnostic=False
            )

    return best_threshold, best_metrics
