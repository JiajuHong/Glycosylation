#!/usr/bin/env python
"""Paired statistical comparison of ordinary GINE and Chiral-GINE predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc", "log_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gine", required=True)
    parser.add_argument("--chiral", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def compute_metrics_arrays(
    y: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "auroc": roc_auc_score(y, prob),
        "auprc": average_precision_score(y, prob),
        "log_loss": log_loss(y, np.column_stack([1.0 - prob, prob]), labels=[0, 1]),
    }


def main() -> int:
    args = parse_args()
    ordinary = pd.read_csv(args.gine)
    chiral = pd.read_csv(args.chiral)
    ordinary = ordinary.loc[ordinary["split"].eq(args.split)].copy()
    chiral = chiral.loc[chiral["split"].eq(args.split)].copy()
    key = ["id", "split"]
    merged = ordinary.merge(chiral, on=key, suffixes=("_gine", "_chiral"), validate="one_to_one")
    if len(merged) != len(ordinary) or len(merged) != len(chiral):
        raise ValueError("Prediction files do not contain the same reaction IDs")
    if not merged["label_gine"].equals(merged["label_chiral"]):
        raise ValueError("Labels differ between prediction files")
    merged["label"] = merged["label_gine"]

    y_all = merged["label"].to_numpy()
    pred_gine_all = merged["prediction_gine"].to_numpy()
    pred_chiral_all = merged["prediction_chiral"].to_numpy()
    prob_gine_all = merged["prob_beta_gine"].to_numpy()
    prob_chiral_all = merged["prob_beta_chiral"].to_numpy()
    metrics_gine = compute_metrics_arrays(y_all, pred_gine_all, prob_gine_all)
    metrics_chiral = compute_metrics_arrays(y_all, pred_chiral_all, prob_chiral_all)
    observed = {metric: metrics_chiral[metric] - metrics_gine[metric] for metric in METRICS}

    gine_correct = merged["correct_gine"].astype(bool)
    chiral_correct = merged["correct_chiral"].astype(bool)
    regressed = int((gine_correct & ~chiral_correct).sum())
    corrected = int((~gine_correct & chiral_correct).sum())
    discordant = regressed + corrected
    mcnemar_p = float(binomtest(min(regressed, corrected), discordant, 0.5).pvalue) if discordant else 1.0

    rng = np.random.default_rng(args.seed)
    group_column = "Pair_Key_gine" if "Pair_Key_gine" in merged.columns else None
    groups = merged[group_column].drop_duplicates().to_numpy() if group_column else None
    group_indices = (
        [np.flatnonzero(merged[group_column].to_numpy() == group) for group in groups]
        if groups is not None
        else None
    )
    bootstrap_rows: list[dict[str, float]] = []
    for iteration in range(args.bootstrap):
        if groups is None:
            sampled_indices = rng.integers(0, len(merged), len(merged))
        else:
            sampled_group_indices = rng.integers(0, len(groups), len(groups))
            sampled_indices = np.concatenate(
                [group_indices[group_idx] for group_idx in sampled_group_indices]
            )
        y = y_all[sampled_indices]
        if np.unique(y).size < 2:
            continue
        left = compute_metrics_arrays(
            y,
            pred_gine_all[sampled_indices],
            prob_gine_all[sampled_indices],
        )
        right = compute_metrics_arrays(
            y,
            pred_chiral_all[sampled_indices],
            prob_chiral_all[sampled_indices],
        )
        bootstrap_rows.append(
            {"iteration": iteration, **{metric: right[metric] - left[metric] for metric in METRICS}}
        )

    bootstrap = pd.DataFrame(bootstrap_rows)
    summary_rows = []
    for metric in METRICS:
        values = bootstrap[metric].to_numpy()
        summary_rows.append(
            {
                "metric": metric,
                "gine": metrics_gine[metric],
                "chiral_gine": metrics_chiral[metric],
                "difference_chiral_minus_gine": observed[metric],
                "bootstrap_ci_low_95": float(np.quantile(values, 0.025)),
                "bootstrap_ci_high_95": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": len(values),
                "bootstrap_unit": "Pair_Key" if groups is not None else "reaction",
            }
        )
    summary = pd.DataFrame(summary_rows)

    merged["comparison"] = np.select(
        [~gine_correct & chiral_correct, gine_correct & ~chiral_correct],
        ["chiral_corrected_gine_error", "chiral_regressed_from_gine"],
        default="same_correctness",
    )
    merged["probability_change_chiral_minus_gine"] = (
        merged["prob_beta_chiral"] - merged["prob_beta_gine"]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"paired_metric_summary_{args.split}.csv", index=False)
    bootstrap.to_csv(output_dir / f"paired_group_bootstrap_{args.split}.csv", index=False)
    merged.loc[merged["comparison"].ne("same_correctness")].to_csv(
        output_dir / f"discordant_reactions_{args.split}.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "split": args.split,
                "n_reactions": len(merged),
                "gine_wrong_chiral_correct": corrected,
                "gine_correct_chiral_wrong": regressed,
                "discordant_total": discordant,
                "mcnemar_exact_pvalue": mcnemar_p,
            }
        ]
    ).to_csv(output_dir / f"mcnemar_{args.split}.csv", index=False)
    print(summary.to_string(index=False))
    print(f"McNemar: corrected={corrected}, regressed={regressed}, exact_p={mcnemar_p:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
