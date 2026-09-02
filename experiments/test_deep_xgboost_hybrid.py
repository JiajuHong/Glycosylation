#!/usr/bin/env python
"""Compare fair XGBoost heads using frozen L3 and optional raw features.

Every neural checkpoint is frozen. XGBoost is fitted on the training split,
the classification threshold is selected on validation, and test is evaluated
once. Donor_Type is deliberately excluded from every raw feature block.
Fitted heads are not persisted; this script is not a production predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from baseline.run_baseline_ml import FP_BITS, FP_INCLUDE_CHIRALITY, build_feature_blocks
from experiments.test_deep_xgboost_head import (
    DEFAULT_CHECKPOINTS,
    best_threshold,
    collect_embeddings,
    load_model_and_data,
    metrics,
)
from layer3.glyco_dataset import glyco_collate_fn


DEFAULT_OUTPUT_DIR = Path("results/deep_xgboost_hybrid_v1")
FEATURE_MODES = ("deep_only", "deep_structure", "deep_structure_condition")


def classifier(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )


def run_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    model, datasets, saved = load_model_and_data(checkpoint_path, device)
    embedded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=glyco_collate_fn,
        )
        embedded[split] = collect_embeddings(model, loader, device)

    frames = []
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for split in ("train", "val", "test"):
        frame = datasets[split].df.copy().reset_index(drop=True)
        frame["_hybrid_split"] = split
        frames.append(frame)
        offsets[split] = (start, start + len(frame))
        start += len(frame)
    combined = pd.concat(frames, ignore_index=True)
    raw = build_feature_blocks(combined, "_hybrid_split")

    deep_all = sparse.csr_matrix(
        np.concatenate([embedded[s][0] for s in ("train", "val", "test")])
    )
    matrices = {
        "deep_only": deep_all,
        "deep_structure": sparse.hstack(
            # deep_all already contains the four condition tokens.
            [deep_all, raw["structure_condition"][:, :2 * FP_BITS]], format="csr"
        ),
        "deep_structure_condition": sparse.hstack(
            [deep_all, raw["structure_condition"]], format="csr"
        ),
    }

    y = {
        split: embedded[split][1]
        for split in ("train", "val", "test")
    }
    scale = float((y["train"] == 0).sum() / (y["train"] == 1).sum())
    seed = int(saved["seed"])
    rows: list[dict[str, object]] = []
    for mode in FEATURE_MODES:
        matrix = matrices[mode]
        sliced = {
            split: matrix[begin:end]
            for split, (begin, end) in offsets.items()
        }
        head = classifier(scale)
        head.fit(sliced["train"], y["train"])
        val_probability = head.predict_proba(sliced["val"])[:, 1]
        threshold = best_threshold(y["val"], val_probability)
        test_probability = head.predict_proba(sliced["test"])[:, 1]

        prediction = datasets["test"].df[
            ["ID", "Reaction_ID", "Label", "Product_Config", "Pair_Key"]
        ].copy()
        prediction["prob_beta"] = test_probability
        prediction["prediction"] = (test_probability >= threshold).astype(int)
        prediction["threshold"] = threshold
        prediction.to_csv(
            output_dir / f"seed{seed}_{mode}_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rows.append(
            {
                "seed": seed,
                "feature_mode": mode,
                "checkpoint": str(checkpoint_path),
                "feature_dim": int(matrix.shape[1]),
                **{
                    f"val_{key}": value
                    for key, value in metrics(y["val"], val_probability, threshold).items()
                },
                **{
                    f"test_{key}": value
                    for key, value in metrics(y["test"], test_probability, threshold).items()
                },
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for checkpoint_path in args.checkpoints:
        rows.extend(
            run_checkpoint(
                checkpoint_path, args.output_dir, torch.device(args.device), args.batch_size
            )
        )
    frame = pd.DataFrame(rows).sort_values(["feature_mode", "seed"])
    frame.to_csv(args.output_dir / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        frame.groupby("feature_mode")
        .agg(
            macro_f1_mean=("test_macro_f1", "mean"),
            macro_f1_std=("test_macro_f1", "std"),
            balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
            auroc_mean=("test_auroc", "mean"),
            auprc_mean=("test_auprc", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment": "deep_xgboost_hybrid_v1",
                "deployment_status": "experimental_only_not_integrated",
                "feature_modes": list(FEATURE_MODES),
                "excluded_features": ["Donor_Type"],
                "fingerprint_include_chirality": FP_INCLUDE_CHIRALITY,
                "selection": "fixed XGBoost parameters; validation threshold only",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
