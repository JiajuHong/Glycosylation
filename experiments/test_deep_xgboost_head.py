#!/usr/bin/env python
"""Pilot: replace the trained L3 model's MLP head with an XGBoost head.

The neural checkpoint is trained only on the training split (with validation
early stopping).  This script freezes it, exports the pre-classifier fusion
vector for train/validation/test, fits XGBoost on train only, tunes only the
decision threshold on validation, and evaluates test once.

This is an experimental comparison, not a deployable inference backend: fitted
XGBoost heads are not persisted and the production pipeline still uses the MLP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from layer3.glyco_dataset import GlycoDataset, glyco_collate_fn
from layer3.train_gine import get_feature_dims, move_batch_to_device
from models.glyco_gine_models import GlycoGINECrossAttnTri, build_glyco_gine_model


DEFAULT_CHECKPOINTS = tuple(
    sorted(Path("artifacts/checkpoints/layer3_v1").glob("*.pt"))
)
DEFAULT_OUTPUT_DIR = Path("results/deep_xgboost_head_pilot_v1")


def best_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        balanced_accuracy_score(y_true, probability >= threshold)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, object]:
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).tolist(),
    }


def fusion_vector(model: GlycoGINECrossAttnTri, batch: dict) -> torch.Tensor:
    enc = model.encode_common(batch)
    interaction = model.encode_local_interaction(batch, enc)
    return torch.cat(
        [
            enc["t_donor"],
            enc["t_acceptor"],
            *model._select_local_tokens(interaction),
            enc["t_solv"],
            enc["t_cat"],
            enc["t_temp"],
            enc["t_time"],
        ],
        dim=-1,
    )


@torch.no_grad()
def collect_embeddings(
    model: GlycoGINECrossAttnTri,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    vectors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        vectors.append(fusion_vector(model, batch).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return np.concatenate(vectors), np.concatenate(labels).astype(int)


def load_model_and_data(
    checkpoint_path: Path, device: torch.device
) -> tuple[GlycoGINECrossAttnTri, dict[str, GlycoDataset], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint["args"]
    if saved["model_type"] != "crossattn_tri" or saved["local_output_mode"] != "l3":
        raise ValueError(f"pilot expects crossattn_tri/l3, got {checkpoint_path}")

    dataset_kwargs = {
        "split_column": saved["split_column"],
        "encoder_type": saved["encoder_type"],
        "chiral_graph_cache_path": saved["chiral_graph_cache"],
        "validate": True,
    }
    train = GlycoDataset(saved["csv"], split="train", **dataset_kwargs)
    datasets = {
        "train": train,
        "val": GlycoDataset(saved["csv"], split="val", stats=train.stats, **dataset_kwargs),
        "test": GlycoDataset(saved["csv"], split="test", stats=train.stats, **dataset_kwargs),
    }
    atom_dim, bond_dim, num_solvent, num_catalyst = get_feature_dims(train)
    model = build_glyco_gine_model(
        model_type=saved["model_type"],
        atom_feat_dim=atom_dim,
        bond_feat_dim=bond_dim,
        num_solvent_tokens=num_solvent,
        num_catalyst_tokens=num_catalyst,
        hidden_dim=saved["hidden_dim"],
        num_layers=saved["num_layers"],
        dropout=saved["dropout"],
        encoder_type=saved["encoder_type"],
        perm_cat_dropout=saved["perm_cat_dropout"],
        perm_cat_normalization=saved["perm_cat_normalization"],
        local_output_mode=saved["local_output_mode"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, datasets, saved


def run_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
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

    x_train, y_train = embedded["train"]
    x_val, y_val = embedded["val"]
    x_test, y_test = embedded["test"]
    n_negative = int((y_train == 0).sum())
    n_positive = int((y_train == 1).sum())
    classifier = XGBClassifier(
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
        scale_pos_weight=n_negative / n_positive,
    )
    classifier.fit(x_train, y_train)
    val_probability = classifier.predict_proba(x_val)[:, 1]
    threshold = best_threshold(y_val, val_probability)
    test_probability = classifier.predict_proba(x_test)[:, 1]
    seed = int(saved["seed"])

    prediction_frame = datasets["test"].df[
        ["ID", "Reaction_ID", "Label", "Product_Config", "Pair_Key"]
    ].copy()
    prediction_frame["prob_beta"] = test_probability
    prediction_frame["prediction"] = (test_probability >= threshold).astype(int)
    prediction_frame["threshold"] = threshold
    prediction_frame.to_csv(
        output_dir / f"seed{seed}_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "split_column": saved["split_column"],
        "embedding_dim": int(x_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        **{f"val_{key}": value for key, value in metrics(y_val, val_probability, threshold).items()},
        **{f"test_{key}": value for key, value in metrics(y_test, test_probability, threshold).items()},
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = [
        run_checkpoint(path, args.output_dir, device, args.batch_size)
        for path in args.checkpoints
    ]
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(args.output_dir / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    metric_names = ["test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_auroc", "test_auprc"]
    summary = {
        "experiment": "deep_xgboost_head_pilot_v1",
        "deployment_status": "experimental_only_not_integrated",
        "purpose": "fixed-XGBoost replacement of the frozen L3 MLP classification head",
        "seeds": frame["seed"].astype(int).tolist(),
        "metrics": {
            name: {
                "mean": float(frame[name].mean()),
                "std": float(frame[name].std(ddof=1)),
                "values": frame[name].astype(float).tolist(),
            }
            for name in metric_names
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
