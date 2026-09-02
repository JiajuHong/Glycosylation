#!/usr/bin/env python
"""Export predictions and build formal summaries for formal_computational_v1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from support.experiments.run_formal_computational_v1 import CONFIGS, EXPERIMENT_ID, SEEDS


METRICS = (
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_auroc",
    "test_auprc",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def binary_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "auroc_mean_score": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def metric_path(root: Path, config_name: str, seed: int) -> Path:
    return (
        root
        / "artifacts"
        / "metrics"
        / EXPERIMENT_ID
        / config_name
        / f"seed{seed}.csv"
    )


def load_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for config in CONFIGS:
        for seed in SEEDS:
            path = metric_path(root, config.name, seed)
            if not path.exists():
                raise FileNotFoundError(f"missing formal metric: {path}")
            frame = pd.read_csv(path)
            if len(frame) != 1:
                raise ValueError(f"formal metric must contain one row: {path}")
            frame.insert(0, "config_name", config.name)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def export_predictions(
    root: Path,
    metrics: pd.DataFrame,
    output_root: Path,
    device: str,
) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {}
    for _, row in metrics.iterrows():
        config_name = str(row["config_name"])
        seed = int(row["seed"])
        output = output_root / "predictions" / config_name / f"seed{seed}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "layer3.export_checkpoint_predictions",
            "--checkpoint",
            str(row["checkpoint"]),
            "--threshold",
            str(float(row["tuned_threshold"])),
            "--output",
            str(output),
            "--splits",
            "val",
            "test",
            "--batch-size",
            "128",
            "--device",
            device,
        ]
        process = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"prediction export failed for {config_name}/seed{seed}\n"
                f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )
        outputs.setdefault(config_name, []).append(output)
    return outputs


def build_ensemble(paths: list[Path], split: str = "test") -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(paths) != 3:
        raise ValueError("three seed prediction files are required")
    merged: pd.DataFrame | None = None
    for seed, path in enumerate(sorted(paths)):
        frame = pd.read_csv(path)
        frame = frame.loc[frame["split"].eq(split)].copy()
        keep = [
            "id",
            "reaction_id",
            "label",
            "Donor_Type",
            "Year",
            "Pair_Key",
            "prob_beta",
            "prediction",
        ]
        available = [column for column in keep if column in frame]
        frame = frame[available].sort_values("id").reset_index(drop=True)
        rename = {
            "prob_beta": f"seed{seed}_prob_beta",
            "prediction": f"seed{seed}_prediction",
        }
        frame = frame.rename(columns=rename)
        identity = [
            column
            for column in ("id", "reaction_id", "label", "Donor_Type", "Year", "Pair_Key")
            if column in frame
        ]
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=identity, how="inner", validate="one_to_one")
    assert merged is not None
    probability_columns = [f"seed{seed}_prob_beta" for seed in range(3)]
    prediction_columns = [f"seed{seed}_prediction" for seed in range(3)]
    merged["mean_beta_score"] = merged[probability_columns].mean(axis=1)
    merged["beta_vote_count"] = merged[prediction_columns].sum(axis=1).astype(int)
    merged["ensemble_prediction"] = (merged["beta_vote_count"] >= 2).astype(int)
    merged["seed_disagreement"] = merged[prediction_columns].nunique(axis=1).gt(1)
    merged["correct"] = merged["ensemble_prediction"].eq(merged["label"])
    metrics = binary_metrics(
        merged["label"].to_numpy(dtype=int),
        merged["ensemble_prediction"].to_numpy(dtype=int),
        merged["mean_beta_score"].to_numpy(dtype=float),
    )
    metrics.update(
        {
            "seed_disagreement_count": int(merged["seed_disagreement"].sum()),
            "seed_disagreement_rate": float(merged["seed_disagreement"].mean()),
            "unanimous_accuracy": float(
                merged.loc[~merged["seed_disagreement"], "correct"].mean()
            ),
            "disagreement_accuracy": float(
                merged.loc[merged["seed_disagreement"], "correct"].mean()
            )
            if merged["seed_disagreement"].any()
            else None,
        }
    )
    return merged, metrics


def bootstrap_cluster_ci(frame: pd.DataFrame, repetitions: int = 2000) -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    group_column = "Pair_Key" if "Pair_Key" in frame and frame["Pair_Key"].notna().all() else "id"
    groups = list(frame.groupby(group_column, sort=False).groups.values())
    values: dict[str, list[float]] = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    for _ in range(repetitions):
        selected = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([np.asarray(groups[index], dtype=int) for index in selected])
        sample = frame.loc[indices]
        labels = sample["label"].to_numpy(dtype=int)
        predictions = sample["ensemble_prediction"].to_numpy(dtype=int)
        if len(np.unique(labels)) < 2:
            continue
        values["accuracy"].append(float(accuracy_score(labels, predictions)))
        values["balanced_accuracy"].append(float(balanced_accuracy_score(labels, predictions)))
        values["macro_f1"].append(float(f1_score(labels, predictions, average="macro")))
    return {
        "method": f"cluster bootstrap by {group_column}",
        "repetitions_requested": repetitions,
        "metrics": {
            metric: {
                "lower_95": float(np.quantile(samples, 0.025)),
                "upper_95": float(np.quantile(samples, 0.975)),
                "valid_repetitions": len(samples),
            }
            for metric, samples in values.items()
        },
    }


def subgroup_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = [("overall", "all", frame)]
    definitions.extend(
        ("label", "Alpha" if int(label) == 0 else "Beta", group)
        for label, group in frame.groupby("label", sort=True)
    )
    if "Donor_Type" in frame:
        definitions.extend(
            ("donor_type", str(name), group)
            for name, group in frame.groupby("Donor_Type", sort=True)
        )
    definitions.extend(
        ("seed_agreement", "disagreement" if bool(name) else "unanimous", group)
        for name, group in frame.groupby("seed_disagreement", sort=True)
    )
    for dimension, group_name, group in definitions:
        labels = group["label"].to_numpy(dtype=int)
        predictions = group["ensemble_prediction"].to_numpy(dtype=int)
        row = {
            "dimension": dimension,
            "group": group_name,
            "n": len(group),
            "errors": int((labels != predictions).sum()),
            "accuracy": float(accuracy_score(labels, predictions)),
            "mean_beta_score": float(group["mean_beta_score"].mean()),
        }
        if len(np.unique(labels)) == 2:
            row["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))
            row["macro_f1"] = float(f1_score(labels, predictions, average="macro"))
        rows.append(row)
    return rows


def summarize_runs(metrics: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    selected = metrics.loc[metrics["config_name"].isin(configs)].copy()
    aggregations: dict[str, list[str]] = {
        metric: ["mean", "std", "min"] for metric in METRICS
    }
    aggregations["parameter_count"] = ["mean"]
    summary = selected.groupby("config_name", sort=False).agg(aggregations).reset_index()
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def paired_baseline_comparisons(
    main_frame: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the main ensemble with each baseline on identical test rows."""
    required = {"ID", "Label", "model_name", "feature_set", "y_pred"}
    missing = sorted(required.difference(baseline_predictions.columns))
    if missing:
        raise ValueError(f"baseline predictions are missing columns: {missing}")
    main = main_frame[["id", "label", "ensemble_prediction"]].copy()
    rows: list[dict[str, Any]] = []
    for (model_name, feature_set), baseline in baseline_predictions.groupby(
        ["model_name", "feature_set"], sort=True
    ):
        joined = main.merge(
            baseline[["ID", "Label", "y_pred"]],
            left_on="id",
            right_on="ID",
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(main) or not joined["label"].eq(joined["Label"]).all():
            raise ValueError(
                f"baseline rows do not align with main ensemble: {model_name}/{feature_set}"
            )
        main_correct = joined["ensemble_prediction"].eq(joined["label"])
        baseline_correct = joined["y_pred"].eq(joined["Label"])
        main_only = int((main_correct & ~baseline_correct).sum())
        baseline_only = int((~main_correct & baseline_correct).sum())
        discordant = main_only + baseline_only
        rows.append(
            {
                "baseline_model": model_name,
                "feature_set": feature_set,
                "n": len(joined),
                "main_accuracy": float(main_correct.mean()),
                "baseline_accuracy": float(baseline_correct.mean()),
                "main_correct_baseline_wrong": main_only,
                "main_wrong_baseline_correct": baseline_only,
                "mcnemar_exact_p": float(binomtest(main_only, discordant, 0.5).pvalue)
                if discordant
                else 1.0,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    root = cli.root.resolve()
    output_root = root / "results" / EXPERIMENT_ID
    output_root.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(root)
    metrics.to_csv(output_root / "all_run_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_paths = export_predictions(root, metrics, output_root, cli.device)

    architecture_configs = [
        "pair_chiral_global",
        "pair_chiral_local",
        "pair_chiral_crossattn",
        "pair_chiral_tri_l1",
        "pair_chiral_tri_l2",
        "pair_chiral_tri_l3",
    ]
    summarize_runs(metrics, architecture_configs).to_csv(
        output_root / "architecture_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_runs(
        metrics, ["pair_chiral_tri_l3", "pair_gine_tri_l3_fair"]
    ).to_csv(
        output_root / "encoder_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_runs(metrics, ["year_chiral_tri_l3"]).to_csv(
        output_root / "temporal_generalization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ensemble_frames: dict[str, pd.DataFrame] = {}
    ensemble_metrics: dict[str, Any] = {}
    for config_name, paths in prediction_paths.items():
        frame, result = build_ensemble(paths, split="test")
        ensemble_frames[config_name] = frame
        ensemble_metrics[config_name] = result
        frame.to_csv(
            output_root / "predictions" / config_name / "ensemble_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
    write_json(output_root / "ensemble_metrics.json", ensemble_metrics)

    main_name = "pair_chiral_tri_l3"
    main_frame = ensemble_frames[main_name]
    main_bootstrap = bootstrap_cluster_ci(main_frame)
    write_json(output_root / "main_ensemble_bootstrap_ci.json", main_bootstrap)
    pd.DataFrame(subgroup_rows(main_frame)).to_csv(
        output_root / "main_ensemble_subgroup_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_frame.loc[~main_frame["correct"]].to_csv(
        output_root / "main_ensemble_error_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparisons = []
    main_correct = main_frame.set_index("id")["correct"]
    for config_name in architecture_configs + ["pair_gine_tri_l3_fair"]:
        other = ensemble_frames[config_name].set_index("id")
        joined = pd.DataFrame(
            {
                "main_correct": main_correct,
                "other_correct": other["correct"],
            }
        ).dropna()
        main_only = int((joined["main_correct"] & ~joined["other_correct"]).sum())
        other_only = int((~joined["main_correct"] & joined["other_correct"]).sum())
        discordant = main_only + other_only
        p_value = (
            float(binomtest(main_only, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        comparisons.append(
            {
                "comparison_model": config_name,
                "n": len(joined),
                "main_accuracy": float(main_correct.mean()),
                "comparison_accuracy": float(other["correct"].mean()),
                "main_correct_comparison_wrong": main_only,
                "main_wrong_comparison_correct": other_only,
                "mcnemar_exact_p": p_value,
            }
        )
    pd.DataFrame(comparisons).to_csv(
        output_root / "paired_ensemble_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline = output_root / "baselines" / "baseline_results.csv"
    if not baseline.exists():
        raise FileNotFoundError(f"formal baseline results are missing: {baseline}")
    baseline_frame = pd.read_csv(baseline)
    pair_baseline = baseline_frame.loc[
        baseline_frame["split_name"].eq("split_pair_group")
    ].sort_values(["feature_set", "balanced_accuracy"], ascending=[True, False])
    pair_baseline.to_csv(
        output_root / "baseline_pair_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_predictions = pd.read_csv(
        output_root / "baselines" / "predictions_split_pair_group.csv"
    )
    paired_baseline_comparisons(main_frame, baseline_predictions).to_csv(
        output_root / "paired_baseline_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    completion = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "run_count": len(metrics),
        "prediction_export_count": sum(len(paths) for paths in prediction_paths.values()),
        "main_ensemble": ensemble_metrics[main_name],
        "temporal_ensemble": ensemble_metrics["year_chiral_tri_l3"],
        "outputs": sorted(
            str(path.relative_to(root)) for path in output_root.rglob("*") if path.is_file()
        ),
    }
    write_json(output_root / "completion_report.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
