#!/usr/bin/env python
"""复评六个冻结 checkpoint，并检查哈希、配置和批大小确定性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.eval_metrics import compute_binary_metrics, search_threshold
from layer1.predict_hard_feasibility import (
    checkpoint_acceptor_role_names,
    load_model as load_layer1_model,
)
from layer1.train_hard_feasibility import (
    HardFeasibilityDataset,
    collate as layer1_collate,
    predict as predict_layer1,
    stress_summary,
)
from layer3.glyco_dataset import GlycoDataset, glyco_collate_fn
from layer3.train_gine import move_batch_to_device
from pipeline.predict_three_layer import (
    build_layer3_model,
    load_layer3_template_and_dims,
    load_manifest,
    resolve_artifact,
)


def validate_manifest_artifacts(value: Any, root: Path) -> int:
    """递归校验清单内所有带 SHA256 的正式文件。"""
    if isinstance(value, dict):
        count = 0
        if "path" in value and "sha256" in value:
            resolve_artifact(root, value)
            count += 1
        for child in value.values():
            count += validate_manifest_artifacts(child, root)
        return count
    if isinstance(value, list):
        return sum(validate_manifest_artifacts(child, root) for child in value)
    return 0


def layer3_predictions(model, dataset: GlycoDataset, device: torch.device, batch_size: int):
    labels: list[int] = []
    scores: list[float] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=glyco_collate_fn)
    with torch.no_grad():
        for batch in loader:
            moved = move_batch_to_device(batch, device)
            scores.extend(torch.softmax(model(moved), dim=-1)[:, 1].cpu().tolist())
            labels.extend(batch["label"].tolist())
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def audit_layer1(manifest: dict[str, Any], root: Path, device: torch.device) -> list[dict[str, Any]]:
    config = manifest["layer1"]
    data_path = resolve_artifact(root, config["training_data"])
    stress_path = resolve_artifact(root, config["stress_data"])
    cache_path = resolve_artifact(root, config["graph_cache"])
    threshold = float(config["ensemble"]["threshold"])
    records = []
    for entry in config["checkpoints"]:
        path = resolve_artifact(root, entry)
        acceptor_role_names = checkpoint_acceptor_role_names([path])
        test_ds = HardFeasibilityDataset(
            data_path,
            cache_path,
            split="test",
            acceptor_role_names=acceptor_role_names,
        )
        stress_ds = HardFeasibilityDataset(
            stress_path,
            cache_path,
            split="test",
            stress=True,
            acceptor_role_names=acceptor_role_names,
        )
        model, seed = load_layer1_model(path, device)
        if seed != int(entry["seed"]):
            raise ValueError(f"第一层 seed 元数据错误: {path}")
        predictions = []
        stress_frame = None
        for batch_size in (32, 64, 32):
            loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=layer1_collate)
            frame, _ = predict_layer1(model, loader, device)
            predictions.append(frame["probability"].to_numpy())
            if batch_size == 32:
                stress_loader = DataLoader(
                    stress_ds, batch_size=batch_size, shuffle=False, collate_fn=layer1_collate
                )
                stress_frame, _ = predict_layer1(model, stress_loader, device)
        max_difference = float(np.max(np.abs(predictions[0] - predictions[1])))
        repeat_difference = float(np.max(np.abs(predictions[0] - predictions[2])))
        repeat_class_match = bool(
            np.array_equal(predictions[0] >= threshold, predictions[2] >= threshold)
        )
        cpu_gpu_class_match = True
        if device.type == "cuda":
            cpu_model, _ = load_layer1_model(path, torch.device("cpu"))
            cpu_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=layer1_collate)
            cpu_frame, _ = predict_layer1(cpu_model, cpu_loader, torch.device("cpu"))
            cpu_gpu_class_match = bool(
                np.array_equal(
                    predictions[0] >= threshold,
                    cpu_frame["probability"].to_numpy() >= threshold,
                )
            )
        metrics = compute_binary_metrics(test_ds.df["hard_feasibility"], predictions[0], threshold=threshold)
        records.append(
            {
                "seed": seed,
                "checkpoint": str(path.relative_to(root)),
                "acceptor_role_names": acceptor_role_names,
                "explicit_o4_external_role_channel": (
                    "O4_EXTERNAL" in acceptor_role_names
                ),
                "test_n": len(test_ds),
                "test_metrics": metrics,
                "stress": stress_summary(stress_frame, threshold),
                "batch_size_max_abs_difference": max_difference,
                "deterministic_batch_check": max_difference <= 1e-6,
                "repeat_max_abs_difference": repeat_difference,
                "repeat_class_match": repeat_class_match,
                "repeat_check": repeat_difference <= 1e-6 and repeat_class_match,
                "cpu_gpu_class_match": cpu_gpu_class_match,
            }
        )
    return records


def audit_layer2(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    """核对冻结的第二层方法边界和既有五池评测结论。"""
    config = manifest["layer2"]
    audit_path = resolve_artifact(root, config["audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = {
        "u_pool_qa_pass": bool(audit["u_generation"]["qa"]["overall_pass"]),
        "weights_match": audit["features"]["weights"] == config["weights"],
        "not_called_probability": "not a calibrated reaction probability"
        in audit["probability_warning"],
        "not_a_hard_gate": config.get("hard_gate") is False,
        "five_u_pools": int(audit["u_generation"]["pool_count"]) == 5,
    }
    return {
        "method": config["method"],
        "checks": checks,
        "overall_status": "passed" if all(checks.values()) else "failed",
        "test_summary": audit["validation_summary"]["test"],
        "pool_stability": audit["validation_summary"]["test_pool_stability"],
        "interpretation": "literature support/applicability domain; not reaction probability",
    }


def audit_layer3(manifest: dict[str, Any], root: Path, device: torch.device) -> list[dict[str, Any]]:
    records = []
    expected_args = {
        "model_type": "crossattn_tri",
        "local_output_mode": "l3",
        "encoder_type": "chiral_gine",
        "split_column": "split_pair_group",
    }
    for entry in manifest["layer3"]["checkpoints"]:
        path = resolve_artifact(root, entry)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        seed = int(entry["seed"])
        if int(args["seed"]) != seed:
            raise ValueError(f"第三层 seed 元数据错误: {path}")
        mismatches = {
            key: {"expected": value, "actual": args.get(key)}
            for key, value in expected_args.items()
            if args.get(key) != value
        }
        if mismatches:
            raise ValueError(f"第三层架构配置不一致 {path}: {mismatches}")
        template, dims = load_layer3_template_and_dims(checkpoint, root)
        model = build_layer3_model(checkpoint, dims, device)
        dataset_kwargs = {
            "csv_path": template.csv_path,
            "split_column": template.split_column,
            "encoder_type": template.encoder_type,
            "solvent_vocab_path": root / "data/processed/solvent_vocab.json",
            "catalyst_vocab_path": root / "data/processed/catalyst_vocab.json",
            "chiral_graph_cache_path": root / manifest["layer3"]["graph_cache"]["path"],
            "stats": template.stats,
            "validate": True,
        }
        split_results: dict[str, Any] = {}
        for split in ("val", "test"):
            dataset = GlycoDataset(split=split, **dataset_kwargs)
            labels_a, scores_a = layer3_predictions(model, dataset, device, 64)
            labels_b, scores_b = layer3_predictions(model, dataset, device, 128)
            labels_c, scores_c = layer3_predictions(model, dataset, device, 64)
            if not np.array_equal(labels_a, labels_b):
                raise ValueError(f"第三层不同 batch size 标签顺序不一致: {path}, {split}")
            max_difference = float(np.max(np.abs(scores_a - scores_b)))
            repeat_difference = float(np.max(np.abs(scores_a - scores_c)))
            repeat_class_match = bool(
                np.array_equal(
                    scores_a >= float(entry["threshold"]),
                    scores_c >= float(entry["threshold"]),
                )
            )
            tuned_threshold = None
            threshold_reproduced = True
            if split == "val":
                tuned_threshold, _ = search_threshold(labels_a, scores_a)
                threshold_reproduced = abs(float(tuned_threshold) - float(entry["threshold"])) <= 1e-12
            split_results[split] = {
                "n": len(dataset),
                "metrics": compute_binary_metrics(
                    labels_a, scores_a, threshold=float(entry["threshold"])
                ),
                "batch_size_max_abs_difference": max_difference,
                "deterministic_batch_check": max_difference <= 1e-6,
                "repeat_max_abs_difference": repeat_difference,
                "repeat_class_match": repeat_class_match,
                "repeat_check": repeat_difference <= 1e-6 and repeat_class_match,
                "reproduced_validation_threshold": tuned_threshold,
                "threshold_reproduced": threshold_reproduced,
            }
        cpu_gpu_class_match = True
        if device.type == "cuda":
            cpu_model = build_layer3_model(checkpoint, dims, torch.device("cpu"))
            test_dataset = GlycoDataset(split="test", **dataset_kwargs)
            cpu_labels, cpu_scores = layer3_predictions(
                cpu_model, test_dataset, torch.device("cpu"), 64
            )
            gpu_labels, gpu_scores = layer3_predictions(model, test_dataset, device, 64)
            cpu_gpu_class_match = bool(
                np.array_equal(cpu_labels, gpu_labels)
                and np.array_equal(
                    cpu_scores >= float(entry["threshold"]),
                    gpu_scores >= float(entry["threshold"]),
                )
            )
        records.append(
            {
                "seed": seed,
                "checkpoint": str(path.relative_to(root)),
                "threshold": float(entry["threshold"]),
                "score_semantics": "class-1 Beta softmax score; not calibrated probability",
                "cpu_gpu_class_match": cpu_gpu_class_match,
                **split_results,
            }
        )
    return records


def audit_condition_screening(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    """审计湿实验条件筛选应用层，不重新评估三层模型。"""
    config = manifest["application"]["condition_screening"]
    library = pd.read_csv(resolve_artifact(root, config["template_library"]))
    template_counts = library.groupby("Donor_Type").size().to_dict()
    template_counts = {str(key): int(value) for key, value in template_counts.items()}
    checks = {
        "template_count": len(library) == int(config["template_count"]) == 539,
        "unique_template_ids": not library["Template_ID"].duplicated().any(),
        "complete_condition_flags": bool(
            library[["has_solvent", "has_catalyst", "has_temp", "has_time"]]
            .eq(1)
            .all(axis=None)
        ),
        "supporting_reaction_count": int(library["Template_Support_Count"].sum())
        == int(config["complete_supporting_reactions"])
        == 1058,
        "all_curated_templates_used": template_counts
        == {
            "thioglycoside": 291,
            "trichloroacetimidate": 200,
            "trifluoroacetimidate": 48,
        },
        "no_catalyst_exclusion_metadata": not {
            "Catalyst_Mechanism_Role",
            "Mechanism_Compatible",
            "Mechanism_Filter_Version",
        }.intersection(library.columns),
        "not_called_probability": "not a success probability" in config["interpretation"],
        "recipe_limitation_recorded": "verify equivalents" in config["recipe_limitation"],
        "original_condition_exclusion_recorded": (
            "Source_Reaction_ID" in config["original_condition_exclusion"]
            and "condition key" in config["original_condition_exclusion"]
        ),
        "unanimous_vote_policy_recorded": (
            "A requires 3/3" in config["ranking"]
            and "B requires 3/3" in config["ranking"]
            and "2/3 disagreement" in config["ranking"]
        ),
        "exploratory_rescue_policy_recorded": (
            "exploratory_rescue" in config["rescue_policy"]
            and "reaction success is not guaranteed" in config["rescue_policy"]
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "checks": checks,
        "template_counts_by_donor_type": template_counts,
        "overall_status": "passed" if all(checks.values()) else "failed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/model_manifest_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/frozen_model_audit_v2.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    manifest, root = load_manifest(cli.manifest.resolve())
    device = torch.device(cli.device)
    validated_artifact_count = validate_manifest_artifacts(manifest, root)
    report = {
        "pipeline_version": manifest["pipeline_version"],
        "manifest": str(cli.manifest),
        "device": str(device),
        "hash_check": {"status": "passed", "artifact_count": validated_artifact_count},
        "layer1": audit_layer1(manifest, root, device),
        "layer2": audit_layer2(manifest, root),
        "layer3": audit_layer3(manifest, root, device),
        "condition_screening": audit_condition_screening(manifest, root),
    }
    all_checks = [
        record[key]
        for record in report["layer1"]
        for key in ("deterministic_batch_check", "repeat_check", "cpu_gpu_class_match")
    ] + [
        record[split][key]
        for record in report["layer3"]
        for split in ("val", "test")
        for key in ("deterministic_batch_check", "repeat_check", "threshold_reproduced")
    ] + [
        record["cpu_gpu_class_match"] for record in report["layer3"]
    ] + [
        report["layer2"]["overall_status"] == "passed",
        report["condition_screening"]["overall_status"] == "passed",
    ]
    report["overall_status"] = "passed" if all(all_checks) else "failed"
    output = cli.output if cli.output.is_absolute() else root / cli.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "output": str(output)}, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
