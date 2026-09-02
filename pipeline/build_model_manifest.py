#!/usr/bin/env python
"""根据冻结产物生成可校验的 V1 模型清单。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from pipeline.predict_three_layer import sha256_file


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def artifact(root: Path, path: Path, **extra: Any) -> dict[str, Any]:
    absolute = path if path.is_absolute() else root / path
    if not absolute.exists():
        raise FileNotFoundError(absolute)
    try:
        stored_path = str(absolute.resolve().relative_to(root.resolve()))
    except ValueError:
        stored_path = str(absolute.resolve())
    return {
        "path": stored_path,
        "sha256": sha256_file(absolute),
        "size_bytes": absolute.stat().st_size,
        **extra,
    }


def result_threshold(path: Path, seed: int) -> float:
    frame = pd.read_csv(path)
    if "seed" in frame:
        frame = frame.loc[pd.to_numeric(frame["seed"], errors="coerce") == seed]
    if frame.empty or "tuned_threshold" not in frame:
        raise ValueError(f"无法从 {path} 读取 seed={seed} 的 tuned_threshold")
    return float(frame.iloc[-1]["tuned_threshold"])


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_manifest_v2.json"))
    parser.add_argument("--pipeline-version", default="1.5.0")
    parser.add_argument(
        "--source-git-commit",
        help=(
            "Clean source snapshot commit when the manifest is generated from a synced "
            "server worktree whose local Git metadata differs."
        ),
    )
    parser.add_argument(
        "--layer1-artifact-version",
        default="layer1_no_o4_external_role_v1",
        help="Directory name used under artifacts/checkpoints and artifacts/metrics.",
    )
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    root = cli.root.resolve()
    output = cli.output if cli.output.is_absolute() else root / cli.output

    layer1_entries = []
    layer1_metrics_payloads = []
    for seed in range(3):
        entry = artifact(
            root,
            Path(
                f"artifacts/checkpoints/{cli.layer1_artifact_version}/seed{seed}.pt"
            ),
            seed=seed,
        )
        metrics_path = Path(
            f"artifacts/metrics/{cli.layer1_artifact_version}/seed{seed}.json"
        )
        entry["metrics"] = artifact(
            root,
            metrics_path,
        )
        entry["history"] = artifact(
            root,
            Path(
                f"artifacts/metrics/{cli.layer1_artifact_version}/"
                f"seed{seed}_history.csv"
            ),
        )
        layer1_entries.append(entry)
        layer1_metrics_payloads.append(
            json.loads((root / metrics_path).read_text(encoding="utf-8"))
        )
    layer3_entries = []
    for seed in range(3):
        checkpoint_path = Path(
            "artifacts/checkpoints/layer3_v1/"
            f"chiral_gine_crossattn_tri_l3_split_pair_group_seed{seed}_v1.pt"
        )
        result_path = root / f"artifacts/metrics/layer3_v1_seed{seed}.csv"
        entry = artifact(
            root,
            checkpoint_path,
            seed=seed,
            threshold=result_threshold(result_path, seed),
        )
        entry["metrics"] = artifact(root, result_path)
        entry["history"] = artifact(root, checkpoint_path.with_suffix(".history.csv"))
        layer3_entries.append(entry)

    generation_git_commit = git_value(root, "rev-parse", "HEAD")
    generation_status = git_value(root, "status", "--porcelain")
    source_git_commit = cli.source_git_commit or generation_git_commit
    source_paths = [
        Path("pipeline/build_model_manifest.py"),
        Path("pipeline/audit_frozen_models.py"),
        Path("pipeline/build_condition_library.py"),
        Path("pipeline/predict_three_layer.py"),
        Path("pipeline/run_e2e_acceptance.py"),
        Path("pipeline/screen_conditions.py"),
        Path("pipeline/status.py"),
        *sorted(Path("pipeline/tests").glob("test_*.py")),
        Path("layer1/predict_hard_feasibility.py"),
        Path("layer1/train_hard_feasibility.py"),
        *sorted(Path("layer1/tests").glob("test_*.py")),
        Path("layer2/retrieve_literature_evidence.py"),
        *sorted(Path("layer2/tests").glob("test_*.py")),
        Path("layer3/glyco_dataset.py"),
        Path("layer3/chiral_graph.py"),
        Path("layer3/extract_donor_rfu.py"),
        Path("layer3/extract_acceptor_4oh.py"),
        Path("layer3/tokenize_conditions.py"),
        Path("layer3/train_gine.py"),
        *sorted(Path("layer3/tests").glob("test_*.py")),
        *sorted(Path("models").glob("*.py")),
    ]
    manifest = {
        "schema_version": 1,
        "pipeline_version": cli.pipeline_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        # 清单位于 artifacts/，因此根目录相对清单为上一级。
        "project_root": "..",
        "task_definition": {
            "layer1": "目标 O4 结构门控：判断目标位点是否满足必要结构条件",
            "layer2": "可追溯的成功文献先例检索；不定义适用域或成功概率",
            "layer3": "在反应发生前提下的Alpha/Beta立体选择性分类",
        },
        "label_mapping": {"0": "Alpha", "1": "Beta"},
        "score_semantics": {
            "layer1_class1": "structurally feasible score",
            "layer2": (
                "identity counts and raw donor/acceptor Tanimoto similarities; "
                "no aggregate compatibility score or applicability-domain label"
            ),
            "layer3_class1": "Beta softmax score; not a calibrated probability",
        },
        "layer1": {
            "checkpoints": layer1_entries,
            "ensemble": {"method": "mean_score", "threshold": 0.5},
            "production_threshold_reason": "fixed binary threshold; ignores threshold-search tie artifact",
            "feature_policy": {
                "acceptor_role_names": layer1_metrics_payloads[0].get(
                    "acceptor_role_names"
                ),
                "explicit_o4_external_role_channel": not bool(
                    layer1_metrics_payloads[0]
                    .get("ablation", {})
                    .get("o4_external_role_removed", False)
                ),
                "external_atom_and_o4_x_bond_retained_in_graph": True,
            },
            "negative_sample_definition": (
                "site-matched counterfactual structural negatives generated by "
                "blocking the designated O4"
            ),
            "scope_limitation": (
                "tests a necessary target-site structural prerequisite; does not "
                "estimate empirical reaction success"
            ),
            "training_data": artifact(root, Path("data/processed/hard_feasibility_pre_model.csv")),
            "stress_data": artifact(root, Path("data/processed/hard_feasibility_stress_test.csv")),
            "graph_cache": artifact(root, Path("data/processed/hard_feasibility_chiral_graph_cache.pt")),
        },
        "layer2": {
            "method": "transparent_condition_transfer_evidence_v3",
            "reference_split": "all",
            "fingerprint": "Morgan radius 2, 2048 bits, chirality-aware Tanimoto",
            "joint_similarity": "minimum(donor_tanimoto, acceptor_tanimoto)",
            "pair_history_types": ["known_pair", "novel_pair"],
            "condition_transfer_evidence_types": [
                "exact_pair_exact_condition",
                "same_donor_condition",
                "same_acceptor_condition",
                "analog_condition",
                "template_only",
            ],
            "positive_reference": artifact(
                root, Path("data/processed/soft_compatibility_positive_1561.csv")
            ),
            "hard_gate": False,
            "scope_limitation": (
                "separates donor-acceptor pair history from evidence for transferring the "
                "current condition; pair history is descriptive and excluded from ranking"
            ),
        },
        "layer3": {
            "checkpoints": layer3_entries,
            "ensemble": {
                "method": "majority_vote",
                "thresholds": "seed_specific_validation_thresholds",
                "mean_beta_score": "auxiliary_only",
            },
            "training_data": artifact(root, Path("data/processed/glyco_model_local.csv")),
            "graph_cache": artifact(root, Path("data/processed/rdkit_chiral_graph_cache.pt")),
            "solvent_vocab": artifact(root, Path("data/processed/solvent_vocab.json")),
            "catalyst_vocab": artifact(root, Path("data/processed/catalyst_vocab.json")),
        },
        "application": {
            "condition_screening": {
                "method": "same-donor-type observed-template ranking",
                "template_library": artifact(
                    root, Path("data/processed/condition_template_library_v1.csv")
                ),
                "template_count": 539,
                "complete_supporting_reactions": 1058,
                "candidate_scope": (
                    "all observed complete templates within the resolved donor type; "
                    "Catalyst records are used as curated without additional exclusion"
                ),
                "original_condition_exclusion": (
                    "exclude by normalized condition key and by membership of Source_Reaction_ID "
                    "in the template supporting reactions"
                ),
                "ranking": (
                    "Alpha/Beta: A requires 3/3 target votes; 2/3 is exploratory; at most "
                    "1/3 is not recommended. Within a vote tier, rank transparent literature "
                    "precedent before joint structure similarity and template support. Any: "
                    "literature-evidence retrieval without stereo ranking"
                ),
                "rescue_policy": (
                    "Task_Mode=exploratory_rescue is always labelled exploratory rescue; "
                    "reaction success is not guaranteed"
                ),
                "diversity": "catalyst, solvent and 20-degree temperature bin",
                "hard_gate": False,
                "interpretation": "wet-lab screening priority; not a success probability",
                "recipe_limitation": (
                    "condition skeleton only; verify equivalents, concentration, addition order, "
                    "atmosphere and work-up in the representative source"
                ),
            }
        },
        "execution_policy": {
            "layer1_reject_skips_layer2": True,
            "layer1_reject_skips_layer3": True,
            "literature_evidence_is_not_model_confidence": True,
            "seed_disagreement_is_reported_separately": True,
        },
        "environment": {
            "scope": "frozen inference, audit and tests; model retraining is out of scope",
            "lock_file": artifact(root, Path("environment.server.yml")),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "rdkit": package_version("rdkit"),
            "torch_geometric": package_version("torch-geometric"),
            "scikit_learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
        },
        "source": {
            "git_commit": source_git_commit,
            "git_dirty": False if cli.source_git_commit else bool(generation_status),
            "generation_worktree": {
                "git_commit": generation_git_commit,
                "git_dirty": bool(generation_status),
                "source_commit_override": bool(cli.source_git_commit),
            },
            "files": [artifact(root, path) for path in source_paths],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "checkpoints": 6}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
