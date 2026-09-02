#!/usr/bin/env python
"""Run the frozen formal computational experiment matrix on two GPUs.

Outputs are kept under versioned artifact, result, and log directories. Existing
valid metric files are treated as completed runs so the matrix can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


EXPERIMENT_ID = "formal_computational_v1"
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    split_column: str
    encoder_type: str
    model_type: str
    local_output_mode: str = "l3"
    fair_gine_training: bool = False


CONFIGS = (
    ExperimentConfig("pair_chiral_global", "split_pair_group", "chiral_gine", "global"),
    ExperimentConfig("pair_chiral_local", "split_pair_group", "chiral_gine", "local"),
    ExperimentConfig(
        "pair_chiral_crossattn", "split_pair_group", "chiral_gine", "crossattn"
    ),
    ExperimentConfig(
        "pair_chiral_tri_l1",
        "split_pair_group",
        "chiral_gine",
        "crossattn_tri",
        "l1",
    ),
    ExperimentConfig(
        "pair_chiral_tri_l2",
        "split_pair_group",
        "chiral_gine",
        "crossattn_tri",
        "l2",
    ),
    ExperimentConfig(
        "pair_chiral_tri_l3",
        "split_pair_group",
        "chiral_gine",
        "crossattn_tri",
        "l3",
    ),
    ExperimentConfig(
        "pair_gine_tri_l3_fair",
        "split_pair_group",
        "gine",
        "crossattn_tri",
        "l3",
        True,
    ),
    ExperimentConfig(
        "year_chiral_tri_l3",
        "split_year",
        "chiral_gine",
        "crossattn_tri",
        "l3",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def paths(root: Path, config: ExperimentConfig, seed: int) -> dict[str, Path]:
    return {
        "checkpoint_dir": root / "artifacts" / "checkpoints" / EXPERIMENT_ID / config.name,
        "metric": root
        / "artifacts"
        / "metrics"
        / EXPERIMENT_ID
        / config.name
        / f"seed{seed}.csv",
        "log": root / "logs" / EXPERIMENT_ID / f"{config.name}_seed{seed}.log",
    }


def is_complete(metric_path: Path, config: ExperimentConfig, seed: int) -> bool:
    if not metric_path.exists():
        return False
    try:
        frame = pd.read_csv(metric_path)
    except Exception:
        return False
    if len(frame) != 1:
        return False
    row = frame.iloc[0]
    return (
        int(row["seed"]) == seed
        and str(row["split_column"]) == config.split_column
        and str(row["encoder_type"]) == config.encoder_type
        and str(row["model_type"]) == config.model_type
        and str(row.get("local_output_mode", "l3")) == config.local_output_mode
        and Path(str(row["checkpoint"])).exists()
    )


def command(
    root: Path,
    config: ExperimentConfig,
    seed: int,
    run_paths: dict[str, Path],
) -> list[str]:
    cli = [
        sys.executable,
        "-m",
        "layer3.train_gine",
        "--csv",
        "data/processed/glyco_model_local.csv",
        "--split-column",
        config.split_column,
        "--encoder-type",
        config.encoder_type,
        "--model-type",
        config.model_type,
        "--local-output-mode",
        config.local_output_mode,
        "--seed",
        str(seed),
        "--run-id",
        EXPERIMENT_ID,
        "--checkpoint-dir",
        str(run_paths["checkpoint_dir"]),
        "--results-csv",
        str(run_paths["metric"]),
        "--device",
        "cuda:0",
        "--batch-size",
        "32",
        "--epochs",
        "200",
    ]
    if config.fair_gine_training:
        cli.extend(
            [
                "--deterministic",
                "--gradient-clip",
                "2.0",
                "--warmup-epochs",
                "10",
                "--lr-scheduler",
                "cosine",
                "--patience",
                "40",
            ]
        )
    return cli


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    root = cli.root.resolve()
    if not cli.gpus:
        raise ValueError("at least one GPU must be supplied")
    data_path = root / "data/processed/glyco_model_local.csv"
    source_paths = [
        root / "layer3/train_gine.py",
        root / "layer3/glyco_dataset.py",
        root / "models/glyco_gine_models.py",
        root / "models/local_modules.py",
        root / "models/chiral_gine_encoder.py",
        root / "models/chiral_gine_conv.py",
        root / "models/tetra_permutation.py",
    ]
    metric_root = root / "artifacts" / "metrics" / EXPERIMENT_ID
    log_root = root / "logs" / EXPERIMENT_ID
    metric_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = [(config, seed) for config in CONFIGS for seed in SEEDS]
    commands = []
    for config, seed in jobs:
        run_paths = paths(root, config, seed)
        commands.append(
            {
                "config": asdict(config),
                "seed": seed,
                "command": command(root, config, seed, run_paths),
                "metric": str(run_paths["metric"].relative_to(root)),
                "log": str(run_paths["log"].relative_to(root)),
            }
        )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "data": {
            "path": str(data_path.relative_to(root)),
            "sha256": sha256(data_path),
            "rows": len(pd.read_csv(data_path, encoding="utf-8-sig")),
        },
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": sha256(path)}
            for path in source_paths
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "rdkit": package_version("rdkit"),
            "torch_geometric": package_version("torch-geometric"),
        },
        "gpus": cli.gpus,
        "commands": commands,
    }
    write_json(metric_root / "run_manifest.json", manifest)

    pending = []
    completed = []
    for config, seed in jobs:
        run_paths = paths(root, config, seed)
        if is_complete(run_paths["metric"], config, seed):
            completed.append(f"{config.name}/seed{seed}")
        else:
            pending.append((config, seed))
    state: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "completed": completed,
        "failed": [],
        "running": {},
        "pending": [f"{config.name}/seed{seed}" for config, seed in pending],
    }
    write_json(metric_root / "run_state.json", state)

    running: dict[int, dict[str, Any]] = {}
    free_gpus = list(cli.gpus)
    while pending or running:
        while pending and free_gpus:
            config, seed = pending.pop(0)
            gpu = free_gpus.pop(0)
            run_paths = paths(root, config, seed)
            for key in ("checkpoint_dir",):
                run_paths[key].mkdir(parents=True, exist_ok=True)
            run_paths["metric"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["log"].parent.mkdir(parents=True, exist_ok=True)
            cli_command = command(root, config, seed, run_paths)
            log_handle = run_paths["log"].open("w", encoding="utf-8")
            log_handle.write("COMMAND: " + " ".join(cli_command) + "\n")
            log_handle.flush()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                cli_command,
                cwd=root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            key = f"{config.name}/seed{seed}"
            running[process.pid] = {
                "process": process,
                "config": config,
                "seed": seed,
                "gpu": gpu,
                "log_handle": log_handle,
                "key": key,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(f"START {key} gpu={gpu} pid={process.pid}", flush=True)

        finished = []
        for pid, item in running.items():
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log_handle"].close()
            key = item["key"]
            if returncode == 0 and is_complete(
                paths(root, item["config"], item["seed"])["metric"],
                item["config"],
                item["seed"],
            ):
                state["completed"].append(key)
                print(f"DONE  {key} gpu={item['gpu']}", flush=True)
            else:
                state["failed"].append(
                    {"key": key, "returncode": returncode, "gpu": item["gpu"]}
                )
                print(f"FAIL  {key} returncode={returncode}", flush=True)
            free_gpus.append(item["gpu"])
            free_gpus.sort()
            finished.append(pid)
        for pid in finished:
            del running[pid]
        state["running"] = {
            item["key"]: {
                "pid": pid,
                "gpu": item["gpu"],
                "started_utc": item["started_utc"],
            }
            for pid, item in running.items()
        }
        state["pending"] = [f"{config.name}/seed{seed}" for config, seed in pending]
        write_json(metric_root / "run_state.json", state)
        if pending or running:
            time.sleep(cli.poll_seconds)

    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    state["status"] = "failed" if state["failed"] else "completed"
    write_json(metric_root / "run_state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
