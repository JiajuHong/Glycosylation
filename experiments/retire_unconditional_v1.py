"""Archive retired variants and filter active baseline tables without retraining."""
import csv
import json
from pathlib import Path
import shutil


def main():
    archive = Path("artifacts/archives/without_conditions_20260902")
    if not (archive / "pre_removal.tar.gz").is_file():
        raise RuntimeError("Create and verify the pre-removal archive first")
    for name in (
        "experiments/run_condition_ablation_v1.py",
        "results/condition_ablation_v1",
        "artifacts/checkpoints/condition_ablation_v1",
        "logs/condition_ablation_v1",
    ):
        source = Path(name)
        target = archive / name
        if source.exists():
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))

    base = Path("results/formal_computational_v1")
    tables = list((base / "baselines").glob("*.csv")) + [
        base / "baseline_pair_group_summary.csv",
        base / "paired_baseline_comparisons.csv",
    ]
    for path in tables:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        if "feature_set" not in fields:
            raise ValueError(f"Missing feature_set: {path}")
        kept = [row for row in rows if row["feature_set"] != "structure_only"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        print(f"{path}: {len(rows)} -> {len(kept)} rows")
    path = base / "baselines/baseline_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["feature_sets"] = ["structure_condition", "condition_only"]
    metadata["retirement_note"] = (
        "Structure-only rows archived, not invalidated. Retained runs unchanged; "
        "source hash describes original training code, not current code. "
        "Archive: artifacts/archives/without_conditions_20260902/pre_removal.tar.gz"
    )
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
