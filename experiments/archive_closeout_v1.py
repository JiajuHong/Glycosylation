"""Create a non-destructive source snapshot without committing unrelated edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    root = Path.cwd()
    output = root / "artifacts/releases/closeout_20260902"
    output.mkdir(parents=True, exist_ok=True)
    extensions = {".py", ".md", ".yml", ".yaml", ".sh"}
    paths = {p for p in root.iterdir() if p.is_file() and p.suffix in extensions}
    for directory in ("models", "layer1", "layer2", "layer3", "pipeline", "baseline", "common", "experiments"):
        paths.update(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in extensions and "__pycache__" not in p.parts)
    paths.add(root / "artifacts/model_manifest_v2.json")
    inventory = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    archive = output / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for p in sorted(paths):
            handle.add(p, arcname=str(p.relative_to(root)), recursive=False)
    git_status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    manifest = {
        "snapshot_type": "source_snapshot_not_a_claim_of_completed_experiments",
        "production_model": "frozen Chiral-GINE Tri-L3 plus MLP",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(git_status), "git_status": git_status,
        "source_files": inventory,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "training_data_sha256": hashlib.sha256((root / "data/processed/glyco_model_local.csv").read_bytes()).hexdigest(),
        "condition_ablation_status_file": "artifacts/archives/without_conditions_20260902/results/condition_ablation_v1/run_manifest.json",
        "weights": "Unchanged production checkpoint paths/hashes are in model_manifest_v2.json.",
    }
    (output / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_files": len(paths), "archive": str(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
