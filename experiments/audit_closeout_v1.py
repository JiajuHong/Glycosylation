"""Read-only verification of frozen artifacts and the default-model regression export."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    manifest = json.loads(Path("artifacts/model_manifest_v2.json").read_text())
    checked = []

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = Path(value["path"])
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != value["sha256"]:
                    raise ValueError(f"Hash mismatch: {path}")
                checked.append(str(path))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(manifest)
    old = pd.read_csv("results/formal_computational_v1/predictions/pair_chiral_tri_l3/seed0.csv")
    new = pd.read_csv("artifacts/audit/closeout_seed0_predictions.csv")
    joined = old.merge(new, on=["id", "split"], suffixes=("_old", "_new"), validate="one_to_one")
    if len(joined) != len(old) or len(joined) != len(new):
        raise ValueError("Prediction row mismatch")
    max_difference = float((joined.prob_beta_old - joined.prob_beta_new).abs().max())
    class_match = bool(joined.prediction_old.eq(joined.prediction_new).all())
    if max_difference > 1e-5 or not class_match:
        raise ValueError(f"Default model changed: max difference={max_difference}")
    result = {
        "status": "passed", "checked_artifact_count": len(checked),
        "seed0_val_test_rows": len(joined), "max_beta_score_difference": max_difference,
        "all_classifications_match": class_match,
        "scope": "artifact hashes and unchanged default full-condition seed0 inference; not a new end-to-end audit",
    }
    Path("artifacts/audit/closeout_v1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
