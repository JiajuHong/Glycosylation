#!/usr/bin/env python
"""用开放、封闭和非法三类输入执行三层端到端验收。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from pipeline.predict_three_layer import load_manifest
from pipeline.status import (
    FINAL_INVALID_INPUT,
    FINAL_PREDICTED,
    FINAL_STRUCTURALLY_INFEASIBLE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/model_manifest_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/e2e_acceptance_v2.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    manifest, root = load_manifest(cli.manifest.resolve())
    positives = pd.read_csv(root / manifest["layer2"]["positive_reference"]["path"])
    stress = pd.read_csv(root / manifest["layer1"]["stress_data"]["path"])
    blocked_rows = stress.loc[stress["hard_feasibility"].eq(0)]
    if blocked_rows.empty:
        raise ValueError("压力测试文件中没有 O4 封闭样本")

    open_row = positives.iloc[0].copy()
    open_row["Acceptance_Case"] = "known_open_positive"
    blocked_row = blocked_rows.iloc[0].copy()
    blocked_row["Acceptance_Case"] = "known_o4_blocked"
    invalid_row = open_row.copy()
    invalid_row["Acceptance_Case"] = "invalid_smiles"
    invalid_row["Donor_Canonical_SMILES"] = "not-a-smiles"
    candidates = pd.DataFrame([open_row, blocked_row, invalid_row])

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        input_path = directory / "acceptance_input.csv"
        prediction_path = directory / "acceptance_predictions.csv"
        candidates.to_csv(input_path, index=False, encoding="utf-8-sig")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pipeline.predict_three_layer",
                "--input",
                str(input_path),
                "--output",
                str(prediction_path),
                "--manifest",
                str(cli.manifest.resolve()),
                "--device",
                cli.device,
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"统一推理返回 {process.returncode}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )
        predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")

        # 使用已知成功的三氟乙酰亚胺酸酯反应验收条件筛选应用层。
        screening_source = positives.loc[
            positives["Donor_Type"].eq("trifluoroacetimidate")
        ].iloc[0]
        screening_task_path = directory / "screening_task.csv"
        screening_output_path = directory / "screening_recommendations.csv"
        pd.DataFrame(
            [
                {
                    "Task_ID": "known_trifluoroacetimidate",
                    "Donor_Canonical_SMILES": screening_source["Donor_Canonical_SMILES"],
                    "Acceptor_Canonical_SMILES": screening_source["Acceptor_Canonical_SMILES"],
                    "Target_O4_Index": screening_source["Acceptor_O4_Index"],
                    "Target_Config": "Beta" if int(screening_source["Label"]) == 1 else "Alpha",
                }
            ]
        ).to_csv(screening_task_path, index=False, encoding="utf-8-sig")
        screening_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pipeline.screen_conditions",
                "--input",
                str(screening_task_path),
                "--output",
                str(screening_output_path),
                "--manifest",
                str(cli.manifest.resolve()),
                "--device",
                cli.device,
                "--top-n",
                "3",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if screening_process.returncode != 0:
            raise RuntimeError(
                f"条件筛选返回 {screening_process.returncode}\n"
                f"STDOUT:\n{screening_process.stdout}\nSTDERR:\n{screening_process.stderr}"
            )
        screening = pd.read_csv(screening_output_path, encoding="utf-8-sig")

    observed = dict(zip(predictions["Acceptance_Case"], predictions["Final_Status"]))
    expected = {
        "known_open_positive": FINAL_PREDICTED,
        "known_o4_blocked": FINAL_STRUCTURALLY_INFEASIBLE,
        "invalid_smiles": FINAL_INVALID_INPUT,
    }
    checks = {
        "three_rows_returned": len(predictions) == 3,
        "statuses_match": observed == expected,
        "blocked_skips_layer2": bool(predictions.loc[
            predictions["Acceptance_Case"].eq("known_o4_blocked"), "Soft_Domain_Status"
        ].eq("not_evaluated").all()),
        "blocked_skips_layer3": bool(predictions.loc[
            predictions["Acceptance_Case"].eq("known_o4_blocked"), "Stereoselectivity_Label"
        ].eq("not_evaluated").all()),
        "open_has_stereo_label": bool(predictions.loc[
            predictions["Acceptance_Case"].eq("known_open_positive"), "Stereoselectivity_Label"
        ].isin(["Alpha", "Beta"]).all()),
        "third_layer_columns_are_beta_scores": all(
            "Beta_Score" in column
            for column in predictions.columns
            if column.startswith("Layer3_Seed") and column.endswith("_Score")
        ),
        "condition_screening_returns_recommendations": 1 <= len(screening) <= 3,
        "condition_screening_uses_target_support_tiers": bool(
            screening["Recommendation_Tier"].isin(["A", "B", "C_EXPLORATORY"]).all()
        ),
        "condition_screening_preserves_curated_catalyst_records": (
            "Catalyst_Mechanism_Role" not in screening.columns
            and "Mechanism_Compatible" not in screening.columns
        ),
        "condition_screening_records_recipe_limitation": bool(
            screening["Wet_Lab_Use_Requirement"].str.contains(
                "verify full equivalents", regex=False
            ).all()
        ),
    }
    report = {
        "pipeline_version": manifest["pipeline_version"],
        "overall_status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "observed_statuses": observed,
        "expected_statuses": expected,
        "predicted_stereoselectivity": predictions.loc[
            predictions["Acceptance_Case"].eq("known_open_positive"),
            "Stereoselectivity_Label",
        ].iloc[0],
        "subprocess_stdout": process.stdout,
        "condition_screening_stdout": screening_process.stdout,
    }
    output = cli.output if cli.output.is_absolute() else root / cli.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "output": str(output)}, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
