#!/usr/bin/env python
"""从冻结数据中构造一组小型三层推理测试样本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_COLUMNS = [
    "Donor_Canonical_SMILES",
    "Acceptor_Canonical_SMILES",
    "Donor_Type",
    "Target_O4_Index",
    "Solvent",
    "Catalyst",
    "Temp_C",
    "Time_min",
    "has_solvent",
    "has_catalyst",
    "has_temp",
    "has_time",
]


def base_record(row: pd.Series) -> dict[str, Any]:
    """只保留正式推理需要或允许提供的输入字段。"""
    record = {column: row.get(column, np.nan) for column in INPUT_COLUMNS}
    for column in ("has_solvent", "has_catalyst", "has_temp", "has_time"):
        if pd.isna(record[column]):
            record[column] = 0
    return record


def add_expectation(
    record: dict[str, Any],
    case_id: str,
    description: str,
    final_status: str,
    layer1_decision: str,
    target_state: str,
    downstream_evaluated: bool,
) -> dict[str, Any]:
    return {
        "Test_Case_ID": case_id,
        "Test_Description": description,
        "Expected_Final_Status": final_status,
        "Expected_Layer1_Decision": layer1_decision,
        "Expected_Target_O4_State": target_state,
        "Expected_Downstream_Evaluated": int(downstream_evaluated),
        **record,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-reference",
        type=Path,
        default=Path("data/processed/soft_compatibility_positive_1561.csv"),
    )
    parser.add_argument(
        "--stress-test",
        type=Path,
        default=Path("data/processed/hard_feasibility_stress_test.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/test/three_layer_inference_test_cases_v1.csv"),
    )
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    positives = pd.read_csv(cli.positive_reference, encoding="utf-8-sig")
    stress = pd.read_csv(cli.stress_test, encoding="utf-8-sig")
    records: list[dict[str, Any]] = []

    # 三种支持的供体各取一条完整条件成功反应。
    donor_types = (
        "trichloroacetimidate",
        "trifluoroacetimidate",
        "thioglycoside",
    )
    positive_examples: dict[str, pd.Series] = {}
    for donor_type in donor_types:
        row = positives.loc[positives["Donor_Type"].eq(donor_type)].sort_values("ID").iloc[0]
        positive_examples[donor_type] = row
        record = base_record(row)
        record["Target_O4_Index"] = int(row["Acceptor_O4_Index"])
        records.append(
            add_expectation(
                record,
                f"OPEN_{donor_type.upper()}",
                f"目标 O4 开放，供体类型为 {donor_type}",
                "predicted",
                "feasible",
                "free",
                True,
            )
        )

    # 非目标羟基被乙酰化，但目标 O4 仍开放，应通过第一层。
    non_target = stress.loc[stress["Stress_Test_Type"].eq("non_target_oh_acetyl")].iloc[0]
    records.append(
        add_expectation(
            base_record(non_target),
            "OPEN_NON_TARGET_OH_ACETYLATED",
            "非目标羟基乙酰化，目标 O4 仍开放",
            "predicted",
            "feasible",
            "free",
            True,
        )
    )

    # 条件全部缺失时仍可运行；模型使用明确的缺失标记。
    missing_conditions = base_record(positive_examples["thioglycoside"])
    missing_conditions["Target_O4_Index"] = int(
        positive_examples["thioglycoside"]["Acceptor_O4_Index"]
    )
    for column in ("Solvent", "Catalyst", "Temp_C", "Time_min"):
        missing_conditions[column] = np.nan
    for column in ("has_solvent", "has_catalyst", "has_temp", "has_time"):
        missing_conditions[column] = 0
    records.append(
        add_expectation(
            missing_conditions,
            "OPEN_MISSING_CONDITIONS",
            "目标 O4 开放，但所有反应条件缺失",
            "predicted",
            "feasible",
            "free",
            True,
        )
    )

    # 三类未在训练负样本中使用的 O4 封闭压力测试。
    for stress_type in ("o4_methyl", "o4_benzyl", "o4_trimethylsilyl"):
        row = stress.loc[stress["Stress_Test_Type"].eq(stress_type)].iloc[0]
        records.append(
            add_expectation(
                base_record(row),
                stress_type.upper(),
                f"目标 O4 被 {stress_type.removeprefix('o4_')} 封闭",
                "structurally_infeasible",
                "infeasible",
                "blocked",
                False,
            )
        )

    valid = base_record(positive_examples["trichloroacetimidate"])
    valid["Target_O4_Index"] = int(
        positive_examples["trichloroacetimidate"]["Acceptor_O4_Index"]
    )

    invalid_donor = dict(valid)
    invalid_donor["Donor_Canonical_SMILES"] = "not-a-smiles"
    records.append(
        add_expectation(
            invalid_donor,
            "INVALID_DONOR_SMILES",
            "供体 SMILES 非法",
            "invalid_input",
            "not_evaluated",
            "parse_error",
            False,
        )
    )

    invalid_acceptor = dict(valid)
    invalid_acceptor["Acceptor_Canonical_SMILES"] = "C1(not-valid"
    records.append(
        add_expectation(
            invalid_acceptor,
            "INVALID_ACCEPTOR_SMILES",
            "受体 SMILES 非法",
            "invalid_input",
            "not_evaluated",
            "parse_error",
            False,
        )
    )

    invalid_index = dict(valid)
    invalid_index["Target_O4_Index"] = 99999
    records.append(
        add_expectation(
            invalid_index,
            "INVALID_TARGET_INDEX",
            "目标 O4 原子编号越界",
            "invalid_input",
            "not_evaluated",
            "parse_error",
            False,
        )
    )

    unsupported_type = dict(valid)
    unsupported_type["Donor_Type"] = "glycosyl_fluoride"
    records.append(
        add_expectation(
            unsupported_type,
            "UNSUPPORTED_DONOR_TYPE",
            "供体类型不在冻结版支持范围内",
            "invalid_input",
            "not_evaluated",
            "parse_error",
            False,
        )
    )

    output = pd.DataFrame(records)
    if len(output) != 12 or not output["Test_Case_ID"].is_unique:
        raise RuntimeError("测试集数量或测试编号异常")
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(cli.output, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "output": str(cli.output),
                "rows": len(output),
                "expected_status_counts": output["Expected_Final_Status"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
