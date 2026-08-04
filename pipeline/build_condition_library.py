#!/usr/bin/env python
"""从成功文献反应中构建可追溯的完整条件模板库。

模板始终保留文献中真实共现的“溶剂+催化剂/活化剂+温度+时间”组合，
不生成任意笛卡尔组合。供体类型是模板身份的一部分，供后续同类条件检索使用。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "Reaction_ID",
    "Donor_Type",
    "Solvent",
    "Solvent_Component_IDs",
    "Catalyst",
    "Catalyst_Component_IDs",
    "Temp_C",
    "Time_min",
    "has_solvent",
    "has_catalyst",
    "has_temp",
    "has_time",
    "Label",
}

def normalize_id_list(value: object) -> str:
    """把组分 ID 规范化为无序集，避免只因书写顺序不同重复建模。"""
    ids = sorted({int(float(part)) for part in str(value).split(";") if part.strip()})
    return ";".join(map(str, ids))


def normalize_names(value: object) -> str:
    names = sorted({part.strip() for part in str(value).split(";") if part.strip()})
    return ";".join(names)


def stable_template_id(row: pd.Series) -> str:
    payload = "|".join(
        [
            str(row["Donor_Type"]),
            str(row["Solvent_Component_IDs"]),
            str(row["Catalyst_Component_IDs"]),
            f"{float(row['Temp_C']):.8g}",
            f"{float(row['Time_min']):.8g}",
        ]
    )
    return "CT_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def join_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def build_library(source: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"成功反应数据缺少字段: {sorted(missing)}")

    frame = source.copy()
    for column in ("has_solvent", "has_catalyst", "has_temp", "has_time"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    frame["Temp_C"] = pd.to_numeric(frame["Temp_C"], errors="coerce")
    frame["Time_min"] = pd.to_numeric(frame["Time_min"], errors="coerce")

    complete = frame.loc[
        frame[["has_solvent", "has_catalyst", "has_temp", "has_time"]].eq(1).all(axis=1)
        & frame["Temp_C"].notna()
        & frame["Time_min"].notna()
        & frame["Solvent"].notna()
        & frame["Catalyst"].notna()
    ].copy()
    if complete.empty:
        raise ValueError("没有可用的完整成功反应条件")

    complete["Solvent_Component_IDs"] = complete["Solvent_Component_IDs"].map(
        normalize_id_list
    )
    complete["Catalyst_Component_IDs"] = complete["Catalyst_Component_IDs"].map(
        normalize_id_list
    )
    complete["Solvent"] = complete["Solvent"].map(normalize_names)
    complete["Catalyst"] = complete["Catalyst"].map(normalize_names)
    if complete["Solvent_Component_IDs"].str.split(";").map(lambda x: "1" in x or "2" in x).any():
        raise ValueError("完整溶剂条件中出现 MISSING/UNKNOWN token")
    if complete["Catalyst_Component_IDs"].str.split(";").map(lambda x: "1" in x or "2" in x).any():
        raise ValueError("完整催化条件中出现 MISSING/UNKNOWN token")

    keys = [
        "Donor_Type",
        "Solvent_Component_IDs",
        "Catalyst_Component_IDs",
        "Temp_C",
        "Time_min",
    ]
    records: list[dict[str, object]] = []
    for key, group in complete.groupby(keys, sort=True, dropna=False):
        donor_type, solvent_ids, catalyst_ids, temperature, time_min = key
        labels = pd.to_numeric(group["Label"], errors="coerce")
        record = {
            "Donor_Type": donor_type,
            "Solvent": sorted(group["Solvent"].astype(str))[0],
            "Solvent_Component_IDs": solvent_ids,
            "Catalyst": sorted(group["Catalyst"].astype(str))[0],
            "Catalyst_Component_IDs": catalyst_ids,
            "Temp_C": float(temperature),
            "Time_min": float(time_min),
            "has_solvent": 1,
            "has_catalyst": 1,
            "has_temp": 1,
            "has_time": 1,
            "Template_Support_Count": int(len(group)),
            "Alpha_Success_Count": int(labels.eq(0).sum()),
            "Beta_Success_Count": int(labels.eq(1).sum()),
            "Representative_Reaction_ID": str(group.iloc[0]["Reaction_ID"]),
            "Supporting_Reaction_IDs": join_unique(group["Reaction_ID"]),
            "Supporting_Sources": join_unique(group.get("Source", pd.Series(dtype=object))),
        }
        record.update(
            {
                "Condition_Recipe_Completeness": (
                    "skeleton_only; equivalents/concentration/addition_order/atmosphere not available"
                ),
            }
        )
        if "Full_Reaction_SHA256" in group:
            record["Unique_Full_Reaction_Count"] = int(group["Full_Reaction_SHA256"].nunique())
        else:
            record["Unique_Full_Reaction_Count"] = int(len(group))
        records.append(record)

    library = pd.DataFrame(records)
    library.insert(0, "Template_ID", library.apply(stable_template_id, axis=1))
    if library["Template_ID"].duplicated().any():
        raise ValueError("条件模板 ID 发生冲突")
    if int(library["Template_Support_Count"].sum()) != len(complete):
        raise ValueError("模板支持数与完整反应数不一致")
    return library.sort_values(
        ["Donor_Type", "Template_Support_Count", "Template_ID"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/soft_compatibility_positive_1561.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/condition_template_library_v1.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = pd.read_csv(args.input, encoding="utf-8-sig")
    library = build_library(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    library.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        {
            "output": str(args.output),
            "templates": len(library),
            "by_donor_type": library.groupby("Donor_Type").size().to_dict(),
            "supporting_complete_reactions": int(library["Template_Support_Count"].sum()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
