#!/usr/bin/env python
"""针对固定供体—受体批量筛选真实文献条件模板。

该程序不生成新化学条件，只在相同供体离去基类型内迁移已成功条件，
然后调用冻结三层流水线。结果是湿实验优先级，不是反应成功概率。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layer1.predict_hard_feasibility import canonicalize, resolve_donor_type
from layer3.tokenize_conditions import (
    CATALYST_TOKEN_MAP,
    CATALYST_VOCAB,
    SOLVENT_TOKEN_MAP,
    SOLVENT_VOCAB,
    map_components,
)
from pipeline.build_condition_library import normalize_id_list


TARGET_CONFIG_MAP = {
    "a": "Alpha",
    "alpha": "Alpha",
    "α": "Alpha",
    "b": "Beta",
    "beta": "Beta",
    "β": "Beta",
    "any": "Any",
    "either": "Any",
    "不限": "Any",
}
TASK_MODE_STANDARD = "standard"
TASK_MODE_EXPLORATORY_RESCUE = "exploratory_rescue"
TASK_MODE_MAP = {
    "standard": TASK_MODE_STANDARD,
    "stereoselective": TASK_MODE_STANDARD,
    "exploratory_rescue": TASK_MODE_EXPLORATORY_RESCUE,
    "rescue": TASK_MODE_EXPLORATORY_RESCUE,
    "探索性救援": TASK_MODE_EXPLORATORY_RESCUE,
}
TIER_ORDER = {
    "A": 0,
    "A_RETRIEVAL": 0,
    "B": 1,
    "B_RETRIEVAL": 1,
    "C_EXPLORATORY": 2,
    "C_RESCUE_EXPLORATORY": 2,
    "D_NOT_TARGET": 3,
    "REJECT": 4,
}
TIER_LABEL = {
    "A": "优先",
    "A_RETRIEVAL": "优先",
    "B": "谨慎",
    "B_RETRIEVAL": "谨慎",
    "C_EXPLORATORY": "探索",
    "C_RESCUE_EXPLORATORY": "探索性救援",
    "D_NOT_TARGET": "不推荐",
    "REJECT": "硬结构不通过",
}
DOMAIN_ORDER = {"in_domain": 0, "borderline": 1, "out_of_domain": 2, "not_evaluated": 3}


def missing(value: object) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def normalize_target_config(value: object) -> str:
    if missing(value):
        return "Any"
    key = str(value).strip().lower()
    if key not in TARGET_CONFIG_MAP:
        raise ValueError(f"不支持的 Target_Config={value!r}，应为 Alpha/Beta/Any")
    return TARGET_CONFIG_MAP[key]


def normalize_task_mode(row: pd.Series) -> str:
    """显式识别第三类探索性救援任务，同时兼容旧任务表。"""
    value = row.get("Task_Mode")
    if not missing(value):
        key = str(value).strip().lower()
        if key not in TASK_MODE_MAP:
            raise ValueError(
                f"不支持的 Task_Mode={value!r}，应为 standard/exploratory_rescue"
            )
        return TASK_MODE_MAP[key]

    task_class = row.get("Task_Class")
    if not missing(task_class):
        key = str(task_class).strip().lower()
        if key in {"3", "3.0", "第三类", "class3", "class_3", "third"}:
            return TASK_MODE_EXPLORATORY_RESCUE
    return TASK_MODE_STANDARD


def original_component_ids(
    row: pd.Series,
    id_column: str,
    raw_column: str,
    token_map: dict[str, str],
    vocab: dict[str, int],
    missing_token: str,
    unknown_token: str,
) -> str | None:
    if not missing(row.get(id_column)):
        return normalize_id_list(row[id_column])
    if missing(row.get(raw_column)):
        return None
    _, ids, _ = map_components(
        row[raw_column], token_map, vocab, missing_token, unknown_token
    )
    return normalize_id_list(ids)


def original_condition_key(row: pd.Series) -> tuple[str, str, float, float] | None:
    solvent_ids = original_component_ids(
        row,
        "Original_Solvent_Component_IDs",
        "Original_Solvent",
        SOLVENT_TOKEN_MAP,
        SOLVENT_VOCAB,
        "SOLV_MISSING",
        "SOLV_UNKNOWN",
    )
    catalyst_ids = original_component_ids(
        row,
        "Original_Catalyst_Component_IDs",
        "Original_Catalyst",
        CATALYST_TOKEN_MAP,
        CATALYST_VOCAB,
        "CAT_MISSING",
        "CAT_UNKNOWN",
    )
    temperature = row.get("Original_Temp_C")
    time_min = row.get("Original_Time_min")
    if solvent_ids is None or catalyst_ids is None or missing(temperature) or missing(time_min):
        return None
    return solvent_ids, catalyst_ids, float(temperature), float(time_min)


def template_condition_key(row: pd.Series) -> tuple[str, str, float, float]:
    return (
        normalize_id_list(row["Solvent_Component_IDs"]),
        normalize_id_list(row["Catalyst_Component_IDs"]),
        float(row["Temp_C"]),
        float(row["Time_min"]),
    )


def prepare_tasks(tasks: pd.DataFrame, library: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"}
    absent = required - set(tasks.columns)
    if absent:
        raise ValueError(f"任务文件缺少字段: {sorted(absent)}")
    if "Task_ID" not in tasks:
        tasks = tasks.copy()
        tasks["Task_ID"] = [f"TASK_{index + 1:03d}" for index in range(len(tasks))]
    if tasks["Task_ID"].astype(str).duplicated().any():
        raise ValueError("Task_ID 必须唯一")

    candidate_rows: list[dict[str, Any]] = []
    excluded_original = 0
    task_summaries: list[dict[str, Any]] = []
    for _, task in tasks.iterrows():
        task_id = str(task["Task_ID"])
        target_config = normalize_target_config(task.get("Target_Config"))
        task_mode = normalize_task_mode(task)
        donor, _, donor_mapping = canonicalize(task["Donor_Canonical_SMILES"])
        acceptor, _, acceptor_mapping = canonicalize(task["Acceptor_Canonical_SMILES"])
        if donor is None or donor_mapping is None:
            raise ValueError(f"{task_id}: 供体 SMILES 无效")
        if acceptor is None or acceptor_mapping is None:
            raise ValueError(f"{task_id}: 受体 SMILES 无效")
        donor_type, source = resolve_donor_type(donor, task.get("Donor_Type"))
        if donor_type is None:
            raise ValueError(f"{task_id}: 供体类型无法解析: {source}")

        target_index: float | int = np.nan
        if not missing(task.get("Target_O4_Index")):
            original_index = int(float(task["Target_O4_Index"]))
            if original_index < 0 or original_index >= len(acceptor_mapping):
                raise ValueError(f"{task_id}: Target_O4_Index 越界")
            target_index = int(acceptor_mapping[original_index])

        # 生工整理的条件记录原样参与候选筛选；这里只限定供体类型，
        # 不对 Catalyst 名称追加机理角色判断或人工排除规则。
        templates = library.loc[library["Donor_Type"].eq(donor_type)]
        if templates.empty:
            raise ValueError(f"{task_id}: 条件库中没有 {donor_type} 模板")
        original_key = original_condition_key(task)
        kept = 0
        for _, template in templates.iterrows():
            matches_original = original_key is not None and template_condition_key(template) == original_key
            if matches_original:
                excluded_original += 1
                continue
            record = task.to_dict()
            record.update(
                {
                    "Task_ID": task_id,
                    "Candidate_ID": f"{task_id}::{template['Template_ID']}",
                    "Target_Config": target_config,
                    "Task_Mode": task_mode,
                    "Donor_Canonical_SMILES": donor,
                    "Acceptor_Canonical_SMILES": acceptor,
                    "Donor_Type": donor_type,
                    "Target_O4_Index": target_index,
                    "Template_ID": template["Template_ID"],
                    "Solvent": template["Solvent"],
                    "Solvent_Component_IDs": template["Solvent_Component_IDs"],
                    "Catalyst": template["Catalyst"],
                    "Catalyst_Component_IDs": template["Catalyst_Component_IDs"],
                    "Temp_C": template["Temp_C"],
                    "Time_min": template["Time_min"],
                    "has_solvent": 1,
                    "has_catalyst": 1,
                    "has_temp": 1,
                    "has_time": 1,
                    "Template_Support_Count": template["Template_Support_Count"],
                    "Template_Alpha_Success_Count": template["Alpha_Success_Count"],
                    "Template_Beta_Success_Count": template["Beta_Success_Count"],
                    "Template_Representative_Reaction_ID": template[
                        "Representative_Reaction_ID"
                    ],
                    "Template_Supporting_Reaction_IDs": template[
                        "Supporting_Reaction_IDs"
                    ],
                    "Condition_Recipe_Completeness": template[
                        "Condition_Recipe_Completeness"
                    ],
                    "Original_Condition_Complete": original_key is not None,
                    "Matches_Original_Condition": False,
                }
            )
            candidate_rows.append(record)
            kept += 1
        task_summaries.append(
            {
                "Task_ID": task_id,
                "Donor_Type": donor_type,
                "Target_Config": target_config,
                "Task_Mode": task_mode,
                "Candidate_Count": kept,
                "Original_Condition_Complete": original_key is not None,
            }
        )

    if not candidate_rows:
        raise ValueError("删除原条件后没有可评分候选")
    return pd.DataFrame(candidate_rows), {
        "tasks": task_summaries,
        "excluded_original_conditions": excluded_original,
    }


def add_ranking_fields(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "Task_Mode" not in frame:
        frame["Task_Mode"] = TASK_MODE_STANDARD
    else:
        frame["Task_Mode"] = frame.apply(normalize_task_mode, axis=1)
    vote_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("Layer3_Seed") and column.endswith("_Vote")
    )
    if not vote_columns:
        raise ValueError("三层输出中没有种子投票字段")
    frame["Target_Vote_Count"] = [
        (
            np.nan
            if str(row["Target_Config"]) == "Any"
            else sum(str(row[column]) == str(row["Target_Config"]) for column in vote_columns)
        )
        for _, row in frame.iterrows()
    ]
    frame["Ensemble_Size"] = len(vote_columns)
    beta_score = pd.to_numeric(frame["Stereoselectivity_Mean_Beta_Score"], errors="coerce")
    frame["Target_Stereo_Score"] = np.select(
        [frame["Target_Config"].eq("Beta"), frame["Target_Config"].eq("Alpha")],
        [beta_score, 1.0 - beta_score],
        default=np.nan,
    )
    majority = len(vote_columns) // 2 + 1

    def tier(row: pd.Series) -> str:
        if row["Final_Status"] != "predicted":
            return "REJECT"
        domain = str(row["Soft_Domain_Status"])
        if row["Task_Mode"] == TASK_MODE_EXPLORATORY_RESCUE:
            return "C_RESCUE_EXPLORATORY"
        if row["Target_Config"] == "Any":
            if domain == "in_domain":
                return "A_RETRIEVAL"
            if domain == "borderline":
                return "B_RETRIEVAL"
            return "C_EXPLORATORY"
        votes = int(row["Target_Vote_Count"])
        if votes == len(vote_columns):
            if domain == "in_domain":
                return "A"
            if domain == "borderline":
                return "B"
            return "C_EXPLORATORY"
        if votes >= majority:
            return "C_EXPLORATORY"
        return "D_NOT_TARGET"

    frame["Recommendation_Tier"] = frame.apply(tier, axis=1)
    frame["Recommendation_Level"] = frame["Recommendation_Tier"].map(TIER_LABEL)
    frame["Tier_Order"] = frame["Recommendation_Tier"].map(TIER_ORDER)
    frame["Domain_Order"] = frame["Soft_Domain_Status"].map(DOMAIN_ORDER).fillna(9)
    frame["Temp_Bin_20C"] = (pd.to_numeric(frame["Temp_C"]) / 20.0).round().astype(int)
    frame["Diversity_Key"] = (
        frame["Catalyst_Component_IDs"].astype(str)
        + "|"
        + frame["Solvent_Component_IDs"].astype(str)
        + "|T"
        + frame["Temp_Bin_20C"].astype(str)
    )
    frame = frame.sort_values(
        [
            "Task_ID",
            "Tier_Order",
            "Target_Vote_Count",
            "Domain_Order",
            "Soft_Compatibility_Score",
            "Target_Stereo_Score",
            "Template_Support_Count",
            "Template_ID",
        ],
        ascending=[True, True, False, True, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    frame["Overall_Target_Rank"] = frame.groupby("Task_ID").cumcount() + 1
    return frame


def select_diverse_recommendations(ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    for _, task_frame in ranked.groupby("Task_ID", sort=False):
        chosen: set[int] = set()
        used_catalysts: set[str] = set()
        used_diversity: set[str] = set()
        for tier in (
            "A",
            "A_RETRIEVAL",
            "B",
            "B_RETRIEVAL",
            "C_RESCUE_EXPLORATORY",
            "C_EXPLORATORY",
        ):
            tier_frame = task_frame.loc[task_frame["Recommendation_Tier"].eq(tier)]
            for phase in ("catalyst", "diversity", "fill"):
                for index, row in tier_frame.iterrows():
                    if index in chosen:
                        continue
                    catalyst = str(row["Catalyst_Component_IDs"])
                    diversity = str(row["Diversity_Key"])
                    if phase == "catalyst" and catalyst in used_catalysts:
                        continue
                    if phase == "diversity" and diversity in used_diversity:
                        continue
                    chosen.add(index)
                    used_catalysts.add(catalyst)
                    used_diversity.add(diversity)
                    selected_rows.append(row)
                    if len(chosen) >= top_n:
                        break
                if len(chosen) >= top_n:
                    break
            if len(chosen) >= top_n:
                break
    if not selected_rows:
        return ranked.iloc[0:0].copy()
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["Recommendation_Rank"] = selected.groupby("Task_ID").cumcount() + 1
    def recommendation_reason(row: pd.Series) -> str:
        if row["Task_Mode"] == TASK_MODE_EXPLORATORY_RESCUE:
            return (
                f"{row['Recommendation_Tier']}; {row['Soft_Domain_Status']}; "
                "exploratory rescue only; reaction success is not guaranteed; "
                f"template support={int(row['Template_Support_Count'])}"
            )
        if row["Target_Config"] == "Any":
            return (
                f"{row['Recommendation_Tier']}; {row['Soft_Domain_Status']}; "
                "stereochemistry not used for retrieval ranking; "
                f"template support={int(row['Template_Support_Count'])}"
            )
        return (
            f"{row['Recommendation_Tier']}; {row['Soft_Domain_Status']}; "
            f"{int(row['Target_Vote_Count'])}/{int(row['Ensemble_Size'])} seeds support "
            f"{row['Target_Config']}; template support={int(row['Template_Support_Count'])}"
        )

    selected["Recommendation_Reason"] = selected.apply(recommendation_reason, axis=1)
    selected["Score_Interpretation"] = (
        "screening priority only; soft compatibility is literature support and Beta/Alpha score "
        "is not calibrated reaction-success probability"
    )
    selected["Wet_Lab_Use_Requirement"] = (
        "verify full equivalents, concentration, addition order, atmosphere and work-up in the "
        "representative source before experiment"
    )
    return selected


def run_frozen_pipeline(
    candidates: pd.DataFrame,
    manifest: Path,
    device: str,
    batch_size: int,
    top_k_neighbors: int,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        input_path = directory_path / "condition_candidates.csv"
        output_path = directory_path / "condition_predictions.csv"
        candidates.to_csv(input_path, index=False, encoding="utf-8-sig")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pipeline.predict_three_layer",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--manifest",
                str(manifest.resolve()),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--top-k-neighbors",
                str(top_k_neighbors),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"三层推理失败（{returncode_text(process.returncode)}）\n"
                f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )
        return pd.read_csv(output_path, encoding="utf-8-sig")


def returncode_text(code: int) -> str:
    return f"returncode={code}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="湿实验任务 CSV")
    parser.add_argument("--output", type=Path, required=True, help="筛选后的推荐条件 CSV")
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("data/processed/condition_template_library_v1.csv"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/model_manifest_v2.json")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--all-candidates-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_n < 1 or args.batch_size < 1 or args.top_k_neighbors < 1:
        raise ValueError("top-n、batch-size 和 top-k-neighbors 必须为正整数")
    tasks = pd.read_csv(args.input, encoding="utf-8-sig")
    library = pd.read_csv(args.library, encoding="utf-8-sig")
    candidates, preparation_audit = prepare_tasks(tasks, library)
    predictions = run_frozen_pipeline(
        candidates, args.manifest, args.device, args.batch_size, args.top_k_neighbors
    )
    ranked = add_ranking_fields(predictions)
    selected = select_diverse_recommendations(ranked, args.top_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")
    if args.all_candidates_output:
        args.all_candidates_output.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.all_candidates_output, index=False, encoding="utf-8-sig")
    summary = {
        "tasks": len(tasks),
        "candidates_scored": len(ranked),
        "recommendations": len(selected),
        "tier_counts": selected["Recommendation_Tier"].value_counts().to_dict(),
        "preparation": preparation_audit,
        "output": str(args.output),
        "all_candidates_output": (
            None if args.all_candidates_output is None else str(args.all_candidates_output)
        ),
        "warning": "recommendation tiers are screening priorities, not reaction-success probabilities",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
