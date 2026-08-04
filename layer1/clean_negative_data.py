#!/usr/bin/env python3
"""第一层数据清洗：构建保守的 861 条 O4 乙酰化负样本。

The source XLSX is extracted to JSON by build_negative_cleaning_outputs.mjs so
the original workbook is never modified.  Every retained row must match an
independently regenerated O4-acetylated acceptor from glyco_model_local.csv.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.error")


def canonicalize(smiles: object) -> str | None:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def repair_extra_brackets(smiles: object) -> tuple[str, list[int]]:
    text = "" if smiles is None else str(smiles)
    repaired: list[str] = []
    removed_positions: list[int] = []
    i = 0
    while i < len(text):
        if i + 1 < len(text) and text[i : i + 2] == "]]":
            repaired.append("]")
            removed_positions.append(i + 2)  # 1-based position of second bracket
            i += 2
        else:
            repaired.append(text[i])
            i += 1
    return "".join(repaired), removed_positions


def generate_o4_acetate(smiles: str, o4_index: int, c4_index: int) -> dict[str, object]:
    original = Chem.MolFromSmiles(smiles)
    if original is None:
        raise ValueError("original_acceptor_parse_failed")
    o4 = original.GetAtomWithIdx(o4_index)
    if o4.GetAtomicNum() != 8 or o4.GetTotalNumHs() < 1:
        raise ValueError("target_o4_is_not_free_oh")
    if original.GetBondBetweenAtoms(o4_index, c4_index) is None:
        raise ValueError("target_o4_not_bonded_to_c4")

    original_chiral_tags = [atom.GetChiralTag() for atom in original.GetAtoms()]
    editable = Chem.RWMol(original)
    carbonyl_c = editable.AddAtom(Chem.Atom(6))
    carbonyl_o = editable.AddAtom(Chem.Atom(8))
    methyl_c = editable.AddAtom(Chem.Atom(6))
    editable.AddBond(o4_index, carbonyl_c, Chem.BondType.SINGLE)
    editable.AddBond(carbonyl_c, carbonyl_o, Chem.BondType.DOUBLE)
    editable.AddBond(carbonyl_c, methyl_c, Chem.BondType.SINGLE)
    blocked = editable.GetMol()
    Chem.SanitizeMol(blocked)
    Chem.AssignStereochemistry(blocked, force=True, cleanIt=True)

    if [blocked.GetAtomWithIdx(i).GetChiralTag() for i in range(original.GetNumAtoms())] != original_chiral_tags:
        raise ValueError("pre_existing_chiral_tag_changed")

    blocked_smiles = Chem.MolToSmiles(blocked, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(blocked_smiles)
    if reparsed is None:
        raise ValueError("generated_acceptor_reparse_failed")
    mapping = reparsed.GetSubstructMatch(blocked, useChirality=True)
    if len(mapping) != blocked.GetNumAtoms():
        raise ValueError("generated_atom_mapping_failed")
    blocked_o4 = int(mapping[o4_index])
    blocked_c4 = int(mapping[c4_index])
    o4_atom = reparsed.GetAtomWithIdx(blocked_o4)
    if o4_atom.GetAtomicNum() != 8 or o4_atom.GetTotalNumHs() != 0:
        raise ValueError("generated_o4_not_blocked")
    if reparsed.GetBondBetweenAtoms(blocked_o4, blocked_c4) is None:
        raise ValueError("generated_o4_c4_bond_missing")

    return {
        "blocked_smiles": blocked_smiles,
        "blocked_o4_index": blocked_o4,
        "blocked_c4_index": blocked_c4,
    }


def normalized_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    return text or None


def normalized_components(value: object) -> tuple[str, ...] | None:
    text = normalized_text(value)
    if text is None:
        return None
    return tuple(sorted(part.strip() for part in text.split(",") if part.strip()))


def normalized_number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


def pair_cost(negative: pd.Series, parent: pd.Series) -> int:
    cost = 0
    comparisons = [
        (normalized_components(negative["溶剂(solvent)"]), normalized_components(parent["Solvent"]), 2),
        (normalized_components(negative["催化剂"]), normalized_components(parent["Catalyst"]), 2),
        (normalized_number(negative["温度℃"]), normalized_number(parent["Temp_C"]), 1),
        (normalized_number(negative["时间min"]), normalized_number(parent["Time_min"]), 1),
    ]
    for left, right, weight in comparisons:
        if left is not None and right is not None and left != right:
            cost += weight
    return cost


def stable_group_assignment(neg_group: pd.DataFrame, base_group: pd.DataFrame) -> list[tuple[int, int, int]]:
    neg_indices = list(neg_group.sort_values(["序号", "Source_Negative_Row"]).index)
    base_indices = list(base_group.sort_values(["ID", "Reaction_ID"]).index)
    best: tuple[int, tuple[int, ...]] | None = None
    for permutation in itertools.permutations(base_indices):
        total = sum(pair_cost(neg_group.loc[n], base_group.loc[b]) for n, b in zip(neg_indices, permutation))
        candidate = (total, tuple(permutation))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return [
        (neg_idx, base_idx, pair_cost(neg_group.loc[neg_idx], base_group.loc[base_idx]))
        for neg_idx, base_idx in zip(neg_indices, best[1])
    ]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, default=Path(".codex-tmp-negative-cleaning/negative_source_rows.json"))
    parser.add_argument("--base-csv", type=Path, default=Path("data/processed/glyco_model_local.csv"))
    parser.add_argument("--cleaned-output", type=Path, default=Path("data/processed/negative_cleaned_861.csv"))
    parser.add_argument("--repair-report", type=Path, default=Path("results/negative_smiles_repair_report.csv"))
    parser.add_argument("--manual-review", type=Path, default=Path("results/negative_unmatched_manual_review.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/negative_cleaning_summary.csv"))
    args = parser.parse_args()

    payload = json.loads(args.source_json.read_text(encoding="utf-8"))
    negative = pd.DataFrame(payload["negativeValues"][1:], columns=payload["negativeValues"][0])
    negative["Source_Negative_Row"] = range(2, len(negative) + 2)

    repair_rows: list[dict[str, object]] = []
    for idx, row in negative.iterrows():
        raw = row["受体smlie"]
        repaired, positions = repair_extra_brackets(raw)
        raw_canonical = canonicalize(raw)
        repaired_canonical = canonicalize(repaired)
        negative.loc[idx, "Repaired_Acceptor_SMILES"] = repaired
        negative.loc[idx, "Blocked_Acceptor_Canonical_SMILES"] = repaired_canonical
        negative.loc[idx, "Donor_Canonical_SMILES"] = canonicalize(row["供体smile"])
        negative.loc[idx, "Bracket_Repair_Used"] = bool(positions)
        repair_rows.append(
            {
                "Source_Negative_Row": int(row["Source_Negative_Row"]),
                "Source_Sequence_ID": int(row["序号"]),
                "Original_Acceptor_SMILES": raw,
                "Repaired_Acceptor_SMILES": repaired,
                "Raw_RDKit_Parse_OK": raw_canonical is not None,
                "Repaired_RDKit_Parse_OK": repaired_canonical is not None,
                "Modified": bool(positions),
                "Repair_Count": len(positions),
                "Removed_Bracket_Positions_1Based": ";".join(map(str, positions)),
                "Repaired_Canonical_SMILES": repaired_canonical,
            }
        )
    repair = pd.DataFrame(repair_rows)

    base = pd.read_csv(args.base_csv, encoding="utf-8-sig")
    generated_rows: list[dict[str, object]] = []
    for idx, row in base.iterrows():
        generated = generate_o4_acetate(
            str(row["Acceptor_Canonical_SMILES"]),
            int(row["Acceptor_O4_Index"]),
            int(row["Acceptor_C4_Index"]),
        )
        generated_rows.append({"base_index": idx, **generated})
    generated = pd.DataFrame(generated_rows).set_index("base_index")
    base = pd.concat([base, generated], axis=1)

    negative["Structure_Key"] = list(
        zip(negative["Donor_Canonical_SMILES"], negative["Blocked_Acceptor_Canonical_SMILES"])
    )
    base["Structure_Key"] = list(zip(base["Donor_Canonical_SMILES"], base["blocked_smiles"]))
    base_groups = {key: group for key, group in base.groupby("Structure_Key", sort=False)}

    assignments: list[tuple[int, int, int, int]] = []
    unmatched_indices: list[int] = []
    group_size_mismatches: list[tuple[object, int, int]] = []
    for key, neg_group in negative.groupby("Structure_Key", sort=False):
        base_group = base_groups.get(key)
        if base_group is None:
            unmatched_indices.extend(neg_group.index.tolist())
            continue
        if len(neg_group) != len(base_group):
            group_size_mismatches.append((key, len(neg_group), len(base_group)))
            unmatched_indices.extend(neg_group.index.tolist())
            continue
        for neg_idx, base_idx, cost in stable_group_assignment(neg_group, base_group):
            assignments.append((neg_idx, base_idx, cost, len(neg_group)))

    if group_size_mismatches:
        raise ValueError(f"structure group cardinality mismatch: {group_size_mismatches[:3]}")

    cleaned_rows: list[dict[str, object]] = []
    used_parent_ids: set[int] = set()
    for neg_idx, base_idx, cost, group_size in assignments:
        neg = negative.loc[neg_idx]
        parent = base.loc[base_idx]
        parent_id = int(parent["ID"])
        if parent_id in used_parent_ids:
            raise ValueError(f"parent ID reused: {parent_id}")
        used_parent_ids.add(parent_id)
        if neg["Blocked_Acceptor_Canonical_SMILES"] != parent["blocked_smiles"]:
            raise ValueError(f"blocked acceptor mismatch at negative row {neg['Source_Negative_Row']}")
        mapping_method = "unique_structure_key" if group_size == 1 else "duplicate_group_min_cost_stable"
        cleaned_rows.append(
            {
                "Sample_ID": f"NEG-{parent_id:04d}",
                "Pair_ID": f"PAIR-{parent['Reaction_ID']}",
                "Parent_ID": parent_id,
                "Parent_Reaction_ID": parent["Reaction_ID"],
                "Variant": "o4_acetylated",
                "hard_feasibility": 0,
                "Donor_Canonical_SMILES": parent["Donor_Canonical_SMILES"],
                "Acceptor_Canonical_SMILES": parent["blocked_smiles"],
                "Original_Acceptor_Canonical_SMILES": parent["Acceptor_Canonical_SMILES"],
                "Donor_Type": parent["Donor_Type"],
                "Solvent": parent["Solvent"],
                "Solvent_SMILES": parent["Solvent_SMILES"],
                "has_solvent": int(parent["has_solvent"]),
                "Catalyst": parent["Catalyst"],
                "Catalyst_SMILES": parent["Catalyst_SMILES"],
                "has_catalyst": int(parent["has_catalyst"]),
                "Temp_C": parent["Temp_C"],
                "has_temp": int(parent["has_temp"]),
                "Time_min": parent["Time_min"],
                "has_time": int(parent["has_time"]),
                "Target_C4_Index": int(parent["blocked_c4_index"]),
                "Target_O4_Index": int(parent["blocked_o4_index"]),
                "split_pair_group": parent["split_pair_group"],
                "Source_Negative_Row": int(neg["Source_Negative_Row"]),
                "Source_Sequence_ID": int(neg["序号"]),
                "Bracket_Repair_Used": bool(neg["Bracket_Repair_Used"]),
                "Structure_Group_Size": group_size,
                "Mapping_Method": mapping_method,
                "Mapping_Cost": cost,
                "Structure_Verification": "exact_regenerated_o4_acetate_match",
            }
        )
    cleaned = pd.DataFrame(cleaned_rows).sort_values("Parent_ID").reset_index(drop=True)

    unmatched = negative.loc[unmatched_indices].copy()
    donor_parent_counts = base.groupby("Donor_Canonical_SMILES").size().to_dict()
    manual = pd.DataFrame(
        {
            "Source_Negative_Row": unmatched["Source_Negative_Row"].astype(int),
            "Source_Sequence_ID": unmatched["序号"].astype(int),
            "Donor_Canonical_SMILES": unmatched["Donor_Canonical_SMILES"],
            "Original_Blocked_Acceptor_SMILES": unmatched["受体smlie"],
            "Repaired_Blocked_Acceptor_SMILES": unmatched["Repaired_Acceptor_SMILES"],
            "Blocked_Acceptor_Canonical_SMILES": unmatched["Blocked_Acceptor_Canonical_SMILES"],
            "Solvent": unmatched["溶剂(solvent)"],
            "Catalyst": unmatched["催化剂"],
            "Temp_C": unmatched["温度℃"],
            "Time_min": unmatched["时间min"],
            "Candidate_Parents_With_Same_Donor": unmatched["Donor_Canonical_SMILES"].map(donor_parent_counts).fillna(0).astype(int),
            "Review_Reason": "no_exact_match_to_regenerated_parent_o4_acetate",
            "Training_Decision": "exclude_pending_manual_review",
        }
    ).sort_values("Source_Sequence_ID")

    summary = pd.DataFrame(
        [
            ("Source O4-blocked rows", len(negative), "raw workbook count"),
            ("Raw acceptor SMILES parseable", int(repair["Raw_RDKit_Parse_OK"].sum()), "RDKit parse before repair"),
            ("Rows repaired by ]] -> ]", int(repair["Modified"].sum()), "deterministic syntax repair"),
            ("Acceptor SMILES parseable after repair", int(repair["Repaired_RDKit_Parse_OK"].sum()), "required for all source rows"),
            ("Exact regenerated O4-acetate matches", len(cleaned), "retained reliable negatives"),
            ("Excluded for manual review", len(manual), "not used for training"),
            ("Unique parent reactions used", cleaned["Parent_ID"].nunique(), "must equal retained negatives"),
            ("Duplicate parent use", int(cleaned["Parent_ID"].duplicated().sum()), "must be zero"),
            ("Unique Sample_ID", cleaned["Sample_ID"].nunique(), "must equal retained negatives"),
            ("Blocked O4 has no hydrogen", len(cleaned), "validated during deterministic generation"),
            ("Product/config/yield leakage columns", 0, "not present in cleaned training table"),
        ],
        columns=["Audit_Item", "Value", "Acceptance_Criterion"],
    )

    expected = {
        "source": 876,
        "raw_parseable": 377,
        "repaired": 499,
        "post_repair_parseable": 876,
        "cleaned": 861,
        "manual": 15,
    }
    actual = {
        "source": len(negative),
        "raw_parseable": int(repair["Raw_RDKit_Parse_OK"].sum()),
        "repaired": int(repair["Modified"].sum()),
        "post_repair_parseable": int(repair["Repaired_RDKit_Parse_OK"].sum()),
        "cleaned": len(cleaned),
        "manual": len(manual),
    }
    if actual != expected:
        raise ValueError(f"QA totals differ from frozen expectations: actual={actual}, expected={expected}")
    if cleaned["Parent_ID"].duplicated().any() or cleaned["Sample_ID"].duplicated().any():
        raise ValueError("cleaned identifiers are not one-to-one")
    forbidden = {"Product_SMILES", "Product_Config", "yield", "yield_value", "Label", "Acceptor_OH_Status"}
    if forbidden.intersection(cleaned.columns):
        raise ValueError(f"forbidden leakage columns retained: {forbidden.intersection(cleaned.columns)}")

    write_csv(cleaned, args.cleaned_output)
    write_csv(repair, args.repair_report)
    write_csv(manual, args.manual_review)
    write_csv(summary, args.summary_output)
    print(json.dumps({"cleaned": len(cleaned), "manual_review": len(manual), "repair_rows": len(repair)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
