#!/usr/bin/env python3
"""第一层规则复现：重新生成并逐项验证 861 条 O4 乙酰化负样本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer1.o4_acetylation import O4AcetylationError, canonicalize_smiles, generate_o4_acetate


CHECK_COLUMNS = [
    "Check_1_Original_Parse",
    "Check_2_Target_Indices_Valid",
    "Check_3_O4_Free_Hydroxyl",
    "Check_4_Acetyl_Connectivity",
    "Check_5_Valence_Legal",
    "Check_6_Stereochemistry_Preserved",
    "Check_7_Only_O4_Acetylation",
    "Check_8_Cooperator_Structure_Match",
]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned-negatives", type=Path, default=Path("data/processed/negative_cleaned_861.csv"))
    parser.add_argument("--parent-data", type=Path, default=Path("data/processed/glyco_model_local.csv"))
    parser.add_argument(
        "--generated-output",
        type=Path,
        default=Path("data/processed/o4_acetylated_regenerated_861.csv"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("results/o4_acetylation_validation_861.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/o4_acetylation_validation_summary.csv"),
    )
    args = parser.parse_args()

    negative = pd.read_csv(args.cleaned_negatives, encoding="utf-8-sig")
    parents = pd.read_csv(args.parent_data, encoding="utf-8-sig")
    required_negative = {
        "Sample_ID",
        "Parent_ID",
        "Parent_Reaction_ID",
        "Acceptor_Canonical_SMILES",
    }
    required_parent = {
        "ID",
        "Reaction_ID",
        "Acceptor_Canonical_SMILES",
        "Acceptor_C4_Index",
        "Acceptor_O4_Index",
    }
    if missing := required_negative - set(negative.columns):
        raise ValueError(f"cleaned negative data missing columns: {sorted(missing)}")
    if missing := required_parent - set(parents.columns):
        raise ValueError(f"parent data missing columns: {sorted(missing)}")
    if len(negative) != 861:
        raise ValueError(f"expected 861 cleaned negatives, found {len(negative)}")
    if negative["Parent_ID"].duplicated().any():
        raise ValueError("Parent_ID must be unique before regeneration")

    parent_subset = parents[list(required_parent)].rename(columns={"ID": "Parent_ID"})
    merged = negative.merge(parent_subset, on="Parent_ID", how="left", validate="one_to_one", suffixes=("_Partner", "_Parent"))
    if merged["Reaction_ID"].isna().any():
        raise ValueError("one or more Parent_ID values do not exist in the parent dataset")
    if not (merged["Parent_Reaction_ID"].astype(str) == merged["Reaction_ID"].astype(str)).all():
        raise ValueError("Parent_Reaction_ID disagrees with parent dataset")

    validation_rows: list[dict[str, object]] = []
    generated_by_parent: dict[int, dict[str, object]] = {}
    for row in merged.itertuples(index=False):
        base_record = {
            "Sample_ID": row.Sample_ID,
            "Parent_ID": int(row.Parent_ID),
            "Parent_Reaction_ID": row.Parent_Reaction_ID,
            "Original_Acceptor_Canonical_SMILES": row.Acceptor_Canonical_SMILES_Parent,
            "Original_Target_C4_Index": int(row.Acceptor_C4_Index),
            "Original_Target_O4_Index": int(row.Acceptor_O4_Index),
            "Cooperator_Blocked_Acceptor_Canonical_SMILES": row.Acceptor_Canonical_SMILES_Partner,
        }
        try:
            result = generate_o4_acetate(
                row.Acceptor_Canonical_SMILES_Parent,
                int(row.Acceptor_O4_Index),
                int(row.Acceptor_C4_Index),
            )
            partner_canonical = canonicalize_smiles(row.Acceptor_Canonical_SMILES_Partner)
            partner_match = result.generated_canonical_smiles == partner_canonical
            generation_checks = [
                result.check_original_parse,
                result.check_target_indices_valid,
                result.check_o4_is_free_hydroxyl,
                result.check_acetyl_connectivity,
                result.check_valence_legal,
                result.check_stereochemistry_preserved,
                result.check_only_o4_acetylation,
            ]
            overall_pass = all(generation_checks) and partner_match
            validation = {
                **base_record,
                "Program_Generated_Acceptor_Canonical_SMILES": result.generated_canonical_smiles,
                "Generated_Target_C4_Index": result.generated_c4_index,
                "Generated_Target_O4_Index": result.generated_o4_index,
                "Original_Atom_Count": result.original_atom_count,
                "Generated_Atom_Count": result.generated_atom_count,
                "Atom_Count_Delta": result.generated_atom_count - result.original_atom_count,
                "Original_Bond_Count": result.original_bond_count,
                "Generated_Bond_Count": result.generated_bond_count,
                "Bond_Count_Delta": result.generated_bond_count - result.original_bond_count,
                "Original_Chiral_Atom_Count": result.original_chiral_atom_count,
                "Generated_Parent_Chiral_Atom_Count": result.generated_original_chiral_atom_count,
                "Check_1_Original_Parse": result.check_original_parse,
                "Check_2_Target_Indices_Valid": result.check_target_indices_valid,
                "Check_3_O4_Free_Hydroxyl": result.check_o4_is_free_hydroxyl,
                "Check_4_Acetyl_Connectivity": result.check_acetyl_connectivity,
                "Check_5_Valence_Legal": result.check_valence_legal,
                "Check_6_Stereochemistry_Preserved": result.check_stereochemistry_preserved,
                "Check_7_Only_O4_Acetylation": result.check_only_o4_acetylation,
                "Check_8_Cooperator_Structure_Match": partner_match,
                "Overall_Validation_Pass": overall_pass,
                "Validation_Error": "" if overall_pass else "program-generated canonical structure differs from cooperator structure",
            }
            generated_by_parent[int(row.Parent_ID)] = {
                "generated_smiles": result.generated_canonical_smiles,
                "generated_c4_index": result.generated_c4_index,
                "generated_o4_index": result.generated_o4_index,
            }
        except (O4AcetylationError, ValueError) as exc:
            validation = {
                **base_record,
                "Program_Generated_Acceptor_Canonical_SMILES": "",
                "Generated_Target_C4_Index": "",
                "Generated_Target_O4_Index": "",
                "Original_Atom_Count": "",
                "Generated_Atom_Count": "",
                "Atom_Count_Delta": "",
                "Original_Bond_Count": "",
                "Generated_Bond_Count": "",
                "Bond_Count_Delta": "",
                "Original_Chiral_Atom_Count": "",
                "Generated_Parent_Chiral_Atom_Count": "",
                **{column: False for column in CHECK_COLUMNS},
                "Overall_Validation_Pass": False,
                "Validation_Error": str(exc),
            }
        validation_rows.append(validation)

    validation = pd.DataFrame(validation_rows).sort_values("Parent_ID").reset_index(drop=True)
    check_summary = [
        {
            "Validation_Check": column,
            "Pass_Count": int(validation[column].sum()),
            "Fail_Count": int((~validation[column].astype(bool)).sum()),
            "Total": len(validation),
            "Pass_Rate": float(validation[column].mean()),
        }
        for column in CHECK_COLUMNS
    ]
    summary = pd.DataFrame(
        check_summary
        + [
            {
                "Validation_Check": "Overall_Validation_Pass",
                "Pass_Count": int(validation["Overall_Validation_Pass"].sum()),
                "Fail_Count": int((~validation["Overall_Validation_Pass"].astype(bool)).sum()),
                "Total": len(validation),
                "Pass_Rate": float(validation["Overall_Validation_Pass"].mean()),
            }
        ]
    )

    generated = negative.copy()
    generated["Acceptor_Canonical_SMILES"] = generated["Parent_ID"].map(
        lambda parent_id: generated_by_parent.get(int(parent_id), {}).get("generated_smiles")
    )
    generated["Target_C4_Index"] = generated["Parent_ID"].map(
        lambda parent_id: generated_by_parent.get(int(parent_id), {}).get("generated_c4_index")
    )
    generated["Target_O4_Index"] = generated["Parent_ID"].map(
        lambda parent_id: generated_by_parent.get(int(parent_id), {}).get("generated_o4_index")
    )
    generated["Construction_Method"] = "deterministic_target_o4_acetylation"
    generated["Construction_Validation"] = generated["Parent_ID"].map(
        validation.set_index("Parent_ID")["Overall_Validation_Pass"]
    )
    generated = generated.sort_values("Parent_ID").reset_index(drop=True)

    write_csv(generated, args.generated_output)
    write_csv(validation, args.validation_output)
    write_csv(summary, args.summary_output)

    failed = validation.loc[~validation["Overall_Validation_Pass"]]
    print(
        {
            "rows": len(validation),
            "passed": int(validation["Overall_Validation_Pass"].sum()),
            "failed": len(failed),
            "generated_output": str(args.generated_output),
            "validation_output": str(args.validation_output),
        }
    )
    if not failed.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
