#!/usr/bin/env python3
"""第一层数据构造：生成完整数据及结构唯一的正负样本对。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer1.o4_acetylation import canonicalize_smiles, generate_o4_acetate


STRUCTURE_KEY_COLUMNS = ["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"]
FORBIDDEN_OUTPUT_COLUMNS = {
    "Solvent",
    "Solvent_SMILES",
    "Catalyst",
    "Catalyst_SMILES",
    "Temp_C",
    "Time_min",
    "has_solvent",
    "has_catalyst",
    "has_temp",
    "has_time",
    "Solvent_Components",
    "Catalyst_Components",
    "Product_SMILES",
    "Product_Config",
    "yield",
    "yield_value",
    "Label",
    "Acceptor_OH_Status",
}


def stable_digest(*values: object) -> str:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def semicolon_join(values: pd.Series) -> str:
    return ";".join(str(value) for value in values)


def make_sample_row(
    *,
    sample_id: str,
    pair_id: str,
    structural_group_id: str,
    structural_group_hash: str,
    parent_id: int,
    reaction_id: str,
    all_parent_ids: str,
    all_reaction_ids: str,
    duplicate_count: int,
    variant: str,
    label: int,
    donor_smiles: str,
    acceptor_smiles: str,
    original_acceptor_smiles: str,
    donor_type: str,
    target_c4_index: int,
    target_o4_index: int,
) -> dict[str, object]:
    input_hash = stable_digest(donor_smiles, acceptor_smiles)
    return {
        "Sample_ID": sample_id,
        "Pair_ID": pair_id,
        "Structural_Group_ID": structural_group_id,
        "Structural_Group_SHA256": structural_group_hash,
        "Input_Structure_SHA256": input_hash,
        "Representative_Parent_ID": parent_id,
        "Representative_Reaction_ID": reaction_id,
        "All_Parent_IDs": all_parent_ids,
        "All_Parent_Reaction_IDs": all_reaction_ids,
        "Duplicate_Count": duplicate_count,
        "Variant": variant,
        "hard_feasibility": label,
        "Donor_Canonical_SMILES": donor_smiles,
        "Acceptor_Canonical_SMILES": acceptor_smiles,
        "Original_Acceptor_Canonical_SMILES": original_acceptor_smiles,
        "Donor_Type": donor_type,
        "Target_C4_Index": target_c4_index,
        "Target_O4_Index": target_o4_index,
    }


def validate_output_chemistry(unique_pairs: pd.DataFrame) -> None:
    for row in unique_pairs.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.Acceptor_Canonical_SMILES)
        if molecule is None:
            raise ValueError(f"output acceptor cannot be parsed: {row.Sample_ID}")
        c4_index = int(row.Target_C4_Index)
        o4_index = int(row.Target_O4_Index)
        if not (0 <= c4_index < molecule.GetNumAtoms() and 0 <= o4_index < molecule.GetNumAtoms()):
            raise ValueError(f"output target index out of range: {row.Sample_ID}")
        c4 = molecule.GetAtomWithIdx(c4_index)
        o4 = molecule.GetAtomWithIdx(o4_index)
        if c4.GetAtomicNum() != 6 or o4.GetAtomicNum() != 8:
            raise ValueError(f"output target elements invalid: {row.Sample_ID}")
        if molecule.GetBondBetweenAtoms(c4_index, o4_index) is None:
            raise ValueError(f"output C4-O4 bond missing: {row.Sample_ID}")
        if row.Variant == "original":
            if o4.GetTotalNumHs() < 1 or o4.GetDegree() != 1:
                raise ValueError(f"positive O4 is not a free hydroxyl: {row.Sample_ID}")
        elif row.Variant == "o4_acetylated":
            if o4.GetTotalNumHs() != 0:
                raise ValueError(f"negative O4 still has hydrogen: {row.Sample_ID}")
            external_neighbors = [atom for atom in o4.GetNeighbors() if atom.GetIdx() != c4_index]
            if len(external_neighbors) != 1 or external_neighbors[0].GetAtomicNum() != 6:
                raise ValueError(f"negative O4 lacks one acetyl carbon: {row.Sample_ID}")
            carbonyl_c = external_neighbors[0]
            has_carbonyl_o = any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(carbonyl_c).GetAtomicNum() == 8
                for bond in carbonyl_c.GetBonds()
            )
            if not has_carbonyl_o:
                raise ValueError(f"negative O4 substituent is not an acetate: {row.Sample_ID}")
        else:
            raise ValueError(f"unknown variant: {row.Variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-data", type=Path, default=Path("data/processed/glyco_model_local.csv"))
    parser.add_argument(
        "--known-negatives",
        type=Path,
        default=Path("data/processed/o4_acetylated_regenerated_861.csv"),
    )
    parser.add_argument(
        "--full-pairs-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_full_pairs_3122.csv"),
    )
    parser.add_argument(
        "--unique-pairs-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_unique_pairs_2632.csv"),
    )
    parser.add_argument(
        "--model-ready-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_model_ready_2632.csv"),
    )
    parser.add_argument(
        "--groups-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_structural_groups_1316.csv"),
    )
    parser.add_argument(
        "--generation-validation-output",
        type=Path,
        default=Path("results/hard_feasibility_generation_validation_1561.csv"),
    )
    parser.add_argument(
        "--duplicate-groups-output",
        type=Path,
        default=Path("results/hard_feasibility_duplicate_groups.csv"),
    )
    parser.add_argument(
        "--qa-summary-output",
        type=Path,
        default=Path("results/hard_feasibility_unique_dataset_qa.csv"),
    )
    args = parser.parse_args()

    parents = pd.read_csv(args.parent_data, encoding="utf-8-sig").sort_values("ID").reset_index(drop=True)
    if len(parents) != 1561:
        raise ValueError(f"expected 1561 parent reactions, found {len(parents)}")
    required = {
        "ID",
        "Reaction_ID",
        "Donor_Canonical_SMILES",
        "Acceptor_Canonical_SMILES",
        "Donor_Type",
        "Acceptor_C4_Index",
        "Acceptor_O4_Index",
    }
    if missing := required - set(parents.columns):
        raise ValueError(f"parent data missing required columns: {sorted(missing)}")
    if parents["ID"].duplicated().any() or parents["Reaction_ID"].duplicated().any():
        raise ValueError("parent ID and Reaction_ID must each be unique")

    known = pd.read_csv(args.known_negatives, encoding="utf-8-sig")
    known_by_parent = known.set_index("Parent_ID")["Acceptor_Canonical_SMILES"].to_dict()
    if len(known_by_parent) != 861:
        raise ValueError(f"expected 861 known negatives, found {len(known_by_parent)}")

    generated_records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []
    for parent in parents.itertuples(index=False):
        result = generate_o4_acetate(
            parent.Acceptor_Canonical_SMILES,
            int(parent.Acceptor_O4_Index),
            int(parent.Acceptor_C4_Index),
        )
        known_partner = known_by_parent.get(int(parent.ID))
        known_match: bool | None = None
        if known_partner is not None:
            known_match = canonicalize_smiles(known_partner) == result.generated_canonical_smiles
        generation_checks = [
            result.check_original_parse,
            result.check_target_indices_valid,
            result.check_o4_is_free_hydroxyl,
            result.check_acetyl_connectivity,
            result.check_valence_legal,
            result.check_stereochemistry_preserved,
            result.check_only_o4_acetylation,
        ]
        overall_pass = all(generation_checks) and (known_match is not False)
        generated_records.append(
            {
                "Parent_ID": int(parent.ID),
                "Generated_Acceptor_Canonical_SMILES": result.generated_canonical_smiles,
                "Generated_Target_C4_Index": result.generated_c4_index,
                "Generated_Target_O4_Index": result.generated_o4_index,
            }
        )
        validation_records.append(
            {
                "Parent_ID": int(parent.ID),
                "Parent_Reaction_ID": parent.Reaction_ID,
                "Original_Acceptor_Canonical_SMILES": result.original_canonical_smiles,
                "Program_Generated_Acceptor_Canonical_SMILES": result.generated_canonical_smiles,
                "Original_Target_C4_Index": result.original_c4_index,
                "Original_Target_O4_Index": result.original_o4_index,
                "Generated_Target_C4_Index": result.generated_c4_index,
                "Generated_Target_O4_Index": result.generated_o4_index,
                "Atom_Count_Delta": result.generated_atom_count - result.original_atom_count,
                "Bond_Count_Delta": result.generated_bond_count - result.original_bond_count,
                "Check_1_Original_Parse": result.check_original_parse,
                "Check_2_Target_Indices_Valid": result.check_target_indices_valid,
                "Check_3_O4_Free_Hydroxyl": result.check_o4_is_free_hydroxyl,
                "Check_4_Acetyl_Connectivity": result.check_acetyl_connectivity,
                "Check_5_Valence_Legal": result.check_valence_legal,
                "Check_6_Stereochemistry_Preserved": result.check_stereochemistry_preserved,
                "Check_7_Only_O4_Acetylation": result.check_only_o4_acetylation,
                "Known_Cooperator_Negative": known_partner is not None,
                "Check_8_Known_Cooperator_Match": known_match,
                "Overall_Generation_Pass": overall_pass,
            }
        )
    generated = pd.DataFrame(generated_records)
    validation = pd.DataFrame(validation_records)
    parents = parents.rename(columns={"ID": "Parent_ID"})
    parents = parents.merge(generated, on="Parent_ID", how="left", validate="one_to_one")

    group_columns = ["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"]
    full_rows: list[dict[str, object]] = []
    unique_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for (donor_smiles, original_acceptor), group in parents.groupby(group_columns, sort=True, dropna=False):
        group = group.sort_values("Parent_ID")
        representative = group.iloc[0]
        if group["Donor_Type"].nunique(dropna=False) != 1:
            raise ValueError("Donor_Type differs within a structural group")
        if group["Acceptor_C4_Index"].nunique(dropna=False) != 1 or group["Acceptor_O4_Index"].nunique(dropna=False) != 1:
            raise ValueError("target indices differ within a structural group")
        if group["Generated_Acceptor_Canonical_SMILES"].nunique(dropna=False) != 1:
            raise ValueError("program-generated negative differs within a structural group")
        if group["Generated_Target_C4_Index"].nunique(dropna=False) != 1 or group["Generated_Target_O4_Index"].nunique(dropna=False) != 1:
            raise ValueError("generated target indices differ within a structural group")

        group_hash = stable_digest(donor_smiles, original_acceptor)
        group_id = f"SG-{group_hash[:16]}"
        pair_id = f"PAIR-{group_hash[:16]}"
        all_parent_ids = semicolon_join(group["Parent_ID"].astype(int))
        all_reaction_ids = semicolon_join(group["Reaction_ID"])
        duplicate_count = len(group)
        blocked_acceptor = representative["Generated_Acceptor_Canonical_SMILES"]

        group_rows.append(
            {
                "Structural_Group_ID": group_id,
                "Structural_Group_SHA256": group_hash,
                "Representative_Parent_ID": int(representative["Parent_ID"]),
                "Representative_Reaction_ID": representative["Reaction_ID"],
                "All_Parent_IDs": all_parent_ids,
                "All_Parent_Reaction_IDs": all_reaction_ids,
                "Duplicate_Count": duplicate_count,
                "Donor_Canonical_SMILES": donor_smiles,
                "Original_Acceptor_Canonical_SMILES": original_acceptor,
                "O4_Acetylated_Acceptor_Canonical_SMILES": blocked_acceptor,
                "Donor_Type": representative["Donor_Type"],
                "Original_Target_C4_Index": int(representative["Acceptor_C4_Index"]),
                "Original_Target_O4_Index": int(representative["Acceptor_O4_Index"]),
                "Generated_Target_C4_Index": int(representative["Generated_Target_C4_Index"]),
                "Generated_Target_O4_Index": int(representative["Generated_Target_O4_Index"]),
            }
        )

        positive_unique = make_sample_row(
            sample_id=f"POS-{group_hash[:16]}",
            pair_id=pair_id,
            structural_group_id=group_id,
            structural_group_hash=group_hash,
            parent_id=int(representative["Parent_ID"]),
            reaction_id=representative["Reaction_ID"],
            all_parent_ids=all_parent_ids,
            all_reaction_ids=all_reaction_ids,
            duplicate_count=duplicate_count,
            variant="original",
            label=1,
            donor_smiles=donor_smiles,
            acceptor_smiles=original_acceptor,
            original_acceptor_smiles=original_acceptor,
            donor_type=representative["Donor_Type"],
            target_c4_index=int(representative["Acceptor_C4_Index"]),
            target_o4_index=int(representative["Acceptor_O4_Index"]),
        )
        negative_unique = make_sample_row(
            sample_id=f"NEG-{group_hash[:16]}",
            pair_id=pair_id,
            structural_group_id=group_id,
            structural_group_hash=group_hash,
            parent_id=int(representative["Parent_ID"]),
            reaction_id=representative["Reaction_ID"],
            all_parent_ids=all_parent_ids,
            all_reaction_ids=all_reaction_ids,
            duplicate_count=duplicate_count,
            variant="o4_acetylated",
            label=0,
            donor_smiles=donor_smiles,
            acceptor_smiles=blocked_acceptor,
            original_acceptor_smiles=original_acceptor,
            donor_type=representative["Donor_Type"],
            target_c4_index=int(representative["Generated_Target_C4_Index"]),
            target_o4_index=int(representative["Generated_Target_O4_Index"]),
        )
        unique_rows.extend([positive_unique, negative_unique])

        for parent in group.itertuples(index=False):
            full_pair_id = f"PAIR-RXN-{parent.Reaction_ID}"
            common = {
                "pair_id": full_pair_id,
                "structural_group_id": group_id,
                "structural_group_hash": group_hash,
                "parent_id": int(parent.Parent_ID),
                "reaction_id": parent.Reaction_ID,
                "all_parent_ids": all_parent_ids,
                "all_reaction_ids": all_reaction_ids,
                "duplicate_count": duplicate_count,
                "donor_smiles": donor_smiles,
                "original_acceptor_smiles": original_acceptor,
                "donor_type": parent.Donor_Type,
            }
            full_rows.append(
                make_sample_row(
                    sample_id=f"POS-RXN-{int(parent.Parent_ID):04d}",
                    variant="original",
                    label=1,
                    acceptor_smiles=original_acceptor,
                    target_c4_index=int(parent.Acceptor_C4_Index),
                    target_o4_index=int(parent.Acceptor_O4_Index),
                    **common,
                )
            )
            full_rows.append(
                make_sample_row(
                    sample_id=f"NEG-RXN-{int(parent.Parent_ID):04d}",
                    variant="o4_acetylated",
                    label=0,
                    acceptor_smiles=parent.Generated_Acceptor_Canonical_SMILES,
                    target_c4_index=int(parent.Generated_Target_C4_Index),
                    target_o4_index=int(parent.Generated_Target_O4_Index),
                    **common,
                )
            )

    full_pairs = pd.DataFrame(full_rows).sort_values(["Representative_Parent_ID", "hard_feasibility"], ascending=[True, False]).reset_index(drop=True)
    unique_pairs = pd.DataFrame(unique_rows).sort_values(["Structural_Group_ID", "hard_feasibility"], ascending=[True, False]).reset_index(drop=True)
    groups = pd.DataFrame(group_rows).sort_values("Structural_Group_ID").reset_index(drop=True)
    duplicate_groups = groups.loc[groups["Duplicate_Count"] > 1].copy()
    model_ready_columns = [
        "Sample_ID",
        "Pair_ID",
        "Structural_Group_ID",
        "hard_feasibility",
        "Donor_Canonical_SMILES",
        "Acceptor_Canonical_SMILES",
        "Donor_Type",
        "Target_C4_Index",
        "Target_O4_Index",
    ]
    model_ready = unique_pairs[model_ready_columns].copy()

    checks: list[tuple[str, int, int, bool, str]] = []

    def add_check(name: str, expected: int, observed: int, details: str) -> None:
        checks.append((name, expected, observed, expected == observed, details))

    add_check("Parent reaction count", 1561, len(parents), "source positive reactions")
    add_check("Generated negative count", 1561, len(validation), "one generation per parent")
    for check_column in [
        "Check_1_Original_Parse",
        "Check_2_Target_Indices_Valid",
        "Check_3_O4_Free_Hydroxyl",
        "Check_4_Acetyl_Connectivity",
        "Check_5_Valence_Legal",
        "Check_6_Stereochemistry_Preserved",
        "Check_7_Only_O4_Acetylation",
    ]:
        add_check(check_column, 1561, int(validation[check_column].sum()), "all generated parent reactions")
    add_check("Atom count delta equals 3", 1561, int((validation["Atom_Count_Delta"] == 3).sum()), "C, O and methyl C added")
    add_check("Bond count delta equals 3", 1561, int((validation["Bond_Count_Delta"] == 3).sum()), "O-C, C=O and C-C added")
    add_check("Overall generation pass", 1561, int(validation["Overall_Generation_Pass"].sum()), "checks 1-7 plus known-partner match where available")
    add_check("Known cooperator negatives", 861, int(validation["Known_Cooperator_Negative"].sum()), "previously validated subset")
    add_check("Known cooperator matches", 861, int(validation["Check_8_Known_Cooperator_Match"].fillna(False).sum()), "canonical structure equality")
    add_check("Unique structural groups", 1316, len(groups), "donor + original acceptor")
    add_check("Unique structural group IDs", 1316, groups["Structural_Group_ID"].nunique(), "no truncated-ID collision")
    add_check("Unique structural group hashes", 1316, groups["Structural_Group_SHA256"].nunique(), "no SHA-256 collision")
    add_check("Full paired archive rows", 3122, len(full_pairs), "1561 positive + 1561 negative")
    add_check("Unique model rows", 2632, len(unique_pairs), "1316 positive + 1316 negative")
    add_check("Model-ready rows", 2632, len(model_ready), "minimal structure-only table")
    add_check("Model-ready columns", 9, len(model_ready.columns), "IDs, label, structures, donor type and target indices")
    add_check("Unique Sample_ID", 2632, unique_pairs["Sample_ID"].nunique(), "no sample identifier collision")
    add_check("Unique Pair_ID", 1316, unique_pairs["Pair_ID"].nunique(), "one positive/negative pair per structural group")
    add_check("Unique input structures", 2632, unique_pairs["Input_Structure_SHA256"].nunique(), "no repeated model input or cross-label collision")
    add_check("Positive rows", 1316, int((unique_pairs["hard_feasibility"] == 1).sum()), "one per structural group")
    add_check("Negative rows", 1316, int((unique_pairs["hard_feasibility"] == 0).sum()), "one per structural group")
    add_check("Duplicate source rows represented", 1561, int(groups["Duplicate_Count"].sum()), "no parent reaction lost from metadata")
    add_check("Condition columns retained", 0, len(FORBIDDEN_OUTPUT_COLUMNS.intersection(unique_pairs.columns)), "model dataset is structure-only")
    direct_leakage_columns = {
        "Variant",
        "Original_Acceptor_Canonical_SMILES",
        "All_Parent_IDs",
        "All_Parent_Reaction_IDs",
        "Duplicate_Count",
        "Construction_Method",
        "Construction_Validation",
        "Structure_Verification",
    }
    add_check("Direct construction/leakage columns in model-ready table", 0, len(direct_leakage_columns.intersection(model_ready.columns)), "metadata excluded from model-ready CSV")

    pair_counts = unique_pairs.groupby("Structural_Group_ID").size()
    label_counts = unique_pairs.groupby("Structural_Group_ID")["hard_feasibility"].nunique()
    variant_counts = unique_pairs.groupby("Structural_Group_ID")["Variant"].nunique()
    add_check("Groups with exactly two rows", 1316, int((pair_counts == 2).sum()), "one positive and one negative")
    add_check("Groups with both labels", 1316, int((label_counts == 2).sum()), "labels 1 and 0")
    add_check("Groups with both variants", 1316, int((variant_counts == 2).sum()), "original and O4-acetylated")
    group_acceptor_difference = (
        groups["Original_Acceptor_Canonical_SMILES"]
        != groups["O4_Acetylated_Acceptor_Canonical_SMILES"]
    )
    add_check("Groups with distinct positive and negative acceptors", 1316, int(group_acceptor_difference.sum()), "O4 acetylation changes input structure")

    archive_pair_counts = full_pairs.groupby("Representative_Parent_ID").size()
    archive_label_counts = full_pairs.groupby("Representative_Parent_ID")["hard_feasibility"].nunique()
    add_check("Full archive parents with two rows", 1561, int((archive_pair_counts == 2).sum()), "one positive and one negative per reaction")
    add_check("Full archive parents with both labels", 1561, int((archive_label_counts == 2).sum()), "labels 1 and 0 per reaction")

    validate_output_chemistry(unique_pairs)
    add_check("Output chemistry and target indices valid", 2632, len(unique_pairs), "RDKit parse, C4-O4, free/blocked O4")

    qa = pd.DataFrame(checks, columns=["QA_Check", "Expected", "Observed", "Pass", "Details"])
    if not qa["Pass"].all():
        failed = qa.loc[~qa["Pass"]]
        raise ValueError(f"QA failed:\n{failed.to_string(index=False)}")
    if FORBIDDEN_OUTPUT_COLUMNS.intersection(full_pairs.columns):
        raise ValueError("full pair archive unexpectedly contains forbidden condition/leakage columns")

    write_csv(full_pairs, args.full_pairs_output)
    write_csv(unique_pairs, args.unique_pairs_output)
    write_csv(model_ready, args.model_ready_output)
    write_csv(groups, args.groups_output)
    write_csv(validation, args.generation_validation_output)
    write_csv(duplicate_groups, args.duplicate_groups_output)
    write_csv(qa, args.qa_summary_output)
    print(
        {
            "structural_groups": len(groups),
            "unique_model_rows": len(unique_pairs),
            "full_archive_rows": len(full_pairs),
            "duplicate_groups": len(duplicate_groups),
            "all_qa_pass": bool(qa["Pass"].all()),
        }
    )


if __name__ == "__main__":
    main()
