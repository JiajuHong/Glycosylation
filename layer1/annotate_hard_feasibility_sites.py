#!/usr/bin/env python3
"""第一层位点标注：为 1316 组结构唯一正负样本标记目标糖基化位点。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer1.o4_acetylation import generate_o4_acetate


ROLE_COLUMNS = {
    "C1": "Acceptor_C1_Index",
    "O5": "Acceptor_O5_Index",
    "C2": "Acceptor_C2_Index",
    "C3": "Acceptor_C3_Index",
    "C4": "Acceptor_C4_Index",
    "C5": "Acceptor_C5_Index",
    "O4": "Acceptor_O4_Index",
}
ROLE_ORDER = ["C1", "O5", "C2", "C3", "C4", "C5", "O4", "O4_EXTERNAL"]
EXTENDED_ROLE_ORDER = [
    "O4",
    "C4",
    "C3",
    "C5",
    "C2",
    "O5",
    "C1",
    "N2_SUB_ENTRY",
    "N2_SUBGRAPH",
    "PG_ENTRY",
    "O4_EXTERNAL",
]
RING_EDGES = [("C1", "O5"), ("O5", "C5"), ("C5", "C4"), ("C4", "C3"), ("C3", "C2"), ("C2", "C1")]
EXPECTED_ELEMENTS = {"C1": 6, "O5": 8, "C2": 6, "C3": 6, "C4": 6, "C5": 6, "O4": 8}
CONDITION_COLUMNS = {
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
}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def role_json(roles: dict[str, int | None]) -> str:
    payload = {role: ([] if roles.get(role) is None else [int(roles[role])]) for role in ROLE_ORDER}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def core_indices(roles: dict[str, int | None]) -> list[int]:
    return [int(roles[role]) for role in ROLE_ORDER if roles.get(role) is not None]


def parse_indices(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(float(part)) for part in text.split(";") if part.strip()]


def join_indices(values: list[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def map_indices(values: list[int], mapping: tuple[int, ...]) -> list[int]:
    return [int(mapping[index]) for index in values]


def extended_role_json(
    base_role_json: str,
    mapping: tuple[int, ...] | None,
    o4_external: int | None,
) -> str:
    base = json.loads(base_role_json)
    payload: dict[str, list[int]] = {}
    for role in EXTENDED_ROLE_ORDER:
        if role == "O4_EXTERNAL":
            payload[role] = [] if o4_external is None else [int(o4_external)]
            continue
        values = [int(value) for value in base.get(role, [])]
        payload[role] = values if mapping is None else map_indices(values, mapping)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def topology_is_valid(molecule: Chem.Mol, roles: dict[str, int | None]) -> bool:
    explicit_roles = [role for role in ROLE_ORDER if roles.get(role) is not None]
    indices = [int(roles[role]) for role in explicit_roles]
    if len(indices) != len(set(indices)):
        return False
    if any(index < 0 or index >= molecule.GetNumAtoms() for index in indices):
        return False
    for role, atomic_number in EXPECTED_ELEMENTS.items():
        if molecule.GetAtomWithIdx(int(roles[role])).GetAtomicNum() != atomic_number:
            return False
    for role in ["C1", "O5", "C2", "C3", "C4", "C5"]:
        if not molecule.GetAtomWithIdx(int(roles[role])).IsInRing():
            return False
    for left, right in RING_EDGES + [("C4", "O4")]:
        if molecule.GetBondBetweenAtoms(int(roles[left]), int(roles[right])) is None:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique-pairs", type=Path, default=Path("data/processed/hard_feasibility_unique_pairs_2632.csv"))
    parser.add_argument("--structural-groups", type=Path, default=Path("data/processed/hard_feasibility_structural_groups_1316.csv"))
    parser.add_argument("--parent-data", type=Path, default=Path("data/processed/glyco_model_local.csv"))
    parser.add_argument(
        "--annotated-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_site_annotated_2632.csv"),
    )
    parser.add_argument(
        "--model-ready-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_site_model_ready_2632.csv"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("results/hard_feasibility_site_annotation_validation_1316.csv"),
    )
    parser.add_argument(
        "--qa-output",
        type=Path,
        default=Path("results/hard_feasibility_site_annotation_qa.csv"),
    )
    args = parser.parse_args()

    unique_pairs = pd.read_csv(args.unique_pairs, encoding="utf-8-sig")
    groups = pd.read_csv(args.structural_groups, encoding="utf-8-sig")
    parents = pd.read_csv(args.parent_data, encoding="utf-8-sig").set_index("ID", drop=False)
    if len(unique_pairs) != 2632 or len(groups) != 1316:
        raise ValueError("expected 2632 unique samples and 1316 structural groups")

    annotated_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for group in groups.itertuples(index=False):
        group_samples = unique_pairs.loc[unique_pairs["Structural_Group_ID"] == group.Structural_Group_ID]
        if len(group_samples) != 2:
            raise ValueError(f"structural group does not contain two samples: {group.Structural_Group_ID}")
        positive = group_samples.loc[group_samples["hard_feasibility"] == 1]
        negative = group_samples.loc[group_samples["hard_feasibility"] == 0]
        if len(positive) != 1 or len(negative) != 1:
            raise ValueError(f"structural group lacks one positive and one negative: {group.Structural_Group_ID}")
        positive = positive.iloc[0]
        negative = negative.iloc[0]

        parent = parents.loc[int(group.Representative_Parent_ID)]
        original_roles = {role: int(parent[column]) for role, column in ROLE_COLUMNS.items()}
        original_roles["O4_EXTERNAL"] = None
        result = generate_o4_acetate(
            parent["Acceptor_Canonical_SMILES"],
            original_roles["O4"],
            original_roles["C4"],
        )
        mapped_roles = {
            role: int(result.original_to_generated_indices[index])
            for role, index in original_roles.items()
            if role != "O4_EXTERNAL"
        }
        mapped_roles["O4_EXTERNAL"] = int(result.generated_o4_external_index)
        original_local_indices = parse_indices(parent["Acceptor_OH_Local_Atom_Indices"])
        mapped_local_indices = map_indices(original_local_indices, result.original_to_generated_indices)
        if mapped_roles["O4_EXTERNAL"] not in mapped_local_indices:
            mapped_local_indices.append(mapped_roles["O4_EXTERNAL"])
        original_extended_roles = extended_role_json(
            parent["Acceptor_OH_Role_Indices"],
            mapping=None,
            o4_external=None,
        )
        mapped_extended_roles = extended_role_json(
            parent["Acceptor_OH_Role_Indices"],
            mapping=result.original_to_generated_indices,
            o4_external=mapped_roles["O4_EXTERNAL"],
        )

        original_molecule = Chem.MolFromSmiles(positive["Acceptor_Canonical_SMILES"])
        blocked_molecule = Chem.MolFromSmiles(negative["Acceptor_Canonical_SMILES"])
        check_original_matches_parent = (
            positive["Acceptor_Canonical_SMILES"] == result.original_canonical_smiles
            and group.Original_Acceptor_Canonical_SMILES == result.original_canonical_smiles
        )
        check_blocked_matches_generator = (
            negative["Acceptor_Canonical_SMILES"] == result.generated_canonical_smiles
            and group.O4_Acetylated_Acceptor_Canonical_SMILES == result.generated_canonical_smiles
        )
        check_original_topology = topology_is_valid(original_molecule, original_roles)
        check_blocked_topology = topology_is_valid(blocked_molecule, mapped_roles)
        original_o4 = original_molecule.GetAtomWithIdx(original_roles["O4"])
        blocked_o4 = blocked_molecule.GetAtomWithIdx(mapped_roles["O4"])
        external = blocked_molecule.GetAtomWithIdx(mapped_roles["O4_EXTERNAL"])
        check_positive_o4_free = original_o4.GetTotalNumHs() >= 1 and original_o4.GetDegree() == 1
        check_negative_o4_blocked = (
            blocked_o4.GetTotalNumHs() == 0
            and blocked_molecule.GetBondBetweenAtoms(mapped_roles["O4"], mapped_roles["O4_EXTERNAL"]) is not None
        )
        check_external_is_acetyl_carbonyl = (
            external.GetAtomicNum() == 6
            and external.GetIdx() == result.generated_o4_external_index
            and any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(external).GetAtomicNum() == 8
                for bond in external.GetBonds()
            )
            and any(
                bond.GetBondType() == Chem.BondType.SINGLE
                and bond.GetOtherAtom(external).GetAtomicNum() == 6
                for bond in external.GetBonds()
            )
        )
        check_role_mapping_complete = all(
            mapped_roles[role] == result.original_to_generated_indices[original_roles[role]]
            for role in ROLE_COLUMNS
        )
        check_target_indices_match_dataset = (
            int(positive["Target_C4_Index"]) == original_roles["C4"]
            and int(positive["Target_O4_Index"]) == original_roles["O4"]
            and int(negative["Target_C4_Index"]) == mapped_roles["C4"]
            and int(negative["Target_O4_Index"]) == mapped_roles["O4"]
        )
        checks = [
            check_original_matches_parent,
            check_blocked_matches_generator,
            check_original_topology,
            check_blocked_topology,
            check_positive_o4_free,
            check_negative_o4_blocked,
            check_external_is_acetyl_carbonyl,
            check_role_mapping_complete,
            check_target_indices_match_dataset,
            result.check_stereochemistry_preserved,
            result.check_only_o4_acetylation,
        ]
        overall_pass = all(checks)
        validation_rows.append(
            {
                "Structural_Group_ID": group.Structural_Group_ID,
                "Representative_Parent_ID": int(group.Representative_Parent_ID),
                "Positive_Sample_ID": positive["Sample_ID"],
                "Negative_Sample_ID": negative["Sample_ID"],
                "Check_1_Original_Matches_Parent": check_original_matches_parent,
                "Check_2_Blocked_Matches_Generator": check_blocked_matches_generator,
                "Check_3_Original_Role_Topology": check_original_topology,
                "Check_4_Blocked_Role_Topology": check_blocked_topology,
                "Check_5_Positive_O4_Free": check_positive_o4_free,
                "Check_6_Negative_O4_Blocked": check_negative_o4_blocked,
                "Check_7_External_Is_Acetyl_Carbonyl": check_external_is_acetyl_carbonyl,
                "Check_8_Role_Mapping_Complete": check_role_mapping_complete,
                "Check_9_Target_Indices_Match_Dataset": check_target_indices_match_dataset,
                "Check_10_Stereochemistry_Preserved": result.check_stereochemistry_preserved,
                "Check_11_Only_O4_Acetylation": result.check_only_o4_acetylation,
                "Overall_Annotation_Pass": overall_pass,
            }
        )

        for sample, roles, local_indices, extended_roles, state, source in [
            (
                positive,
                original_roles,
                original_local_indices,
                original_extended_roles,
                "free_oh",
                "validated_parent_annotation",
            ),
            (
                negative,
                mapped_roles,
                mapped_local_indices,
                mapped_extended_roles,
                "o4_acetate",
                "transferred_by_deterministic_atom_mapping",
            ),
        ]:
            indices = core_indices(roles)
            annotated = sample.to_dict()
            annotated.update(
                {
                    "Acceptor_Site_Annotation_Status": "ok" if overall_pass else "failed",
                    "Acceptor_Site_Annotation_Source": source,
                    "Acceptor_Target_Core_Atom_Indices": ";".join(map(str, indices)),
                    "Acceptor_Target_Core_Num_Atoms": len(indices),
                    "Acceptor_Target_Core_Role_Indices": role_json(roles),
                    "Acceptor_Target_Local_Atom_Indices": join_indices(local_indices),
                    "Acceptor_Target_Local_Num_Atoms": len(local_indices),
                    "Acceptor_Target_Role_Indices": extended_roles,
                    "Acceptor_C1_Index": roles["C1"],
                    "Acceptor_O5_Index": roles["O5"],
                    "Acceptor_C2_Index": roles["C2"],
                    "Acceptor_C3_Index": roles["C3"],
                    "Acceptor_C4_Index": roles["C4"],
                    "Acceptor_C5_Index": roles["C5"],
                    "Acceptor_O4_Index": roles["O4"],
                    "Acceptor_O4_External_Atom_Index": -1 if roles["O4_EXTERNAL"] is None else roles["O4_EXTERNAL"],
                    "Acceptor_O4_External_Atom_Role": "implicit_H" if roles["O4_EXTERNAL"] is None else "acetyl_carbonyl_C",
                    "Acceptor_Target_O4_State": state,
                    "Donor_RFU_Atom_Indices": parent["Donor_RFU_Atom_Indices"],
                    "Donor_RFU_Role_Indices": parent["Donor_RFU_Role_Indices"],
                    "Donor_C1_Index": int(parent["Donor_C1_Index"]),
                    "Donor_O5_Index": int(parent["Donor_O5_Index"]),
                    "Donor_C2_Index": int(parent["Donor_C2_Index"]),
                    "Donor_LG_Entry_Index": int(parent["Donor_LG_Entry_Index"]),
                    "Donor_LG_Core_Indices": parent["Donor_LG_Core_Indices"],
                    "Donor_Ring_Context_Indices": parent["Donor_Ring_Context_Indices"],
                    "Donor_C2_Sub_Entry_Index": int(parent["Donor_C2_Sub_Entry_Index"]),
                    "Donor_C2_Sub_Entry_Status": parent["Donor_C2_Sub_Entry_Status"],
                }
            )
            annotated_rows.append(annotated)

    annotated = pd.DataFrame(annotated_rows).sort_values(["Structural_Group_ID", "hard_feasibility"], ascending=[True, False]).reset_index(drop=True)
    validation = pd.DataFrame(validation_rows).sort_values("Structural_Group_ID").reset_index(drop=True)

    site_model_columns = [
        "Sample_ID",
        "Pair_ID",
        "Structural_Group_ID",
        "hard_feasibility",
        "Donor_Canonical_SMILES",
        "Acceptor_Canonical_SMILES",
        "Donor_Type",
        "Donor_RFU_Atom_Indices",
        "Donor_RFU_Role_Indices",
        "Donor_C1_Index",
        "Donor_O5_Index",
        "Donor_C2_Index",
        "Donor_LG_Entry_Index",
        "Donor_LG_Core_Indices",
        "Donor_Ring_Context_Indices",
        "Donor_C2_Sub_Entry_Index",
        "Donor_C2_Sub_Entry_Status",
        "Acceptor_Target_Core_Atom_Indices",
        "Acceptor_Target_Core_Num_Atoms",
        "Acceptor_Target_Core_Role_Indices",
        "Acceptor_Target_Local_Atom_Indices",
        "Acceptor_Target_Local_Num_Atoms",
        "Acceptor_Target_Role_Indices",
        "Acceptor_C1_Index",
        "Acceptor_O5_Index",
        "Acceptor_C2_Index",
        "Acceptor_C3_Index",
        "Acceptor_C4_Index",
        "Acceptor_C5_Index",
        "Acceptor_O4_Index",
        "Acceptor_O4_External_Atom_Index",
    ]
    model_ready = annotated[site_model_columns].copy()

    qa_records: list[dict[str, object]] = []

    def add_qa(name: str, expected: int, observed: int, details: str) -> None:
        qa_records.append({"QA_Check": name, "Expected": expected, "Observed": observed, "Pass": expected == observed, "Details": details})

    add_qa("Structural groups", 1316, len(validation), "one validation row per pair")
    add_qa("Annotated samples", 2632, len(annotated), "1316 positive + 1316 negative")
    for column in [column for column in validation.columns if column.startswith("Check_")]:
        add_qa(column, 1316, int(validation[column].sum()), "all structural groups")
    add_qa("Overall annotation pass", 1316, int(validation["Overall_Annotation_Pass"].sum()), "all 11 pair checks")
    add_qa("Annotation status ok", 2632, int((annotated["Acceptor_Site_Annotation_Status"] == "ok").sum()), "all samples")
    add_qa("Unique Sample_ID", 2632, annotated["Sample_ID"].nunique(), "no row duplication")
    input_unique = annotated[["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"]].drop_duplicates()
    add_qa("Unique donor-acceptor model inputs", 2632, len(input_unique), "no repeated or cross-label input")
    add_qa("Positive external index sentinel", 1316, int(((annotated["hard_feasibility"] == 1) & (annotated["Acceptor_O4_External_Atom_Index"] == -1)).sum()), "implicit hydrogen has no explicit atom")
    add_qa("Negative explicit external atom", 1316, int(((annotated["hard_feasibility"] == 0) & (annotated["Acceptor_O4_External_Atom_Index"] >= 0)).sum()), "acetyl carbonyl carbon")
    add_qa("Condition columns in annotated output", 0, len(CONDITION_COLUMNS.intersection(annotated.columns)), "structure-only task")
    direct_status_columns = {
        "Variant",
        "Original_Acceptor_Canonical_SMILES",
        "Acceptor_Target_O4_State",
        "Acceptor_O4_External_Atom_Role",
        "Acceptor_Site_Annotation_Status",
        "Acceptor_Site_Annotation_Source",
    }
    add_qa("Direct status/leakage columns in site model-ready output", 0, len(direct_status_columns.intersection(model_ready.columns)), "audit-only metadata removed")
    add_qa("Site model-ready rows", 2632, len(model_ready), "final annotated model table")
    add_qa("Site model-ready columns", len(site_model_columns), len(model_ready.columns), "fixed schema")

    qa = pd.DataFrame(qa_records)
    if not validation["Overall_Annotation_Pass"].all() or not qa["Pass"].all():
        raise ValueError("site annotation QA failed")

    write_csv(annotated, args.annotated_output)
    write_csv(model_ready, args.model_ready_output)
    write_csv(validation, args.validation_output)
    write_csv(qa, args.qa_output)
    print(
        {
            "structural_groups": len(validation),
            "annotated_samples": len(annotated),
            "model_ready_rows": len(model_ready),
            "validation_checks_per_group": len([column for column in validation.columns if column.startswith("Check_")]),
            "all_qa_pass": bool(qa["Pass"].all()),
        }
    )


if __name__ == "__main__":
    main()
