#!/usr/bin/env python3
"""第一层训练准备：生成无泄漏划分、手性图缓存和捷径压力测试。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer3.chiral_graph import CHIRAL_GRAPH_FEATURE_VERSION, mol_to_chiral_pyg_graph
from layer3.glyco_dataset import (
    ATOM_FEATURE_NAMES,
    BOND_FEATURE_NAMES,
    DONOR_ROLE_NAMES,
    atom_features,
    bond_features,
    make_role_matrix,
    mol_from_smiles,
)


ACCEPTOR_SITE_ROLE_NAMES = [
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
STRESS_FRAGMENTS = {
    "o4_methyl": ("[*]C", 0),
    "o4_benzyl": ("[*]Cc1ccccc1", 0),
    "o4_trimethylsilyl": ("[*][Si](C)(C)C", 0),
    "non_target_oh_acetyl": ("[*]C(=O)C", 1),
}


def parse_indices(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(float(part)) for part in text.split(";") if part.strip()]


def join_indices(values: list[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def stable_hash(*values: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, values)).encode("utf-8")).hexdigest()


def build_acceptor_group_split(data: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_frame = groups[["Structural_Group_ID", "Original_Acceptor_Canonical_SMILES"]].copy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_index, temp_index = next(
        first.split(
            group_frame,
            groups=group_frame["Original_Acceptor_Canonical_SMILES"],
        )
    )
    group_frame["split_acceptor_group"] = "train"
    temp = group_frame.iloc[temp_index]
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
    val_relative, test_relative = next(
        second.split(
            temp,
            groups=temp["Original_Acceptor_Canonical_SMILES"],
        )
    )
    group_frame.loc[temp.index[val_relative], "split_acceptor_group"] = "val"
    group_frame.loc[temp.index[test_relative], "split_acceptor_group"] = "test"

    split_map = group_frame.set_index("Structural_Group_ID")["split_acceptor_group"]
    out = data.copy()
    out["split_acceptor_group"] = out["Structural_Group_ID"].map(split_map)
    if out["split_acceptor_group"].isna().any():
        raise ValueError("one or more structural groups were not assigned a split")

    acceptors_by_split = {
        split: set(group_frame.loc[group_frame["split_acceptor_group"] == split, "Original_Acceptor_Canonical_SMILES"])
        for split in ["train", "val", "test"]
    }
    group_ids_by_split = {
        split: set(out.loc[out["split_acceptor_group"] == split, "Structural_Group_ID"])
        for split in ["train", "val", "test"]
    }
    input_keys_by_split = {
        split: set(
            zip(
                out.loc[out["split_acceptor_group"] == split, "Donor_Canonical_SMILES"],
                out.loc[out["split_acceptor_group"] == split, "Acceptor_Canonical_SMILES"],
            )
        )
        for split in ["train", "val", "test"]
    }

    overlap: dict[str, dict[str, int]] = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap[f"{left}-{right}"] = {
            "acceptor_overlap": len(acceptors_by_split[left] & acceptors_by_split[right]),
            "pair_overlap": len(group_ids_by_split[left] & group_ids_by_split[right]),
            "exact_input_overlap": len(input_keys_by_split[left] & input_keys_by_split[right]),
        }
    if any(value for pair in overlap.values() for value in pair.values()):
        raise ValueError(f"split leakage detected: {overlap}")

    summary = {
        "random_state": [42, 43],
        "sample_counts": out["split_acceptor_group"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "structural_group_counts": group_frame["split_acceptor_group"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "original_acceptor_counts": {
            split: len(acceptors_by_split[split]) for split in ["train", "val", "test"]
        },
        "label_counts": {
            split: out.loc[out["split_acceptor_group"] == split, "hard_feasibility"].value_counts().sort_index().astype(int).to_dict()
            for split in ["train", "val", "test"]
        },
        "donor_type_counts": {
            split: out.loc[out["split_acceptor_group"] == split, "Donor_Type"].value_counts().sort_index().astype(int).to_dict()
            for split in ["train", "val", "test"]
        },
        "sample_proportions": {
            split: float((out["split_acceptor_group"] == split).mean())
            for split in ["train", "val", "test"]
        },
        "overlap": overlap,
    }
    return out, summary


def attach_fragment(
    original_smiles: str,
    oxygen_index: int,
    fragment_smiles: str,
) -> dict[str, Any]:
    original = Chem.MolFromSmiles(original_smiles)
    if original is None:
        raise ValueError("original stress-test acceptor cannot be parsed")
    Chem.AssignStereochemistry(original, force=True, cleanIt=True)
    oxygen = original.GetAtomWithIdx(int(oxygen_index))
    if oxygen.GetAtomicNum() != 8 or oxygen.GetTotalNumHs() < 1:
        raise ValueError("stress-test attachment oxygen is not a free hydroxyl")
    original_chiral_tags = [atom.GetChiralTag() for atom in original.GetAtoms()]

    fragment = Chem.MolFromSmiles(fragment_smiles)
    dummy_atoms = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        raise ValueError("stress fragment must contain exactly one dummy attachment atom")
    dummy_index_fragment = dummy_atoms[0]
    dummy = fragment.GetAtomWithIdx(dummy_index_fragment)
    if dummy.GetDegree() != 1:
        raise ValueError("stress fragment dummy atom must have one neighbor")
    attachment_index_fragment = dummy.GetNeighbors()[0].GetIdx()

    original_atom_count = original.GetNumAtoms()
    combined = Chem.CombineMols(original, fragment)
    editable = Chem.RWMol(combined)
    dummy_index = original_atom_count + dummy_index_fragment
    attachment_index = original_atom_count + attachment_index_fragment
    editable.AddBond(int(oxygen_index), attachment_index, Chem.BondType.SINGLE)
    editable.RemoveAtom(dummy_index)
    if attachment_index > dummy_index:
        attachment_index -= 1
    modified = editable.GetMol()
    Chem.SanitizeMol(modified)
    Chem.AssignStereochemistry(modified, force=True, cleanIt=True)
    if [modified.GetAtomWithIdx(i).GetChiralTag() for i in range(original_atom_count)] != original_chiral_tags:
        raise ValueError("stress-test modification changed parent chiral tags")

    canonical = Chem.MolToSmiles(modified, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(canonical)
    mapping = reparsed.GetSubstructMatch(modified, useChirality=True)
    if len(mapping) != modified.GetNumAtoms():
        raise ValueError("stress-test canonical atom mapping failed")
    return {
        "canonical_smiles": canonical,
        "original_to_modified": tuple(int(mapping[i]) for i in range(original_atom_count)),
        "attachment_atom_index": int(mapping[attachment_index]),
    }


def mapped_role_json(
    role_json: str,
    mapping: tuple[int, ...],
    o4_external_index: int | None,
) -> str:
    roles = json.loads(role_json)
    mapped: dict[str, list[int]] = {}
    for role in ACCEPTOR_SITE_ROLE_NAMES:
        if role == "O4_EXTERNAL":
            mapped[role] = [] if o4_external_index is None else [int(o4_external_index)]
        else:
            mapped[role] = [int(mapping[int(index)]) for index in roles.get(role, [])]
    return json.dumps(mapped, ensure_ascii=False, separators=(",", ":"))


def build_stress_tests(pre_model: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    test_positive = pre_model.loc[
        (pre_model["split_acceptor_group"] == "test")
        & (pre_model["hard_feasibility"] == 1)
    ].copy()
    records: list[dict[str, Any]] = []
    for row in test_positive.itertuples(index=False):
        original = Chem.MolFromSmiles(row.Acceptor_Canonical_SMILES)
        original_roles = json.loads(row.Acceptor_Target_Role_Indices)
        target_o4 = int(original_roles["O4"][0])
        original_local = parse_indices(row.Acceptor_Target_Local_Atom_Indices)

        for stress_type in ["o4_methyl", "o4_benzyl", "o4_trimethylsilyl"]:
            fragment, label = STRESS_FRAGMENTS[stress_type]
            modified = attach_fragment(row.Acceptor_Canonical_SMILES, target_o4, fragment)
            mapping = modified["original_to_modified"]
            mapped_local = [int(mapping[index]) for index in original_local]
            if modified["attachment_atom_index"] not in mapped_local:
                mapped_local.append(modified["attachment_atom_index"])
            roles_json = mapped_role_json(
                row.Acceptor_Target_Role_Indices,
                mapping,
                modified["attachment_atom_index"],
            )
            roles = json.loads(roles_json)
            records.append(
                {
                    "Sample_ID": f"STRESS-{stress_type}-{row.Structural_Group_ID}",
                    "Parent_Structural_Group_ID": row.Structural_Group_ID,
                    "Stress_Test_Type": stress_type,
                    "hard_feasibility": label,
                    "Donor_Canonical_SMILES": row.Donor_Canonical_SMILES,
                    "Acceptor_Canonical_SMILES": modified["canonical_smiles"],
                    "Donor_Type": row.Donor_Type,
                    "Donor_RFU_Atom_Indices": row.Donor_RFU_Atom_Indices,
                    "Donor_RFU_Role_Indices": row.Donor_RFU_Role_Indices,
                    "Acceptor_Target_Local_Atom_Indices": join_indices(mapped_local),
                    "Acceptor_Target_Role_Indices": roles_json,
                    "Target_C4_Index": int(roles["C4"][0]),
                    "Target_O4_Index": int(roles["O4"][0]),
                    "Target_O4_External_Atom_Index": int(modified["attachment_atom_index"]),
                    "Modification_Site_Index": int(mapping[target_o4]),
                    "Modification_Attachment_Atom_Index": int(modified["attachment_atom_index"]),
                    "split_acceptor_group": "test",
                }
            )

        non_target_oh = [
            atom.GetIdx()
            for atom in original.GetAtoms()
            if atom.GetAtomicNum() == 8
            and atom.GetTotalNumHs() >= 1
            and atom.GetDegree() == 1
            and atom.GetIdx() != target_o4
        ]
        if non_target_oh:
            modification_index = min(non_target_oh)
            fragment, label = STRESS_FRAGMENTS["non_target_oh_acetyl"]
            modified = attach_fragment(row.Acceptor_Canonical_SMILES, modification_index, fragment)
            mapping = modified["original_to_modified"]
            mapped_local = [int(mapping[index]) for index in original_local]
            roles_json = mapped_role_json(
                row.Acceptor_Target_Role_Indices,
                mapping,
                None,
            )
            roles = json.loads(roles_json)
            records.append(
                {
                    "Sample_ID": f"STRESS-non-target-ac-{row.Structural_Group_ID}",
                    "Parent_Structural_Group_ID": row.Structural_Group_ID,
                    "Stress_Test_Type": "non_target_oh_acetyl",
                    "hard_feasibility": label,
                    "Donor_Canonical_SMILES": row.Donor_Canonical_SMILES,
                    "Acceptor_Canonical_SMILES": modified["canonical_smiles"],
                    "Donor_Type": row.Donor_Type,
                    "Donor_RFU_Atom_Indices": row.Donor_RFU_Atom_Indices,
                    "Donor_RFU_Role_Indices": row.Donor_RFU_Role_Indices,
                    "Acceptor_Target_Local_Atom_Indices": join_indices(mapped_local),
                    "Acceptor_Target_Role_Indices": roles_json,
                    "Target_C4_Index": int(roles["C4"][0]),
                    "Target_O4_Index": int(roles["O4"][0]),
                    "Target_O4_External_Atom_Index": -1,
                    "Modification_Site_Index": int(mapping[modification_index]),
                    "Modification_Attachment_Atom_Index": int(modified["attachment_atom_index"]),
                    "split_acceptor_group": "test",
                }
            )

    stress = pd.DataFrame(records).sort_values(["Stress_Test_Type", "Parent_Structural_Group_ID"]).reset_index(drop=True)
    if stress["Sample_ID"].duplicated().any():
        raise ValueError("duplicate stress-test Sample_ID")
    if stress[["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"]].duplicated().any():
        raise ValueError("duplicate stress-test model input")

    base_input_keys = set(
        zip(pre_model["Donor_Canonical_SMILES"], pre_model["Acceptor_Canonical_SMILES"])
    )
    overlaps_base = stress.apply(
        lambda row: (row["Donor_Canonical_SMILES"], row["Acceptor_Canonical_SMILES"])
        in base_input_keys,
        axis=1,
    )
    excluded_existing_input_overlap = int(overlaps_base.sum())
    stress = stress.loc[~overlaps_base].reset_index(drop=True)
    remaining_stress_keys = set(
        zip(stress["Donor_Canonical_SMILES"], stress["Acceptor_Canonical_SMILES"])
    )
    if remaining_stress_keys & base_input_keys:
        raise ValueError("stress-test input still overlaps the main dataset")

    for row in stress.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.Acceptor_Canonical_SMILES)
        roles = json.loads(row.Acceptor_Target_Role_Indices)
        o4 = molecule.GetAtomWithIdx(int(row.Target_O4_Index))
        c4 = molecule.GetAtomWithIdx(int(row.Target_C4_Index))
        if molecule.GetBondBetweenAtoms(o4.GetIdx(), c4.GetIdx()) is None:
            raise ValueError("stress-test C4-O4 bond missing")
        modification_site = molecule.GetAtomWithIdx(int(row.Modification_Site_Index))
        modification_attachment = molecule.GetAtomWithIdx(int(row.Modification_Attachment_Atom_Index))
        if molecule.GetBondBetweenAtoms(modification_site.GetIdx(), modification_attachment.GetIdx()) is None:
            raise ValueError("stress-test modification bond missing")
        if row.Stress_Test_Type == "non_target_oh_acetyl":
            if o4.GetTotalNumHs() < 1 or roles["O4_EXTERNAL"]:
                raise ValueError("non-target acetylation control closed target O4")
            if modification_site.GetIdx() == o4.GetIdx() or modification_site.GetAtomicNum() != 8:
                raise ValueError("non-target acetylation modified the target O4")
            if modification_attachment.GetAtomicNum() != 6 or not any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(modification_attachment).GetAtomicNum() == 8
                for bond in modification_attachment.GetBonds()
            ):
                raise ValueError("non-target acetylation did not add a carbonyl carbon")
        else:
            if o4.GetTotalNumHs() != 0 or not roles["O4_EXTERNAL"]:
                raise ValueError("O4 protection stress test left target O4 open")
            if modification_site.GetIdx() != o4.GetIdx():
                raise ValueError("O4 protection stress test modified a non-target atom")
            if row.Stress_Test_Type == "o4_methyl":
                if modification_attachment.GetAtomicNum() != 6 or modification_attachment.GetDegree() != 1:
                    raise ValueError("O4-Me stress structure is not a methyl ether")
            elif row.Stress_Test_Type == "o4_benzyl":
                carbon_neighbors = [
                    atom
                    for atom in modification_attachment.GetNeighbors()
                    if atom.GetIdx() != o4.GetIdx() and atom.GetAtomicNum() == 6
                ]
                if modification_attachment.GetAtomicNum() != 6 or not any(atom.GetIsAromatic() for atom in carbon_neighbors):
                    raise ValueError("O4-Bn stress structure is not a benzyl ether")
            elif row.Stress_Test_Type == "o4_trimethylsilyl":
                if modification_attachment.GetAtomicNum() != 14:
                    raise ValueError("O4-Si stress structure is not a silyl ether")

    summary = {
        "test_parent_groups": int(test_positive["Structural_Group_ID"].nunique()),
        "counts_by_type": stress["Stress_Test_Type"].value_counts().sort_index().astype(int).to_dict(),
        "label_counts": stress["hard_feasibility"].value_counts().sort_index().astype(int).to_dict(),
        "unique_inputs": int(len(stress)),
        "exact_input_overlap_with_main_dataset": 0,
        "excluded_existing_input_overlap": excluded_existing_input_overlap,
        "all_structures_valid": True,
    }
    return stress, summary


def validate_chiral_graph(graph: Any) -> None:
    required = {
        "x",
        "edge_index",
        "edge_attr",
        "heavy_atom_mask",
        "parity_atoms",
        "tetra_center_index",
        "tetra_neighbor_index",
        "tetra_edge_index",
        "num_heavy_atoms",
    }
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise ValueError(f"chiral graph missing fields: {missing}")
    if graph.x.shape[0] != graph.num_nodes or graph.heavy_atom_mask.shape[0] != graph.num_nodes:
        raise ValueError("chiral graph node tensor shape mismatch")
    if graph.tetra_center_index.shape[0] != graph.tetra_neighbor_index.shape[0]:
        raise ValueError("tetrahedral center/neighbor count mismatch")
    if graph.tetra_center_index.shape[0] != graph.tetra_edge_index.shape[0]:
        raise ValueError("tetrahedral center/edge count mismatch")
    if graph.tetra_neighbor_index.numel() and int(graph.tetra_neighbor_index.max()) >= graph.num_nodes:
        raise ValueError("tetrahedral neighbor index out of bounds")
    if graph.tetra_edge_index.numel() and int(graph.tetra_edge_index.max()) >= graph.num_edges:
        raise ValueError("tetrahedral edge index out of bounds")


def build_graph_cache(pre_model: pd.DataFrame, stress: pd.DataFrame, output: Path) -> dict[str, Any]:
    smiles = sorted(
        set(pre_model["Donor_Canonical_SMILES"])
        | set(pre_model["Acceptor_Canonical_SMILES"])
        | set(stress["Donor_Canonical_SMILES"])
        | set(stress["Acceptor_Canonical_SMILES"])
    )
    graphs: dict[str, Any] = {}
    for molecule_smiles in smiles:
        molecule = mol_from_smiles(molecule_smiles)
        graph = mol_to_chiral_pyg_graph(
            molecule,
            molecule_smiles,
            atom_feature_fn=atom_features,
            bond_feature_fn=bond_features,
        )
        validate_chiral_graph(graph)
        graphs[molecule_smiles] = graph

    samples: dict[str, dict[str, Any]] = {}
    for row in pre_model.itertuples(index=False):
        donor_local = parse_indices(row.Donor_RFU_Atom_Indices)
        donor_roles = json.loads(row.Donor_RFU_Role_Indices)
        acceptor_local = parse_indices(row.Acceptor_Target_Local_Atom_Indices)
        acceptor_roles = json.loads(row.Acceptor_Target_Role_Indices)
        donor_graph = graphs[row.Donor_Canonical_SMILES]
        acceptor_graph = graphs[row.Acceptor_Canonical_SMILES]
        if any(index < 0 or index >= donor_graph.num_heavy_atoms for index in donor_local):
            raise ValueError(f"donor RFU index out of bounds: {row.Sample_ID}")
        if any(index < 0 or index >= acceptor_graph.num_heavy_atoms for index in acceptor_local):
            raise ValueError(f"acceptor site index out of bounds: {row.Sample_ID}")
        if int(row.Acceptor_C4_Index) not in acceptor_local or int(row.Acceptor_O4_Index) not in acceptor_local:
            raise ValueError(f"C4/O4 missing from acceptor local site: {row.Sample_ID}")
        samples[row.Sample_ID] = {
            "pair_id": row.Pair_ID,
            "structural_group_id": row.Structural_Group_ID,
            "split": row.split_acceptor_group,
            "label": int(row.hard_feasibility),
            "donor_smiles": row.Donor_Canonical_SMILES,
            "acceptor_smiles": row.Acceptor_Canonical_SMILES,
            "donor_rfu_atom_indices": torch.tensor(donor_local, dtype=torch.long),
            "donor_rfu_role_matrix": make_role_matrix(donor_local, donor_roles, DONOR_ROLE_NAMES),
            "acceptor_site_atom_indices": torch.tensor(acceptor_local, dtype=torch.long),
            "acceptor_site_role_matrix": make_role_matrix(
                acceptor_local,
                acceptor_roles,
                ACCEPTOR_SITE_ROLE_NAMES,
            ),
            "acceptor_c4_index": int(row.Acceptor_C4_Index),
            "acceptor_o4_index": int(row.Acceptor_O4_Index),
        }
    if len(samples) != len(pre_model):
        raise ValueError("graph-cache sample metadata is incomplete")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_version": CHIRAL_GRAPH_FEATURE_VERSION,
        "graph_type": "chiral_gine",
        "atom_feature_names": ATOM_FEATURE_NAMES,
        "bond_feature_names": BOND_FEATURE_NAMES,
        "donor_role_names": DONOR_ROLE_NAMES,
        "acceptor_site_role_names": ACCEPTOR_SITE_ROLE_NAMES,
        "n_unique_smiles": len(graphs),
        "n_samples": len(samples),
        "graphs": graphs,
        "samples": samples,
    }
    torch.save(payload, output)
    return {
        "unique_graphs": len(graphs),
        "sample_metadata_rows": len(samples),
        "atom_feature_dim": len(ATOM_FEATURE_NAMES),
        "bond_feature_dim": len(BOND_FEATURE_NAMES),
        "donor_role_dim": len(DONOR_ROLE_NAMES),
        "acceptor_site_role_dim": len(ACCEPTOR_SITE_ROLE_NAMES),
        "all_graphs_valid": True,
    }


def paired_shortcut_balance(pre_model: pd.DataFrame) -> dict[str, Any]:
    positive = pre_model.loc[pre_model["hard_feasibility"] == 1]
    negative = pre_model.loc[pre_model["hard_feasibility"] == 0]
    donor_positive = positive["Donor_Canonical_SMILES"].value_counts().sort_index()
    donor_negative = negative["Donor_Canonical_SMILES"].value_counts().sort_index()
    donor_delta = donor_positive.subtract(donor_negative, fill_value=0)
    donor_type_positive = positive["Donor_Type"].value_counts().sort_index()
    donor_type_negative = negative["Donor_Type"].value_counts().sort_index()
    donor_type_delta = donor_type_positive.subtract(donor_type_negative, fill_value=0)
    condition_columns_present = sorted(CONDITION_COLUMNS.intersection(pre_model.columns))
    if donor_delta.abs().sum() != 0 or donor_type_delta.abs().sum() != 0:
        raise ValueError("paired donor distribution is not identical across labels")
    if condition_columns_present:
        raise ValueError(f"condition columns unexpectedly present: {condition_columns_present}")
    return {
        "positive_count": len(positive),
        "negative_count": len(negative),
        "donor_distribution_total_absolute_delta": int(donor_delta.abs().sum()),
        "donor_type_distribution_total_absolute_delta": int(donor_type_delta.abs().sum()),
        "condition_columns_present": condition_columns_present,
        "conditions_not_used_by_design": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-data",
        type=Path,
        default=Path("data/processed/hard_feasibility_site_model_ready_2632.csv"),
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("data/processed/hard_feasibility_structural_groups_1316.csv"),
    )
    parser.add_argument(
        "--pre-model-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_pre_model.csv"),
    )
    parser.add_argument(
        "--graph-cache-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_chiral_graph_cache.pt"),
    )
    parser.add_argument(
        "--stress-output",
        type=Path,
        default=Path("data/processed/hard_feasibility_stress_test.csv"),
    )
    parser.add_argument(
        "--qa-output",
        type=Path,
        default=Path("results/hard_feasibility_pre_model_qa.json"),
    )
    args = parser.parse_args()

    site_data = pd.read_csv(args.site_data, encoding="utf-8-sig")
    groups = pd.read_csv(args.groups, encoding="utf-8-sig")
    if len(site_data) != 2632 or len(groups) != 1316:
        raise ValueError("unexpected site-data or structural-group count")

    pre_model, split_summary = build_acceptor_group_split(site_data, groups)
    stress, stress_summary = build_stress_tests(pre_model)
    shortcut_summary = paired_shortcut_balance(pre_model)

    args.pre_model_output.parent.mkdir(parents=True, exist_ok=True)
    pre_model.to_csv(args.pre_model_output, index=False, encoding="utf-8-sig")
    stress.to_csv(args.stress_output, index=False, encoding="utf-8-sig")
    graph_summary = build_graph_cache(pre_model, stress, args.graph_cache_output)

    qa = {
        "schema_version": "hard_feasibility_pre_model_v1",
        "overall_pass": True,
        "input": {
            "samples": len(pre_model),
            "structural_groups": int(pre_model["Structural_Group_ID"].nunique()),
            "unique_model_inputs": int(len(pre_model[["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"]].drop_duplicates())),
        },
        "split_acceptor_group": split_summary,
        "graph_cache": graph_summary,
        "shortcut_balance": shortcut_summary,
        "stress_tests": stress_summary,
        "outputs": {
            "pre_model_csv": str(args.pre_model_output),
            "graph_cache_pt": str(args.graph_cache_output),
            "stress_test_csv": str(args.stress_output),
        },
    }
    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    args.qa_output.write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
