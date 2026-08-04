#!/usr/bin/env python
"""第一层推理入口：批量判断目标 O4 是否满足硬结构可行性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from layer3.chiral_graph import mol_to_chiral_pyg_graph
from layer3.extract_acceptor_4oh import (
    build_local_result,
    find_c2_n_sub_entries,
    find_sugar_rings,
    traverse_ring_from_o5_c1,
)
from layer3.extract_donor_rfu import PATTERNS, extract_donor_rfu
from layer3.glyco_dataset import atom_features, bond_features
from layer1.train_hard_feasibility import (
    ACCEPTOR_ROLE_NAMES,
    DONOR_ROLE_NAMES,
    StructureOnlyChiralGINE,
    collate,
    make_role_matrix,
    parse_int_list,
    parse_role_dict,
    safe_torch_load,
)


SUPPORTED_DONOR_TYPES = tuple(PATTERNS)


def canonicalize(smiles: object) -> tuple[str | None, Chem.Mol | None, tuple[int, ...] | None]:
    if smiles is None or pd.isna(smiles) or not str(smiles).strip():
        return None, None, None
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return None, None, None
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(canonical)
    Chem.AssignStereochemistry(reparsed, force=True, cleanIt=True)
    mapping = reparsed.GetSubstructMatch(molecule, useChirality=True)
    if len(mapping) != molecule.GetNumAtoms():
        return None, None, None
    return canonical, reparsed, tuple(int(mapping[index]) for index in range(len(mapping)))


def resolve_donor_type(donor_smiles: str, supplied: object = None) -> tuple[str | None, str]:
    if supplied is not None and not pd.isna(supplied) and str(supplied).strip():
        donor_type = str(supplied).strip().lower()
        if donor_type not in SUPPORTED_DONOR_TYPES:
            return None, f"unsupported Donor_Type={donor_type!r}"
        result = extract_donor_rfu(donor_smiles, donor_type)
        if result["Donor_RFU_Status"] != "ok":
            return None, f"donor RFU extraction failed for supplied type: {result['Donor_RFU_Status']}"
        return donor_type, "supplied"

    matches = []
    for donor_type in SUPPORTED_DONOR_TYPES:
        result = extract_donor_rfu(donor_smiles, donor_type)
        if result["Donor_RFU_Status"] == "ok":
            matches.append(donor_type)
    if len(matches) == 1:
        return matches[0], "inferred"
    if not matches:
        return None, "cannot infer a supported donor type"
    return None, f"ambiguous donor type: {matches}"


def enumerate_target_site_candidates(molecule: Chem.Mol) -> list[dict[str, Any]]:
    """Locate the glucosamine C4 oxygen whether it is free or substituted."""

    candidates: list[dict[str, Any]] = []
    for ring in find_sugar_rings(molecule):
        ring_set = set(ring)
        o5_atoms = [idx for idx in ring if molecule.GetAtomWithIdx(idx).GetAtomicNum() == 8]
        if len(o5_atoms) != 1:
            continue
        o5 = o5_atoms[0]
        c1_options = [
            atom.GetIdx()
            for atom in molecule.GetAtomWithIdx(o5).GetNeighbors()
            if atom.GetIdx() in ring_set and atom.GetAtomicNum() == 6
        ]
        for c1 in c1_options:
            roles = traverse_ring_from_o5_c1(molecule, ring, o5, c1)
            if roles is None:
                continue
            n2_entries = find_c2_n_sub_entries(molecule, roles["C2"], ring_set)
            o4_entries = []
            for atom in molecule.GetAtomWithIdx(roles["C4"]).GetNeighbors():
                if atom.GetIdx() in ring_set or atom.GetAtomicNum() != 8:
                    continue
                bond = molecule.GetBondBetweenAtoms(roles["C4"], atom.GetIdx())
                if atom.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE:
                    o4_entries.append(atom.GetIdx())
            if n2_entries and len(o4_entries) == 1:
                candidates.append({
                    **roles,
                    "O4": o4_entries[0],
                    "N2_SUB_ENTRY": n2_entries,
                    "ring": list(ring),
                })

    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = tuple(int(candidate[name]) for name in ["C1", "C2", "C3", "C4", "C5", "O5", "O4"])
        unique[key] = candidate
    return list(unique.values())


def extract_acceptor_target_site(
    acceptor_smiles: str,
    target_o4_index: int | None = None,
) -> dict[str, Any]:
    """定位目标 O4。

    未显式指定原子编号时，优先选择唯一的游离 O4，而不是要求受体
    只能含有一个氨基糖 C4-O 骨架。这使多糖环受体中的已保护 O4
    不会阻碍对唯一游离 O4 的自动定位。如果用户显式指定了位点，
    则保留对封闭 O4 的识别，以便第一层将其判为结构不可行。
    """

    molecule = Chem.MolFromSmiles(acceptor_smiles)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    candidates = enumerate_target_site_candidates(molecule)

    def describe(candidate: dict[str, Any]) -> dict[str, Any]:
        o4 = int(candidate["O4"])
        c4 = int(candidate["C4"])
        o4_atom = molecule.GetAtomWithIdx(o4)
        external = [
            atom.GetIdx()
            for atom in o4_atom.GetNeighbors()
            if atom.GetIdx() != c4
        ]
        if not external and o4_atom.GetTotalNumHs() >= 1:
            state = "free"
            external_index = None
        elif len(external) == 1 and o4_atom.GetTotalNumHs() == 0:
            state = "blocked"
            external_index = int(external[0])
        else:
            state = "invalid"
            external_index = None
        return {
            "candidate": candidate,
            "state": state,
            "external_index": external_index,
        }

    described = [describe(candidate) for candidate in candidates]
    counts = {
        "candidate_count": len(described),
        "free_candidate_count": sum(item["state"] == "free" for item in described),
        "blocked_candidate_count": sum(item["state"] == "blocked" for item in described),
    }

    if target_o4_index is not None:
        selected = [
            item
            for item in described
            if int(item["candidate"]["O4"]) == target_o4_index
        ]
        if len(selected) != 1:
            return {"status": "supplied_target_o4_not_found", **counts}
        selection_source = "supplied_target"
    else:
        free_candidates = [item for item in described if item["state"] == "free"]
        if len(free_candidates) == 1:
            selected = free_candidates
            selection_source = "auto_unique_free"
        elif len(free_candidates) > 1:
            return {"status": "multiple_free_target_sites", **counts}
        elif len(described) == 1:
            # 单一但已封闭的 O4 仍需交给第一层判断。
            selected = described
            selection_source = "auto_unique_structural"
        elif not described:
            return {"status": "no_target_site", **counts}
        else:
            # 存在多个结构 O4，但没有任何游离位点。
            return {"status": "no_free_target_site", **counts}

    selected_item = selected[0]
    candidate = selected_item["candidate"]
    base = build_local_result(molecule, candidate, "inference_rule_unique")
    if base["Acceptor_OH_Status"] != "ok":
        return {"status": str(base["Acceptor_OH_Status"]), **counts}

    o4 = int(candidate["O4"])
    state = str(selected_item["state"])
    external_index = selected_item["external_index"]
    if state == "invalid":
        return {"status": "invalid_o4_valence", **counts}

    local_indices = parse_int_list(base["Acceptor_OH_Local_Atom_Indices"])
    if external_index is not None and external_index not in local_indices:
        local_indices.append(external_index)
    roles = parse_role_dict(base["Acceptor_OH_Role_Indices"])
    roles["O4_EXTERNAL"] = [] if external_index is None else [external_index]
    for role in ACCEPTOR_ROLE_NAMES:
        roles.setdefault(role, [])
    return {
        "status": "ok",
        **counts,
        "selection_source": selection_source,
        "state": state,
        "external_index": external_index,
        "external_element": "H" if external_index is None else molecule.GetAtomWithIdx(external_index).GetSymbol(),
        "o4_index": o4,
        "local_indices": local_indices,
        "roles": roles,
    }


def make_graph(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    return mol_to_chiral_pyg_graph(
        molecule,
        smiles,
        atom_feature_fn=atom_features,
        bond_feature_fn=bond_features,
    )


def annotate_row(
    row: pd.Series,
    row_number: int,
    acceptor_role_names: list[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    acceptor_role_names = list(acceptor_role_names or ACCEPTOR_ROLE_NAMES)
    audit: dict[str, Any] = {"Inference_Status": "error", "Inference_Error": ""}
    donor_smiles, _, _ = canonicalize(row.get("Donor_Canonical_SMILES"))
    acceptor_smiles, _, acceptor_mapping = canonicalize(row.get("Acceptor_Canonical_SMILES"))
    if donor_smiles is None:
        audit["Inference_Error"] = "invalid or missing Donor_Canonical_SMILES"
        return None, audit
    if acceptor_smiles is None:
        audit["Inference_Error"] = "invalid or missing Acceptor_Canonical_SMILES"
        return None, audit

    donor_type, donor_type_source = resolve_donor_type(donor_smiles, row.get("Donor_Type"))
    if donor_type is None:
        audit["Inference_Error"] = donor_type_source
        return None, audit
    donor = extract_donor_rfu(donor_smiles, donor_type)
    supplied_target = None
    target_source = "auto"
    for column in ("Target_O4_Index", "Acceptor_O4_Index"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            original_index = int(float(value))
            if original_index < 0 or original_index >= len(acceptor_mapping):
                audit["Inference_Error"] = f"{column} is out of bounds"
                return None, audit
            supplied_target = int(acceptor_mapping[original_index])
            target_source = column
            break
    acceptor = extract_acceptor_target_site(acceptor_smiles, supplied_target)
    if acceptor["status"] != "ok":
        if supplied_target is None and acceptor["status"] == "no_free_target_site":
            audit.update({
                "Inference_Status": "structurally_infeasible",
                "Inference_Error": "",
                "Canonical_Donor_SMILES": donor_smiles,
                "Canonical_Acceptor_SMILES": acceptor_smiles,
                "Resolved_Donor_Type": donor_type,
                "Donor_Type_Source": donor_type_source,
                "Target_O4_Source": "auto_no_free_target",
                "Target_O4_State": "all_blocked",
                "Target_O4_Candidate_Count": acceptor["candidate_count"],
                "Target_O4_Free_Candidate_Count": acceptor["free_candidate_count"],
                "Target_O4_Blocked_Candidate_Count": acceptor["blocked_candidate_count"],
            })
            return None, audit
        audit["Inference_Error"] = f"acceptor target-site extraction failed: {acceptor['status']}"
        return None, audit
    if supplied_target is None:
        target_source = str(acceptor["selection_source"])

    donor_indices = parse_int_list(donor["Donor_RFU_Atom_Indices"])
    donor_roles = parse_role_dict(donor["Donor_RFU_Role_Indices"])
    acceptor_indices = [int(value) for value in acceptor["local_indices"]]
    donor_role_matrix = make_role_matrix(donor_indices, donor_roles, DONOR_ROLE_NAMES)
    acceptor_role_matrix = make_role_matrix(
        acceptor_indices,
        acceptor["roles"],
        acceptor_role_names,
    )
    donor_c1 = int(donor["Donor_C1_Index"])
    acceptor_o4 = int(acceptor["o4_index"])
    item = {
        "sample_id": str(row.get("Sample_ID", row.get("ID", row_number))),
        "stress_type": "inference",
        "parent_group": "",
        "donor_graph": make_graph(donor_smiles),
        "acceptor_graph": make_graph(acceptor_smiles),
        "donor_indices": torch.tensor(donor_indices, dtype=torch.long),
        "acceptor_indices": torch.tensor(acceptor_indices, dtype=torch.long),
        "donor_roles": donor_role_matrix,
        "acceptor_roles": acceptor_role_matrix,
        "donor_c1_local_pos": donor_indices.index(donor_c1),
        "acceptor_o4_local_pos": acceptor_indices.index(acceptor_o4),
        "label": -1,
    }
    audit.update({
        "Inference_Status": "ok",
        "Inference_Error": "",
        "Canonical_Donor_SMILES": donor_smiles,
        "Canonical_Acceptor_SMILES": acceptor_smiles,
        "Resolved_Donor_Type": donor_type,
        "Donor_Type_Source": donor_type_source,
        "Target_O4_Source": target_source,
        "Target_O4_Index": acceptor_o4,
        "Target_O4_State": acceptor["state"],
        "Target_O4_Candidate_Count": acceptor["candidate_count"],
        "Target_O4_Free_Candidate_Count": acceptor["free_candidate_count"],
        "Target_O4_Blocked_Candidate_Count": acceptor["blocked_candidate_count"],
        "Target_O4_External_Index": -1 if acceptor["external_index"] is None else acceptor["external_index"],
        "Target_O4_External_Element": acceptor["external_element"],
    })
    return item, audit


class InferenceDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[StructureOnlyChiralGINE, int]:
    checkpoint = safe_torch_load(checkpoint_path, device)
    args = checkpoint["args"]
    acceptor_role_names = list(checkpoint.get("acceptor_role_names") or [])
    if acceptor_role_names != ACCEPTOR_ROLE_NAMES:
        raise ValueError(
            f"{checkpoint_path}: unsupported Layer-1 role schema; "
            "expected the 10-role schema without O4_EXTERNAL"
        )
    model = StructureOnlyChiralGINE(
        atom_dim=15,
        bond_dim=18,
        hidden_dim=int(args["hidden_dim"]),
        num_layers=int(args["num_layers"]),
        dropout=float(args["dropout"]),
        acceptor_role_dim=len(acceptor_role_names),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, int(args["seed"])


def run_probabilities(model: StructureOnlyChiralGINE, loader: DataLoader, device: torch.device) -> list[float]:
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            moved = dict(batch)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) or key in {"donor_graph", "acceptor_graph"}:
                    moved[key] = value.to(device)
            probabilities.extend(torch.softmax(model(moved), dim=-1)[:, 1].cpu().tolist())
    return probabilities


def discover_checkpoints(explicit: list[Path] | None, directory: Path) -> list[Path]:
    checkpoints = explicit or sorted(directory.glob("seed*.pt"))
    checkpoints = [Path(path) for path in checkpoints]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {directory}")
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    return checkpoints


def checkpoint_acceptor_role_names(checkpoints: list[Path]) -> list[str]:
    schemas = []
    for checkpoint_path in checkpoints:
        checkpoint = safe_torch_load(checkpoint_path, "cpu")
        schema = list(checkpoint.get("acceptor_role_names") or [])
        if schema != ACCEPTOR_ROLE_NAMES:
            raise ValueError(
                f"{checkpoint_path}: unsupported Layer-1 role schema; "
                "expected the 10-role schema without O4_EXTERNAL"
            )
        schemas.append(schema)
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("ensemble checkpoints use incompatible acceptor role schemas")
    return schemas[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with donor and acceptor SMILES.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/hard_feasibility_chiral_gine"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"}
    if missing := required - set(source.columns):
        parser.error(f"input is missing required columns: {sorted(missing)}")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between zero and one")

    checkpoints = discover_checkpoints(args.checkpoints, args.checkpoint_dir)
    acceptor_role_names = checkpoint_acceptor_role_names(checkpoints)

    items: list[dict[str, Any]] = []
    valid_rows: list[int] = []
    audits: list[dict[str, Any]] = []
    for row_number, (_, row) in enumerate(source.iterrows()):
        item, audit = annotate_row(
            row,
            row_number,
            acceptor_role_names=acceptor_role_names,
        )
        audits.append(audit)
        if item is not None:
            items.append(item)
            valid_rows.append(row_number)

    output = source.copy()
    audit_frame = pd.DataFrame(audits, index=output.index)
    for column in audit_frame.columns:
        output[column] = audit_frame[column]
    output["Hard_Feasibility_Probability"] = np.nan
    output["Hard_Feasibility_Probability_SD"] = np.nan
    output["Hard_Feasibility_Prediction"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["Hard_Feasibility_Decision"] = "manual_review"

    if items:
        loader = DataLoader(InferenceDataset(items), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        device = torch.device(args.device)
        all_probabilities: list[list[float]] = []
        seeds: list[int] = []
        for checkpoint_path in checkpoints:
            model, seed = load_model(checkpoint_path, device)
            probabilities = run_probabilities(model, loader, device)
            all_probabilities.append(probabilities)
            seeds.append(seed)
            output.loc[valid_rows, f"Probability_Seed_{seed}"] = probabilities
        matrix = np.asarray(all_probabilities, dtype=float)
        mean_probability = matrix.mean(axis=0)
        probability_sd = matrix.std(axis=0, ddof=0)
        prediction = (mean_probability >= args.threshold).astype(int)
        output.loc[valid_rows, "Hard_Feasibility_Probability"] = mean_probability
        output.loc[valid_rows, "Hard_Feasibility_Probability_SD"] = probability_sd
        output.loc[valid_rows, "Hard_Feasibility_Prediction"] = prediction
        output.loc[valid_rows, "Hard_Feasibility_Decision"] = np.where(prediction == 1, "pass", "reject")
        output["Inference_Seeds"] = ";".join(map(str, seeds))
    else:
        output["Inference_Seeds"] = ""
    output["Decision_Threshold"] = args.threshold
    output["Ensemble_Size"] = len(checkpoints)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(json.dumps({
        "input_rows": int(len(output)),
        "predicted_rows": int(len(valid_rows)),
        "manual_review_rows": int(len(output) - len(valid_rows)),
        "ensemble_size": len(checkpoints),
        "acceptor_role_names": acceptor_role_names,
        "threshold": args.threshold,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
