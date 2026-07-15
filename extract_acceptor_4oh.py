from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem


INPUT_PATH = Path("data/processed/donor_rfu_extracted.csv")
OVERRIDE_PATH = Path("data/processed/acceptor_4oh_overrides.csv")
OUTPUT_PATH = Path("data/processed/glyco_model_local.csv")
REPORT_PATH = Path("data/processed/acceptor_4oh_report.csv")
MANUAL_CHECK_PATH = Path("data/processed/acceptor_4oh_manual_check.csv")


OUTPUT_COLUMNS = [
    "Acceptor_OH_Status",
    "Acceptor_OH_Selection_Method",
    "Acceptor_OH_Local_Atom_Indices",
    "Acceptor_OH_Local_SMILES",
    "Acceptor_OH_Local_Num_Atoms",
    "Acceptor_OH_Role_Indices",
    "Acceptor_C1_Index",
    "Acceptor_O5_Index",
    "Acceptor_C2_Index",
    "Acceptor_C3_Index",
    "Acceptor_C4_Index",
    "Acceptor_C5_Index",
    "Acceptor_O4_Index",
    "Acceptor_N2_Sub_Entry_Indices",
    "Acceptor_N2_Subgraph_Indices",
    "Acceptor_PG_Entry_Indices",
]


def empty_result(status: str, method: str = "failed") -> dict[str, object]:
    roles = {
        "O4": [],
        "C4": [],
        "C3": [],
        "C5": [],
        "C2": [],
        "O5": [],
        "C1": [],
        "N2_SUB_ENTRY": [],
        "N2_SUBGRAPH": [],
        "PG_ENTRY": [],
    }
    return {
        "Acceptor_OH_Status": status,
        "Acceptor_OH_Selection_Method": method,
        "Acceptor_OH_Local_Atom_Indices": "",
        "Acceptor_OH_Local_SMILES": "",
        "Acceptor_OH_Local_Num_Atoms": 0,
        "Acceptor_OH_Role_Indices": json.dumps(roles, separators=(",", ":")),
        "Acceptor_C1_Index": "",
        "Acceptor_O5_Index": "",
        "Acceptor_C2_Index": "",
        "Acceptor_C3_Index": "",
        "Acceptor_C4_Index": "",
        "Acceptor_C5_Index": "",
        "Acceptor_O4_Index": "",
        "Acceptor_N2_Sub_Entry_Indices": "",
        "Acceptor_N2_Subgraph_Indices": "",
        "Acceptor_PG_Entry_Indices": "",
    }


def join_indices(indices: list[int]) -> str:
    return ";".join(str(idx) for idx in sorted(set(indices)))


def fragment_smiles(mol: Chem.Mol, atom_indices: list[int]) -> str:
    if not atom_indices:
        return ""
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=sorted(set(atom_indices)),
        canonical=True,
        isomericSmiles=True,
    )


def find_sugar_rings(mol: Chem.Mol) -> list[tuple[int, ...]]:
    sugar_rings: list[tuple[int, ...]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in {5, 6}:
            continue
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        num_o = sum(atom.GetAtomicNum() == 8 for atom in atoms)
        num_c = sum(atom.GetAtomicNum() == 6 for atom in atoms)
        if num_o == 1 and num_c == len(ring) - 1:
            sugar_rings.append(tuple(ring))
    return sugar_rings


def traverse_ring_from_o5_c1(
    mol: Chem.Mol,
    ring: tuple[int, ...],
    o5_idx: int,
    c1_idx: int,
) -> dict[str, int] | None:
    ring_set = set(ring)
    c1_ring_neighbors = [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(c1_idx).GetNeighbors()
        if neighbor.GetIdx() in ring_set
    ]
    c2_candidates = [idx for idx in c1_ring_neighbors if idx != o5_idx]
    if len(c2_candidates) != 1:
        return None

    path = [c1_idx, c2_candidates[0]]
    previous = c1_idx
    current = c2_candidates[0]

    while True:
        next_ring_neighbors = [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors()
            if neighbor.GetIdx() in ring_set and neighbor.GetIdx() != previous
        ]
        if not next_ring_neighbors:
            return None
        if o5_idx in next_ring_neighbors:
            break
        next_idx = next_ring_neighbors[0]
        path.append(next_idx)
        previous, current = current, next_idx
        if len(path) > len(ring):
            return None

    if len(path) != len(ring) - 1:
        return None

    if len(path) == 5:
        c1, c2, c3, c4, c5 = path
        return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "O5": o5_idx}

    # Kept explicit so five-member cases go to manual check instead of being silently mis-numbered.
    return None


def find_c2_n_sub_entries(mol: Chem.Mol, c2_idx: int, ring_set: set[int]) -> list[int]:
    entries: list[int] = []
    for neighbor in mol.GetAtomWithIdx(c2_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx in ring_set:
            continue
        if neighbor.GetAtomicNum() == 7:
            entries.append(idx)
    return sorted(entries)


def find_free_oh_on_atom(mol: Chem.Mol, carbon_idx: int, ring_set: set[int]) -> list[int]:
    oh_atoms: list[int] = []
    for neighbor in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx in ring_set:
            continue
        if neighbor.GetAtomicNum() != 8:
            continue
        if neighbor.GetFormalCharge() != 0:
            continue
        if neighbor.GetTotalNumHs() <= 0:
            continue
        oh_atoms.append(idx)
    return sorted(oh_atoms)


def enumerate_candidates(mol: Chem.Mol) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for ring in find_sugar_rings(mol):
        ring_set = set(ring)
        o_atoms = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8]
        if len(o_atoms) != 1:
            continue
        o5 = o_atoms[0]
        c1_options = [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(o5).GetNeighbors()
            if neighbor.GetIdx() in ring_set and neighbor.GetAtomicNum() == 6
        ]
        for c1 in c1_options:
            roles = traverse_ring_from_o5_c1(mol, ring, o5, c1)
            if roles is None:
                continue
            n2_entries = find_c2_n_sub_entries(mol, roles["C2"], ring_set)
            o4_entries = find_free_oh_on_atom(mol, roles["C4"], ring_set)
            if n2_entries and len(o4_entries) == 1:
                candidate = {
                    **roles,
                    "O4": o4_entries[0],
                    "N2_SUB_ENTRY": n2_entries,
                    "ring": list(ring),
                }
                candidates.append(candidate)

    unique: list[dict[str, object]] = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate["O5"],
            candidate["C1"],
            candidate["C2"],
            candidate["C3"],
            candidate["C4"],
            candidate["C5"],
            candidate["O4"],
            tuple(candidate["N2_SUB_ENTRY"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def expand_from_entries(
    mol: Chem.Mol,
    entries: list[int],
    blocked: set[int],
    max_depth: int,
) -> list[int]:
    selected: set[int] = set(entries)
    queue: list[tuple[int, int]] = [(idx, 0) for idx in entries]
    seen = set(blocked)

    while queue:
        atom_idx, depth = queue.pop(0)
        if atom_idx in seen:
            continue
        seen.add(atom_idx)
        selected.add(atom_idx)
        if depth >= max_depth:
            continue
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            next_idx = neighbor.GetIdx()
            if next_idx in seen or next_idx in blocked:
                continue
            selected.add(next_idx)
            # Keep aromatic/ring systems as entry atoms only; the full group is represented by the global acceptor graph.
            if neighbor.GetIsAromatic() or neighbor.IsInRing():
                continue
            queue.append((next_idx, depth + 1))
    return sorted(selected)


def find_pg_entries(
    mol: Chem.Mol,
    candidate: dict[str, object],
    ring_set: set[int],
) -> list[int]:
    protected_centers = [candidate["C1"], candidate["C3"], candidate["C5"]]
    excluded = set(ring_set)
    excluded.add(candidate["O4"])
    excluded.update(candidate["N2_SUB_ENTRY"])

    entries: set[int] = set()
    for center_idx in protected_centers:
        for neighbor in mol.GetAtomWithIdx(int(center_idx)).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx in excluded:
                continue
            entries.add(idx)
    return sorted(entries)


def expand_pg_entries(mol: Chem.Mol, pg_entries: list[int], blocked: set[int]) -> list[int]:
    selected: set[int] = set(pg_entries)
    for entry in pg_entries:
        entry_atom = mol.GetAtomWithIdx(entry)
        for neighbor in entry_atom.GetNeighbors():
            idx = neighbor.GetIdx()
            if idx in blocked or idx in selected:
                continue
            selected.add(idx)
            # For O-protecting groups, this captures O plus the first external atom.
            # For aromatic/ring substituents, do not expand beyond the entry atom's immediate neighbor.
            break
    return sorted(selected)


def build_local_result(
    mol: Chem.Mol,
    candidate: dict[str, object],
    method: str,
) -> dict[str, object]:
    ring_set = set(candidate["ring"])
    core_atoms = {
        candidate["O4"],
        candidate["C4"],
        candidate["C3"],
        candidate["C5"],
        candidate["C2"],
        candidate["O5"],
        candidate["C1"],
    }

    n2_subgraph = expand_from_entries(
        mol,
        list(candidate["N2_SUB_ENTRY"]),
        blocked=ring_set,
        max_depth=2,
    )
    pg_entries = find_pg_entries(mol, candidate, ring_set)
    pg_subgraph = expand_pg_entries(
        mol,
        pg_entries,
        blocked=ring_set | {candidate["O4"]} | set(candidate["N2_SUB_ENTRY"]),
    )

    local_atoms = sorted(core_atoms | set(n2_subgraph) | set(pg_subgraph))
    if not local_atoms:
        return empty_result("empty_local_subgraph", "failed")

    role_indices = {
        "O4": [candidate["O4"]],
        "C4": [candidate["C4"]],
        "C3": [candidate["C3"]],
        "C5": [candidate["C5"]],
        "C2": [candidate["C2"]],
        "O5": [candidate["O5"]],
        "C1": [candidate["C1"]],
        "N2_SUB_ENTRY": list(candidate["N2_SUB_ENTRY"]),
        "N2_SUBGRAPH": n2_subgraph,
        "PG_ENTRY": pg_entries,
    }

    return {
        "Acceptor_OH_Status": "ok",
        "Acceptor_OH_Selection_Method": method,
        "Acceptor_OH_Local_Atom_Indices": join_indices(local_atoms),
        "Acceptor_OH_Local_SMILES": fragment_smiles(mol, local_atoms),
        "Acceptor_OH_Local_Num_Atoms": len(local_atoms),
        "Acceptor_OH_Role_Indices": json.dumps(role_indices, separators=(",", ":")),
        "Acceptor_C1_Index": candidate["C1"],
        "Acceptor_O5_Index": candidate["O5"],
        "Acceptor_C2_Index": candidate["C2"],
        "Acceptor_C3_Index": candidate["C3"],
        "Acceptor_C4_Index": candidate["C4"],
        "Acceptor_C5_Index": candidate["C5"],
        "Acceptor_O4_Index": candidate["O4"],
        "Acceptor_N2_Sub_Entry_Indices": join_indices(list(candidate["N2_SUB_ENTRY"])),
        "Acceptor_N2_Subgraph_Indices": join_indices(n2_subgraph),
        "Acceptor_PG_Entry_Indices": join_indices(pg_entries),
    }


def load_overrides(path: Path) -> dict[int, tuple[int, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    overrides: dict[int, tuple[int, str]] = {}
    for _, row in df.iterrows():
        sample_id = int(row["ID"])
        rank = int(row["Selected_Candidate_Rank"])
        method = str(row.get("Selection_Method", "manual_resolved"))
        overrides[sample_id] = (rank, method)
    return overrides


def extract_acceptor_4oh(
    acceptor_smiles: object,
    sample_id: int,
    overrides: dict[int, tuple[int, str]],
) -> dict[str, object]:
    if pd.isna(acceptor_smiles) or str(acceptor_smiles).strip() == "":
        return empty_result("mol_parse_fail")

    mol = Chem.MolFromSmiles(str(acceptor_smiles))
    if mol is None:
        return empty_result("mol_parse_fail")
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    sugar_rings = find_sugar_rings(mol)
    if not sugar_rings:
        return empty_result("no_sugar_ring")

    candidates = enumerate_candidates(mol)
    if not candidates:
        return empty_result("no_c2_n_ring_or_no_c4_free_oh")

    if len(candidates) == 1:
        return build_local_result(mol, candidates[0], "rule_unique")

    if sample_id in overrides:
        rank, method = overrides[sample_id]
        if 1 <= rank <= len(candidates):
            return build_local_result(mol, candidates[rank - 1], method)

    return empty_result("multiple_candidates")


def validate_output(df: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict[str, object]] = []

    for _, row in df.iterrows():
        if row["Acceptor_OH_Status"] != "ok":
            continue
        mol = Chem.MolFromSmiles(row["Acceptor_Canonical_SMILES"])
        roles = json.loads(row["Acceptor_OH_Role_Indices"])
        local_atoms = set(int(float(idx)) for idx in str(row["Acceptor_OH_Local_Atom_Indices"]).split(";") if idx)

        failed: list[str] = []
        o4 = roles["O4"][0]
        c4 = roles["C4"][0]
        c2 = roles["C2"][0]
        n2_entries = roles["N2_SUB_ENTRY"]
        role_atoms = set()
        for value in roles.values():
            role_atoms.update(value)

        if not role_atoms.issubset(local_atoms):
            failed.append("roles_not_in_local_atoms")
        if mol.GetAtomWithIdx(o4).GetAtomicNum() != 8 or mol.GetAtomWithIdx(o4).GetTotalNumHs() <= 0:
            failed.append("o4_not_free_oh")
        if mol.GetBondBetweenAtoms(o4, c4) is None:
            failed.append("o4_not_bonded_to_c4")
        if mol.GetAtomWithIdx(c4).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(c4).IsInRing():
            failed.append("c4_not_ring_carbon")
        if mol.GetAtomWithIdx(c2).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(c2).IsInRing():
            failed.append("c2_not_ring_carbon")
        for n_idx in n2_entries:
            if mol.GetAtomWithIdx(n_idx).GetAtomicNum() != 7:
                failed.append("n2_entry_not_n")
            if mol.GetBondBetweenAtoms(c2, n_idx) is None:
                failed.append("n2_entry_not_bonded_to_c2")

        if failed:
            issues.append(
                {
                    "ID": row["ID"],
                    "failed_checks": ";".join(sorted(set(failed))),
                    "Acceptor_OH_Local_SMILES": row["Acceptor_OH_Local_SMILES"],
                    "Acceptor_Canonical_SMILES": row["Acceptor_Canonical_SMILES"],
                }
            )

    return pd.DataFrame(issues)


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    overrides = load_overrides(OVERRIDE_PATH)
    results = [
        extract_acceptor_4oh(row.Acceptor_Canonical_SMILES, int(row.ID), overrides)
        for row in df.itertuples(index=False)
    ]
    result_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out = pd.concat([df, result_df], axis=1)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    for key, count in Counter(out["Acceptor_OH_Status"]).items():
        summary_rows.append({"metric": "Acceptor_OH_Status", "key": key, "count": count})
    for key, count in Counter(out["Acceptor_OH_Selection_Method"]).items():
        summary_rows.append({"metric": "Acceptor_OH_Selection_Method", "key": key, "count": count})
    size_ok = out.loc[out["Acceptor_OH_Status"] == "ok", "Acceptor_OH_Local_Num_Atoms"].astype(int)
    if len(size_ok):
        summary_rows.extend(
            [
                {"metric": "Acceptor_OH_Local_Num_Atoms", "key": "min", "count": int(size_ok.min())},
                {"metric": "Acceptor_OH_Local_Num_Atoms", "key": "median", "count": float(size_ok.median())},
                {"metric": "Acceptor_OH_Local_Num_Atoms", "key": "max", "count": int(size_ok.max())},
            ]
        )

    validation_issues = validate_output(out)
    summary_rows.append({"metric": "Validation_Issues", "key": "rows", "count": len(validation_issues)})
    pd.DataFrame(summary_rows).to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    manual = out[
        (out["Acceptor_OH_Status"] != "ok")
        | (out["Acceptor_OH_Selection_Method"] != "rule_unique")
    ].copy()
    if len(validation_issues):
        validation_issues.insert(0, "Manual_Check_Type", "validation_issue")
        manual.insert(0, "Manual_Check_Type", "status_or_selection")
        pd.concat([manual, validation_issues], ignore_index=True, sort=False).to_csv(
            MANUAL_CHECK_PATH,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        manual.to_csv(MANUAL_CHECK_PATH, index=False, encoding="utf-8-sig")

    print("Acceptor_OH_Status")
    print(out["Acceptor_OH_Status"].value_counts(dropna=False).to_string())
    print("Acceptor_OH_Selection_Method")
    print(out["Acceptor_OH_Selection_Method"].value_counts(dropna=False).to_string())
    print("Local size")
    print(size_ok.describe().to_string())
    print("Validation issue rows:", len(validation_issues))
    print("Manual check rows:", len(manual))


if __name__ == "__main__":
    main()
