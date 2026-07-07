from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import pandas as pd
from rdkit import Chem


INPUT_PATH = Path("condition_tokenized_with_splits.csv")
FALLBACK_INPUT_PATH = Path("condition_tokenized.csv")
OUTPUT_PATH = Path("donor_rfu_extracted.csv")
REPORT_PATH = Path("donor_rfu_report.csv")
MANUAL_CHECK_PATH = Path("donor_rfu_manual_check.csv")
DONOR_SMILES_COLUMN = "Donor_Canonical_SMILES"


PATTERNS = {
    "trichloroacetimidate": [
        (
            "linear_or_cyclic_o_imidate",
            Chem.MolFromSmarts("[#6;R]-[#8]-[#6](=[#7])-[#6]([Cl])([Cl])[Cl]"),
        ),
        (
            "cyclic_n_imidate",
            Chem.MolFromSmarts("[#6;R]-[#7]=[#6](-[#8])-[#6]([Cl])([Cl])[Cl]"),
        ),
    ],
    "trifluoroacetimidate": [
        (
            "linear_or_cyclic_o_imidate",
            Chem.MolFromSmarts("[#6;R]-[#8]-[#6](=[#7])-[#6]([F])([F])[F]"),
        ),
        (
            "cyclic_n_imidate",
            Chem.MolFromSmarts("[#6;R]-[#7]=[#6](-[#8])-[#6]([F])([F])[F]"),
        ),
    ],
    "thioglycoside": [
        ("c1_s", Chem.MolFromSmarts("[#6;R]-[#16]")),
    ],
}


RFU_COLUMNS = [
    "Donor_RFU_Status",
    "Donor_RFU_Atom_Indices",
    "Donor_RFU_SMILES",
    "Donor_RFU_Num_Atoms",
    "Donor_RFU_Role_Indices",
    "Donor_C1_Index",
    "Donor_O5_Index",
    "Donor_C2_Index",
    "Donor_LG_Entry_Index",
    "Donor_LG_Core_Indices",
    "Donor_Ring_Context_Indices",
    "Donor_Bridge_Core_Indices",
    "Donor_C2_Sub_Entry_Index",
    "Donor_C2_Sub_Entry_Status",
]


def empty_result(status: str) -> dict[str, object]:
    result: dict[str, object] = {
        "Donor_RFU_Status": status,
        "Donor_RFU_Atom_Indices": "",
        "Donor_RFU_SMILES": "",
        "Donor_RFU_Num_Atoms": 0,
        "Donor_RFU_Role_Indices": json.dumps(
            {
                "C1": [],
                "O5": [],
                "C2": [],
                "LG_ENTRY": [],
                "LG_CORE": [],
                "RING_CONTEXT": [],
                "BRIDGE_CORE": [],
                "C2_SUB_ENTRY": [],
            },
            separators=(",", ":"),
        ),
        "Donor_C1_Index": "",
        "Donor_O5_Index": "",
        "Donor_C2_Index": "",
        "Donor_LG_Entry_Index": "",
        "Donor_LG_Core_Indices": "",
        "Donor_Ring_Context_Indices": "",
        "Donor_Bridge_Core_Indices": "",
        "Donor_C2_Sub_Entry_Index": "",
        "Donor_C2_Sub_Entry_Status": "",
    }
    return result


def same_ring(mol: Chem.Mol, atom_a: int, atom_b: int) -> bool:
    return any(atom_a in ring and atom_b in ring for ring in mol.GetRingInfo().AtomRings())


def ring_neighbors_by_role(mol: Chem.Mol, c1_idx: int) -> tuple[list[int], list[int]]:
    o5_candidates: list[int] = []
    c2_candidates: list[int] = []
    for neighbor in mol.GetAtomWithIdx(c1_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if not neighbor.IsInRing() or not same_ring(mol, c1_idx, idx):
            continue
        if neighbor.GetAtomicNum() == 8:
            o5_candidates.append(idx)
        elif neighbor.GetAtomicNum() == 6:
            c2_candidates.append(idx)
    return sorted(o5_candidates), sorted(c2_candidates)


def ring_context_indices(mol: Chem.Mol, c1_idx: int, o5_idx: int, c2_idx: int) -> list[int]:
    del mol, c1_idx
    # Keep the RFU local: O5 and C2 are the direct C1 ring context.
    # Expanding to the next ring atoms makes this subgraph drift toward a larger sugar-ring fragment.
    return sorted({o5_idx, c2_idx})


def c2_sub_entry_indices(mol: Chem.Mol, c1_idx: int, c2_idx: int) -> list[int]:
    entries: list[int] = []
    for neighbor in mol.GetAtomWithIdx(c2_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx == c1_idx:
            continue
        if neighbor.IsInRing() and same_ring(mol, c1_idx, idx):
            continue
        entries.append(idx)
    return sorted(entries)


def imidate_bridge_indices(mol: Chem.Mol, c2_idx: int, lg_core: list[int]) -> list[int]:
    core = set(lg_core)
    bridge: set[int] = set()
    for neighbor in mol.GetAtomWithIdx(c2_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx in core:
            bridge.add(idx)
    return sorted(bridge)


def lg_roles(mol: Chem.Mol, donor_type: str, match: tuple[int, ...]) -> tuple[int, list[int]]:
    if donor_type == "thioglycoside":
        return match[1], []
    c1_idx = match[0]
    core = set(match[2:])
    n_idx = match[3]
    match_atoms = set(match)
    for neighbor in mol.GetAtomWithIdx(n_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx in match_atoms:
            continue
        if same_ring(mol, c1_idx, idx):
            continue
        # Keep only the first atom of an N-substituent such as N-phenyl; the full group stays in the global donor graph.
        core.add(idx)
    return match[1], sorted(core)


def add_acyclic_tail(
    mol: Chem.Mol,
    core: set[int],
    start_idx: int,
    blocked: set[int],
    max_depth: int = 2,
) -> None:
    queue: list[tuple[int, int]] = [(start_idx, 0)]
    seen = set(blocked)
    while queue:
        atom_idx, depth = queue.pop(0)
        if atom_idx in seen:
            continue
        seen.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetIsAromatic() or atom.IsInRing():
            core.add(atom_idx)
            continue
        core.add(atom_idx)
        if depth >= max_depth:
            continue
        for neighbor in atom.GetNeighbors():
            next_idx = neighbor.GetIdx()
            if next_idx in seen:
                continue
            if neighbor.GetIsAromatic() or neighbor.IsInRing():
                core.add(next_idx)
                continue
            queue.append((next_idx, depth + 1))


def thioglycoside_lg_core(mol: Chem.Mol, c1_idx: int, sulfur_idx: int) -> list[int]:
    core: set[int] = set()
    entry_neighbors = [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors()
        if neighbor.GetIdx() != c1_idx
    ]

    for idx in entry_neighbors:
        atom = mol.GetAtomWithIdx(idx)
        core.add(idx)
        if atom.GetIsAromatic() or atom.IsInRing():
            continue

        # Functionalized sulfur leaving groups such as xanthates
        # (S-C(=S)-OR) and disulfone-like groups (S-S(=O)2-Ar) need
        # their local heteroatom-rich core, while ordinary S-aryl groups
        # should not expand into a full aromatic ring.
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in {sulfur_idx, c1_idx}:
                continue
            bond = mol.GetBondBetweenAtoms(idx, neighbor_idx)
            if neighbor.GetIsAromatic() or neighbor.IsInRing():
                core.add(neighbor_idx)
            elif neighbor.GetAtomicNum() != 6 or bond.GetBondType() != Chem.BondType.SINGLE:
                core.add(neighbor_idx)
                if neighbor.GetAtomicNum() in {8, 16}:
                    for tail_atom in neighbor.GetNeighbors():
                        tail_idx = tail_atom.GetIdx()
                        if tail_idx in {idx, sulfur_idx, c1_idx}:
                            continue
                        add_acyclic_tail(
                            mol,
                            core,
                            tail_idx,
                            blocked={idx, sulfur_idx, c1_idx},
                            max_depth=2,
                        )

    return sorted(core)


def join_indices(indices: list[int]) -> str:
    return ";".join(str(idx) for idx in indices)


def fragment_smiles(mol: Chem.Mol, atom_indices: list[int]) -> str:
    if not atom_indices:
        return ""
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=atom_indices,
        canonical=True,
        isomericSmiles=True,
    )


def extract_donor_rfu(donor_smiles: object, donor_type: object) -> dict[str, object]:
    if pd.isna(donor_smiles) or str(donor_smiles).strip() == "":
        return empty_result("mol_parse_fail")

    donor_type_str = str(donor_type).strip()
    mol = Chem.MolFromSmiles(str(donor_smiles))
    if mol is None:
        return empty_result("mol_parse_fail")

    if donor_type_str not in PATTERNS:
        return empty_result("no_lg_match")

    raw_matches: list[tuple[str, tuple[int, ...]]] = []
    candidates: dict[int, tuple[str, tuple[int, ...], list[int], list[int]]] = {}
    for pattern_name, pattern in PATTERNS[donor_type_str]:
        for match in mol.GetSubstructMatches(pattern):
            raw_matches.append((pattern_name, match))
            c1_idx = match[0]
            c1_atom = mol.GetAtomWithIdx(c1_idx)
            if c1_atom.GetAtomicNum() != 6 or not c1_atom.IsInRing():
                continue
            o5_candidates, c2_candidates = ring_neighbors_by_role(mol, c1_idx)
            if not o5_candidates or not c2_candidates:
                continue
            candidates.setdefault(c1_idx, (pattern_name, match, o5_candidates, c2_candidates))

    if not raw_matches:
        return empty_result("no_lg_match")
    if not candidates:
        return empty_result("no_anomeric_c1")
    if len(candidates) > 1:
        return empty_result("multiple_candidates")

    c1_idx, (pattern_name, match, o5_candidates, c2_candidates) = next(iter(candidates.items()))
    o5_idx = o5_candidates[0]
    c2_idx = c2_candidates[0]
    lg_entry_idx, lg_core_indices = lg_roles(mol, donor_type_str, match)
    if donor_type_str == "thioglycoside":
        lg_core_indices = thioglycoside_lg_core(mol, c1_idx, lg_entry_idx)

    ring_context = ring_context_indices(mol, c1_idx, o5_idx, c2_idx)
    c2_entries = c2_sub_entry_indices(mol, c1_idx, c2_idx)
    bridge_core = imidate_bridge_indices(mol, c2_idx, lg_core_indices)

    if c2_entries:
        c2_sub_status = "found"
    elif donor_type_str in {"trichloroacetimidate", "trifluoroacetimidate"} and bridge_core:
        c2_sub_status = "cyclic_bridge"
    else:
        c2_sub_status = "absent_deoxy_or_no_substituent"

    atom_set = {
        c1_idx,
        o5_idx,
        c2_idx,
        lg_entry_idx,
        *lg_core_indices,
        *ring_context,
        *bridge_core,
        *c2_entries,
    }
    atom_indices = sorted(atom_set)

    if not atom_indices:
        return empty_result("empty_rfu")

    roles = {
        "C1": [c1_idx],
        "O5": [o5_idx],
        "C2": [c2_idx],
        "LG_ENTRY": [lg_entry_idx],
        "LG_CORE": lg_core_indices,
        "RING_CONTEXT": ring_context,
        "BRIDGE_CORE": bridge_core,
        "C2_SUB_ENTRY": c2_entries,
    }

    return {
        "Donor_RFU_Status": "ok",
        "Donor_RFU_Atom_Indices": join_indices(atom_indices),
        "Donor_RFU_SMILES": fragment_smiles(mol, atom_indices),
        "Donor_RFU_Num_Atoms": len(atom_indices),
        "Donor_RFU_Role_Indices": json.dumps(roles, separators=(",", ":")),
        "Donor_C1_Index": c1_idx,
        "Donor_O5_Index": o5_idx,
        "Donor_C2_Index": c2_idx,
        "Donor_LG_Entry_Index": lg_entry_idx,
        "Donor_LG_Core_Indices": join_indices(lg_core_indices),
        "Donor_Ring_Context_Indices": join_indices(ring_context),
        "Donor_Bridge_Core_Indices": join_indices(bridge_core),
        "Donor_C2_Sub_Entry_Index": join_indices(c2_entries),
        "Donor_C2_Sub_Entry_Status": c2_sub_status,
    }


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for status, count in df["Donor_RFU_Status"].value_counts(dropna=False).items():
        rows.append({"section": "Donor_RFU_Status", "key": status, "value": int(count)})
    for status, count in df["Donor_C2_Sub_Entry_Status"].value_counts(dropna=False).items():
        rows.append({"section": "Donor_C2_Sub_Entry_Status", "key": status, "value": int(count)})
    for donor_type, group in df.groupby("Donor_Type"):
        sizes = group["Donor_RFU_Num_Atoms"].astype(int).tolist()
        rows.append(
            {
                "section": "Donor_RFU_Num_Atoms",
                "key": donor_type,
                "value": f"n={len(sizes)};min={min(sizes)};median={median(sizes)};max={max(sizes)}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    input_path = INPUT_PATH if INPUT_PATH.exists() else FALLBACK_INPUT_PATH
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    results = [
        extract_donor_rfu(getattr(row, DONOR_SMILES_COLUMN), row.Donor_Type)
        for row in df.itertuples(index=False)
    ]
    rfu_df = pd.DataFrame(results, columns=RFU_COLUMNS)
    output_df = pd.concat([df, rfu_df], axis=1)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    report_df = build_report(output_df)
    report_df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    manual_check_df = output_df[
        (output_df["Donor_RFU_Status"] != "ok")
        | (output_df["Donor_C2_Sub_Entry_Status"] != "found")
    ].copy()
    manual_check_df.to_csv(MANUAL_CHECK_PATH, index=False, encoding="utf-8-sig")

    print(f"Input: {input_path}")
    print(f"Wrote {OUTPUT_PATH} with shape {output_df.shape}")
    print("Donor_RFU_Status")
    print(output_df["Donor_RFU_Status"].value_counts(dropna=False).to_string())
    print("Donor_C2_Sub_Entry_Status")
    print(output_df["Donor_C2_Sub_Entry_Status"].value_counts(dropna=False).to_string())
    print("Donor_RFU_Num_Atoms by Donor_Type")
    print(
        output_df.groupby("Donor_Type")["Donor_RFU_Num_Atoms"]
        .agg(["count", "min", "median", "max"])
        .to_string()
    )
    print(f"Manual check rows: {len(manual_check_df)}")


if __name__ == "__main__":
    main()
