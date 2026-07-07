from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem


INPUT_PATH = Path("donor_rfu_extracted.csv")
OUTPUT_PATH = Path("acceptor_prior_check.csv")
SUMMARY_PATH = Path("acceptor_prior_summary.csv")
MANUAL_PATH = Path("acceptor_prior_manual_check.csv")


def same_ring(mol: Chem.Mol, a: int, b: int) -> bool:
    return any(a in ring and b in ring for ring in mol.GetRingInfo().AtomRings())


def is_free_oh(mol: Chem.Mol, atom_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 8:
        return False
    if atom.GetFormalCharge() != 0:
        return False
    if atom.GetTotalNumHs() < 1:
        return False
    if atom.GetDegree() != 1:
        return False
    return True


def has_exocyclic_n(mol: Chem.Mol, carbon_idx: int, main_ring: set[int]) -> bool:
    for neighbor in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if neighbor.GetAtomicNum() != 7:
            continue
        if idx in main_ring:
            continue
        return True
    return False


def free_oh_neighbors(mol: Chem.Mol, carbon_idx: int) -> list[int]:
    hits: list[int] = []
    for neighbor in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if is_free_oh(mol, idx):
            hits.append(idx)
    return sorted(hits)


def ring_order_from_o(mol: Chem.Mol, ring: tuple[int, ...], oxygen_idx: int) -> list[list[int]]:
    ring_set = set(ring)
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    carbon_neighbors = [
        neighbor.GetIdx()
        for neighbor in oxygen.GetNeighbors()
        if neighbor.GetIdx() in ring_set and neighbor.GetAtomicNum() == 6
    ]
    orders: list[list[int]] = []
    for first in carbon_neighbors:
        order = [oxygen_idx, first]
        previous = oxygen_idx
        current = first
        while len(order) < len(ring):
            next_candidates = [
                neighbor.GetIdx()
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors()
                if neighbor.GetIdx() in ring_set
                and neighbor.GetIdx() != previous
                and neighbor.GetIdx() not in order
            ]
            if not next_candidates:
                break
            previous, current = current, next_candidates[0]
            order.append(current)
        if len(order) == len(ring):
            orders.append(order)
    return orders


def pyranose_like_rings(mol: Chem.Mol) -> list[tuple[int, ...]]:
    rings: list[tuple[int, ...]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        symbols = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in ring]
        if symbols.count(8) == 1 and symbols.count(6) == 5:
            rings.append(tuple(ring))
    return rings


def check_acceptor(smiles: str) -> dict[str, object]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "Acceptor_Prior_Status": "mol_parse_fail",
            "Acceptor_Prior_Num_Candidates": 0,
            "Acceptor_Prior_Candidates_JSON": "[]",
            "Acceptor_Prior_Ring_Count": 0,
        }

    candidates: list[dict[str, object]] = []
    rings = pyranose_like_rings(mol)
    for ring in rings:
        oxygen_atoms = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8]
        if len(oxygen_atoms) != 1:
            continue
        oxygen_idx = oxygen_atoms[0]
        for order in ring_order_from_o(mol, ring, oxygen_idx):
            # order = [O5, C1, C2, C3, C4, C5] in one direction around the ring.
            if len(order) != 6:
                continue
            c1, c2, c3, c4, c5 = order[1], order[2], order[3], order[4], order[5]
            n2 = has_exocyclic_n(mol, c2, set(ring))
            oh4 = free_oh_neighbors(mol, c4)
            if n2 and oh4:
                candidates.append(
                    {
                        "O5": oxygen_idx,
                        "C1": c1,
                        "C2": c2,
                        "C3": c3,
                        "C4": c4,
                        "C5": c5,
                        "C2_has_N": True,
                        "C4_OH": oh4,
                        "ring": list(ring),
                    }
                )

    unique_candidates: list[dict[str, object]] = []
    seen = set()
    for candidate in candidates:
        key = (candidate["O5"], candidate["C1"], candidate["C2"], candidate["C4"], tuple(candidate["C4_OH"]))
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    if len(unique_candidates) == 0:
        status = "no_c2n_c4oh_candidate"
    elif len(unique_candidates) == 1:
        status = "ok_unique"
    else:
        status = "multiple_candidates"

    return {
        "Acceptor_Prior_Status": status,
        "Acceptor_Prior_Num_Candidates": len(unique_candidates),
        "Acceptor_Prior_Candidates_JSON": json.dumps(unique_candidates, separators=(",", ":")),
        "Acceptor_Prior_Ring_Count": len(rings),
    }


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    results = [check_acceptor(smiles) for smiles in df["Acceptor_Canonical_SMILES"]]
    result_df = pd.DataFrame(results)
    out = pd.concat([df, result_df], axis=1)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary_rows = []
    for key, count in Counter(out["Acceptor_Prior_Status"]).items():
        summary_rows.append({"metric": "Acceptor_Prior_Status", "key": key, "count": count})
    for key, count in Counter(out["Acceptor_Prior_Ring_Count"]).items():
        summary_rows.append({"metric": "Acceptor_Prior_Ring_Count", "key": key, "count": count})
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    manual = out[out["Acceptor_Prior_Status"] != "ok_unique"].copy()
    manual.to_csv(MANUAL_PATH, index=False, encoding="utf-8-sig")

    print("Status counts")
    print(out["Acceptor_Prior_Status"].value_counts().to_string())
    print("Ring count")
    print(out["Acceptor_Prior_Ring_Count"].value_counts().sort_index().to_string())
    print(f"Manual check rows: {len(manual)}")


if __name__ == "__main__":
    main()
