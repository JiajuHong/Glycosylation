#!/usr/bin/env python
"""第二层：检索可追溯的成功文献先例。

本模块不估计反应成功概率，不定义适用域，也不输出人为加权的兼容性总分。
它只报告候选条件是否有直接/结构相关的成功先例，并返回最近文献反应。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from layer1.predict_hard_feasibility import (
    canonicalize,
    extract_acceptor_target_site,
    resolve_donor_type,
)
from layer3.tokenize_conditions import (
    CATALYST_TOKEN_MAP,
    CATALYST_VOCAB,
    SOLVENT_TOKEN_MAP,
    SOLVENT_VOCAB,
    map_components,
)


INTERPRETATION = (
    "traceable precedent among successful literature reactions; identity counts and "
    "Tanimoto similarities are evidence descriptors, not reaction probabilities or "
    "applicability-domain labels"
)
CONDITION_TRANSFER_EVIDENCE_TYPES = (
    "exact_pair_exact_condition",
    "same_donor_condition",
    "same_acceptor_condition",
    "analog_condition",
    "template_only",
)
PAIR_HISTORY_TYPES = ("known_pair", "novel_pair")


def missing(value: object) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def normalize_ids(value: object) -> str:
    if missing(value):
        return ""
    values = {int(float(part)) for part in str(value).split(";") if part.strip()}
    return ";".join(map(str, sorted(values)))


def condition_ids(
    row: pd.Series,
    id_column: str,
    raw_column: str,
    token_map: dict[str, str],
    vocab: dict[str, int],
    missing_token: str,
    unknown_token: str,
) -> str:
    if id_column in row and not missing(row[id_column]):
        return normalize_ids(row[id_column])
    _, ids, _ = map_components(
        row.get(raw_column), token_map, vocab, missing_token, unknown_token
    )
    return normalize_ids(ids)


def prepare_row(row: pd.Series) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """规范化结构、目标位点和条件，供证据检索与第三层共同使用。"""
    audit: dict[str, Any] = {
        "Evidence_Retrieval_Status": "manual_review",
        "Evidence_Retrieval_Error": "",
    }
    donor_smiles, _, _ = canonicalize(row.get("Donor_Canonical_SMILES"))
    acceptor_smiles, acceptor_molecule, acceptor_mapping = canonicalize(
        row.get("Acceptor_Canonical_SMILES")
    )
    if donor_smiles is None or acceptor_smiles is None:
        audit["Evidence_Retrieval_Error"] = "invalid or missing donor/acceptor SMILES"
        return None, audit

    donor_type, donor_source = resolve_donor_type(donor_smiles, row.get("Donor_Type"))
    if donor_type is None:
        audit["Evidence_Retrieval_Error"] = donor_source
        return None, audit

    supplied_target = None
    target_source = "auto"
    for column in ("Target_O4_Index", "Acceptor_O4_Index"):
        value = row.get(column)
        if not missing(value):
            original_index = int(float(value))
            if original_index < 0 or original_index >= len(acceptor_mapping):
                audit["Evidence_Retrieval_Error"] = f"{column} is out of bounds"
                return None, audit
            supplied_target = int(acceptor_mapping[original_index])
            target_source = column
            break
    target = extract_acceptor_target_site(acceptor_smiles, supplied_target)
    if target["status"] != "ok":
        audit["Evidence_Retrieval_Error"] = (
            f"target-site extraction failed: {target['status']}"
        )
        return None, audit
    if supplied_target is None:
        target_source = str(target["selection_source"])
    if target["state"] != "free":
        audit.update(
            {
                "Evidence_Retrieval_Status": "layer1_reject",
                "Evidence_Retrieval_Error": "target O4 is structurally blocked",
                "Target_O4_State": target["state"],
            }
        )
        return None, audit

    target_index = int(target["o4_index"])
    additional_free_oh_count = sum(
        atom.GetAtomicNum() == 8
        and atom.GetFormalCharge() == 0
        and atom.GetTotalNumHs() >= 1
        and atom.GetDegree() == 1
        and atom.GetIdx() != target_index
        for atom in acceptor_molecule.GetAtoms()
    )
    solvent_ids = condition_ids(
        row,
        "Solvent_Component_IDs",
        "Solvent",
        SOLVENT_TOKEN_MAP,
        SOLVENT_VOCAB,
        "SOLV_MISSING",
        "SOLV_UNKNOWN",
    )
    catalyst_ids = condition_ids(
        row,
        "Catalyst_Component_IDs",
        "Catalyst",
        CATALYST_TOKEN_MAP,
        CATALYST_VOCAB,
        "CAT_MISSING",
        "CAT_UNKNOWN",
    )
    has_temp = int(row.get("has_temp", 0 if missing(row.get("Temp_C")) else 1))
    has_time = int(row.get("has_time", 0 if missing(row.get("Time_min")) else 1))
    prepared = {
        "Donor_Canonical_SMILES": donor_smiles,
        "Acceptor_Canonical_SMILES": acceptor_smiles,
        "Donor_Type": donor_type,
        "Solvent_Component_IDs": solvent_ids,
        "Catalyst_Component_IDs": catalyst_ids,
        "Temp_C": np.nan if not has_temp else float(row["Temp_C"]),
        "has_temp": has_temp,
        "Time_min": np.nan if not has_time else float(row["Time_min"]),
        "has_time": has_time,
        "additional_free_oh_count": additional_free_oh_count,
    }
    audit.update(
        {
            "Evidence_Retrieval_Status": "ok",
            "Canonical_Donor_SMILES": donor_smiles,
            "Canonical_Acceptor_SMILES": acceptor_smiles,
            "Resolved_Donor_Type": donor_type,
            "Donor_Type_Source": donor_source,
            "Target_O4_Index_Canonical": target_index,
            "Target_O4_Source": target_source,
            "Target_O4_State": "free",
            "Target_O4_Candidate_Count": target["candidate_count"],
            "Target_O4_Free_Candidate_Count": target["free_candidate_count"],
            "Target_O4_Blocked_Candidate_Count": target["blocked_candidate_count"],
            "additional_free_oh_count": additional_free_oh_count,
        }
    )
    return prepared, audit


def unique_reference(positives: pd.DataFrame, reference_split: str) -> pd.DataFrame:
    reference = (
        positives
        if reference_split == "all"
        else positives.loc[positives["PU_Split"] == reference_split]
    )
    key = "Full_Reaction_SHA256"
    if key not in reference:
        raise ValueError(f"positive reference lacks {key}")
    return reference.drop_duplicates(key, keep="first").reset_index(drop=True).copy()


def numeric_condition_key(value: object, present: object) -> str:
    if not int(present) or missing(value):
        return "MISSING"
    return format(float(value), ".12g")


def condition_key(row: pd.Series) -> tuple[str, str, str, str, str]:
    return (
        str(row["Donor_Type"]),
        normalize_ids(row["Solvent_Component_IDs"]),
        normalize_ids(row["Catalyst_Component_IDs"]),
        numeric_condition_key(row["Temp_C"], row["has_temp"]),
        numeric_condition_key(row["Time_min"], row["has_time"]),
    )


class Fingerprints:
    def __init__(self, radius: int = 2, fp_size: int = 2048) -> None:
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size, includeChirality=True
        )
        self.cache: dict[str, Any] = {}

    def get(self, smiles: str):
        if smiles not in self.cache:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"invalid SMILES: {smiles}")
            self.cache[smiles] = self.generator.GetFingerprint(molecule)
        return self.cache[smiles]


def condition_transfer_evidence(
    exact_pair_condition_count: int,
    same_donor_condition_count: int,
    same_acceptor_condition_count: int,
    condition_count: int,
) -> str:
    if exact_pair_condition_count:
        return "exact_pair_exact_condition"
    if same_donor_condition_count:
        return "same_donor_condition"
    if same_acceptor_condition_count:
        return "same_acceptor_condition"
    if condition_count:
        return "analog_condition"
    return "template_only"


EVIDENCE_COLUMNS = [
    "Pair_History",
    "Condition_Transfer_Evidence",
    "Exact_Pair_Condition_Precedent",
    "Exact_Pair_Condition_Count",
    "Exact_Pair_Precedent_Count",
    "Same_Donor_Condition_Count",
    "Same_Acceptor_Condition_Count",
    "Condition_Precedent_Count",
    "Nearest_Evidence_Reaction_ID",
    "Nearest_Donor_Tanimoto",
    "Nearest_Acceptor_Tanimoto",
    "Nearest_Joint_Similarity",
]


def score_prepared_queries(
    query: pd.DataFrame,
    positives: pd.DataFrame,
    *,
    reference_split: str = "all",
    top_k_neighbors: int = 5,
) -> tuple[pd.DataFrame, list[list[dict[str, Any]]], dict[str, Any]]:
    """返回身份计数和结构相似度，不输出兼容性总分或域标签。"""
    if top_k_neighbors < 1:
        raise ValueError("top_k_neighbors must be positive")
    reference = unique_reference(positives, reference_split)
    if reference.empty:
        raise ValueError("positive reference is empty")
    if query.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS), [], {
            "reference_rows_unique": len(reference),
            "reference_scope": reference_split,
        }

    reference = reference.copy()
    reference["_condition_key"] = [condition_key(row) for _, row in reference.iterrows()]
    by_condition = {
        key: np.asarray(indices, dtype=int)
        for key, indices in reference.groupby("_condition_key", sort=False).groups.items()
    }
    fingerprints = Fingerprints()
    reference_donor_fp = [
        fingerprints.get(str(value)) for value in reference["Donor_Canonical_SMILES"]
    ]
    reference_acceptor_fp = [
        fingerprints.get(str(value)) for value in reference["Acceptor_Canonical_SMILES"]
    ]

    records: list[dict[str, Any]] = []
    all_neighbors: list[list[dict[str, Any]]] = []
    for _, row in query.reset_index(drop=True).iterrows():
        donor = str(row["Donor_Canonical_SMILES"])
        acceptor = str(row["Acceptor_Canonical_SMILES"])
        key = condition_key(row)
        condition_indices = by_condition.get(key, np.asarray([], dtype=int))
        same_pair = reference["Donor_Canonical_SMILES"].eq(donor) & reference[
            "Acceptor_Canonical_SMILES"
        ].eq(acceptor)
        same_condition = reference.index.isin(condition_indices)
        exact_pair_condition_count = int((same_pair & same_condition).sum())
        exact_pair_count = int(same_pair.sum())
        same_donor_condition_count = int(
            (reference["Donor_Canonical_SMILES"].eq(donor) & same_condition).sum()
        )
        same_acceptor_condition_count = int(
            (reference["Acceptor_Canonical_SMILES"].eq(acceptor) & same_condition).sum()
        )
        condition_count = int(len(condition_indices))
        transfer_evidence = condition_transfer_evidence(
            exact_pair_condition_count,
            same_donor_condition_count,
            same_acceptor_condition_count,
            condition_count,
        )
        pair_history = "known_pair" if exact_pair_count else "novel_pair"

        # 结构相似度只在“当前条件”的成功反应中计算。若条件无先例，
        # 不用其他条件下的相似底物伪充当前条件证据。
        search_indices = condition_indices
        donor_fp = fingerprints.get(donor)
        acceptor_fp = fingerprints.get(acceptor)
        donor_sim = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                donor_fp, [reference_donor_fp[index] for index in search_indices]
            ),
            dtype=float,
        )
        acceptor_sim = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                acceptor_fp, [reference_acceptor_fp[index] for index in search_indices]
            ),
            dtype=float,
        )
        joint = np.minimum(donor_sim, acceptor_sim)
        combined = donor_sim + acceptor_sim
        order = np.lexsort((-combined, -joint))[: min(top_k_neighbors, len(search_indices))]
        neighbors: list[dict[str, Any]] = []
        for rank, local_index in enumerate(order, start=1):
            reference_index = int(search_indices[int(local_index)])
            ref = reference.iloc[reference_index]
            neighbors.append(
                {
                    "rank": rank,
                    "reaction_id": str(ref.get("Reaction_ID", ref.get("ID", reference_index))),
                    "source": str(ref.get("Source", "")),
                    "condition_match": bool(condition_count),
                    "donor_tanimoto": round(float(donor_sim[local_index]), 6),
                    "acceptor_tanimoto": round(float(acceptor_sim[local_index]), 6),
                    "joint_similarity": round(float(joint[local_index]), 6),
                    "exact_pair": bool(
                        str(ref["Donor_Canonical_SMILES"]) == donor
                        and str(ref["Acceptor_Canonical_SMILES"]) == acceptor
                    ),
                    "literature_support_count": int(
                        ref.get("Exact_Reaction_Duplicate_Count", 1)
                    ),
                }
            )
        nearest = neighbors[0] if neighbors else {}
        records.append(
            {
                "Pair_History": pair_history,
                "Condition_Transfer_Evidence": transfer_evidence,
                "Exact_Pair_Condition_Precedent": bool(exact_pair_condition_count),
                "Exact_Pair_Condition_Count": exact_pair_condition_count,
                "Exact_Pair_Precedent_Count": exact_pair_count,
                "Same_Donor_Condition_Count": same_donor_condition_count,
                "Same_Acceptor_Condition_Count": same_acceptor_condition_count,
                "Condition_Precedent_Count": condition_count,
                "Nearest_Evidence_Reaction_ID": nearest.get("reaction_id", ""),
                "Nearest_Donor_Tanimoto": nearest.get("donor_tanimoto", np.nan),
                "Nearest_Acceptor_Tanimoto": nearest.get("acceptor_tanimoto", np.nan),
                "Nearest_Joint_Similarity": nearest.get("joint_similarity", np.nan),
            }
        )
        all_neighbors.append(neighbors)
    return pd.DataFrame(records), all_neighbors, {
        "reference_rows_unique": len(reference),
        "reference_scope": reference_split,
        "fingerprint": "Morgan radius 2, 2048 bits, chirality-aware Tanimoto",
        "joint_similarity": (
            "minimum of donor and acceptor Tanimoto; no compensating weighted sum"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--positive-reference",
        type=Path,
        default=Path("data/processed/soft_compatibility_positive_1561.csv"),
    )
    parser.add_argument("--reference-split", choices=["all", "train"], default="all")
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k_neighbors < 1:
        raise ValueError("--top-k-neighbors must be positive")
    source = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"}
    if missing_columns := required - set(source.columns):
        raise ValueError(f"input is missing required columns: {sorted(missing_columns)}")
    positives = pd.read_csv(args.positive_reference, encoding="utf-8-sig")

    prepared_rows: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    audits: list[dict[str, Any]] = []
    for row_number, (_, row) in enumerate(source.iterrows()):
        prepared, audit = prepare_row(row)
        audits.append(audit)
        if prepared is not None:
            prepared_rows.append(prepared)
            valid_indices.append(row_number)

    output = source.copy()
    audit_frame = pd.DataFrame(audits, index=output.index)
    for column in audit_frame:
        output[column] = audit_frame[column]
    text_columns = {
        "Pair_History",
        "Condition_Transfer_Evidence",
        "Nearest_Evidence_Reaction_ID",
    }
    for column in EVIDENCE_COLUMNS:
        output[column] = "" if column in text_columns else np.nan
    output["Nearest_Evidence_Reactions_JSON"] = ""
    output["Literature_Evidence_Interpretation"] = INTERPRETATION

    if prepared_rows:
        scored, neighbor_lists, _ = score_prepared_queries(
            pd.DataFrame(prepared_rows),
            positives,
            reference_split=args.reference_split,
            top_k_neighbors=args.top_k_neighbors,
        )
        for query_pos, output_pos in enumerate(valid_indices):
            for column in scored.columns:
                output.loc[output_pos, column] = scored.loc[query_pos, column]
            output.loc[output_pos, "Nearest_Evidence_Reactions_JSON"] = json.dumps(
                neighbor_lists[query_pos], ensure_ascii=False, separators=(",", ":")
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "input_rows": len(output),
                "retrieved_rows": len(valid_indices),
                "condition_transfer_evidence_counts": output[
                    "Condition_Transfer_Evidence"
                ].value_counts().to_dict(),
                "pair_history_counts": output["Pair_History"].value_counts().to_dict(),
                "output": str(args.output),
                "interpretation": INTERPRETATION,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
