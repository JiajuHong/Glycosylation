#!/usr/bin/env python
"""第二层推理入口：计算文献支持度和适用域评分。

该分数是与成功反应的加权近邻相似度，不是校准后的反应概率，也不能作为硬门控。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.preprocessing import normalize

from layer1.predict_hard_feasibility import canonicalize, extract_acceptor_target_site, resolve_donor_type
from layer3.tokenize_conditions import (
    CATALYST_TOKEN_MAP,
    CATALYST_VOCAB,
    SOLVENT_TOKEN_MAP,
    SOLVENT_VOCAB,
    map_components,
)


DONOR_TYPES = ("thioglycoside", "trichloroacetimidate", "trifluoroacetimidate")
WEIGHTS = {"donor": 0.4, "acceptor": 0.4, "context": 0.2}
INTERPRETATION = (
    "relative similarity to literature successful reactions; this is an applicability-domain "
    "and literature-support score, not a reaction probability"
)


def missing(value: object) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def parse_ids(value: object) -> list[int]:
    if missing(value):
        return []
    return [int(float(part)) for part in str(value).split(";") if part.strip()]


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
        return str(row[id_column])
    _, ids, _ = map_components(
        row.get(raw_column), token_map, vocab, missing_token, unknown_token
    )
    return ids


class CompatibilityFeatures:
    """Morgan donor/acceptor blocks plus reaction-context features."""

    def __init__(self, reference: pd.DataFrame, n_bits: int = 1024) -> None:
        self.n_bits = n_bits
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        self.fingerprint_cache: dict[str, np.ndarray] = {}
        self.n_solvent = max(SOLVENT_VOCAB.values()) + 1
        self.n_catalyst = max(CATALYST_VOCAB.values()) + 1
        temp = pd.to_numeric(reference.loc[reference["has_temp"] == 1, "Temp_C"], errors="coerce").dropna()
        time = (
            pd.to_numeric(reference.loc[reference["has_time"] == 1, "Time_min"], errors="coerce")
            .dropna()
            .map(np.log1p)
        )
        self.temp_mean = float(temp.mean()) if len(temp) else 0.0
        self.temp_std = float(temp.std(ddof=0) or 1.0) if len(temp) else 1.0
        self.time_mean = float(time.mean()) if len(time) else 0.0
        self.time_std = float(time.std(ddof=0) or 1.0) if len(time) else 1.0

    def fingerprint(self, smiles: str) -> np.ndarray:
        if smiles not in self.fingerprint_cache:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"invalid SMILES: {smiles}")
            bit_vector = self.generator.GetFingerprint(molecule)
            array = np.zeros(self.n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bit_vector, array)
            self.fingerprint_cache[smiles] = array
        return self.fingerprint_cache[smiles]

    def fingerprint_matrix(self, values: pd.Series) -> sparse.csr_matrix:
        return sparse.csr_matrix(np.vstack([self.fingerprint(str(value)) for value in values]))

    def context_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        # solvent + catalyst + donor type + temperature/time + free-OH multiplicity
        width = self.n_solvent + self.n_catalyst + len(DONOR_TYPES) + 4 + 3
        matrix = np.zeros((len(frame), width), dtype=np.float32)
        donor_offset = self.n_solvent + self.n_catalyst
        numeric_offset = donor_offset + len(DONOR_TYPES)
        oh_offset = numeric_offset + 4
        donor_position = {name: index for index, name in enumerate(DONOR_TYPES)}
        for row_pos, row in enumerate(frame.itertuples(index=False)):
            for token in parse_ids(row.Solvent_Component_IDs):
                if 0 <= token < self.n_solvent:
                    matrix[row_pos, token] = 1.0
            for token in parse_ids(row.Catalyst_Component_IDs):
                if 0 <= token < self.n_catalyst:
                    matrix[row_pos, self.n_solvent + token] = 1.0
            donor_type = str(row.Donor_Type)
            if donor_type in donor_position:
                matrix[row_pos, donor_offset + donor_position[donor_type]] = 1.0
            has_temp = float(row.has_temp)
            has_time = float(row.has_time)
            matrix[row_pos, numeric_offset] = has_temp
            matrix[row_pos, numeric_offset + 1] = (
                (float(row.Temp_C) - self.temp_mean) / self.temp_std if has_temp else 0.0
            )
            matrix[row_pos, numeric_offset + 2] = has_time
            matrix[row_pos, numeric_offset + 3] = (
                (math.log1p(float(row.Time_min)) - self.time_mean) / self.time_std
                if has_time else 0.0
            )
            matrix[row_pos, oh_offset + min(int(row.additional_free_oh_count), 2)] = 1.0
        return sparse.csr_matrix(matrix)

    def normalized_blocks(
        self, frame: pd.DataFrame
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        return (
            normalize(self.fingerprint_matrix(frame["Donor_Canonical_SMILES"])),
            normalize(self.fingerprint_matrix(frame["Acceptor_Canonical_SMILES"])),
            normalize(self.context_matrix(frame)),
        )

    @staticmethod
    def weighted_matrix(
        blocks: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]
    ) -> sparse.csr_matrix:
        donor, acceptor, context = blocks
        return sparse.hstack(
            [
                donor * math.sqrt(WEIGHTS["donor"]),
                acceptor * math.sqrt(WEIGHTS["acceptor"]),
                context * math.sqrt(WEIGHTS["context"]),
            ],
            format="csr",
        )


def prepare_row(row: pd.Series) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit: dict[str, Any] = {"Soft_Score_Status": "manual_review", "Soft_Score_Error": ""}
    donor_smiles, _, _ = canonicalize(row.get("Donor_Canonical_SMILES"))
    acceptor_smiles, acceptor_molecule, acceptor_mapping = canonicalize(
        row.get("Acceptor_Canonical_SMILES")
    )
    if donor_smiles is None or acceptor_smiles is None:
        audit["Soft_Score_Error"] = "invalid or missing donor/acceptor SMILES"
        return None, audit

    donor_type, donor_source = resolve_donor_type(donor_smiles, row.get("Donor_Type"))
    if donor_type is None:
        audit["Soft_Score_Error"] = donor_source
        return None, audit

    supplied_target = None
    target_source = "auto"
    for column in ("Target_O4_Index", "Acceptor_O4_Index"):
        value = row.get(column)
        if not missing(value):
            original_index = int(float(value))
            if original_index < 0 or original_index >= len(acceptor_mapping):
                audit["Soft_Score_Error"] = f"{column} is out of bounds"
                return None, audit
            supplied_target = int(acceptor_mapping[original_index])
            target_source = column
            break
    target = extract_acceptor_target_site(acceptor_smiles, supplied_target)
    if target["status"] != "ok":
        audit["Soft_Score_Error"] = f"target-site extraction failed: {target['status']}"
        return None, audit
    if supplied_target is None:
        target_source = str(target["selection_source"])
    if target["state"] != "free":
        audit.update(
            {
                "Soft_Score_Status": "layer1_reject",
                "Soft_Score_Error": "target O4 is structurally blocked",
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
        row, "Solvent_Component_IDs", "Solvent", SOLVENT_TOKEN_MAP, SOLVENT_VOCAB,
        "SOLV_MISSING", "SOLV_UNKNOWN",
    )
    catalyst_ids = condition_ids(
        row, "Catalyst_Component_IDs", "Catalyst", CATALYST_TOKEN_MAP, CATALYST_VOCAB,
        "CAT_MISSING", "CAT_UNKNOWN",
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
            "Soft_Score_Status": "ok",
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
    reference = positives if reference_split == "all" else positives.loc[positives["PU_Split"] == reference_split]
    key = "Full_Reaction_SHA256"
    if key not in reference:
        raise ValueError(f"positive reference lacks {key}")
    reference = reference.drop_duplicates(key, keep="first").reset_index(drop=True).copy()
    return reference


def loo_calibration(
    reference: pd.DataFrame, matrix: sparse.csr_matrix
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    similarities = np.clip((matrix @ matrix.T).toarray(), -1.0, 1.0)
    np.fill_diagonal(similarities, -np.inf)
    nearest = similarities.max(axis=1)
    by_type = {
        donor_type: nearest[reference["Donor_Type"].eq(donor_type).to_numpy()]
        for donor_type in DONOR_TYPES
    }
    return nearest, by_type


def calibration_for_type(
    donor_type: str, overall: np.ndarray, by_type: dict[str, np.ndarray]
) -> tuple[np.ndarray, float, float]:
    distribution = by_type.get(donor_type, np.array([]))
    if len(distribution) < 30:
        distribution = overall
    return distribution, float(np.quantile(distribution, 0.05)), float(np.quantile(distribution, 0.10))


def component_similarity(query_row: sparse.csr_matrix, reference_row: sparse.csr_matrix) -> float:
    return float(np.clip(query_row.multiply(reference_row).sum(), -1.0, 1.0))


def score_prepared_queries(
    query: pd.DataFrame,
    positives: pd.DataFrame,
    *,
    reference_split: str = "all",
    top_k_neighbors: int = 5,
) -> tuple[pd.DataFrame, list[list[dict[str, Any]]], dict[str, Any]]:
    """给已经标准化的候选计算第二层分数。

    这个函数是统一推理接口与命令行脚本共用的唯一实现。输入行必须已经由
    :func:`prepare_row` 验证，返回的分数只表示与成功文献反应的相似程度。
    """
    if top_k_neighbors < 1:
        raise ValueError("top_k_neighbors must be positive")
    reference = unique_reference(positives, reference_split)
    if reference.empty:
        raise ValueError("positive reference is empty")

    features = CompatibilityFeatures(reference)
    reference_blocks = features.normalized_blocks(reference)
    reference_matrix = features.weighted_matrix(reference_blocks)
    loo_overall, loo_by_type = loo_calibration(reference, reference_matrix)

    columns = [
        "Soft_Compatibility_Score",
        "Soft_TopK_Mean_Score",
        "Soft_Support_Percentile",
        "Soft_Domain_Q05",
        "Soft_Domain_Q10",
        "Soft_Domain_Status",
    ]
    if query.empty:
        return pd.DataFrame(columns=columns), [], {
            "reference_rows_unique": len(reference),
            "reference_scope": reference_split,
            "weights": WEIGHTS,
        }

    query = query.reset_index(drop=True)
    query_blocks = features.normalized_blocks(query)
    query_matrix = features.weighted_matrix(query_blocks)
    similarities = np.clip((query_matrix @ reference_matrix.T).toarray(), -1.0, 1.0)
    top_k = min(top_k_neighbors, len(reference))
    neighbor_indices = np.argsort(-similarities, axis=1, kind="stable")[:, :top_k]
    neighbor_values = np.take_along_axis(similarities, neighbor_indices, axis=1)

    records: list[dict[str, Any]] = []
    all_neighbors: list[list[dict[str, Any]]] = []
    for query_pos in range(len(query)):
        score = float(neighbor_values[query_pos, 0])
        donor_type = str(query.iloc[query_pos]["Donor_Type"])
        distribution, q05, q10 = calibration_for_type(donor_type, loo_overall, loo_by_type)
        percentile = 100.0 * float(np.mean(distribution <= score))
        domain_status = (
            "in_domain" if score >= q10 else "borderline" if score >= q05 else "out_of_domain"
        )
        neighbors: list[dict[str, Any]] = []
        for rank, reference_pos in enumerate(neighbor_indices[query_pos], start=1):
            ref = reference.iloc[int(reference_pos)]
            neighbors.append(
                {
                    "rank": rank,
                    "reaction_id": str(ref.get("Reaction_ID", ref.get("ID", reference_pos))),
                    "source": str(ref.get("Source", "")),
                    "total_similarity": round(float(similarities[query_pos, reference_pos]), 6),
                    "donor_similarity": round(
                        component_similarity(
                            query_blocks[0][query_pos], reference_blocks[0][reference_pos]
                        ),
                        6,
                    ),
                    "acceptor_similarity": round(
                        component_similarity(
                            query_blocks[1][query_pos], reference_blocks[1][reference_pos]
                        ),
                        6,
                    ),
                    "condition_context_similarity": round(
                        component_similarity(
                            query_blocks[2][query_pos], reference_blocks[2][reference_pos]
                        ),
                        6,
                    ),
                    "literature_support_count": int(ref.get("Exact_Reaction_Duplicate_Count", 1)),
                }
            )
        records.append(
            {
                "Soft_Compatibility_Score": score,
                "Soft_TopK_Mean_Score": float(neighbor_values[query_pos].mean()),
                "Soft_Support_Percentile": percentile,
                "Soft_Domain_Q05": q05,
                "Soft_Domain_Q10": q10,
                "Soft_Domain_Status": domain_status,
            }
        )
        all_neighbors.append(neighbors)
    return pd.DataFrame(records), all_neighbors, {
        "reference_rows_unique": len(reference),
        "reference_scope": reference_split,
        "weights": WEIGHTS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--positive-reference", type=Path,
        default=Path("data/processed/soft_compatibility_positive_1561.csv"),
    )
    parser.add_argument("--reference-split", choices=["all", "train"], default="all")
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    args = parser.parse_args()
    if args.top_k_neighbors < 1:
        parser.error("--top-k-neighbors must be positive")

    source = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"}
    if missing_columns := required - set(source.columns):
        parser.error(f"input is missing required columns: {sorted(missing_columns)}")
    positives = pd.read_csv(args.positive_reference, encoding="utf-8-sig")
    reference = unique_reference(positives, args.reference_split)

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
    numeric_columns = [
        "Soft_Compatibility_Score", "Soft_TopK_Mean_Score", "Soft_Support_Percentile",
        "Soft_Domain_Q05", "Soft_Domain_Q10",
    ]
    for column in numeric_columns:
        output[column] = np.nan
    output["Soft_Domain_Status"] = "not_scored"
    output["Nearest_Success_Reactions_JSON"] = ""
    output["Soft_Score_Interpretation"] = INTERPRETATION

    if prepared_rows:
        query = pd.DataFrame(prepared_rows)
        scored, neighbor_lists, _ = score_prepared_queries(
            query,
            positives,
            reference_split=args.reference_split,
            top_k_neighbors=args.top_k_neighbors,
        )
        for query_pos, output_pos in enumerate(valid_indices):
            for column in scored.columns:
                output.loc[output_pos, column] = scored.loc[query_pos, column]
            output.loc[output_pos, "Nearest_Success_Reactions_JSON"] = json.dumps(
                neighbor_lists[query_pos], ensure_ascii=False, separators=(",", ":")
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "input_rows": len(output),
                "scored_rows": len(valid_indices),
                "layer1_reject_rows": int((output["Soft_Score_Status"] == "layer1_reject").sum()),
                "manual_review_rows": int((output["Soft_Score_Status"] == "manual_review").sum()),
                "domain_counts": output["Soft_Domain_Status"].value_counts().to_dict(),
                "reference_rows_raw": len(positives),
                "reference_rows_unique": len(reference),
                "reference_scope": args.reference_split,
                "weights": WEIGHTS,
                "output": str(args.output),
                "warning": INTERPRETATION,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
