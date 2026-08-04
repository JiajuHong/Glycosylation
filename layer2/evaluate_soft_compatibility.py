#!/usr/bin/env python
"""第二层正式评测：复现 kNN 排序指标并输出可审计 JSON。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr

from layer2.build_soft_compatibility_u import CONDITION_COLUMNS, validate_pools
from layer2.score_soft_compatibility import (
    CompatibilityFeatures,
    DONOR_TYPES,
    WEIGHTS,
    calibration_for_type,
    loo_calibration,
    prepare_row,
    unique_reference,
)


def materialize_u(references: pd.DataFrame, positives: pd.DataFrame) -> pd.DataFrame:
    source = positives.set_index("ID", drop=False)
    records: list[dict[str, Any]] = []
    for row in references.itertuples(index=False):
        donor = source.loc[int(row.Donor_Source_ID)]
        acceptor = source.loc[int(row.Acceptor_Source_ID)]
        condition = source.loc[int(row.Condition_Source_ID)]
        record: dict[str, Any] = {
            "PU_Sample_ID": row.PU_Sample_ID,
            "PU_Class": "U",
            "PU_Pool_ID": int(row.PU_Pool_ID),
            "PU_Split": str(row.PU_Split),
            "Donor_Type": str(row.Donor_Type),
            "Donor_Canonical_SMILES": donor.Donor_Canonical_SMILES,
            "Acceptor_Canonical_SMILES": acceptor.Acceptor_Canonical_SMILES,
            "additional_free_oh_count": int(row.additional_free_oh_count),
        }
        for column in CONDITION_COLUMNS:
            record[column] = condition[column]
        records.append(record)
    return pd.DataFrame(records)


def candidate_frame(positives: pd.DataFrame, unlabeled: pd.DataFrame, split: str) -> pd.DataFrame:
    positive = positives.loc[positives["PU_Split"] == split].copy()
    unknown = unlabeled.loc[unlabeled["PU_Split"] == split].copy()
    positive["Hidden_Positive"] = 1
    unknown["Hidden_Positive"] = 0
    return pd.concat([positive, unknown], ignore_index=True, sort=False)


def nearest_scores(train: sparse.csr_matrix, query: sparse.csr_matrix) -> np.ndarray:
    return np.clip((query @ train.T).max(axis=1).toarray().ravel(), -1.0, 1.0)


def ranking_metrics(frame: pd.DataFrame, scores: np.ndarray) -> tuple[dict[str, Any], dict[str, float]]:
    labels = frame["Hidden_Positive"].to_numpy(dtype=int)
    n_positive = int(labels.sum())
    n_total = len(labels)
    # Candidate frames list P before U. Stable sorting would therefore favor P at
    # tied score boundaries. Break ties by sample-ID hash, independent of labels.
    tie_keys = np.fromiter(
        (
            int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16], 16)
            for sample_id in frame["PU_Sample_ID"]
        ),
        dtype=np.uint64,
        count=n_total,
    )
    order = np.lexsort((tie_keys, -scores))
    ranks = rankdata(-scores, method="average")
    positive_ranks = ranks[labels == 1]

    def recall_at(k: int) -> float:
        k = min(max(int(k), 1), n_total)
        return float(labels[order[:k]].sum() / n_positive)

    metrics = {
        "n_candidates": n_total,
        "n_hidden_positive": n_positive,
        "n_unlabeled": n_total - n_positive,
        "random_recall_at_n_positive": n_positive / n_total,
        "recall_at_5pct": recall_at(math.ceil(0.05 * n_total)),
        "recall_at_10pct": recall_at(math.ceil(0.10 * n_total)),
        "recall_at_20pct": recall_at(math.ceil(0.20 * n_total)),
        "recall_at_n_positive": recall_at(n_positive),
        "mean_positive_rank": float(positive_ranks.mean()),
        "median_positive_rank": float(np.median(positive_ranks)),
        "mean_positive_percentile": float(
            np.mean(1.0 - (positive_ranks - 1.0) / max(n_total - 1, 1))
        ),
    }
    positive_ids = frame.loc[labels == 1, "PU_Sample_ID"].astype(str)
    percentiles = {
        sample_id: float(1.0 - (rank - 1.0) / max(n_total - 1, 1))
        for sample_id, rank in zip(positive_ids, positive_ranks)
    }
    return metrics, percentiles


def aggregate_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    frame = pd.DataFrame([row for row in rows if row["split"] == split])
    output: dict[str, Any] = {}
    for column in (
        "recall_at_5pct",
        "recall_at_10pct",
        "recall_at_20pct",
        "recall_at_n_positive",
        "mean_positive_percentile",
    ):
        values = frame[column].to_numpy(dtype=float)
        output[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def pool_stability(percentiles: dict[int, dict[str, float]]) -> dict[str, float]:
    correlations = []
    pool_ids = sorted(percentiles)
    for left_pos, left in enumerate(pool_ids):
        for right in pool_ids[left_pos + 1 :]:
            common = sorted(set(percentiles[left]) & set(percentiles[right]))
            correlation = spearmanr(
                [percentiles[left][key] for key in common],
                [percentiles[right][key] for key in common],
            ).statistic
            correlations.append(float(correlation))
    return {
        "positive_rank_spearman_mean": float(np.mean(correlations)),
        "positive_rank_spearman_min": float(np.min(correlations)),
    }


def ablation(
    features: CompatibilityFeatures,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, float]:
    train_blocks = features.normalized_blocks(train)
    query_blocks = features.normalized_blocks(candidates)
    configs = {
        "donor_only": ((0,), (1.0,)),
        "acceptor_only": ((1,), (1.0,)),
        "condition_only": ((2,), (1.0,)),
        "donor_acceptor": ((0, 1), (0.5, 0.5)),
        "full_reaction": ((0, 1, 2), (0.4, 0.4, 0.2)),
    }
    output = {}
    for name, (indices, weights) in configs.items():
        train_matrix = sparse.hstack(
            [train_blocks[index] * math.sqrt(weight) for index, weight in zip(indices, weights)],
            format="csr",
        )
        query_matrix = sparse.hstack(
            [query_blocks[index] * math.sqrt(weight) for index, weight in zip(indices, weights)],
            format="csr",
        )
        metrics, _ = ranking_metrics(candidates, nearest_scores(train_matrix, query_matrix))
        output[f"{name}_recall_at_n_positive"] = metrics["recall_at_n_positive"]
    return output


def calibration_summary(positives: pd.DataFrame) -> dict[str, Any]:
    reference = unique_reference(positives, "all")
    features = CompatibilityFeatures(reference)
    matrix = features.weighted_matrix(features.normalized_blocks(reference))
    overall, by_type = loo_calibration(reference, matrix)
    summary: dict[str, Any] = {}
    for donor_type, values in [("overall", overall), *by_type.items()]:
        summary[donor_type] = {
            "n": len(values),
            "q05": float(np.quantile(values, 0.05)),
            "q10": float(np.quantile(values, 0.10)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.50)),
        }
    summary["domain_rule"] = (
        "below donor-type q05 = out_of_domain; q05 to q10 = borderline; "
        "at or above q10 = in_domain"
    )
    return summary


def split_summary(positives: pd.DataFrame) -> dict[str, Any]:
    counts = {}
    for split in ("train", "val", "test"):
        frame = positives.loc[positives["PU_Split"] == split]
        counts[split] = {
            "rows": len(frame),
            "unique_full_reactions": int(frame["Full_Reaction_SHA256"].nunique()),
            "unique_pairs": int(frame[["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"]].drop_duplicates().shape[0]),
        }
    sets = {
        split: {
            "donor": set(frame["Donor_Canonical_SMILES"]),
            "acceptor": set(frame["Acceptor_Canonical_SMILES"]),
            "pair": set(zip(frame["Donor_Canonical_SMILES"], frame["Acceptor_Canonical_SMILES"])),
        }
        for split, frame in positives.groupby("PU_Split")
    }
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}-{right}"] = {
            key: len(sets[left][key] & sets[right][key]) for key in ("donor", "acceptor", "pair")
        }
    return {"counts": counts, "cross_split_overlap": overlaps}


def stress_summary(
    stress_path: Path | None,
    positives: pd.DataFrame,
) -> dict[str, Any] | None:
    if stress_path is None or not stress_path.exists():
        return None
    stress = pd.read_csv(stress_path, encoding="utf-8-sig")
    prepared_rows = []
    statuses = []
    for _, row in stress.iterrows():
        prepared, audit = prepare_row(row)
        statuses.append(audit["Soft_Score_Status"])
        if prepared is not None:
            prepared_rows.append(prepared)
    result: dict[str, Any] = {
        "rows": len(stress),
        "layer1_reject": statuses.count("layer1_reject"),
        "manual_review": statuses.count("manual_review"),
        "scored": len(prepared_rows),
    }
    if prepared_rows:
        reference = unique_reference(positives, "all")
        features = CompatibilityFeatures(reference)
        reference_matrix = features.weighted_matrix(features.normalized_blocks(reference))
        overall, by_type = loo_calibration(reference, reference_matrix)
        query = pd.DataFrame(prepared_rows)
        scores = nearest_scores(
            reference_matrix, features.weighted_matrix(features.normalized_blocks(query))
        )
        domain_counts = {"in_domain": 0, "borderline": 0, "out_of_domain": 0}
        for score, donor_type in zip(scores, query["Donor_Type"]):
            _, q05, q10 = calibration_for_type(str(donor_type), overall, by_type)
            status = "in_domain" if score >= q10 else "borderline" if score >= q05 else "out_of_domain"
            domain_counts[status] += 1
        result["domain_counts"] = domain_counts
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-reference", type=Path,
        default=Path("data/processed/soft_compatibility_positive_1561.csv"),
    )
    parser.add_argument(
        "--unlabeled", type=Path,
        default=Path("data/processed/soft_compatibility_u_candidates.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/soft_compatibility_knn_audit.json"),
    )
    parser.add_argument(
        "--stress-test", type=Path,
        default=Path("data/processed/hard_feasibility_stress_test.csv"),
    )
    parser.add_argument("--pool-count", type=int, default=5)
    parser.add_argument("--u-per-positive", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    positives = pd.read_csv(args.positive_reference, encoding="utf-8-sig")
    references = pd.read_csv(args.unlabeled, encoding="utf-8-sig")
    qa = validate_pools(positives, references, args.pool_count, args.u_per_positive)
    if not qa["overall_pass"]:
        raise RuntimeError("U pool QA failed before evaluation")

    train = unique_reference(positives, "train")
    features = CompatibilityFeatures(train)
    train_matrix = features.weighted_matrix(features.normalized_blocks(train))
    metric_rows: list[dict[str, Any]] = []
    test_percentiles: dict[int, dict[str, float]] = {}
    pool0_test_candidates = None
    for pool_id in range(args.pool_count):
        pool_u = materialize_u(
            references.loc[references["PU_Pool_ID"] == pool_id], positives
        )
        for split in ("val", "test"):
            candidates = candidate_frame(positives, pool_u, split)
            query_matrix = features.weighted_matrix(features.normalized_blocks(candidates))
            metrics, percentiles = ranking_metrics(
                candidates, nearest_scores(train_matrix, query_matrix)
            )
            metric_rows.append({"pool_id": pool_id, "split": split, **metrics})
            if split == "test":
                test_percentiles[pool_id] = percentiles
                if pool_id == 0:
                    pool0_test_candidates = candidates

    if pool0_test_candidates is None:
        raise RuntimeError("pool 0 test candidates were not generated")
    audit = {
        "layer": 2,
        "task_name": "target-4-O glycosylation literature-support and applicability-domain assessment",
        "model": "weighted nearest-neighbour retrieval over literature successful reactions",
        "probability_warning": "The score is not a calibrated reaction probability and is not a hard feasibility gate.",
        "reference_data": {
            "successful_reaction_rows": len(positives),
            "unique_full_reactions": int(positives["Full_Reaction_SHA256"].nunique()),
            "exact_duplicate_rows": int(len(positives) - positives["Full_Reaction_SHA256"].nunique()),
            "unique_donor_acceptor_pairs": int(positives[["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"]].drop_duplicates().shape[0]),
            "additional_free_oh_reaction_rows": int(positives["additional_free_oh_count"].gt(0).sum()),
        },
        "split": split_summary(positives),
        "features": {
            "donor": "Morgan radius 2, 1024 bits, cosine similarity",
            "acceptor": "Morgan radius 2, 1024 bits, cosine similarity",
            "condition_context": "solvent, catalyst/activator, temperature, time, donor type and additional free-OH count",
            "weights": WEIGHTS,
        },
        "u_generation": {
            "meaning": "unobserved/unknown controlled recombinations, not failed reactions",
            "base_seed": args.seed,
            "pool_seed_formula": "base_seed + pool_id * 1009",
            "pool_count": args.pool_count,
            "u_per_positive": args.u_per_positive,
            "qa": qa,
        },
        "leave_one_out_domain_calibration": calibration_summary(positives),
        "validation_metrics_by_pool": metric_rows,
        "validation_summary": {
            "validation": aggregate_metrics(metric_rows, "val"),
            "test": aggregate_metrics(metric_rows, "test"),
            "test_pool_stability": pool_stability(test_percentiles),
        },
        "pool0_test_ablation": ablation(features, train, pool0_test_candidates),
        "stress_interface": stress_summary(args.stress_test, positives),
        "validated_scope": "donor-acceptor pair disjoint; individual donor or acceptor components may recur across splits",
        "not_yet_validated": [
            "donor-cold",
            "acceptor-cold",
            "double-cold",
            "prospective experimental success/failure",
        ],
        "production_decision": "Use as a soft ranking and evidence-retrieval layer after layer-one structural filtering; never reject solely because of a low score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "test_metrics": audit["validation_summary"]["test"],
                "u_qa_pass": qa["overall_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
