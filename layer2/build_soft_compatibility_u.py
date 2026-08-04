#!/usr/bin/env python
"""第二层数据构造：确定性生成用于排序评测的未标注反应池。

U 表示未观察/未知，不表示实验失败；候选仅在同类供体的成功反应组件间重组。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
CONDITION_COLUMNS = (
    "Solvent_Component_IDs",
    "Catalyst_Component_IDs",
    "Temp_C",
    "has_temp",
    "Time_min",
    "has_time",
)
OUTPUT_COLUMNS = (
    "PU_Sample_ID",
    "PU_Pool_ID",
    "PU_Split",
    "Donor_Type",
    "Donor_Source_ID",
    "Acceptor_Source_ID",
    "Condition_Source_ID",
    "additional_free_oh_count",
    "Pair_SHA256",
    "Condition_SHA256",
    "Full_Reaction_SHA256",
)


def normalized(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    return str(value)


def stable_hash(*values: object) -> str:
    payload = "\x1f".join(normalized(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def condition_key(row: pd.Series | dict[str, Any]) -> tuple[str, ...]:
    return tuple(normalized(row[column]) for column in CONDITION_COLUMNS)


def full_key(donor: str, acceptor: str, condition: tuple[str, ...]) -> tuple[str, ...]:
    return donor, acceptor, *condition


def generate_pool(
    positives: pd.DataFrame,
    pool_id: int,
    u_per_positive: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    known_pairs = set(
        zip(positives["Donor_Canonical_SMILES"], positives["Acceptor_Canonical_SMILES"])
    )
    known_full = {
        full_key(
            str(row.Donor_Canonical_SMILES),
            str(row.Acceptor_Canonical_SMILES),
            condition_key(row._asdict()),
        )
        for row in positives.itertuples(index=False)
    }
    used_pairs: set[tuple[str, str]] = set()
    used_full: set[tuple[str, ...]] = set()
    records: list[dict[str, Any]] = []

    for split in SPLITS:
        split_data = positives.loc[positives["PU_Split"] == split]
        target_count = len(split_data) * u_per_positive
        by_type = {
            donor_type: group.reset_index(drop=True)
            for donor_type, group in split_data.groupby("Donor_Type")
        }
        type_names = sorted(by_type)
        probabilities = np.asarray([len(by_type[name]) for name in type_names], dtype=float)
        probabilities /= probabilities.sum()
        accepted = 0
        attempts = 0
        max_attempts = max(target_count * 200, 10_000)
        while accepted < target_count and attempts < max_attempts:
            attempts += 1
            donor_type = str(rng.choice(type_names, p=probabilities))
            library = by_type[donor_type]
            donor_source = library.iloc[int(rng.integers(len(library)))]
            acceptor_source = library.iloc[int(rng.integers(len(library)))]
            condition_source = library.iloc[int(rng.integers(len(library)))]
            donor = str(donor_source["Donor_Canonical_SMILES"])
            acceptor = str(acceptor_source["Acceptor_Canonical_SMILES"])
            condition = condition_key(condition_source)
            pair = donor, acceptor
            reaction = full_key(donor, acceptor, condition)
            if (
                pair in known_pairs
                or reaction in known_full
                or pair in used_pairs
                or reaction in used_full
            ):
                continue
            used_pairs.add(pair)
            used_full.add(reaction)
            digest = stable_hash(pool_id, split, donor, acceptor, *condition)
            records.append(
                {
                    "PU_Sample_ID": f"U{pool_id}-{split}-{digest[:16]}",
                    "PU_Pool_ID": pool_id,
                    "PU_Split": split,
                    "Donor_Type": donor_type,
                    "Donor_Source_ID": int(donor_source["ID"]),
                    "Acceptor_Source_ID": int(acceptor_source["ID"]),
                    "Condition_Source_ID": int(condition_source["ID"]),
                    "additional_free_oh_count": int(
                        acceptor_source["additional_free_oh_count"]
                    ),
                    "Pair_SHA256": stable_hash(donor, acceptor),
                    "Condition_SHA256": stable_hash(*condition),
                    "Full_Reaction_SHA256": stable_hash(donor, acceptor, *condition),
                }
            )
            accepted += 1
        if accepted != target_count:
            raise RuntimeError(
                f"pool {pool_id} split {split}: generated {accepted}/{target_count} "
                f"after {attempts} attempts"
            )
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def validate_pools(
    positives: pd.DataFrame,
    unlabeled: pd.DataFrame,
    pool_count: int,
    u_per_positive: int,
) -> dict[str, Any]:
    source = positives.set_index("ID", drop=False)
    known_pairs = set(
        zip(positives["Donor_Canonical_SMILES"], positives["Acceptor_Canonical_SMILES"])
    )
    expected = (positives["PU_Split"].value_counts() * u_per_positive).to_dict()
    checks: dict[str, bool] = {
        "unique_u_sample_ids": not unlabeled["PU_Sample_ID"].duplicated().any(),
        "all_source_ids_exist": set(unlabeled["Donor_Source_ID"]).issubset(source.index)
        and set(unlabeled["Acceptor_Source_ID"]).issubset(source.index)
        and set(unlabeled["Condition_Source_ID"]).issubset(source.index),
    }
    summaries: dict[str, Any] = {}
    for pool_id in range(pool_count):
        pool = unlabeled.loc[unlabeled["PU_Pool_ID"] == pool_id]
        pairs = [
            (
                str(source.loc[int(row.Donor_Source_ID), "Donor_Canonical_SMILES"]),
                str(source.loc[int(row.Acceptor_Source_ID), "Acceptor_Canonical_SMILES"]),
            )
            for row in pool.itertuples(index=False)
        ]
        type_valid = all(
            str(source.loc[int(row.Donor_Source_ID), "Donor_Type"])
            == str(source.loc[int(row.Acceptor_Source_ID), "Donor_Type"])
            == str(source.loc[int(row.Condition_Source_ID), "Donor_Type"])
            == str(row.Donor_Type)
            for row in pool.itertuples(index=False)
        )
        split_counts = (
            pool["PU_Split"].value_counts().reindex(SPLITS).fillna(0).astype(int).to_dict()
        )
        split_pair_sets = {
            split: set(pool.loc[pool["PU_Split"] == split, "Pair_SHA256"])
            for split in SPLITS
        }
        overlap = {
            f"{left}-{right}": len(split_pair_sets[left] & split_pair_sets[right])
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        }
        checks[f"pool_{pool_id}_expected_counts"] = split_counts == {
            split: int(expected[split]) for split in SPLITS
        }
        checks[f"pool_{pool_id}_unique_pairs"] = not pool["Pair_SHA256"].duplicated().any()
        checks[f"pool_{pool_id}_unique_full_reactions"] = not pool[
            "Full_Reaction_SHA256"
        ].duplicated().any()
        checks[f"pool_{pool_id}_no_known_positive_pairs"] = not any(
            pair in known_pairs for pair in pairs
        )
        checks[f"pool_{pool_id}_same_donor_type_sources"] = type_valid
        checks[f"pool_{pool_id}_no_cross_split_pair_overlap"] = not any(overlap.values())
        summaries[str(pool_id)] = {
            "rows": len(pool),
            "split_counts": split_counts,
            "donor_type_counts": pool["Donor_Type"].value_counts().to_dict(),
            "cross_split_pair_overlap": overlap,
        }
    return {
        "pool_count": pool_count,
        "u_per_positive": u_per_positive,
        "rows": len(unlabeled),
        "checks": checks,
        "pool_summaries": summaries,
        "overall_pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-reference",
        type=Path,
        default=Path("data/processed/soft_compatibility_positive_1561.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/soft_compatibility_u_candidates.csv"),
    )
    parser.add_argument("--pool-count", type=int, default=5)
    parser.add_argument("--u-per-positive", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    positives = pd.read_csv(args.positive_reference, encoding="utf-8-sig")
    required = {
        "ID",
        "PU_Split",
        "Donor_Type",
        "Donor_Canonical_SMILES",
        "Acceptor_Canonical_SMILES",
        "additional_free_oh_count",
        *CONDITION_COLUMNS,
    }
    if missing := required - set(positives.columns):
        parser.error(f"positive reference is missing columns: {sorted(missing)}")
    pools = [
        generate_pool(
            positives,
            pool_id=pool_id,
            u_per_positive=args.u_per_positive,
            seed=args.seed + pool_id * 1009,
        )
        for pool_id in range(args.pool_count)
    ]
    unlabeled = pd.concat(pools, ignore_index=True)
    qa = validate_pools(positives, unlabeled, args.pool_count, args.u_per_positive)
    if not qa["overall_pass"]:
        raise RuntimeError("U pool QA failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    unlabeled.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seed": args.seed,
                **qa,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
