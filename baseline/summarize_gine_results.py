#!/usr/bin/env python
"""基线与模型结果汇总：统计多型号、多种子的均值和波动。"""

from __future__ import annotations

import argparse

import pandas as pd


METRIC_COLUMNS = [
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_auroc",
    "test_auprc",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize GINE experiment results.")
    parser.add_argument("--results-csv", default="results/gine_results_pair.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    if "encoder_type" not in df.columns:
        df["encoder_type"] = "gine"
    else:
        df["encoder_type"] = df["encoder_type"].fillna("gine")
    group_cols = ["split_column", "encoder_type", "model_type"]
    summary = (
        df.groupby(group_cols)[METRIC_COLUMNS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    print(summary.to_string(index=False))
    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
