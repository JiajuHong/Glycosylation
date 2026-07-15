from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_PATH = Path("data/processed/condition_tokenized.csv")
OUTPUT_PATH = Path("data/processed/condition_tokenized_with_splits.csv")

SPLIT_FILES = {
    "split_random_stratified": Path("data/processed/split_random_stratified.csv"),
    "split_pair_group": Path("data/processed/split_pair_group.csv"),
    "split_year": Path("data/processed/split_year.csv"),
}


def main() -> None:
    df = pd.read_csv(BASE_PATH, encoding="utf-8-sig")

    for column_name, split_path in SPLIT_FILES.items():
        split_df = pd.read_csv(split_path, encoding="utf-8-sig")
        if "ID" not in split_df.columns or "split" not in split_df.columns:
            raise ValueError(f"{split_path} must contain ID and split columns")
        split_map = split_df.set_index("ID")["split"]
        df[column_name] = df["ID"].map(split_map)
        if df[column_name].isna().any():
            missing = df.loc[df[column_name].isna(), "ID"].head(10).tolist()
            raise ValueError(f"Missing split assignments from {split_path}: {missing}")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_PATH} with shape {df.shape}")
    for column_name in SPLIT_FILES:
        print(column_name)
        print(df[column_name].value_counts().reindex(["train", "val", "test"]).to_string())


if __name__ == "__main__":
    main()
