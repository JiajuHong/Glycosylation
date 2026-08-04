"""第三层数据划分：生成随机、结构组隔离和年份外推三套划分。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


RANDOM_STATE = 42
INPUT = Path("data/raw/raw.csv")


def add_random_stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    strat = out["Label"].astype(str) + "__" + out["Donor_Type"].astype(str)

    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    train_idx, temp_idx = next(sss1.split(out, strat))

    temp = out.iloc[temp_idx]
    temp_strat = strat.iloc[temp_idx]
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )
    val_rel, test_rel = next(sss2.split(temp, temp_strat))
    val_idx = temp.index[val_rel]
    test_idx = temp.index[test_rel]

    out["split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def add_pair_group_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    strat = out["Label"].astype(str) + "__" + out["Donor_Type"].astype(str)
    groups = out["Pair_Key"].astype(str)

    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    train_idx, temp_idx = next(gss1.split(out, strat, groups))

    temp = out.iloc[temp_idx]
    temp_groups = groups.iloc[temp_idx]
    temp_strat = strat.iloc[temp_idx]
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )
    val_rel, test_rel = next(gss2.split(temp, temp_strat, temp_groups))
    val_idx = temp.index[val_rel]
    test_idx = temp.index[test_rel]

    out["split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def add_year_split(df: pd.DataFrame, cutoff: int = 2017) -> pd.DataFrame:
    out = df.copy()
    out["split"] = "test"

    old_idx = out.index[out["Year"] <= cutoff]
    old = out.loc[old_idx]
    strat = old["Label"].astype(str) + "__" + old["Donor_Type"].astype(str)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )
    train_rel, val_rel = next(sss.split(old, strat))
    train_idx = old.index[train_rel]
    val_idx = old.index[val_rel]

    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "val"
    return out


def summarize(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    print("counts")
    print(df["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_string())

    print("label")
    print(pd.crosstab(df["split"], df["Label"], normalize="index").round(3).to_string())

    print("donor_type")
    print(pd.crosstab(df["split"], df["Donor_Type"], normalize="index").round(3).to_string())

    if name == "split_pair_group.csv":
        overlap = {}
        for a in ["train", "val", "test"]:
            a_groups = set(df.loc[df["split"] == a, "Pair_Key"].astype(str))
            for b in ["train", "val", "test"]:
                if a < b:
                    b_groups = set(df.loc[df["split"] == b, "Pair_Key"].astype(str))
                    overlap[f"{a}-{b}"] = len(a_groups & b_groups)
        print("pair_key_overlap", overlap)

    if name == "split_year.csv":
        print("year")
        print(df.groupby("split")["Year"].agg(["min", "max"]).reindex(["train", "val", "test"]).to_string())


def main() -> None:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    required = {"Label", "Donor_Type", "Pair_Key", "Year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    splitters = {
        "split_random_stratified.csv": add_random_stratified_split,
        "split_pair_group.csv": add_pair_group_split,
        "split_year.csv": add_year_split,
    }

    for filename, fn in splitters.items():
        split_df = fn(df)
        split_df.to_csv(filename, index=False, encoding="utf-8-sig")
        summarize(filename, split_df)


if __name__ == "__main__":
    main()
