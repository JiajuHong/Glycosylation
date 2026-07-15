from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


INPUT_PATH = Path("data/processed/glyco_model_local.csv")
SOLVENT_VOCAB_PATH = Path("data/processed/solvent_vocab.json")
CATALYST_VOCAB_PATH = Path("data/processed/catalyst_vocab.json")
RESULTS_PATH = Path("results/baseline_results.csv")

SPLITS = {
    "split_random_stratified": "results/baseline_predictions_random.csv",
    "split_pair_group": "results/baseline_predictions_pair.csv",
    "split_year": "results/baseline_predictions_year.csv",
}

FEATURE_SETS = ["structure_condition", "structure_only", "condition_only"]
FP_BITS = 2048
FP_RADIUS = 2
RANDOM_STATE = 42


def parse_id_list(value: object) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(float(item)) for item in str(value).split(";") if item.strip()]


def load_vocab(path: Path) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def smiles_to_fp_matrix(smiles: pd.Series, n_bits: int = FP_BITS) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=n_bits)
    rows: list[sparse.csr_matrix] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros(n_bits, dtype=np.float32)
        else:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros(n_bits, dtype=np.float32)
            arr[list(fp.GetOnBits())] = 1.0
        rows.append(sparse.csr_matrix(arr))
    return sparse.vstack(rows, format="csr")


def multi_hot(series: pd.Series, vocab_size: int) -> sparse.csr_matrix:
    rows = []
    cols = []
    data = []
    for row_idx, value in enumerate(series):
        ids = parse_id_list(value)
        for token_id in ids:
            if 0 <= token_id < vocab_size:
                rows.append(row_idx)
                cols.append(token_id)
                data.append(1.0)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(series), vocab_size), dtype=np.float32)


def donor_type_one_hot(df: pd.DataFrame) -> sparse.csr_matrix:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return encoder.fit_transform(df[["Donor_Type"]])


def train_scaled_numeric(
    df: pd.DataFrame,
    split_col: str,
) -> sparse.csr_matrix:
    train_mask = df[split_col].eq("train")

    temp = pd.to_numeric(df["Temp_C"], errors="coerce")
    temp_train = temp[train_mask].dropna()
    temp_mean = float(temp_train.mean()) if len(temp_train) else 0.0
    temp_std = float(temp_train.std(ddof=0)) if len(temp_train) else 1.0
    if temp_std == 0 or math.isnan(temp_std):
        temp_std = 1.0
    temp_norm = ((temp.fillna(temp_mean) - temp_mean) / temp_std).to_numpy(dtype=np.float32)

    time_min = pd.to_numeric(df["Time_min"], errors="coerce")
    log_time = np.log1p(time_min)
    log_time_train = log_time[train_mask].dropna()
    time_mean = float(log_time_train.mean()) if len(log_time_train) else 0.0
    time_std = float(log_time_train.std(ddof=0)) if len(log_time_train) else 1.0
    if time_std == 0 or math.isnan(time_std):
        time_std = 1.0
    log_time_norm = ((log_time.fillna(time_mean) - time_mean) / time_std).to_numpy(dtype=np.float32)

    numeric = np.vstack(
        [
            temp_norm,
            pd.to_numeric(df["has_temp"], errors="coerce").fillna(0).to_numpy(dtype=np.float32),
            log_time_norm,
            pd.to_numeric(df["has_time"], errors="coerce").fillna(0).to_numpy(dtype=np.float32),
        ]
    ).T
    return sparse.csr_matrix(numeric)


def build_feature_blocks(df: pd.DataFrame, split_col: str) -> dict[str, sparse.csr_matrix]:
    solvent_vocab = load_vocab(SOLVENT_VOCAB_PATH)
    catalyst_vocab = load_vocab(CATALYST_VOCAB_PATH)

    donor_fp = smiles_to_fp_matrix(df["Donor_Canonical_SMILES"])
    acceptor_fp = smiles_to_fp_matrix(df["Acceptor_Canonical_SMILES"])
    donor_type = donor_type_one_hot(df)
    solvent = multi_hot(df["Solvent_Component_IDs"], max(solvent_vocab.values()) + 1)
    catalyst = multi_hot(df["Catalyst_Component_IDs"], max(catalyst_vocab.values()) + 1)
    numeric = train_scaled_numeric(df, split_col)

    structure = sparse.hstack([donor_fp, acceptor_fp, donor_type], format="csr")
    condition = sparse.hstack([solvent, catalyst, numeric, donor_type], format="csr")
    structure_condition = sparse.hstack([donor_fp, acceptor_fp, solvent, catalyst, numeric, donor_type], format="csr")

    return {
        "structure_condition": structure_condition,
        "structure_only": structure,
        "condition_only": condition,
    }


def make_models(scale_pos_weight: float) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        ),
    }


def best_threshold_from_val(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y_true, (proba >= thr).astype(int)) for thr in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def safe_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "auroc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else np.nan,
        "auprc": average_precision_score(y_true, proba) if len(np.unique(y_true)) == 2 else np.nan,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_one_split(df: pd.DataFrame, split_col: str) -> tuple[list[dict[str, object]], pd.DataFrame]:
    y = df["Label"].astype(int).to_numpy()
    train_idx = np.where(df[split_col].eq("train").to_numpy())[0]
    val_idx = np.where(df[split_col].eq("val").to_numpy())[0]
    test_idx = np.where(df[split_col].eq("test").to_numpy())[0]

    feature_blocks = build_feature_blocks(df, split_col)
    results: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []

    y_train = y[train_idx]
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    for feature_set, X in feature_blocks.items():
        for model_name, model in make_models(scale_pos_weight).items():
            model.fit(X[train_idx], y_train)
            val_proba = model.predict_proba(X[val_idx])[:, 1]
            threshold = best_threshold_from_val(y[val_idx], val_proba)
            test_proba = model.predict_proba(X[test_idx])[:, 1]
            metrics = safe_metrics(y[test_idx], test_proba, threshold)
            results.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "split_name": split_col,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                }
            )

            pred_df = df.iloc[test_idx][["ID", "Reaction_ID", "Label", "Product_Config", "Donor_Type"]].copy()
            pred_df["model_name"] = model_name
            pred_df["feature_set"] = feature_set
            pred_df["split_name"] = split_col
            pred_df["y_proba"] = test_proba
            pred_df["y_pred"] = (test_proba >= threshold).astype(int)
            pred_df["threshold"] = threshold
            pred_frames.append(pred_df)

    return results, pd.concat(pred_frames, ignore_index=True)


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    all_results: list[dict[str, object]] = []

    for split_col, pred_path in SPLITS.items():
        print(f"Running {split_col} ...", flush=True)
        split_results, pred_df = run_one_split(df, split_col)
        all_results.extend(split_results)
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {pred_path}", flush=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {RESULTS_PATH}")
    print(
        results_df.sort_values(["split_name", "feature_set", "model_name"])[
            [
                "model_name",
                "feature_set",
                "split_name",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "auroc",
                "auprc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
