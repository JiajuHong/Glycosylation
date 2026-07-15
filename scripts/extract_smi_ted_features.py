#!/usr/bin/env python3
"""Extract frozen, molecule-level SMI-TED features for glycosylation data.

The source stereochemical SMILES are never modified. SMI-TED receives a
separate canonical, non-isomeric representation matching its official
normalization. Donors and acceptors are deduplicated jointly, encoded once,
and mapped back to reaction rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase


MODEL_REPOSITORY = "ibm/materials.smi-ted"
MODEL_CHECKPOINT = "smi-ted-Light_40.pt"
MODEL_VOCAB = "bert_vocab_curated.txt"
EMBEDDING_DIM = 768
DONOR_COLUMN = "Donor_Canonical_SMILES"
ACCEPTOR_COLUMN = "Acceptor_Canonical_SMILES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smi-ted-code-dir",
        type=Path,
        required=True,
        help="IBM materials repository's models/smi_ted directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the official checkpoint and vocabulary",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--source-revision",
        default="unknown",
        help="Git revision of the IBM materials source used for extraction",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_smiles(value: Any, *, isomeric: bool) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    csv_path = args.csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    code_dir = args.smi_ted_code_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    required = [
        csv_path,
        code_dir / "smi_ted_light" / "load.py",
        model_dir / MODEL_CHECKPOINT,
        model_dir / MODEL_VOCAB,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required files are missing: {missing}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, output_dir, code_dir, model_dir


def load_official_model(code_dir: Path, model_dir: Path) -> tuple[Any, Any]:
    """Use IBM's loader while resolving already-downloaded files offline."""
    sys.path.insert(0, str(code_dir))
    module = importlib.import_module("smi_ted_light.load")

    def local_hf_resolver(*, repo_id: str, filename: str, **_: Any) -> str:
        if repo_id != MODEL_REPOSITORY:
            raise ValueError(f"Unexpected Hugging Face repository: {repo_id}")
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    # IBM's official load_smi_ted() calls this symbol for the two artifacts.
    # Replacing only the resolver makes extraction reproducible without server
    # internet access; model construction, checkpoint loading, and encode() are
    # unchanged official code.
    module.hf_hub_download = local_hf_resolver
    model = module.load_smi_ted()
    model.eval()
    return model, module


def token_length(model: Any, smiles: str) -> int:
    return len(model.tokenizer.tokenize(smiles)) + 2  # <bos> and <eos>


@torch.inference_mode()
def encode_batches(model: Any, smiles: Sequence[str], batch_size: int) -> np.ndarray:
    if not smiles:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    chunks: list[np.ndarray] = []
    total = len(smiles)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = list(smiles[start:end])
        encoded = model.encode(batch, batch_size=len(batch), return_torch=True)
        array = encoded.detach().cpu().float().numpy()
        expected = (len(batch), EMBEDDING_DIM)
        if array.shape != expected:
            raise RuntimeError(f"Unexpected embedding shape {array.shape}; expected {expected}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite embeddings found in unique rows {start}:{end}")
        chunks.append(array)
        print(f"Encoded {end}/{total} unique molecules", flush=True)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def build_collision_audit(
    frame: pd.DataFrame,
    donor_noniso: pd.Series,
    donor_iso: pd.Series,
    acceptor_noniso: pd.Series,
    acceptor_iso: pd.Series,
) -> pd.DataFrame:
    donor = pd.DataFrame(
        {
            "Role": "donor",
            "SMI_TED_SMILES": donor_noniso,
            "Stereo_Canonical_SMILES": donor_iso,
        }
    )
    acceptor = pd.DataFrame(
        {
            "Role": "acceptor",
            "SMI_TED_SMILES": acceptor_noniso,
            "Stereo_Canonical_SMILES": acceptor_iso,
        }
    )
    long = pd.concat([donor, acceptor], ignore_index=True)
    records: list[dict[str, Any]] = []
    for normalized, group in long.groupby("SMI_TED_SMILES", sort=False):
        variants = sorted(group["Stereo_Canonical_SMILES"].dropna().unique().tolist())
        if len(variants) > 1:
            records.append(
                {
                    "SMI_TED_SMILES": normalized,
                    "Stereo_Variant_Count": len(variants),
                    "Roles": ";".join(sorted(group["Role"].unique().tolist())),
                    "Stereo_Canonical_SMILES_Variants": " || ".join(variants),
                    "Reaction_Occurrences": int(len(group)),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "SMI_TED_SMILES",
            "Stereo_Variant_Count",
            "Roles",
            "Stereo_Canonical_SMILES_Variants",
            "Reaction_Occurrences",
        ],
    )


def save_outputs(
    output_dir: Path,
    source: pd.DataFrame,
    molecule_table: pd.DataFrame,
    unique_embeddings: np.ndarray,
    donor_indices: np.ndarray,
    acceptor_indices: np.ndarray,
    collision_audit: pd.DataFrame,
) -> dict[str, Path]:
    donor_embeddings = unique_embeddings[donor_indices]
    acceptor_embeddings = unique_embeddings[acceptor_indices]
    if not np.array_equal(donor_embeddings, unique_embeddings[donor_indices]):
        raise RuntimeError("Donor feature mapping verification failed")
    if not np.array_equal(acceptor_embeddings, unique_embeddings[acceptor_indices]):
        raise RuntimeError("Acceptor feature mapping verification failed")

    paths = {
        "unique_embeddings": output_dir / "smi_ted_unique_embeddings.npy",
        "donor_embeddings": output_dir / "donor_smi_ted_768.npy",
        "acceptor_embeddings": output_dir / "acceptor_smi_ted_768.npy",
        "donor_indices": output_dir / "donor_smi_ted_indices.npy",
        "acceptor_indices": output_dir / "acceptor_smi_ted_indices.npy",
        "molecule_index": output_dir / "smi_ted_molecule_index.csv",
        "reaction_mapping": output_dir / "reaction_smi_ted_mapping.csv",
        "stereo_collision_audit": output_dir / "stereo_collision_audit.csv",
    }
    np.save(paths["unique_embeddings"], unique_embeddings)
    np.save(paths["donor_embeddings"], donor_embeddings)
    np.save(paths["acceptor_embeddings"], acceptor_embeddings)
    np.save(paths["donor_indices"], donor_indices)
    np.save(paths["acceptor_indices"], acceptor_indices)
    molecule_table.to_csv(paths["molecule_index"], index=False)
    mapping_columns = [
        "Source_Row",
        "Donor_SMITED_SMILES",
        "Acceptor_SMITED_SMILES",
        "Donor_SMITED_Index",
        "Acceptor_SMITED_Index",
    ]
    if "ID" in source.columns:
        mapping_columns.insert(1, "ID")
    source[mapping_columns].to_csv(paths["reaction_mapping"], index=False)
    collision_audit.to_csv(paths["stereo_collision_audit"], index=False)
    return paths


def main() -> None:
    args = parse_args()
    csv_path, output_dir, code_dir, model_dir = validate_paths(args)
    frame = pd.read_csv(csv_path)
    missing_columns = [c for c in (DONOR_COLUMN, ACCEPTOR_COLUMN) if c not in frame]
    if missing_columns:
        raise KeyError(f"Input CSV is missing columns: {missing_columns}")

    donor_noniso = frame[DONOR_COLUMN].map(
        lambda value: canonicalize_smiles(value, isomeric=False)
    )
    acceptor_noniso = frame[ACCEPTOR_COLUMN].map(
        lambda value: canonicalize_smiles(value, isomeric=False)
    )
    donor_iso = frame[DONOR_COLUMN].map(
        lambda value: canonicalize_smiles(value, isomeric=True)
    )
    acceptor_iso = frame[ACCEPTOR_COLUMN].map(
        lambda value: canonicalize_smiles(value, isomeric=True)
    )
    invalid = donor_noniso.isna() | acceptor_noniso.isna() | donor_iso.isna() | acceptor_iso.isna()
    if invalid.any():
        invalid_path = output_dir / "invalid_smiles_rows.csv"
        frame.loc[invalid, [DONOR_COLUMN, ACCEPTOR_COLUMN]].to_csv(invalid_path, index=True)
        raise ValueError(f"Found {int(invalid.sum())} invalid rows; see {invalid_path}")

    frame = frame.copy()
    frame.insert(0, "Source_Row", np.arange(len(frame), dtype=np.int64))
    frame["Donor_SMITED_SMILES"] = donor_noniso
    frame["Acceptor_SMITED_SMILES"] = acceptor_noniso

    all_smiles = pd.concat([donor_noniso, acceptor_noniso], ignore_index=True)
    unique_smiles = all_smiles.drop_duplicates().reset_index(drop=True)
    molecule_table = pd.DataFrame(
        {
            "Feature_Index": np.arange(len(unique_smiles), dtype=np.int64),
            "SMI_TED_SMILES": unique_smiles,
        }
    )
    donor_counts = donor_noniso.value_counts()
    acceptor_counts = acceptor_noniso.value_counts()
    molecule_table["Donor_Occurrences"] = molecule_table["SMI_TED_SMILES"].map(donor_counts).fillna(0).astype(int)
    molecule_table["Acceptor_Occurrences"] = molecule_table["SMI_TED_SMILES"].map(acceptor_counts).fillna(0).astype(int)

    print(f"Reactions: {len(frame)}", flush=True)
    print(f"Unique non-isomeric molecules: {len(molecule_table)}", flush=True)
    model, _ = load_official_model(code_dir, model_dir)
    max_length = int(model.max_len)
    molecule_table["Token_Length"] = molecule_table["SMI_TED_SMILES"].map(
        lambda smiles: token_length(model, smiles)
    )
    too_long = molecule_table["Token_Length"] > max_length
    if too_long.any():
        overlength_path = output_dir / "overlength_smiles.csv"
        molecule_table.loc[too_long].to_csv(overlength_path, index=False)
        raise ValueError(
            f"Found {int(too_long.sum())} molecules longer than {max_length} tokens; "
            f"see {overlength_path}"
        )

    probe_smiles = molecule_table["SMI_TED_SMILES"].head(min(3, len(molecule_table))).tolist()
    probe_first = encode_batches(model, probe_smiles, len(probe_smiles))
    probe_second = encode_batches(model, probe_smiles, len(probe_smiles))
    repeat_max_abs_diff = float(np.max(np.abs(probe_first - probe_second)))
    repeat_exact = bool(np.array_equal(probe_first, probe_second))
    if repeat_max_abs_diff > 1e-6:
        raise RuntimeError(
            f"Repeated encoding is not deterministic enough: max abs diff={repeat_max_abs_diff}"
        )

    unique_embeddings = encode_batches(
        model,
        molecule_table["SMI_TED_SMILES"].astype(str).tolist(),
        args.batch_size,
    )
    expected_unique_shape = (len(molecule_table), EMBEDDING_DIM)
    if unique_embeddings.shape != expected_unique_shape:
        raise RuntimeError(
            f"Unique feature shape {unique_embeddings.shape}; expected {expected_unique_shape}"
        )

    smiles_to_index = dict(
        zip(molecule_table["SMI_TED_SMILES"], molecule_table["Feature_Index"])
    )
    frame["Donor_SMITED_Index"] = donor_noniso.map(smiles_to_index).astype(np.int64)
    frame["Acceptor_SMITED_Index"] = acceptor_noniso.map(smiles_to_index).astype(np.int64)
    donor_indices = frame["Donor_SMITED_Index"].to_numpy(dtype=np.int64)
    acceptor_indices = frame["Acceptor_SMITED_Index"].to_numpy(dtype=np.int64)
    collision_audit = build_collision_audit(
        frame, donor_noniso, donor_iso, acceptor_noniso, acceptor_iso
    )
    paths = save_outputs(
        output_dir,
        frame,
        molecule_table,
        unique_embeddings,
        donor_indices,
        acceptor_indices,
        collision_audit,
    )

    metadata = {
        "model": "SMI-TED 289M",
        "repository": MODEL_REPOSITORY,
        "checkpoint": MODEL_CHECKPOINT,
        "official_source_revision": args.source_revision,
        "embedding_dimension": EMBEDDING_DIM,
        "maximum_token_length": max_length,
        "normalization": {"canonical": True, "isomericSmiles": False},
        "stereochemistry_role": "Removed only for SMI-TED; preserved in graph-model input CSV",
        "input_file": str(csv_path),
        "input_sha256": sha256_file(csv_path),
        "number_of_reactions": int(len(frame)),
        "number_of_unique_molecules": int(len(molecule_table)),
        "unique_donor_smiles_before_nonisomeric_normalization": int(donor_iso.nunique()),
        "unique_acceptor_smiles_before_nonisomeric_normalization": int(acceptor_iso.nunique()),
        "stereo_collision_groups": int(len(collision_audit)),
        "maximum_observed_token_length": int(molecule_table["Token_Length"].max()),
        "batch_size": int(args.batch_size),
        "dtype": str(unique_embeddings.dtype),
        "device": "cuda" if bool(model.is_cuda_available) else "cpu",
        "repeatability_probe_exact": repeat_exact,
        "repeatability_probe_max_abs_diff": repeat_max_abs_diff,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "files": {
            key: {"name": path.name, "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_alignment": "Feature row i aligns with source CSV row i; verified through index arrays",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("SMI-TED feature extraction completed", flush=True)
    print(f"Unique features: {unique_embeddings.shape}", flush=True)
    print(f"Donor features: {(len(frame), EMBEDDING_DIM)}", flush=True)
    print(f"Acceptor features: {(len(frame), EMBEDDING_DIM)}", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
