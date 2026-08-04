#!/usr/bin/env python
"""第三层数据集与校验工具：把完整反应转换为模型可用的图批次。

该模块不定义模型，只负责读取数据、解析和校验局部原子角色，并组装训练批次。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from layer3.chiral_graph import CHIRAL_GRAPH_FEATURE_VERSION, mol_to_chiral_pyg_graph


DONOR_ROLE_NAMES = [
    "C1",
    "O5",
    "C2",
    "LG_ENTRY",
    "LG_CORE",
    "RING_CONTEXT",
    "C2_SUB_ENTRY",
]

ACCEPTOR_ROLE_NAMES = [
    "O4",
    "C4",
    "C3",
    "C5",
    "C2",
    "O5",
    "C1",
    "N2_SUB_ENTRY",
    "N2_SUBGRAPH",
    "PG_ENTRY",
]

GRAPH_FEATURE_VERSION = "rdkit_heavy_v2_chiral"
ENCODER_TYPES = ("gine", "chiral_gine")

ATOM_FEATURE_NAMES = [
    "atomic_num",
    "total_degree",
    "formal_charge",
    "total_num_hs",
    "hybridization",
    "is_aromatic",
    "mass_scaled",
    "chiral_unspecified",
    "chiral_tetrahedral_cw",
    "chiral_tetrahedral_ccw",
    "chiral_other",
    "cip_R",
    "cip_S",
    "cip_unknown",
    "is_in_ring",
]

BOND_FEATURE_NAMES = [
    "bond_single",
    "bond_double",
    "bond_triple",
    "bond_aromatic",
    "is_conjugated",
    "is_in_ring",
    "stereo_none",
    "stereo_any",
    "stereo_z",
    "stereo_e",
    "stereo_cis",
    "stereo_trans",
    "stereo_other",
    "dir_none",
    "dir_endupright",
    "dir_enddownright",
    "dir_eitherdouble",
    "dir_unknown",
]

REQUIRED_COLUMNS = [
    "Donor_Canonical_SMILES",
    "Acceptor_Canonical_SMILES",
    "Donor_RFU_Atom_Indices",
    "Donor_RFU_Role_Indices",
    "Donor_C1_Index",
    "Acceptor_OH_Local_Atom_Indices",
    "Acceptor_OH_Role_Indices",
    "Acceptor_O4_Index",
    "Solvent_Component_IDs",
    "Catalyst_Component_IDs",
    "Temp_C",
    "has_temp",
    "Time_min",
    "has_time",
    "Label",
    "split_pair_group",
    "split_random_stratified",
    "split_year",
]


@dataclass(frozen=True)
class NormalizationStats:
    temp_mean: float
    temp_std: float
    log_time_mean: float
    log_time_std: float


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def parse_int_list(value: Any) -> list[int]:
    """Parse list-like CSV values.

    Preferred formats such as ``[1, 2]`` are parsed with ``ast.literal_eval``.
    The current curated table stores several list fields as ``"1;2;3"``, so a
    semicolon fallback is retained for compatibility.
    """

    if _is_missing(value):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, float):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]

    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return [int(v) for v in parsed]
    if isinstance(parsed, (int, float)):
        return [int(parsed)]

    separator = ";" if ";" in text else ","
    return [int(float(part.strip())) for part in text.split(separator) if part.strip()]


def parse_role_dict(value: Any) -> dict[str, list[int]]:
    if _is_missing(value):
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        raw = ast.literal_eval(str(value))
    return {str(key): [int(v) for v in vals] for key, vals in raw.items()}


def load_vocab(path: str | Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return mol


def one_hot_value(value: Any, choices: list[Any]) -> list[float]:
    values = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    values[idx] = 1.0
    return values


def atom_features(atom: Chem.Atom) -> list[float]:
    chiral_choices = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    cip_code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "UNKNOWN"
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(int(atom.GetHybridization())),
        float(atom.GetIsAromatic()),
        float(atom.GetMass() * 0.01),
    ] + one_hot_value(atom.GetChiralTag(), chiral_choices) + [
        float(cip_code == "R"),
        float(cip_code == "S"),
        float(cip_code not in {"R", "S"}),
        float(atom.IsInRing()),
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    direction_choices = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE,
    ]
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ] + one_hot_value(bond.GetStereo(), stereo_choices) + one_hot_value(bond.GetBondDir(), direction_choices)


def mol_to_pyg_graph(mol: Chem.Mol, smiles: str) -> Data:
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edge_pairs: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_pairs.extend([[begin, end], [end, begin]])
        edge_attrs.extend([features, features])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    data.num_heavy_atoms = mol.GetNumAtoms()
    data.graph_feature_version = GRAPH_FEATURE_VERSION
    return data


def clone_graph(data: Data) -> Data:
    return data.clone()


def safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def make_role_matrix(
    local_atom_indices: list[int],
    role_indices: dict[str, list[int]],
    role_names: list[str],
) -> torch.Tensor:
    local_pos = {atom_idx: pos for pos, atom_idx in enumerate(local_atom_indices)}
    matrix = torch.zeros((len(local_atom_indices), len(role_names)), dtype=torch.float32)
    for role_pos, role_name in enumerate(role_names):
        for atom_idx in role_indices.get(role_name, []):
            if atom_idx in local_pos:
                matrix[local_pos[atom_idx], role_pos] = 1.0
    return matrix


def compute_train_stats(df: pd.DataFrame, split_column: str) -> NormalizationStats:
    train_df = df[df[split_column] == "train"]
    if train_df.empty:
        raise ValueError(f"No train rows found in split column {split_column}")

    temp_values = pd.to_numeric(train_df.loc[train_df["has_temp"] == 1, "Temp_C"], errors="coerce").dropna()
    time_values = pd.to_numeric(train_df.loc[train_df["has_time"] == 1, "Time_min"], errors="coerce").dropna()
    if temp_values.empty:
        raise ValueError("No non-missing training temperatures available")
    if time_values.empty:
        raise ValueError("No non-missing training times available")

    log_time = time_values.map(lambda value: math.log1p(float(value)))
    temp_std = float(temp_values.std(ddof=0)) or 1.0
    log_time_std = float(log_time.std(ddof=0)) or 1.0
    return NormalizationStats(
        temp_mean=float(temp_values.mean()),
        temp_std=temp_std,
        log_time_mean=float(log_time.mean()),
        log_time_std=log_time_std,
    )


class GlycoDataset(Dataset):
    """Stable first-stage dataset for glycosylation reaction modeling."""

    def __init__(
        self,
        csv_path: str | Path = "data/processed/glyco_model_local.csv",
        split_column: str = "split_pair_group",
        split: str | None = None,
        solvent_vocab_path: str | Path = "data/processed/solvent_vocab.json",
        catalyst_vocab_path: str | Path = "data/processed/catalyst_vocab.json",
        graph_cache_path: str | Path | None = "data/processed/rdkit_graph_cache.pt",
        chiral_graph_cache_path: str | Path | None = "data/processed/rdkit_chiral_graph_cache.pt",
        encoder_type: str = "gine",
        stats: NormalizationStats | None = None,
        validate: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.split_column = split_column
        self.split = split
        if encoder_type not in ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type {encoder_type!r}; choose from {ENCODER_TYPES}")
        self.encoder_type = encoder_type
        self.df = pd.read_csv(self.csv_path)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        if split_column not in self.df.columns:
            raise ValueError(f"Unknown split_column: {split_column}")

        if split is not None:
            self.df = self.df[self.df[split_column] == split].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        full_df = pd.read_csv(self.csv_path)
        self.stats = stats or compute_train_stats(full_df, split_column)
        self.solvent_vocab = load_vocab(solvent_vocab_path)
        self.catalyst_vocab = load_vocab(catalyst_vocab_path)
        self.solv_missing_id = int(self.solvent_vocab["SOLV_MISSING"])
        self.cat_missing_id = int(self.catalyst_vocab["CAT_MISSING"])
        selected_cache_path = graph_cache_path if encoder_type == "gine" else chiral_graph_cache_path
        expected_version = (
            GRAPH_FEATURE_VERSION if encoder_type == "gine" else CHIRAL_GRAPH_FEATURE_VERSION
        )
        self.graph_cache = self.load_graph_cache(selected_cache_path, expected_version)

        if validate:
            self.validate_all()

    def __len__(self) -> int:
        return len(self.df)

    def normalize_temp(self, value: Any, has_temp: int) -> float:
        if int(has_temp) == 0 or _is_missing(value):
            return 0.0
        return (float(value) - self.stats.temp_mean) / self.stats.temp_std

    def normalize_log_time(self, value: Any, has_time: int) -> float:
        if int(has_time) == 0 or _is_missing(value):
            return 0.0
        return (math.log1p(float(value)) - self.stats.log_time_mean) / self.stats.log_time_std

    def parse_condition_ids(self, value: Any, missing_id: int, has_value: int) -> list[int]:
        if int(has_value) == 0:
            return [missing_id]
        ids = parse_int_list(value)
        return ids or [missing_id]

    def load_graph_cache(
        self,
        graph_cache_path: str | Path | None,
        expected_version: str,
    ) -> dict[str, Data]:
        if graph_cache_path is None:
            return {}
        path = Path(graph_cache_path)
        if not path.exists():
            return {}
        payload = safe_torch_load(path)
        if payload.get("feature_version") != expected_version:
            raise ValueError(
                f"Graph cache {path} has feature_version={payload.get('feature_version')}, "
                f"expected {expected_version}. Rebuild it with python -m layer3.build_rdkit_graph_cache."
            )
        graphs = payload.get("graphs")
        if not isinstance(graphs, dict):
            raise ValueError(f"Graph cache {path} does not contain a graph dictionary")
        return graphs

    def get_graph(self, smiles: str) -> Data:
        if smiles in self.graph_cache:
            return clone_graph(self.graph_cache[smiles])
        mol = mol_from_smiles(smiles)
        if self.encoder_type == "chiral_gine":
            return mol_to_chiral_pyg_graph(
                mol,
                smiles,
                atom_feature_fn=atom_features,
                bond_feature_fn=bond_features,
            )
        return mol_to_pyg_graph(mol, smiles)

    def build_item(self, row: pd.Series) -> dict[str, Any]:
        donor_smiles = str(row["Donor_Canonical_SMILES"])
        acceptor_smiles = str(row["Acceptor_Canonical_SMILES"])

        donor_rfu_indices = parse_int_list(row["Donor_RFU_Atom_Indices"])
        acceptor_oh_indices = parse_int_list(row["Acceptor_OH_Local_Atom_Indices"])
        donor_roles = parse_role_dict(row["Donor_RFU_Role_Indices"])
        acceptor_roles = parse_role_dict(row["Acceptor_OH_Role_Indices"])
        donor_c1 = int(row["Donor_C1_Index"])
        acceptor_o4 = int(row["Acceptor_O4_Index"])

        donor_role_matrix = make_role_matrix(donor_rfu_indices, donor_roles, DONOR_ROLE_NAMES)
        acceptor_role_matrix = make_role_matrix(acceptor_oh_indices, acceptor_roles, ACCEPTOR_ROLE_NAMES)

        has_temp = int(row["has_temp"])
        has_time = int(row["has_time"])
        solvent_ids = self.parse_condition_ids(
            row["Solvent_Component_IDs"],
            missing_id=self.solv_missing_id,
            has_value=int(row.get("has_solvent", 1)),
        )
        catalyst_ids = self.parse_condition_ids(
            row["Catalyst_Component_IDs"],
            missing_id=self.cat_missing_id,
            has_value=int(row.get("has_catalyst", 1)),
        )

        return {
            "id": int(row["ID"]) if "ID" in row and not _is_missing(row["ID"]) else -1,
            "reaction_id": row.get("Reaction_ID", ""),
            "donor_smiles": donor_smiles,
            "acceptor_smiles": acceptor_smiles,
            "donor_graph": self.get_graph(donor_smiles),
            "acceptor_graph": self.get_graph(acceptor_smiles),
            "donor_rfu_atom_indices": torch.tensor(donor_rfu_indices, dtype=torch.long),
            "donor_rfu_role_matrix": donor_role_matrix,
            "donor_c1_index": donor_c1,
            "donor_c1_local_pos": donor_rfu_indices.index(donor_c1),
            "acceptor_oh_local_atom_indices": torch.tensor(acceptor_oh_indices, dtype=torch.long),
            "acceptor_oh_role_matrix": acceptor_role_matrix,
            "acceptor_o4_index": acceptor_o4,
            "acceptor_o4_local_pos": acceptor_oh_indices.index(acceptor_o4),
            "solvent_component_ids": torch.tensor(solvent_ids, dtype=torch.long),
            "catalyst_component_ids": torch.tensor(catalyst_ids, dtype=torch.long),
            "temp_c_raw": float(row["Temp_C"]) if has_temp and not _is_missing(row["Temp_C"]) else float("nan"),
            "has_temp": has_temp,
            "temp_norm": self.normalize_temp(row["Temp_C"], has_temp),
            "time_min_raw": float(row["Time_min"]) if has_time and not _is_missing(row["Time_min"]) else float("nan"),
            "has_time": has_time,
            "log_time_norm": self.normalize_log_time(row["Time_min"], has_time),
            "label": int(row["Label"]),
            "split_pair_group": row["split_pair_group"],
            "split_random_stratified": row["split_random_stratified"],
            "split_year": row["split_year"],
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.build_item(self.df.iloc[index])

    def validate_row(self, row: pd.Series) -> list[str]:
        errors: list[str] = []
        row_id = row.get("ID", "?")

        try:
            donor_mol = mol_from_smiles(str(row["Donor_Canonical_SMILES"]))
        except ValueError as exc:
            return [f"ID={row_id}: {exc}"]
        try:
            acceptor_mol = mol_from_smiles(str(row["Acceptor_Canonical_SMILES"]))
        except ValueError as exc:
            return [f"ID={row_id}: {exc}"]

        donor_indices = parse_int_list(row["Donor_RFU_Atom_Indices"])
        acceptor_indices = parse_int_list(row["Acceptor_OH_Local_Atom_Indices"])
        donor_c1 = int(row["Donor_C1_Index"])
        acceptor_o4 = int(row["Acceptor_O4_Index"])
        donor_roles = parse_role_dict(row["Donor_RFU_Role_Indices"])
        acceptor_roles = parse_role_dict(row["Acceptor_OH_Role_Indices"])

        if any(idx < 0 or idx >= donor_mol.GetNumAtoms() for idx in donor_indices):
            errors.append(f"ID={row_id}: donor RFU atom index out of bounds")
        if any(idx < 0 or idx >= acceptor_mol.GetNumAtoms() for idx in acceptor_indices):
            errors.append(f"ID={row_id}: acceptor OH atom index out of bounds")
        if donor_c1 not in donor_indices:
            errors.append(f"ID={row_id}: Donor_C1_Index not in Donor_RFU_Atom_Indices")
        if acceptor_o4 not in acceptor_indices:
            errors.append(f"ID={row_id}: Acceptor_O4_Index not in Acceptor_OH_Local_Atom_Indices")

        donor_role_matrix = make_role_matrix(donor_indices, donor_roles, DONOR_ROLE_NAMES)
        acceptor_role_matrix = make_role_matrix(acceptor_indices, acceptor_roles, ACCEPTOR_ROLE_NAMES)
        if donor_role_matrix.shape != (len(donor_indices), len(DONOR_ROLE_NAMES)):
            errors.append(f"ID={row_id}: donor role matrix shape mismatch")
        if acceptor_role_matrix.shape != (len(acceptor_indices), len(ACCEPTOR_ROLE_NAMES)):
            errors.append(f"ID={row_id}: acceptor role matrix shape mismatch")
        for role_name in ["C1", "O5", "C2", "LG_ENTRY"]:
            if not donor_roles.get(role_name):
                errors.append(f"ID={row_id}: donor required role {role_name} is empty")
        for role_name in ["O4", "C4", "C2", "O5"]:
            if not acceptor_roles.get(role_name):
                errors.append(f"ID={row_id}: acceptor required role {role_name} is empty")

        solvent_ids = self.parse_condition_ids(
            row["Solvent_Component_IDs"],
            missing_id=self.solv_missing_id,
            has_value=int(row.get("has_solvent", 1)),
        )
        catalyst_ids = self.parse_condition_ids(
            row["Catalyst_Component_IDs"],
            missing_id=self.cat_missing_id,
            has_value=int(row.get("has_catalyst", 1)),
        )
        if int(row.get("has_solvent", 1)) == 0 and solvent_ids != [self.solv_missing_id]:
            errors.append(f"ID={row_id}: missing solvent did not map to SOLV_MISSING")
        if int(row.get("has_catalyst", 1)) == 0 and catalyst_ids != [self.cat_missing_id]:
            errors.append(f"ID={row_id}: missing catalyst did not map to CAT_MISSING")

        return errors

    def validate_all(self) -> None:
        errors: list[str] = []
        for _, row in self.df.iterrows():
            errors.extend(self.validate_row(row))
        if errors:
            preview = "\n".join(errors[:20])
            raise ValueError(f"GlycoDataset validation failed with {len(errors)} errors:\n{preview}")


def pad_1d(sequences: Iterable[torch.Tensor], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = list(sequences)
    max_len = max((seq.numel() for seq in seqs), default=0)
    values = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for row_idx, seq in enumerate(seqs):
        length = seq.numel()
        if length:
            values[row_idx, :length] = seq
            mask[row_idx, :length] = True
    return values, mask


def pad_role_matrices(matrices: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mats = list(matrices)
    max_len = max((mat.shape[0] for mat in mats), default=0)
    role_dim = mats[0].shape[1] if mats else 0
    values = torch.zeros((len(mats), max_len, role_dim), dtype=torch.float32)
    mask = torch.zeros((len(mats), max_len), dtype=torch.bool)
    for row_idx, mat in enumerate(mats):
        length = mat.shape[0]
        if length:
            values[row_idx, :length, :] = mat
            mask[row_idx, :length] = True
    return values, mask


def glyco_collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    donor_rfu_indices, donor_rfu_mask = pad_1d(item["donor_rfu_atom_indices"] for item in items)
    acceptor_oh_indices, acceptor_oh_mask = pad_1d(
        item["acceptor_oh_local_atom_indices"] for item in items
    )
    donor_roles, donor_role_mask = pad_role_matrices(item["donor_rfu_role_matrix"] for item in items)
    acceptor_roles, acceptor_role_mask = pad_role_matrices(item["acceptor_oh_role_matrix"] for item in items)
    assert torch.equal(donor_role_mask, donor_rfu_mask), "donor role mask and RFU index mask diverged"
    assert torch.equal(acceptor_role_mask, acceptor_oh_mask), "acceptor role mask and OH index mask diverged"
    solvent_ids, solvent_mask = pad_1d(item["solvent_component_ids"] for item in items)
    catalyst_ids, catalyst_mask = pad_1d(item["catalyst_component_ids"] for item in items)

    return {
        "ids": torch.tensor([item["id"] for item in items], dtype=torch.long),
        "reaction_ids": [item["reaction_id"] for item in items],
        "donor_smiles": [item["donor_smiles"] for item in items],
        "acceptor_smiles": [item["acceptor_smiles"] for item in items],
        "donor_graph": Batch.from_data_list([item["donor_graph"] for item in items]),
        "acceptor_graph": Batch.from_data_list([item["acceptor_graph"] for item in items]),
        "donor_rfu_atom_indices": donor_rfu_indices,
        "donor_rfu_mask": donor_rfu_mask,
        "donor_rfu_role_matrix": donor_roles,
        "donor_c1_index": torch.tensor([item["donor_c1_index"] for item in items], dtype=torch.long),
        "donor_c1_local_pos": torch.tensor(
            [item["donor_c1_local_pos"] for item in items], dtype=torch.long
        ),
        "acceptor_oh_local_atom_indices": acceptor_oh_indices,
        "acceptor_oh_mask": acceptor_oh_mask,
        "acceptor_oh_role_matrix": acceptor_roles,
        "acceptor_o4_index": torch.tensor([item["acceptor_o4_index"] for item in items], dtype=torch.long),
        "acceptor_o4_local_pos": torch.tensor(
            [item["acceptor_o4_local_pos"] for item in items], dtype=torch.long
        ),
        "solvent_component_ids": solvent_ids,
        "solvent_mask": solvent_mask,
        "catalyst_component_ids": catalyst_ids,
        "catalyst_mask": catalyst_mask,
        "temp_c_raw": torch.tensor([item["temp_c_raw"] for item in items], dtype=torch.float32),
        "has_temp": torch.tensor([item["has_temp"] for item in items], dtype=torch.float32),
        "temp_norm": torch.tensor([item["temp_norm"] for item in items], dtype=torch.float32),
        "time_min_raw": torch.tensor([item["time_min_raw"] for item in items], dtype=torch.float32),
        "has_time": torch.tensor([item["has_time"] for item in items], dtype=torch.float32),
        "log_time_norm": torch.tensor([item["log_time_norm"] for item in items], dtype=torch.float32),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "split_pair_group": [item["split_pair_group"] for item in items],
        "split_random_stratified": [item["split_random_stratified"] for item in items],
        "split_year": [item["split_year"] for item in items],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and preview GlycoDataset batches.")
    parser.add_argument("--csv", default="glyco_model_local.csv")
    parser.add_argument("--split-column", default="split_pair_group")
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--graph-cache", default="rdkit_graph_cache.pt")
    parser.add_argument("--chiral-graph-cache", default="rdkit_chiral_graph_cache.pt")
    parser.add_argument("--encoder-type", choices=ENCODER_TYPES, default="gine")
    args = parser.parse_args()

    dataset = GlycoDataset(
        csv_path=args.csv,
        split_column=args.split_column,
        split=args.split,
        graph_cache_path=args.graph_cache,
        chiral_graph_cache_path=args.chiral_graph_cache,
        encoder_type=args.encoder_type,
        validate=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=glyco_collate_fn,
    )
    batch = next(iter(loader))

    print("GlycoDataset validation passed")
    print(f"rows: {len(dataset)}")
    print(f"split_column: {args.split_column}")
    print(f"split: {args.split or 'all'}")
    print(f"encoder_type: {args.encoder_type}")
    print(f"temp_mean(train): {dataset.stats.temp_mean:.6g}")
    print(f"temp_std(train): {dataset.stats.temp_std:.6g}")
    print(f"log_time_mean(train): {dataset.stats.log_time_mean:.6g}")
    print(f"log_time_std(train): {dataset.stats.log_time_std:.6g}")
    print(f"batch donor graphs: {batch['donor_graph'].num_graphs}")
    print(f"batch acceptor graphs: {batch['acceptor_graph'].num_graphs}")
    print(f"atom feature dim: {batch['donor_graph'].x.shape[-1]}")
    print(f"bond feature dim: {batch['donor_graph'].edge_attr.shape[-1]}")
    print(f"graph cache loaded: {len(dataset.graph_cache)}")
    print(f"donor_rfu_atom_indices shape: {tuple(batch['donor_rfu_atom_indices'].shape)}")
    print(f"acceptor_oh_local_atom_indices shape: {tuple(batch['acceptor_oh_local_atom_indices'].shape)}")
    print(f"donor_rfu_role_matrix shape: {tuple(batch['donor_rfu_role_matrix'].shape)}")
    print(f"acceptor_oh_role_matrix shape: {tuple(batch['acceptor_oh_role_matrix'].shape)}")
    print(f"solvent_component_ids shape: {tuple(batch['solvent_component_ids'].shape)}")
    print(f"catalyst_component_ids shape: {tuple(batch['catalyst_component_ids'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
