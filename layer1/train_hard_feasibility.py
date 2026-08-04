#!/usr/bin/env python
"""第一层训练入口：训练并评测仅使用结构信息的 O4 Chiral-GINE。"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from common.eval_metrics import compute_binary_metrics, search_threshold
from models.chiral_gine_encoder import ChiralGINEEncoder
from models.local_modules import RFUOHCrossAttentionTri, RoleEmbedding, gather_local_nodes
from models.pooling import GlobalAttnPool


DONOR_ROLE_NAMES = [
    "C1", "O5", "C2", "LG_ENTRY", "LG_CORE", "RING_CONTEXT", "C2_SUB_ENTRY"
]
CACHED_ACCEPTOR_ROLE_NAMES = [
    "O4", "C4", "C3", "C5", "C2", "O5", "C1", "N2_SUB_ENTRY",
    "N2_SUBGRAPH", "PG_ENTRY", "O4_EXTERNAL",
]
ACCEPTOR_ROLE_NAMES = [
    role for role in CACHED_ACCEPTOR_ROLE_NAMES if role != "O4_EXTERNAL"
]


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def parse_int_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        return [int(v) for v in ast.literal_eval(text)]
    separator = ";" if ";" in text else ","
    return [int(part) for part in text.split(separator) if part.strip()]


def parse_role_dict(value: Any) -> dict[str, list[int]]:
    if isinstance(value, dict):
        raw = value
    else:
        raw = json.loads(str(value))
    return {str(key): [int(v) for v in values] for key, values in raw.items()}


def make_role_matrix(
    local_indices: list[int], roles: dict[str, list[int]], role_names: list[str]
) -> torch.Tensor:
    local_position = {atom_idx: pos for pos, atom_idx in enumerate(local_indices)}
    matrix = torch.zeros((len(local_indices), len(role_names)), dtype=torch.float32)
    for role_pos, role_name in enumerate(role_names):
        for atom_idx in roles.get(role_name, []):
            if atom_idx in local_position:
                matrix[local_position[atom_idx], role_pos] = 1.0
    return matrix


def clone_graph(graph):
    return graph.clone()


class HardFeasibilityDataset(Dataset):
    """Read curated annotations and cached chiral molecular graphs only."""

    def __init__(
        self,
        csv_path: str | Path,
        cache_path: str | Path,
        split: str | None = None,
        stress: bool = False,
        acceptor_role_names: list[str] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        if split is not None:
            self.df = self.df[self.df["split_acceptor_group"] == split].reset_index(drop=True)
        payload = safe_torch_load(cache_path)
        self.graphs = payload["graphs"]
        self.cached_samples = payload.get("samples", {})
        self.stress = stress
        self.acceptor_role_names = list(acceptor_role_names or ACCEPTOR_ROLE_NAMES)
        if self.acceptor_role_names != ACCEPTOR_ROLE_NAMES:
            raise ValueError(
                "Only the 10-role acceptor schema without O4_EXTERNAL is supported"
            )
        self._validate()

    def __len__(self) -> int:
        return len(self.df)

    def _metadata(self, row: pd.Series) -> dict[str, Any]:
        sample_id = str(row["Sample_ID"])
        if not self.stress:
            metadata = dict(self.cached_samples[sample_id])
            role_matrix = torch.as_tensor(
                metadata["acceptor_site_role_matrix"], dtype=torch.float32
            )
            if role_matrix.shape[1] == len(CACHED_ACCEPTOR_ROLE_NAMES):
                selected_columns = [
                    CACHED_ACCEPTOR_ROLE_NAMES.index(role)
                    for role in self.acceptor_role_names
                ]
                role_matrix = role_matrix[:, selected_columns]
            elif role_matrix.shape[1] != len(self.acceptor_role_names):
                raise ValueError(
                    f"{sample_id}: unsupported cached acceptor role width "
                    f"{role_matrix.shape[1]}"
                )
            metadata["acceptor_site_role_matrix"] = role_matrix
            return metadata
        donor_indices = parse_int_list(row["Donor_RFU_Atom_Indices"])
        acceptor_indices = parse_int_list(row["Acceptor_Target_Local_Atom_Indices"])
        return {
            "split": str(row["split_acceptor_group"]),
            "label": int(row["hard_feasibility"]),
            "donor_smiles": str(row["Donor_Canonical_SMILES"]),
            "acceptor_smiles": str(row["Acceptor_Canonical_SMILES"]),
            "donor_rfu_atom_indices": donor_indices,
            "donor_rfu_role_matrix": make_role_matrix(
                donor_indices,
                parse_role_dict(row["Donor_RFU_Role_Indices"]),
                DONOR_ROLE_NAMES,
            ),
            "acceptor_site_atom_indices": acceptor_indices,
            "acceptor_site_role_matrix": make_role_matrix(
                acceptor_indices,
                parse_role_dict(row["Acceptor_Target_Role_Indices"]),
                self.acceptor_role_names,
            ),
            "acceptor_o4_index": int(row["Target_O4_Index"]),
        }

    def _validate(self) -> None:
        errors: list[str] = []
        for _, row in self.df.iterrows():
            sample_id = str(row["Sample_ID"])
            if not self.stress and sample_id not in self.cached_samples:
                errors.append(f"{sample_id}: missing cached sample metadata")
                continue
            meta = self._metadata(row)
            donor_smiles = meta["donor_smiles"]
            acceptor_smiles = meta["acceptor_smiles"]
            if donor_smiles not in self.graphs or acceptor_smiles not in self.graphs:
                errors.append(f"{sample_id}: missing molecular graph")
                continue
            donor_indices = [int(v) for v in meta["donor_rfu_atom_indices"]]
            acceptor_indices = [int(v) for v in meta["acceptor_site_atom_indices"]]
            donor_n = int(self.graphs[donor_smiles].num_heavy_atoms)
            acceptor_n = int(self.graphs[acceptor_smiles].num_heavy_atoms)
            if not donor_indices or any(v < 0 or v >= donor_n for v in donor_indices):
                errors.append(f"{sample_id}: invalid donor local indices")
            if not acceptor_indices or any(v < 0 or v >= acceptor_n for v in acceptor_indices):
                errors.append(f"{sample_id}: invalid acceptor local indices")
            if int(meta["acceptor_o4_index"]) not in acceptor_indices:
                errors.append(f"{sample_id}: O4 is absent from acceptor local region")
            donor_roles = torch.as_tensor(meta["donor_rfu_role_matrix"])
            acceptor_roles = torch.as_tensor(meta["acceptor_site_role_matrix"])
            if tuple(donor_roles.shape) != (len(donor_indices), len(DONOR_ROLE_NAMES)):
                errors.append(f"{sample_id}: donor role matrix shape mismatch")
            if tuple(acceptor_roles.shape) != (
                len(acceptor_indices),
                len(self.acceptor_role_names),
            ):
                errors.append(f"{sample_id}: acceptor role matrix shape mismatch")
        if errors:
            raise ValueError("Dataset validation failed:\n" + "\n".join(errors[:20]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        meta = self._metadata(row)
        donor_indices = [int(v) for v in meta["donor_rfu_atom_indices"]]
        acceptor_indices = [int(v) for v in meta["acceptor_site_atom_indices"]]
        donor_roles = torch.as_tensor(meta["donor_rfu_role_matrix"], dtype=torch.float32)
        acceptor_roles = torch.as_tensor(meta["acceptor_site_role_matrix"], dtype=torch.float32)
        donor_c1_positions = torch.nonzero(donor_roles[:, DONOR_ROLE_NAMES.index("C1")]).flatten()
        if donor_c1_positions.numel() != 1:
            raise ValueError(f"{row['Sample_ID']}: expected exactly one donor C1")
        acceptor_o4 = int(meta["acceptor_o4_index"])
        return {
            "sample_id": str(row["Sample_ID"]),
            "stress_type": str(row.get("Stress_Test_Type", "main")),
            "parent_group": str(row.get("Parent_Structural_Group_ID", row.get("Structural_Group_ID", ""))),
            "donor_graph": clone_graph(self.graphs[meta["donor_smiles"]]),
            "acceptor_graph": clone_graph(self.graphs[meta["acceptor_smiles"]]),
            "donor_indices": torch.tensor(donor_indices, dtype=torch.long),
            "acceptor_indices": torch.tensor(acceptor_indices, dtype=torch.long),
            "donor_roles": donor_roles,
            "acceptor_roles": acceptor_roles,
            "donor_c1_local_pos": int(donor_c1_positions.item()),
            "acceptor_o4_local_pos": acceptor_indices.index(acceptor_o4),
            "label": int(meta["label"]),
        }


def pad_indices(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(value.numel() for value in values)
    padded = torch.zeros((len(values), max_len), dtype=torch.long)
    mask = torch.zeros((len(values), max_len), dtype=torch.bool)
    for row, value in enumerate(values):
        padded[row, : value.numel()] = value
        mask[row, : value.numel()] = True
    return padded, mask


def pad_roles(values: list[torch.Tensor], max_len: int) -> torch.Tensor:
    padded = torch.zeros((len(values), max_len, values[0].shape[1]), dtype=torch.float32)
    for row, value in enumerate(values):
        padded[row, : value.shape[0]] = value
    return padded


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    donor_indices, donor_mask = pad_indices([item["donor_indices"] for item in items])
    acceptor_indices, acceptor_mask = pad_indices([item["acceptor_indices"] for item in items])
    return {
        "sample_ids": [item["sample_id"] for item in items],
        "stress_types": [item["stress_type"] for item in items],
        "parent_groups": [item["parent_group"] for item in items],
        "donor_graph": Batch.from_data_list([item["donor_graph"] for item in items]),
        "acceptor_graph": Batch.from_data_list([item["acceptor_graph"] for item in items]),
        "donor_rfu_atom_indices": donor_indices,
        "donor_rfu_mask": donor_mask,
        "donor_rfu_role_matrix": pad_roles([item["donor_roles"] for item in items], donor_indices.shape[1]),
        "donor_c1_local_pos": torch.tensor([item["donor_c1_local_pos"] for item in items]),
        "acceptor_oh_local_atom_indices": acceptor_indices,
        "acceptor_oh_mask": acceptor_mask,
        "acceptor_oh_role_matrix": pad_roles([item["acceptor_roles"] for item in items], acceptor_indices.shape[1]),
        "acceptor_o4_local_pos": torch.tensor([item["acceptor_o4_local_pos"] for item in items]),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }


class StructureOnlyChiralGINE(nn.Module):
    """Donor/acceptor Chiral-GINE with RFU-site interaction and no condition path."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        acceptor_role_dim: int = len(ACCEPTOR_ROLE_NAMES),
    ):
        super().__init__()
        encoder_args = dict(
            atom_feat_dim=atom_dim,
            bond_feat_dim=bond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            perm_cat_dropout=0.1,
            perm_cat_normalization="reference",
        )
        self.donor_encoder = ChiralGINEEncoder(**encoder_args)
        self.acceptor_encoder = ChiralGINEEncoder(**encoder_args)
        self.global_pool = GlobalAttnPool(hidden_dim)
        self.role_embedding = RoleEmbedding(
            donor_role_dim=len(DONOR_ROLE_NAMES),
            acceptor_role_dim=acceptor_role_dim,
            hidden_dim=hidden_dim,
        )
        self.local_interaction = RFUOHCrossAttentionTri(hidden_dim, num_heads=4, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        h_d = self.donor_encoder(batch["donor_graph"])
        h_a = self.acceptor_encoder(batch["acceptor_graph"])
        z_d_global = self.global_pool(h_d, batch["donor_graph"].batch, batch["donor_graph"].heavy_atom_mask)
        z_a_global = self.global_pool(h_a, batch["acceptor_graph"].batch, batch["acceptor_graph"].heavy_atom_mask)
        local_d = gather_local_nodes(h_d, batch["donor_graph"], batch["donor_rfu_atom_indices"], batch["donor_rfu_mask"])
        local_a = gather_local_nodes(h_a, batch["acceptor_graph"], batch["acceptor_oh_local_atom_indices"], batch["acceptor_oh_mask"])
        local_d = self.role_embedding.donor(local_d, batch["donor_rfu_role_matrix"])
        local_a = self.role_embedding.acceptor(local_a, batch["acceptor_oh_role_matrix"])
        interaction = self.local_interaction(
            local_d, local_a,
            batch["donor_rfu_mask"], batch["acceptor_oh_mask"],
            batch["donor_c1_local_pos"], batch["acceptor_o4_local_pos"],
        )
        return self.classifier(torch.cat([
            z_d_global, z_a_global, interaction.z_d_local,
            interaction.z_a_local, interaction.z_int,
        ], dim=-1))


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) or key in {"donor_graph", "acceptor_graph"}:
            result[key] = value.to(device)
    return result


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[pd.DataFrame, float | None]:
    model.eval()
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            logits = model(moved)
            if criterion is not None:
                count = int(moved["label"].numel())
                total_loss += float(criterion(logits, moved["label"]).item()) * count
                total_count += count
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            labels = batch["label"].tolist()
            for idx, probability in enumerate(probabilities):
                rows.append({
                    "sample_id": batch["sample_ids"][idx],
                    "stress_type": batch["stress_types"][idx],
                    "parent_group": batch["parent_groups"][idx],
                    "label": int(labels[idx]),
                    "probability": float(probability),
                })
    mean_loss = total_loss / total_count if criterion is not None else None
    return pd.DataFrame(rows), mean_loss


def run_train_epoch(model, loader, optimizer, criterion, device, gradient_clip: float) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), batch["label"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        count = batch["label"].numel()
        total_loss += float(loss.item()) * count
        total_count += count
    return total_loss / total_count


def evaluate_predictions(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    return compute_binary_metrics(frame["label"], frame["probability"], threshold=threshold)


def stress_summary(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    output: dict[str, Any] = {
        "overall": evaluate_predictions(frame, threshold),
        "n": int(len(frame)),
    }
    for stress_type, part in frame.groupby("stress_type", sort=True):
        prediction = (part["probability"].to_numpy() >= threshold).astype(int)
        output[str(stress_type)] = {
            "n": int(len(part)),
            "expected_label": int(part["label"].iloc[0]),
            "accuracy": float((prediction == part["label"].to_numpy()).mean()),
            "mean_probability_feasible": float(part["probability"].mean()),
            "min_probability_feasible": float(part["probability"].min()),
            "max_probability_feasible": float(part["probability"].max()),
        }
    return output


def rule_baseline_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the transparent O4-open/O4-substituted structural rule."""

    probabilities = []
    for value in dataframe["Acceptor_Target_Role_Indices"]:
        roles = parse_role_dict(value)
        probabilities.append(float(not roles.get("O4_EXTERNAL", [])))
    return pd.DataFrame(
        {
            "label": dataframe["hard_feasibility"].astype(int).to_numpy(),
            "probability": probabilities,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/processed/hard_feasibility_pre_model.csv")
    parser.add_argument("--stress-csv", default="data/processed/hard_feasibility_stress_test.csv")
    parser.add_argument("--graph-cache", default="data/processed/hard_feasibility_chiral_graph_cache.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss decrease counted as an improvement.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--output-dir", default="results/hard_feasibility_chiral_gine")
    parser.add_argument("--checkpoint-dir", default="checkpoints/hard_feasibility_chiral_gine")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Run one forward/backward batch and exit.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    acceptor_role_names = list(ACCEPTOR_ROLE_NAMES)
    dataset_args = {"acceptor_role_names": acceptor_role_names}
    train_ds = HardFeasibilityDataset(args.csv, args.graph_cache, split="train", **dataset_args)
    val_ds = HardFeasibilityDataset(args.csv, args.graph_cache, split="val", **dataset_args)
    test_ds = HardFeasibilityDataset(args.csv, args.graph_cache, split="test", **dataset_args)
    stress_ds = HardFeasibilityDataset(
        args.stress_csv,
        args.graph_cache,
        split="test",
        stress=True,
        **dataset_args,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    stress_loader = DataLoader(stress_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    first = train_ds[0]
    atom_dim = int(first["donor_graph"].x.shape[1])
    bond_dim = int(first["donor_graph"].edge_attr.shape[1])
    model = StructureOnlyChiralGINE(
        atom_dim,
        bond_dim,
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        acceptor_role_dim=len(acceptor_role_names),
    ).to(device)
    perm_params, base_params = [], []
    for name, parameter in model.named_parameters():
        (perm_params if "perm_cat" in name else base_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": perm_params, "lr": args.lr * 0.5},
    ], weight_decay=args.weight_decay)

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < args.warmup_epochs:
            return float(epoch_index + 1) / args.warmup_epochs
        progress = (epoch_index - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    if args.smoke_test:
        batch = move_batch(next(iter(train_loader)), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        print(json.dumps({
            "smoke_test": "pass",
            "device": str(device),
            "batch_size": int(batch["label"].numel()),
            "logit_shape": list(logits.shape),
            "loss": float(loss.item()),
        }, indent=2), flush=True)
        return 0

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"seed{args.seed}.pt"

    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, criterion, device, args.gradient_clip)
        val_frame, val_loss = predict(model, val_loader, device, criterion=criterion)
        assert val_loss is not None
        val_metrics = evaluate_predictions(val_frame, 0.5)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_auroc={val_metrics['auroc']:.4f}",
            flush=True,
        )
        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "acceptor_role_names": acceptor_role_names,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}", flush=True)
                break
        scheduler.step()

    checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    val_frame, _ = predict(model, val_loader, device)
    threshold, _ = search_threshold(val_frame["label"], val_frame["probability"])
    test_frame, _ = predict(model, test_loader, device)
    stress_frame, _ = predict(model, stress_loader, device)
    rule_test = rule_baseline_frame(test_ds.df)
    rule_stress = rule_baseline_frame(stress_ds.df)
    result = {
        "model": "structure_only_chiral_gine_crossattn_tri_no_o4_external_role",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_metric": "validation_cross_entropy_loss",
        "best_validation_loss": best_val_loss,
        "early_stop_patience": args.patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "threshold": threshold,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "validation": evaluate_predictions(val_frame, threshold),
        "test": evaluate_predictions(test_frame, threshold),
        "stress": stress_summary(stress_frame, threshold),
        "rule_baseline": {
            "definition": "feasible iff O4_EXTERNAL is absent from the curated site annotation",
            "test": evaluate_predictions(rule_test, 0.5),
            "stress": evaluate_predictions(rule_stress, 0.5),
        },
        "checkpoint": str(checkpoint_path),
        "acceptor_role_names": acceptor_role_names,
        "ablation": {
            "o4_external_role_removed": True,
            "external_atom_retained_in_graph": True,
        },
        "input_fields": ["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES", "target_site_annotations"],
        "excluded_fields": ["solvent", "catalyst", "temperature", "time"],
    }
    result_path = output_dir / f"seed{args.seed}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(output_dir / f"seed{args.seed}_history.csv", index=False)
    val_frame.to_csv(output_dir / f"seed{args.seed}_validation_predictions.csv", index=False)
    test_frame.to_csv(output_dir / f"seed{args.seed}_test_predictions.csv", index=False)
    stress_frame.to_csv(output_dir / f"seed{args.seed}_stress_predictions.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
