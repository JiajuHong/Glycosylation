#!/usr/bin/env python
"""第三层训练入口：训练供体—受体—条件联合的立体选择性图模型。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.eval_metrics import compute_binary_metrics, search_threshold
from layer3.glyco_dataset import ENCODER_TYPES, GlycoDataset, glyco_collate_fn
from models.glyco_gine_models import (
    LOCAL_OUTPUT_MODES,
    MODEL_TYPES,
    build_glyco_gine_model,
)


def set_seed(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """Use an optional lower learning rate for the newly added PERM_CAT parameters."""

    perm_cat_params = []
    base_params = []
    for name, parameter in model.named_parameters():
        (perm_cat_params if "perm_cat" in name else base_params).append(parameter)
    groups = [{"params": base_params, "lr": args.lr, "name": "base"}]
    if perm_cat_params:
        groups.append(
            {
                "params": perm_cat_params,
                "lr": args.lr * args.perm_cat_lr_scale,
                "name": "perm_cat",
            }
        )
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.lr_scheduler == "none" and args.warmup_epochs == 0:
        return None

    def lr_factor(epoch_index: int) -> float:
        if args.warmup_epochs and epoch_index < args.warmup_epochs:
            return float(epoch_index + 1) / float(args.warmup_epochs)
        if args.lr_scheduler == "cosine":
            remaining = max(args.epochs - args.warmup_epochs, 1)
            progress = min(max(epoch_index - args.warmup_epochs, 0) / remaining, 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def apply_encoder_training_defaults(args: argparse.Namespace) -> None:
    """Resolve encoder-specific defaults while preserving explicit CLI overrides."""

    if args.encoder_type == "chiral_gine":
        defaults = {
            "deterministic": True,
            "perm_cat_dropout": 0.1,
            "perm_cat_normalization": "reference",
            "perm_cat_lr_scale": 0.5,
            "gradient_clip": 2.0,
            "warmup_epochs": min(10, max(args.epochs - 1, 0)),
            "lr_scheduler": "cosine",
            "patience": 40,
        }
        args.training_defaults = "chiral_stable_v1"
    else:
        defaults = {
            "deterministic": False,
            "perm_cat_dropout": None,
            "perm_cat_normalization": "reference",
            "perm_cat_lr_scale": 1.0,
            "gradient_clip": 5.0,
            "warmup_epochs": 0,
            "lr_scheduler": "none",
            "patience": 20,
        }
        args.training_defaults = "gine_original"

    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    tensor_keys = [
        "donor_rfu_atom_indices",
        "donor_rfu_mask",
        "donor_rfu_role_matrix",
        "donor_c1_local_pos",
        "acceptor_oh_local_atom_indices",
        "acceptor_oh_mask",
        "acceptor_oh_role_matrix",
        "acceptor_o4_local_pos",
        "solvent_component_ids",
        "solvent_mask",
        "catalyst_component_ids",
        "catalyst_mask",
        "has_temp",
        "temp_norm",
        "has_time",
        "log_time_norm",
        "label",
    ]
    moved = {key: value for key, value in batch.items()}
    moved["donor_graph"] = batch["donor_graph"].to(device)
    moved["acceptor_graph"] = batch["acceptor_graph"].to(device)
    for key in tensor_keys:
        moved[key] = batch[key].to(device)
    return moved


def get_feature_dims(dataset: GlycoDataset) -> tuple[int, int, int, int]:
    item = dataset[0]
    atom_dim = int(item["donor_graph"].x.shape[-1])
    bond_dim = int(item["donor_graph"].edge_attr.shape[-1])
    num_solvent = max(dataset.solvent_vocab.values()) + 1
    num_catalyst = max(dataset.catalyst_vocab.values()) + 1
    return atom_dim, bond_dim, num_solvent, num_catalyst


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.item()) * batch["label"].numel()
        total_count += batch["label"].numel()
    return total_loss / max(total_count, 1)


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device) -> tuple[list, list]:
    model.eval()
    labels, probs = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        labels.extend(batch["label"].detach().cpu().tolist())
        probs.extend(prob.detach().cpu().tolist())
    return labels, probs


@torch.no_grad()
def evaluate(
    model, loader, criterion, device: torch.device, threshold: float = 0.5
) -> dict:
    model.eval()
    total_loss = 0.0
    total_count = 0
    labels, probs = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        prob = torch.softmax(logits, dim=-1)[:, 1]
        labels.extend(batch["label"].detach().cpu().tolist())
        probs.extend(prob.detach().cpu().tolist())
        total_loss += float(loss.item()) * batch["label"].numel()
        total_count += batch["label"].numel()
    metrics = compute_binary_metrics(labels, probs, threshold=threshold)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GINE glycosylation models.")
    parser.add_argument("--csv", default="data/processed/glyco_model_local.csv")
    parser.add_argument("--split-column", default="split_pair_group")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="global")
    parser.add_argument(
        "--local-output-mode",
        choices=LOCAL_OUTPUT_MODES,
        default="l3",
        help="Local ablation for crossattn_tri: l1=local states, l2=interaction, l3=both.",
    )
    parser.add_argument("--encoder-type", choices=ENCODER_TYPES, default="gine")
    parser.add_argument(
        "--chiral-graph-cache",
        default="data/processed/rdkit_chiral_graph_cache.pt",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--perm-cat-dropout", type=float, default=None)
    parser.add_argument(
        "--perm-cat-normalization",
        choices=["reference", "mean"],
        default=None,
    )
    parser.add_argument("--perm-cat-lr-scale", type=float, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    determinism = parser.add_mutually_exclusive_group()
    determinism.add_argument("--deterministic", dest="deterministic", action="store_true")
    determinism.add_argument(
        "--non-deterministic", dest="deterministic", action="store_false"
    )
    parser.set_defaults(deterministic=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-csv", default="results/gine_results.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    apply_encoder_training_defaults(args)

    if args.gradient_clip <= 0:
        parser.error("--gradient-clip must be greater than zero")
    if not 0.0 <= args.perm_cat_lr_scale:
        parser.error("--perm-cat-lr-scale must be non-negative")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        parser.error("--warmup-epochs must be in [0, epochs)")

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)

    dataset_kwargs = {
        "split_column": args.split_column,
        "encoder_type": args.encoder_type,
        "chiral_graph_cache_path": args.chiral_graph_cache,
        "validate": True,
    }
    train_ds = GlycoDataset(args.csv, split="train", **dataset_kwargs)
    val_ds = GlycoDataset(args.csv, split="val", stats=train_ds.stats, **dataset_kwargs)
    test_ds = GlycoDataset(args.csv, split="test", stats=train_ds.stats, **dataset_kwargs)
    atom_dim, bond_dim, num_solvent, num_catalyst = get_feature_dims(train_ds)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=glyco_collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=glyco_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=glyco_collate_fn)

    model = build_glyco_gine_model(
        model_type=args.model_type,
        atom_feat_dim=atom_dim,
        bond_feat_dim=bond_dim,
        num_solvent_tokens=num_solvent,
        num_catalyst_tokens=num_catalyst,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        perm_cat_dropout=args.perm_cat_dropout,
        perm_cat_normalization=args.perm_cat_normalization,
        local_output_mode=args.local_output_mode,
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args)

    train_labels = train_ds.df["Label"].values
    n_alpha = int((train_labels == 0).sum())
    n_beta = int((train_labels == 1).sum())
    n_total = len(train_labels)
    w_alpha = n_total / (2 * n_alpha)
    w_beta = n_total / (2 * n_beta)
    class_weight = torch.tensor([w_alpha, w_beta], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.03)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    encoder_prefix = "" if args.encoder_type == "gine" else f"{args.encoder_type}_"
    model_checkpoint_name = args.model_type
    if args.model_type == "crossattn_tri":
        model_checkpoint_name = f"{model_checkpoint_name}_{args.local_output_mode}"
    run_suffix = f"_{args.run_id}" if args.run_id else ""
    ckpt_path = checkpoint_dir / (
        f"{encoder_prefix}{model_checkpoint_name}_{args.split_column}_seed{args.seed}{run_suffix}.pt"
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_clip=args.gradient_clip,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr_base": optimizer.param_groups[0]["lr"],
            "lr_perm_cat": next(
                (group["lr"] for group in optimizer.param_groups if group["name"] == "perm_cat"),
                float("nan"),
            ),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f}",
            flush=True,
        )
        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            patience_left = args.patience
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # --- threshold tuning on validation set ---
    val_labels, val_probs = collect_predictions(model, val_loader, device)
    best_threshold, val_tuned_metrics = search_threshold(val_labels, val_probs)
    print(f"tuned_threshold={best_threshold:.4f} val_tuned_macro_f1={val_tuned_metrics['macro_f1']:.4f}")

    val_full_metrics = evaluate(model, val_loader, criterion, device, threshold=best_threshold)
    test_metrics = evaluate(model, test_loader, criterion, device, threshold=best_threshold)
    # auROC / AUPRC are threshold-agnostic — recompute with include_threshold_agnostic=True
    val_full_metrics.update(
        {k: v for k, v in compute_binary_metrics(val_labels, val_probs).items() if k in ("auroc", "auprc")}
    )
    test_labels, test_probs = collect_predictions(model, test_loader, device)
    test_metrics.update(
        {k: v for k, v in compute_binary_metrics(test_labels, test_probs).items() if k in ("auroc", "auprc")}
    )

    result = {
        "model_type": args.model_type,
        "local_output_mode": args.local_output_mode,
        "use_conditions": True,
        "encoder_type": args.encoder_type,
        "split_column": args.split_column,
        "seed": args.seed,
        "run_id": args.run_id,
        "deterministic": args.deterministic,
        "training_defaults": args.training_defaults,
        "dropout": args.dropout,
        "perm_cat_dropout": args.perm_cat_dropout,
        "perm_cat_normalization": args.perm_cat_normalization,
        "perm_cat_lr_scale": args.perm_cat_lr_scale,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "warmup_epochs": args.warmup_epochs,
        "lr_scheduler": args.lr_scheduler,
        "patience": args.patience,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "best_epoch": best_epoch,
        "tuned_threshold": best_threshold,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        **{f"val_{k}": v for k, v in val_full_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "checkpoint": str(ckpt_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    result_df = pd.DataFrame([result])
    output_path = Path(args.results_csv)
    if output_path.exists():
        result_df = pd.concat([pd.read_csv(output_path), result_df], ignore_index=True)
    result_df.to_csv(output_path, index=False)
    pd.DataFrame(history).to_csv(ckpt_path.with_suffix(".history.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
