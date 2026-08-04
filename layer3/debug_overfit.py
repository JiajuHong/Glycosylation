#!/usr/bin/env python
"""第三层诊断工具：在极小数据集上过拟合以检查数据流和梯度。"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from layer3.glyco_dataset import ENCODER_TYPES, GlycoDataset, glyco_collate_fn
from models.glyco_gine_models import MODEL_TYPES, build_glyco_gine_model
from layer3.train_gine import get_feature_dims, move_batch_to_device, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfit a tiny subset to debug GINE data flow.")
    parser.add_argument("--csv", default="data/processed/glyco_model_local.csv")
    parser.add_argument("--split-column", default="split_pair_group")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="global")
    parser.add_argument("--encoder-type", choices=ENCODER_TYPES, default="gine")
    parser.add_argument(
        "--chiral-graph-cache",
        default="data/processed/rdkit_chiral_graph_cache.pt",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    base_ds = GlycoDataset(
        args.csv,
        split_column=args.split_column,
        split="train",
        encoder_type=args.encoder_type,
        chiral_graph_cache_path=args.chiral_graph_cache,
        validate=True,
    )
    ds = Subset(base_ds, list(range(min(args.num_samples, len(base_ds)))))
    loader = DataLoader(ds, batch_size=len(ds), shuffle=True, collate_fn=glyco_collate_fn)
    atom_dim, bond_dim, num_solvent, num_catalyst = get_feature_dims(base_ds)
    model = build_glyco_gine_model(
        args.model_type,
        atom_feat_dim=atom_dim,
        bond_feat_dim=bond_dim,
        num_solvent_tokens=num_solvent,
        num_catalyst_tokens=num_catalyst,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch = move_batch_to_device(next(iter(loader)), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        pred = logits.argmax(dim=-1)
        acc = (pred == batch["label"]).float().mean().item()
        if epoch == 1 or epoch % 10 == 0 or acc >= 0.999:
            print(f"epoch={epoch:03d} loss={loss.item():.6f} acc={acc:.3f}")
        if acc >= 0.999:
            print("tiny overfit passed")
            return 0

    raise SystemExit("tiny overfit failed")


if __name__ == "__main__":
    raise SystemExit(main())
