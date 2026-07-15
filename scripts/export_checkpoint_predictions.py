#!/usr/bin/env python
"""Export per-reaction probabilities from a saved GINE/Chiral-GINE checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from glyco_dataset import GlycoDataset, glyco_collate_fn
from models.glyco_gine_models import build_glyco_gine_model
from train_gine import get_feature_dims, move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    device = torch.device(cli.device)
    checkpoint = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]

    dataset_kwargs = {
        "csv_path": train_args["csv"],
        "split_column": train_args["split_column"],
        "encoder_type": train_args.get("encoder_type", "gine"),
        "chiral_graph_cache_path": train_args.get(
            "chiral_graph_cache", "data/processed/rdkit_chiral_graph_cache.pt"
        ),
        "validate": True,
    }
    train_ds = GlycoDataset(split="train", **dataset_kwargs)
    atom_dim, bond_dim, num_solvent, num_catalyst = get_feature_dims(train_ds)
    model = build_glyco_gine_model(
        model_type=train_args["model_type"],
        atom_feat_dim=atom_dim,
        bond_feat_dim=bond_dim,
        num_solvent_tokens=num_solvent,
        num_catalyst_tokens=num_catalyst,
        hidden_dim=train_args["hidden_dim"],
        num_layers=train_args["num_layers"],
        dropout=train_args["dropout"],
        encoder_type=train_args.get("encoder_type", "gine"),
        perm_cat_dropout=train_args.get("perm_cat_dropout"),
        perm_cat_normalization=train_args.get("perm_cat_normalization", "reference"),
        local_output_mode=train_args.get("local_output_mode", "l3"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records: list[dict[str, object]] = []
    for split in cli.splits:
        dataset = GlycoDataset(split=split, stats=train_ds.stats, **dataset_kwargs)
        metadata = dataset.df.set_index("ID", drop=False)
        loader = DataLoader(
            dataset,
            batch_size=cli.batch_size,
            shuffle=False,
            collate_fn=glyco_collate_fn,
        )
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                probability = torch.softmax(model(batch), dim=-1)[:, 1].cpu()
                labels = batch["label"].cpu()
                ids = batch["ids"].cpu().tolist()
                for row_idx, reaction_id in enumerate(ids):
                    source = metadata.loc[reaction_id]
                    prob_beta = float(probability[row_idx])
                    prediction = int(prob_beta >= cli.threshold)
                    record = {
                        "id": reaction_id,
                        "reaction_id": batch["reaction_ids"][row_idx],
                        "split": split,
                        "label": int(labels[row_idx]),
                        "prob_beta": prob_beta,
                        "threshold": cli.threshold,
                        "prediction": prediction,
                        "correct": int(prediction == int(labels[row_idx])),
                        "donor_smiles": batch["donor_smiles"][row_idx],
                        "acceptor_smiles": batch["acceptor_smiles"][row_idx],
                        "encoder_type": train_args.get("encoder_type", "gine"),
                        "model_type": train_args["model_type"],
                        "seed": train_args["seed"],
                    }
                    for column in ("Pair_Key", "Donor_Type", "Year"):
                        if column in metadata.columns:
                            record[column] = source[column]
                    records.append(record)

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"wrote {len(records)} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
