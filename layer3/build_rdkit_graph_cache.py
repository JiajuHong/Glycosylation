#!/usr/bin/env python
"""第三层数据加速：构建可复用的普通 GINE 或 Chiral-GINE 图缓存。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from layer3.chiral_graph import CHIRAL_GRAPH_FEATURE_VERSION, mol_to_chiral_pyg_graph
from layer3.glyco_dataset import (
    ATOM_FEATURE_NAMES,
    BOND_FEATURE_NAMES,
    GRAPH_FEATURE_VERSION,
    atom_features,
    bond_features,
    mol_from_smiles,
    mol_to_pyg_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/processed/glyco_model_local.csv")
    parser.add_argument("--graph-type", choices=["gine", "chiral_gine"], default="gine")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or (
        "data/processed/rdkit_graph_cache.pt"
        if args.graph_type == "gine"
        else "data/processed/rdkit_chiral_graph_cache.pt"
    )
    feature_version = (
        GRAPH_FEATURE_VERSION if args.graph_type == "gine" else CHIRAL_GRAPH_FEATURE_VERSION
    )

    df = pd.read_csv(args.csv)
    required = ["Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    smiles_list = sorted(
        set(df["Donor_Canonical_SMILES"].astype(str)).union(
            set(df["Acceptor_Canonical_SMILES"].astype(str))
        )
    )

    graphs = {}
    failures = []
    for idx, smiles in enumerate(smiles_list, start=1):
        try:
            mol = mol_from_smiles(smiles)
            if args.graph_type == "chiral_gine":
                graphs[smiles] = mol_to_chiral_pyg_graph(
                    mol,
                    smiles,
                    atom_feature_fn=atom_features,
                    bond_feature_fn=bond_features,
                )
            else:
                graphs[smiles] = mol_to_pyg_graph(mol, smiles)
        except Exception as exc:  # noqa: BLE001 - report all cache build failures.
            failures.append({"smiles": smiles, "error": repr(exc)})
        if idx % 200 == 0:
            print(f"cached {idx}/{len(smiles_list)} unique SMILES", flush=True)

    if failures:
        failure_path = Path(output).with_suffix(".failures.csv")
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        raise RuntimeError(f"Failed to cache {len(failures)} SMILES. See {failure_path}")

    payload = {
        "feature_version": feature_version,
        "graph_type": args.graph_type,
        "atom_feature_names": ATOM_FEATURE_NAMES,
        "bond_feature_names": BOND_FEATURE_NAMES,
        "source_csv": str(Path(args.csv).resolve()),
        "n_unique_smiles": len(smiles_list),
        "graphs": graphs,
    }
    torch.save(payload, output)

    print("RDKit graph cache built")
    print(f"unique_smiles: {len(smiles_list)}")
    print(f"atom_feature_dim: {len(ATOM_FEATURE_NAMES)}")
    print(f"bond_feature_dim: {len(BOND_FEATURE_NAMES)}")
    print(f"graph_type: {args.graph_type}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
