"""普通 GINE 编码器：作为结构编码基础和手性模型对照。"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv


class GINEEncoder(nn.Module):
    """Atom-level GINE encoder for cached RDKit heavy-atom PyG graphs."""

    def __init__(
        self,
        atom_feat_dim: int = 15,
        bond_feat_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.atom_proj = nn.Linear(atom_feat_dim, hidden_dim)
        self.bond_proj = nn.Linear(bond_feat_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINEConv(mlp))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, graph_batch):
        x = self.atom_proj(graph_batch.x.float())
        edge_attr = self.bond_proj(graph_batch.edge_attr.float())
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, graph_batch.edge_index, edge_attr)
            x = norm(self.activation(x) + residual)
            x = self.dropout(x)
        return x
