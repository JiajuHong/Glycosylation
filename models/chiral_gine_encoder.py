"""原子级 Chiral-GINE 编码器，与普通 GINE 模型保持相同输入输出接口。"""

from __future__ import annotations

from torch import nn

from models.chiral_gine_conv import ChiralGINEConv


class ChiralGINEEncoder(nn.Module):
    """Drop-in encoder that retains the current GINE dimensions and residual stack."""

    def __init__(
        self,
        atom_feat_dim: int = 15,
        bond_feat_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        perm_cat_dropout: float | None = None,
        perm_cat_normalization: str = "reference",
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
            self.layers.append(
                ChiralGINEConv(
                    mlp,
                    hidden_dim=hidden_dim,
                    dropout=dropout if perm_cat_dropout is None else perm_cat_dropout,
                    perm_cat_normalization=perm_cat_normalization,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, graph_batch):
        required = (
            "parity_atoms",
            "tetra_center_index",
            "tetra_neighbor_index",
            "tetra_edge_index",
            "heavy_atom_mask",
        )
        missing = [name for name in required if not hasattr(graph_batch, name)]
        if missing:
            raise ValueError(f"Chiral-GINE graph is missing required tensors: {missing}")

        x = self.atom_proj(graph_batch.x.float())
        edge_attr = self.bond_proj(graph_batch.edge_attr.float())
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, graph_batch, edge_attr)
            x = norm(self.activation(x) + residual)
            x = self.dropout(x)
        return x
