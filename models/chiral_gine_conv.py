"""A GINE convolution whose tetrahedral centres use PERM_CAT aggregation."""

from __future__ import annotations

import torch
from torch import nn

from models.tetra_permutation import PermCatAggregator


class ChiralGINEConv(nn.Module):
    """Preserve ordinary GINE everywhere except labelled tetrahedral centres."""

    def __init__(
        self,
        mlp: nn.Module,
        hidden_dim: int,
        dropout: float = 0.2,
        perm_cat_normalization: str = "reference",
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.register_buffer("eps", torch.tensor(0.0))
        self.perm_cat = PermCatAggregator(
            hidden_dim=hidden_dim,
            dropout=dropout,
            normalization=perm_cat_normalization,
        )

    def forward(self, x: torch.Tensor, graph_batch, edge_attr: torch.Tensor) -> torch.Tensor:
        source, target = graph_batch.edge_index
        messages = torch.relu(x[source] + edge_attr)
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(0, target, messages)

        centres = graph_batch.tetra_center_index
        if centres.numel():
            neighbours = graph_batch.tetra_neighbor_index
            edge_ids = graph_batch.tetra_edge_index
            ordered_messages = x[neighbours] + edge_attr[edge_ids]

            # The reference convention canonicalizes CCW by one odd swap, after
            # which the 12 even permutations describe the same handedness.
            ccw = graph_batch.parity_atoms[centres].eq(-1)
            if ccw.any():
                ordered_messages = ordered_messages.clone()
                ordered_messages[ccw] = ordered_messages[ccw][:, [1, 0, 2, 3], :]
            aggregated[centres] = self.perm_cat(ordered_messages)

        return self.mlp((1.0 + self.eps) * x + aggregated)
