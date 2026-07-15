"""PERM_CAT aggregation for tetrahedral chirality.

This is an independent, device-safe implementation of the aggregation idea in
Pattanaik et al., *Message Passing Networks for Molecules with Tetrahedral
Chirality* (2020).  It does not import their model or training framework.
"""

from __future__ import annotations

import torch
from torch import nn


# The alternating group A4: the 12 even permutations that preserve handedness.
EVEN_TETRAHEDRAL_PERMUTATIONS = torch.tensor(
    [
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 2, 1, 0],
    ],
    dtype=torch.long,
)


class PermCatAggregator(nn.Module):
    """Aggregate four ordered neighbour messages over chirality-preserving permutations."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.2,
        normalization: str = "reference",
    ) -> None:
        super().__init__()
        if normalization not in {"reference", "mean"}:
            raise ValueError("PERM_CAT normalization must be 'reference' or 'mean'")
        self.normalization = normalization
        self.register_buffer("even_permutations", EVEN_TETRAHEDRAL_PERMUTATIONS.clone())
        self.permutation_projection = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use explicit Xavier initialization to reduce run-to-run sensitivity."""

        nn.init.xavier_normal_(self.permutation_projection.weight, gain=1.0)
        nn.init.zeros_(self.permutation_projection.bias)
        for layer in self.output_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(layer.bias)

    def forward(self, ordered_messages: torch.Tensor) -> torch.Tensor:
        if ordered_messages.ndim != 3 or ordered_messages.shape[1] != 4:
            raise ValueError(
                "PERM_CAT expects ordered_messages with shape [num_centres, 4, hidden_dim]"
            )
        if ordered_messages.shape[0] == 0:
            return ordered_messages.new_empty((0, ordered_messages.shape[-1]))

        permuted = ordered_messages[:, self.even_permutations, :]
        concatenated = permuted.flatten(start_dim=2)
        projected = self.dropout(torch.relu(self.permutation_projection(concatenated)))
        # ``reference`` reproduces Pattanaik et al.'s /3 scaling. ``mean`` is a
        # lower-variance ablation that averages the 12 equivalent permutations.
        divisor = 3.0 if self.normalization == "reference" else float(self.even_permutations.size(0))
        aggregated = projected.sum(dim=1) / divisor
        return self.output_mlp(aggregated)
