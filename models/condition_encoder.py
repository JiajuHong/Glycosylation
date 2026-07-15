from __future__ import annotations

import torch
from torch import nn

from models.pooling import masked_mean_pool


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        num_solvent_tokens: int,
        num_catalyst_tokens: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.solvent_embedding = nn.Embedding(num_solvent_tokens, hidden_dim)
        self.catalyst_embedding = nn.Embedding(num_catalyst_tokens, hidden_dim)
        self.temp_mlp = self._scalar_mlp(hidden_dim, dropout)
        self.time_mlp = self._scalar_mlp(hidden_dim, dropout)

    @staticmethod
    def _scalar_mlp(hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        solv = self.solvent_embedding(batch["solvent_component_ids"])
        cat = self.catalyst_embedding(batch["catalyst_component_ids"])
        t_solv = masked_mean_pool(solv, batch["solvent_mask"])
        t_cat = masked_mean_pool(cat, batch["catalyst_mask"])

        temp_input = torch.stack([batch["temp_norm"], batch["has_temp"]], dim=-1).float()
        time_input = torch.stack([batch["log_time_norm"], batch["has_time"]], dim=-1).float()
        t_temp = self.temp_mlp(temp_input)
        t_time = self.time_mlp(time_input)
        return t_solv, t_cat, t_temp, t_time
