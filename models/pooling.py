"""通用池化组件：提供掩码平均、全局注意力和 PyG 图池化。"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(eps)


class GlobalAttnPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_mask is not None:
            node_mask = node_mask.bool()
            x = x[node_mask]
            batch = batch[node_mask]
        scores = self.score(x).squeeze(-1)
        weights = softmax(scores, batch).unsqueeze(-1)
        pooled = torch.zeros(
            int(batch.max().item()) + 1,
            x.size(-1),
            dtype=x.dtype,
            device=x.device,
        )
        pooled.index_add_(0, batch, weights * x)
        return pooled


def pyg_global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return global_mean_pool(x, batch)
