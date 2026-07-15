from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.pooling import masked_mean_pool


def gather_local_nodes(
    h_atom: torch.Tensor,
    graph_batch,
    local_indices: torch.Tensor,
    local_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather per-graph local atom embeddings from a PyG batch output."""

    batch_size, max_len = local_indices.shape
    hidden_dim = h_atom.size(-1)
    out = h_atom.new_zeros((batch_size, max_len, hidden_dim))
    ptr = graph_batch.ptr.to(local_indices.device)
    for b in range(batch_size):
        valid = local_mask[b]
        if valid.any():
            global_idx = ptr[b] + local_indices[b, valid]
            out[b, valid] = h_atom[global_idx]
    return out


class RoleEmbedding(nn.Module):
    def __init__(self, donor_role_dim: int = 7, acceptor_role_dim: int = 10, hidden_dim: int = 128) -> None:
        super().__init__()
        self.donor_role_embedding = nn.Parameter(torch.empty(donor_role_dim, hidden_dim))
        self.acceptor_role_embedding = nn.Parameter(torch.empty(acceptor_role_dim, hidden_dim))
        self.donor_norm = nn.LayerNorm(hidden_dim)
        self.acceptor_norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.donor_role_embedding)
        nn.init.xavier_uniform_(self.acceptor_role_embedding)

    def donor(self, h_local: torch.Tensor, role_matrix: torch.Tensor) -> torch.Tensor:
        role_emb = role_matrix.float() @ self.donor_role_embedding
        return self.donor_norm(h_local + role_emb)

    def acceptor(self, h_local: torch.Tensor, role_matrix: torch.Tensor) -> torch.Tensor:
        role_emb = role_matrix.float() @ self.acceptor_role_embedding
        return self.acceptor_norm(h_local + role_emb)


class RFUOHCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.rfu_to_oh = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.oh_to_rfu = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_rfu = nn.LayerNorm(hidden_dim)
        self.norm_oh = nn.LayerNorm(hidden_dim)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        h_rfu: torch.Tensor,
        h_oh: torch.Tensor,
        rfu_mask: torch.Tensor,
        oh_mask: torch.Tensor,
        donor_c1_local_pos: torch.Tensor,
        acceptor_o4_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rfu_from_oh, _ = self.rfu_to_oh(
            query=h_rfu,
            key=h_oh,
            value=h_oh,
            key_padding_mask=~oh_mask,
            need_weights=False,
        )
        oh_from_rfu, _ = self.oh_to_rfu(
            query=h_oh,
            key=h_rfu,
            value=h_rfu,
            key_padding_mask=~rfu_mask,
            need_weights=False,
        )
        rfu_from_oh = self.norm_rfu(h_rfu + rfu_from_oh)
        oh_from_rfu = self.norm_oh(h_oh + oh_from_rfu)

        t_rfu_from_oh = masked_mean_pool(rfu_from_oh, rfu_mask)
        t_oh_from_rfu = masked_mean_pool(oh_from_rfu, oh_mask)
        batch_idx = torch.arange(h_rfu.size(0), device=h_rfu.device)
        h_c1 = rfu_from_oh[batch_idx, donor_c1_local_pos]
        h_o4 = oh_from_rfu[batch_idx, acceptor_o4_local_pos]
        t_int = self.interaction_mlp(
            torch.cat(
                [
                    t_rfu_from_oh,
                    t_oh_from_rfu,
                    h_c1,
                    h_o4,
                    h_c1 * h_o4,
                    torch.abs(h_c1 - h_o4),
                ],
                dim=-1,
            )
        )
        return t_int, rfu_from_oh, oh_from_rfu


@dataclass
class LocalInteractionOutput:
    """Traceable outputs of the RFU–4-OH three-representation interaction block."""

    z_d_local: torch.Tensor
    z_a_local: torch.Tensor
    z_int: torch.Tensor
    h_d_updated: torch.Tensor
    h_a_updated: torch.Tensor
    message_d_from_a: torch.Tensor
    message_a_from_d: torch.Tensor


class AnchorGuidedAttentionPool(nn.Module):
    """Pool a local region using its reaction-centre atom as the query."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        local_states: torch.Tensor,
        local_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(local_states.size(0), device=local_states.device)
        anchor = local_states[batch_idx, anchor_positions]
        attended, _ = self.attention(
            query=anchor.unsqueeze(1),
            key=local_states,
            value=local_states,
            key_padding_mask=~local_mask,
            need_weights=False,
        )
        return self.norm(anchor + self.dropout(attended.squeeze(1)))


class RFUOHCrossAttentionTri(nn.Module):
    """Bidirectional RFU–4-OH interaction with three non-redundant outputs.

    ``z_d_local`` and ``z_a_local`` summarize the residual-updated local states,
    while ``z_int`` is constructed only from the messages exchanged by the two
    independent attention directions.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.d_from_a_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.a_from_d_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.message_dropout = nn.Dropout(dropout)
        self.d_message_norm = nn.LayerNorm(hidden_dim)
        self.a_message_norm = nn.LayerNorm(hidden_dim)
        self.d_ffn = self._make_ffn(hidden_dim, dropout)
        self.a_ffn = self._make_ffn(hidden_dim, dropout)
        self.d_ffn_norm = nn.LayerNorm(hidden_dim)
        self.a_ffn_norm = nn.LayerNorm(hidden_dim)
        self.d_pool = AnchorGuidedAttentionPool(hidden_dim, num_heads, dropout)
        self.a_pool = AnchorGuidedAttentionPool(hidden_dim, num_heads, dropout)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _make_ffn(hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_d: torch.Tensor,
        h_a: torch.Tensor,
        d_mask: torch.Tensor,
        a_mask: torch.Tensor,
        donor_c1_local_pos: torch.Tensor,
        acceptor_o4_local_pos: torch.Tensor,
    ) -> LocalInteractionOutput:
        message_d_from_a, _ = self.d_from_a_attention(
            query=h_d,
            key=h_a,
            value=h_a,
            key_padding_mask=~a_mask,
            need_weights=False,
        )
        message_a_from_d, _ = self.a_from_d_attention(
            query=h_a,
            key=h_d,
            value=h_d,
            key_padding_mask=~d_mask,
            need_weights=False,
        )

        y_d = self.d_message_norm(h_d + self.message_dropout(message_d_from_a))
        y_a = self.a_message_norm(h_a + self.message_dropout(message_a_from_d))
        h_d_updated = self.d_ffn_norm(y_d + self.d_ffn(y_d))
        h_a_updated = self.a_ffn_norm(y_a + self.a_ffn(y_a))

        # Keep padded positions inert in returned node-level representations.
        h_d_updated = h_d_updated * d_mask.unsqueeze(-1)
        h_a_updated = h_a_updated * a_mask.unsqueeze(-1)
        message_d_from_a = message_d_from_a * d_mask.unsqueeze(-1)
        message_a_from_d = message_a_from_d * a_mask.unsqueeze(-1)

        z_d_local = self.d_pool(h_d_updated, d_mask, donor_c1_local_pos)
        z_a_local = self.a_pool(h_a_updated, a_mask, acceptor_o4_local_pos)

        pooled_d_message = masked_mean_pool(message_d_from_a, d_mask)
        pooled_a_message = masked_mean_pool(message_a_from_d, a_mask)
        z_int = self.interaction_mlp(
            torch.cat(
                [
                    pooled_d_message,
                    pooled_a_message,
                    torch.abs(pooled_d_message - pooled_a_message),
                    pooled_d_message * pooled_a_message,
                ],
                dim=-1,
            )
        )
        return LocalInteractionOutput(
            z_d_local=z_d_local,
            z_a_local=z_a_local,
            z_int=z_int,
            h_d_updated=h_d_updated,
            h_a_updated=h_a_updated,
            message_d_from_a=message_d_from_a,
            message_a_from_d=message_a_from_d,
        )
