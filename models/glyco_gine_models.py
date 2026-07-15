from __future__ import annotations

import torch
from torch import nn

from models.condition_encoder import ConditionEncoder
from models.chiral_gine_encoder import ChiralGINEEncoder
from models.gine_encoder import GINEEncoder
from models.local_modules import (
    LocalInteractionOutput,
    RFUOHCrossAttention,
    RFUOHCrossAttentionTri,
    RoleEmbedding,
    gather_local_nodes,
)
from models.pooling import GlobalAttnPool, masked_mean_pool


MODEL_TYPES = ("global", "local", "crossattn", "crossattn_tri")
LOCAL_OUTPUT_MODES = ("l1", "l2", "l3")


def make_classifier(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 2),
    )


class GlycoGINEBase(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int,
        bond_feat_dim: int,
        num_solvent_tokens: int,
        num_catalyst_tokens: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        encoder_type: str = "gine",
        perm_cat_dropout: float | None = None,
        perm_cat_normalization: str = "reference",
    ) -> None:
        super().__init__()
        encoder_classes = {
            "gine": GINEEncoder,
            "chiral_gine": ChiralGINEEncoder,
        }
        if encoder_type not in encoder_classes:
            raise ValueError(f"Unknown encoder_type {encoder_type!r}; choose from {sorted(encoder_classes)}")
        encoder_cls = encoder_classes[encoder_type]
        self.encoder_type = encoder_type
        encoder_kwargs = {}
        if encoder_type == "chiral_gine":
            encoder_kwargs = {
                "perm_cat_dropout": perm_cat_dropout,
                "perm_cat_normalization": perm_cat_normalization,
            }
        self.donor_encoder = encoder_cls(
            atom_feat_dim, bond_feat_dim, hidden_dim, num_layers, dropout, **encoder_kwargs
        )
        self.acceptor_encoder = encoder_cls(
            atom_feat_dim, bond_feat_dim, hidden_dim, num_layers, dropout, **encoder_kwargs
        )
        self.global_pool = GlobalAttnPool(hidden_dim)
        self.condition_encoder = ConditionEncoder(
            num_solvent_tokens=num_solvent_tokens,
            num_catalyst_tokens=num_catalyst_tokens,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def encode_common(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h_donor = self.donor_encoder(batch["donor_graph"])
        h_acceptor = self.acceptor_encoder(batch["acceptor_graph"])
        donor_heavy_mask = getattr(batch["donor_graph"], "heavy_atom_mask", None)
        acceptor_heavy_mask = getattr(batch["acceptor_graph"], "heavy_atom_mask", None)
        t_donor = self.global_pool(
            h_donor, batch["donor_graph"].batch, node_mask=donor_heavy_mask
        )
        t_acceptor = self.global_pool(
            h_acceptor, batch["acceptor_graph"].batch, node_mask=acceptor_heavy_mask
        )
        t_solv, t_cat, t_temp, t_time = self.condition_encoder(batch)
        return {
            "h_donor": h_donor,
            "h_acceptor": h_acceptor,
            "t_donor": t_donor,
            "t_acceptor": t_acceptor,
            "t_solv": t_solv,
            "t_cat": t_cat,
            "t_temp": t_temp,
            "t_time": t_time,
        }


class GlycoGINEGlobal(GlycoGINEBase):
    def __init__(self, *args, hidden_dim: int = 128, dropout: float = 0.2, **kwargs) -> None:
        super().__init__(*args, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.classifier = make_classifier(hidden_dim * 6, hidden_dim, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        enc = self.encode_common(batch)
        z = torch.cat(
            [enc["t_donor"], enc["t_acceptor"], enc["t_solv"], enc["t_cat"], enc["t_temp"], enc["t_time"]],
            dim=-1,
        )
        return self.classifier(z)


class GlycoGINELocal(GlycoGINEBase):
    def __init__(self, *args, hidden_dim: int = 128, dropout: float = 0.2, **kwargs) -> None:
        super().__init__(*args, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.role_embedding = RoleEmbedding(hidden_dim=hidden_dim)
        self.classifier = make_classifier(hidden_dim * 8, hidden_dim, dropout)

    def local_tokens(self, batch: dict[str, torch.Tensor], enc: dict[str, torch.Tensor]):
        h_rfu = gather_local_nodes(
            enc["h_donor"],
            batch["donor_graph"],
            batch["donor_rfu_atom_indices"],
            batch["donor_rfu_mask"],
        )
        h_oh = gather_local_nodes(
            enc["h_acceptor"],
            batch["acceptor_graph"],
            batch["acceptor_oh_local_atom_indices"],
            batch["acceptor_oh_mask"],
        )
        h_rfu_role = self.role_embedding.donor(h_rfu, batch["donor_rfu_role_matrix"])
        h_oh_role = self.role_embedding.acceptor(h_oh, batch["acceptor_oh_role_matrix"])
        t_rfu = masked_mean_pool(h_rfu_role, batch["donor_rfu_mask"])
        t_oh = masked_mean_pool(h_oh_role, batch["acceptor_oh_mask"])
        return h_rfu_role, h_oh_role, t_rfu, t_oh

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        enc = self.encode_common(batch)
        _, _, t_rfu, t_oh = self.local_tokens(batch, enc)
        z = torch.cat(
            [
                enc["t_donor"],
                enc["t_acceptor"],
                t_rfu,
                t_oh,
                enc["t_solv"],
                enc["t_cat"],
                enc["t_temp"],
                enc["t_time"],
            ],
            dim=-1,
        )
        return self.classifier(z)


class GlycoGINECrossAttn(GlycoGINELocal):
    def __init__(self, *args, hidden_dim: int = 128, dropout: float = 0.2, num_heads: int = 4, **kwargs) -> None:
        super().__init__(*args, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.cross_attention = RFUOHCrossAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.classifier = make_classifier(hidden_dim * 9, hidden_dim, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        enc = self.encode_common(batch)
        h_rfu, h_oh, t_rfu, t_oh = self.local_tokens(batch, enc)
        t_int, _, _ = self.cross_attention(
            h_rfu,
            h_oh,
            batch["donor_rfu_mask"],
            batch["acceptor_oh_mask"],
            batch["donor_c1_local_pos"],
            batch["acceptor_o4_local_pos"],
        )
        z = torch.cat(
            [
                enc["t_donor"],
                enc["t_acceptor"],
                t_rfu,
                t_oh,
                t_int,
                enc["t_solv"],
                enc["t_cat"],
                enc["t_temp"],
                enc["t_time"],
            ],
            dim=-1,
        )
        return self.classifier(z)


class GlycoGINECrossAttnTri(GlycoGINELocal):
    """Cross-attention model with explicit zD_local, zA_local and zINT outputs."""

    def __init__(
        self,
        *args,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_heads: int = 4,
        local_output_mode: str = "l3",
        **kwargs,
    ) -> None:
        if local_output_mode not in LOCAL_OUTPUT_MODES:
            raise ValueError(
                f"Unknown local_output_mode {local_output_mode!r}; "
                f"choose from {LOCAL_OUTPUT_MODES}"
            )
        super().__init__(*args, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.local_output_mode = local_output_mode
        self.local_interaction = RFUOHCrossAttentionTri(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        local_token_count = {"l1": 2, "l2": 1, "l3": 3}[local_output_mode]
        # Two molecular graph tokens + selected local tokens + four condition tokens.
        self.classifier = make_classifier(
            hidden_dim * (2 + local_token_count + 4),
            hidden_dim,
            dropout,
        )

    def encode_local_interaction(
        self,
        batch: dict[str, torch.Tensor],
        enc: dict[str, torch.Tensor],
    ) -> LocalInteractionOutput:
        h_d, h_a, _, _ = self.local_tokens(batch, enc)
        return self.local_interaction(
            h_d,
            h_a,
            batch["donor_rfu_mask"],
            batch["acceptor_oh_mask"],
            batch["donor_c1_local_pos"],
            batch["acceptor_o4_local_pos"],
        )

    def _select_local_tokens(self, interaction: LocalInteractionOutput) -> list[torch.Tensor]:
        if self.local_output_mode == "l1":
            return [interaction.z_d_local, interaction.z_a_local]
        if self.local_output_mode == "l2":
            return [interaction.z_int]
        return [interaction.z_d_local, interaction.z_a_local, interaction.z_int]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        enc = self.encode_common(batch)
        interaction = self.encode_local_interaction(batch, enc)
        z = torch.cat(
            [
                enc["t_donor"],
                enc["t_acceptor"],
                *self._select_local_tokens(interaction),
                enc["t_solv"],
                enc["t_cat"],
                enc["t_temp"],
                enc["t_time"],
            ],
            dim=-1,
        )
        return self.classifier(z)


def build_glyco_gine_model(
    model_type: str,
    atom_feat_dim: int,
    bond_feat_dim: int,
    num_solvent_tokens: int,
    num_catalyst_tokens: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.2,
    encoder_type: str = "gine",
    perm_cat_dropout: float | None = None,
    perm_cat_normalization: str = "reference",
    local_output_mode: str = "l3",
):
    cls_map = {
        "global": GlycoGINEGlobal,
        "local": GlycoGINELocal,
        "crossattn": GlycoGINECrossAttn,
        "crossattn_tri": GlycoGINECrossAttnTri,
    }
    if model_type not in cls_map:
        raise ValueError(f"Unknown model_type {model_type}. Choose from {sorted(cls_map)}")
    model_kwargs = dict(
        atom_feat_dim=atom_feat_dim,
        bond_feat_dim=bond_feat_dim,
        num_solvent_tokens=num_solvent_tokens,
        num_catalyst_tokens=num_catalyst_tokens,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        encoder_type=encoder_type,
        perm_cat_dropout=perm_cat_dropout,
        perm_cat_normalization=perm_cat_normalization,
    )
    if model_type == "crossattn_tri":
        model_kwargs["local_output_mode"] = local_output_mode
    return cls_map[model_type](**model_kwargs)
