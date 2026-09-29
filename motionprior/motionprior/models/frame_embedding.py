"""Full q0-q15 temporal history representation from frozen codec embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .semoco_token_interface import FrozenSeMoCoTokenInterface


@dataclass
class FrameEmbeddingOutput:
    hidden: Tensor
    main: Tensor
    residual: Tensor
    semantic: Tensor
    kinematic: Tensor


class FullFrameEmbedding(nn.Module):
    def __init__(
        self,
        token_interface: FrozenSeMoCoTokenInterface,
        *,
        d_model: int = 512,
        detail_dim: int = 64,
    ) -> None:
        super().__init__()
        if d_model != token_interface.latent_dim:
            raise ValueError("pilot requires d_model equal to SeMoCo latent_dim")
        self.token_interface = token_interface
        self.d_model = int(d_model)
        self.detail_dim = int(detail_dim)
        self.main_norm = nn.RMSNorm(self.d_model)
        self.detail_projections = nn.ModuleList(
            nn.Linear(self.d_model, self.detail_dim) for _ in range(16)
        )
        self.detail_norm = nn.RMSNorm(16 * self.detail_dim)
        self.detail_output = nn.Linear(16 * self.detail_dim, self.d_model)
        self.detail_gate = nn.Parameter(torch.zeros(()))
        self.bos_frame_embedding = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.bos_frame_embedding, std=0.02)

    def forward(self, tokens: Tensor) -> FrameEmbeddingOutput:
        if tokens.ndim != 3 or tokens.shape[-1] != 16:
            raise ValueError("held tokens must be [B,T,16]")
        values = tokens.long()
        bos_members = values.eq(self.token_interface.bos_id)
        bos_frames = bos_members.all(dim=-1)
        if torch.any(bos_members.any(dim=-1) & ~bos_frames):
            raise ValueError("BOS must occupy an entire frame; mixed BOS/codec frames are invalid")
        if torch.any((values < 0) | (values > self.token_interface.bos_id)):
            raise ValueError("held token outside codec/BOS range")

        flat = values.reshape(-1, 16)
        flat_bos = bos_frames.reshape(-1)
        main_flat = self.bos_frame_embedding.expand(flat.shape[0], -1).clone()
        residual_flat = torch.zeros_like(main_flat)
        if (~flat_bos).any():
            codec_tokens = flat[~flat_bos].view(-1, 1, 16)
            layers = self.token_interface.layer_embeddings(codec_tokens).squeeze(1)
            main_value = self.main_norm(layers.sum(dim=1))
            main_flat[~flat_bos] = main_value.to(main_flat.dtype)
            detail = torch.cat(
                [projection(layers[:, depth]) for depth, projection in enumerate(self.detail_projections)],
                dim=-1,
            )
            residual_value = self.detail_output(self.detail_norm(detail))
            # Autocast may produce BF16 linear outputs while the learned BOS
            # parameter (and therefore these scatter buffers) remains FP32.
            residual_flat[~flat_bos] = residual_value.to(residual_flat.dtype)
        shape = (*values.shape[:2], self.d_model)
        main = main_flat.view(shape)
        residual = residual_flat.view(shape)
        return FrameEmbeddingOutput(
            hidden=main + self.detail_gate * residual,
            main=main,
            residual=residual,
            semantic=main,
            kinematic=self.detail_gate * residual,
        )


class AxisSeparatedFrameEmbedding(nn.Module):
    """Explicit semantic-q0 and layer-aware kinematic-q1:q15 history."""

    def __init__(
        self,
        token_interface: FrozenSeMoCoTokenInterface,
        *,
        d_model: int = 512,
        detail_dim: int = 64,
    ) -> None:
        super().__init__()
        self.token_interface = token_interface
        self.d_model = int(d_model)
        self.codec_dim = int(token_interface.latent_dim)
        self.detail_dim = int(detail_dim)
        self.semantic_input_projection = (
            nn.Identity()
            if self.codec_dim == self.d_model
            else nn.Linear(self.codec_dim, self.d_model, bias=False)
        )
        self.semantic_norm = nn.RMSNorm(self.d_model)
        self.semantic_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.kinematic_projections = nn.ModuleList(
            nn.Linear(self.codec_dim, self.detail_dim) for _ in range(15)
        )
        self.kinematic_norm = nn.RMSNorm(15 * self.detail_dim)
        self.kinematic_output = nn.Linear(15 * self.detail_dim, self.d_model, bias=False)
        self.fusion_norm = nn.RMSNorm(2 * self.d_model)
        self.fusion_output = nn.Linear(2 * self.d_model, self.d_model, bias=False)
        self.semantic_bos_embedding = nn.Parameter(torch.empty(self.d_model))
        self.kinematic_bos_embedding = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.semantic_bos_embedding, std=0.02)
        nn.init.normal_(self.kinematic_bos_embedding, std=0.02)

    def forward(self, tokens: Tensor) -> FrameEmbeddingOutput:
        if tokens.ndim != 3 or tokens.shape[-1] != 16:
            raise ValueError("held tokens must be [B,T,16]")
        values = tokens.long()
        bos_members = values.eq(self.token_interface.bos_id)
        bos_frames = bos_members.all(dim=-1)
        if torch.any(bos_members.any(dim=-1) & ~bos_frames):
            raise ValueError("BOS must occupy an entire frame; mixed BOS/codec frames are invalid")
        if torch.any((values < 0) | (values > self.token_interface.bos_id)):
            raise ValueError("held token outside codec/BOS range")

        flat = values.reshape(-1, 16)
        flat_bos = bos_frames.reshape(-1)
        semantic_flat = self.semantic_bos_embedding.expand(flat.shape[0], -1).clone()
        kinematic_flat = self.kinematic_bos_embedding.expand(flat.shape[0], -1).clone()
        if (~flat_bos).any():
            codec_tokens = flat[~flat_bos].view(-1, 1, 16)
            layers = self.token_interface.layer_embeddings(codec_tokens).squeeze(1)
            semantic = self.semantic_projection(
                self.semantic_norm(self.semantic_input_projection(layers[:, 0]))
            )
            kinematic_parts = [
                projection(layers[:, depth + 1])
                for depth, projection in enumerate(self.kinematic_projections)
            ]
            kinematic = self.kinematic_output(
                self.kinematic_norm(torch.cat(kinematic_parts, dim=-1))
            )
            semantic_flat[~flat_bos] = semantic.to(semantic_flat.dtype)
            kinematic_flat[~flat_bos] = kinematic.to(kinematic_flat.dtype)
        shape = (*values.shape[:2], self.d_model)
        semantic = semantic_flat.view(shape)
        kinematic = kinematic_flat.view(shape)
        hidden = self.fusion_output(self.fusion_norm(torch.cat((semantic, kinematic), dim=-1)))
        return FrameEmbeddingOutput(
            hidden=hidden,
            main=semantic,
            residual=kinematic,
            semantic=semantic,
            kinematic=kinematic,
        )
