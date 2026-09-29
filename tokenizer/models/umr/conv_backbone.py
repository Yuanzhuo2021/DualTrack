"""Shared 1D-conv encoder/decoder backbone for the UMR structured codec.

Progressive dilated-conv encoder/decoder primitives consumed by
:mod:`models.umr.structured_vq` (the native, config-driven codec). These blocks
are quantizer-agnostic; quantization is handled separately by
:mod:`models.quantizers`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    DIM_JOINTS76_ROT6D,
    DIM_ROOT_ROT6D,
    DIM_ROOT_TRAJ,
    DIM_SPARSE_VEL,
    SLICE_FOOT_CONTACT,
    SLICE_JOINTS76_ROT6D,
    SLICE_ROOT_ROT6D,
    SLICE_ROOT_TRAJ,
    SLICE_SPARSE_VEL,
)
from models.causal_layers import (
    CausalConv1d,
    DownsampleBlock1d,
    UpsampleBlock1d,
    _ChannelNorm,
)


def _project(in_ch: int, out_ch: int, *, causal: bool = True) -> nn.Module:
    return CausalConv1d(in_ch, out_ch, kernel_size=1, causal=causal)


def _norm_layer(channels: int, mode: str) -> nn.Module:
    if mode == "channel":
        return _ChannelNorm(channels)
    if mode == "none":
        return nn.Identity()
    raise ValueError(f"unknown residual_norm={mode!r}; expected 'channel' or 'none'")


def _factor_stride(stride: int, layers: int) -> list[int]:
    if stride < 1:
        raise ValueError(f"temporal_stride must be >= 1; got {stride}")
    if layers < 1:
        raise ValueError(f"num_downsample_layers must be >= 1; got {layers}")
    if stride == 1:
        return [1]
    factors: list[int] = []
    remaining = stride
    for _ in range(layers - 1):
        if remaining % 2 != 0:
            raise ValueError(f"temporal_stride={stride} cannot be split into {layers} stride-2 stages")
        factors.append(2)
        remaining //= 2
    factors.append(remaining)
    if any(f < 1 for f in factors):
        raise ValueError(f"invalid stride factorization for temporal_stride={stride}, layers={layers}")
    return factors


class DilatedResBlock1d(nn.Module):
    """Residual block with dilation and dropout."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        causal: bool,
        dropout: float,
        norm: str,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _norm_layer(channels, norm),
            nn.ReLU(),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation, causal=causal),
            _norm_layer(channels, norm),
            nn.ReLU(),
            CausalConv1d(channels, channels, kernel_size=1, causal=causal),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


def _resnet_stage(
    channels: int,
    *,
    depth: int,
    kernel_size: int,
    dilation_growth_rate: int,
    causal: bool,
    dropout: float,
    norm: str,
    reverse_dilation: bool = False,
) -> nn.Module:
    dilations = [dilation_growth_rate ** i for i in range(depth)]
    if reverse_dilation:
        dilations = list(reversed(dilations))
    return nn.Sequential(
        *[
            DilatedResBlock1d(
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                causal=causal,
                dropout=dropout,
                norm=norm,
            )
            for dilation in dilations
        ]
    )


class NearestUpsampleBlock1d(nn.Module):
    def __init__(self, channels: int, stride: int, *, causal: bool) -> None:
        super().__init__()
        self.stride = stride
        self.conv = CausalConv1d(channels, channels, kernel_size=3, causal=causal)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(torch.repeat_interleave(x, self.stride, dim=-1))


@dataclass
class StructuredEncoderConfig:
    input_dim: int = DIM_FEATURES                # 499
    latent_dim: int = 512
    temporal_stride: int = 2
    num_residual_blocks: int = 6
    kernel_size: int = 3
    causal: bool = True
    num_downsample_layers: int = 1
    residual_blocks_per_stage: int | None = None
    dilation_growth_rate: int = 3
    residual_dropout: float = 0.0
    residual_norm: str = "channel"               # channel | none


class StructuredEncoder(nn.Module):
    """499D record → ``latent_dim`` token-rate latent."""

    def __init__(self, cfg: StructuredEncoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or StructuredEncoderConfig()
        if cfg.input_dim != DIM_FEATURES:
            raise ValueError(f"StructuredEncoder.input_dim must be {DIM_FEATURES}; got {cfg.input_dim}")
        self.cfg = cfg

        strides = _factor_stride(cfg.temporal_stride, cfg.num_downsample_layers)
        depth = cfg.residual_blocks_per_stage or cfg.num_residual_blocks
        self.input_proj = nn.Sequential(
            CausalConv1d(cfg.input_dim, cfg.latent_dim, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        stages: list[nn.Module] = []
        for stride in strides:
            if stride > 1:
                stages.append(DownsampleBlock1d(cfg.latent_dim, stride=stride, causal=cfg.causal))
            stages.append(
                _resnet_stage(
                    cfg.latent_dim,
                    depth=depth,
                    kernel_size=cfg.kernel_size,
                    dilation_growth_rate=cfg.dilation_growth_rate,
                    causal=cfg.causal,
                    dropout=cfg.residual_dropout,
                    norm=cfg.residual_norm,
                )
            )
        self.stages = nn.Sequential(*stages)
        self.out_proj = nn.Sequential(
            _norm_layer(cfg.latent_dim, cfg.residual_norm),
            nn.ReLU(),
            CausalConv1d(cfg.latent_dim, cfg.latent_dim, kernel_size=3, causal=cfg.causal),
        )

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_dim

    @property
    def temporal_stride(self) -> int:
        return self.cfg.temporal_stride

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.shape[1] != self.cfg.input_dim:
            raise ValueError(
                f"StructuredEncoder expects [B, {self.cfg.input_dim}, T]; got {tuple(x.shape)}"
            )
        if x.shape[-1] % self.cfg.temporal_stride != 0:
            raise ValueError(
                f"T={x.shape[-1]} is not divisible by temporal_stride={self.cfg.temporal_stride}"
            )
        return self.out_proj(self.stages(self.input_proj(x)))


@dataclass
class StructuredDecoderConfig:
    output_dim: int = DIM_FEATURES               # 499
    latent_dim: int = 512
    temporal_stride: int = 2
    num_residual_blocks: int = 6
    kernel_size: int = 3
    causal: bool = True
    num_upsample_layers: int = 1
    residual_blocks_per_stage: int | None = None
    dilation_growth_rate: int = 3
    residual_dropout: float = 0.0
    upsample_mode: str = "transpose"
    residual_norm: str = "channel"               # channel | none
    decoder_head_mode: str = "split"             # split | single


class StructuredDecoder(nn.Module):
    """``latent_dim`` token-rate latent → 5 named heads → packed 499D feature."""

    def __init__(self, cfg: StructuredDecoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or StructuredDecoderConfig()
        if cfg.output_dim != DIM_FEATURES:
            raise ValueError(f"StructuredDecoder.output_dim must be {DIM_FEATURES}; got {cfg.output_dim}")
        self.cfg = cfg

        strides = _factor_stride(cfg.temporal_stride, cfg.num_upsample_layers)
        depth = cfg.residual_blocks_per_stage or cfg.num_residual_blocks
        self.input_proj = nn.Sequential(
            CausalConv1d(cfg.latent_dim, cfg.latent_dim, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        stages: list[nn.Module] = []
        for stride in reversed(strides):
            stages.append(
                _resnet_stage(
                    cfg.latent_dim,
                    depth=depth,
                    kernel_size=cfg.kernel_size,
                    dilation_growth_rate=cfg.dilation_growth_rate,
                    causal=cfg.causal,
                    dropout=cfg.residual_dropout,
                    norm=cfg.residual_norm,
                    reverse_dilation=True,
                )
            )
            if stride > 1:
                if cfg.upsample_mode == "nearest":
                    stages.append(NearestUpsampleBlock1d(cfg.latent_dim, stride=stride, causal=cfg.causal))
                elif cfg.upsample_mode == "transpose":
                    stages.append(UpsampleBlock1d(cfg.latent_dim, stride=stride))
                else:
                    raise ValueError(f"unknown upsample_mode={cfg.upsample_mode!r}")
        self.stages = nn.Sequential(*stages)
        self.refine = nn.Sequential(
            CausalConv1d(cfg.latent_dim, cfg.latent_dim, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        if cfg.decoder_head_mode == "single":
            # Single output head: one conv predicts the full packed feature vector.
            self.head_features = CausalConv1d(cfg.latent_dim, cfg.output_dim, kernel_size=3, causal=cfg.causal)
        elif cfg.decoder_head_mode == "split":
            # 5 task-specific heads concatenated to the 499D packed view.
            self.head_root_traj = _project(cfg.latent_dim, DIM_ROOT_TRAJ, causal=cfg.causal)
            self.head_root_rot6d = _project(cfg.latent_dim, DIM_ROOT_ROT6D, causal=cfg.causal)
            self.head_joints76_rot6d = _project(cfg.latent_dim, DIM_JOINTS76_ROT6D, causal=cfg.causal)
            self.head_sparse_vel = _project(cfg.latent_dim, DIM_SPARSE_VEL, causal=cfg.causal)
            self.head_foot_contact = _project(cfg.latent_dim, DIM_FOOT_CONTACT, causal=cfg.causal)
        else:
            raise ValueError(f"unknown decoder_head_mode={cfg.decoder_head_mode!r}")

    def forward(self, z_q: Tensor) -> dict[str, Tensor]:
        if z_q.dim() != 3 or z_q.shape[1] != self.cfg.latent_dim:
            raise ValueError(
                f"StructuredDecoder expects [B, {self.cfg.latent_dim}, T_token]; got {tuple(z_q.shape)}"
            )
        h = self.refine(self.stages(self.input_proj(z_q)))
        if self.cfg.decoder_head_mode == "single":
            features_rec = self.head_features(h)
            root_traj = features_rec[:, SLICE_ROOT_TRAJ]
            root_rot6d = features_rec[:, SLICE_ROOT_ROT6D]
            joints76_rot6d = features_rec[:, SLICE_JOINTS76_ROT6D]
            sparse_vel = features_rec[:, SLICE_SPARSE_VEL]
            foot_contact_logits = features_rec[:, SLICE_FOOT_CONTACT]
        else:
            root_traj = self.head_root_traj(h)
            root_rot6d = self.head_root_rot6d(h)
            joints76_rot6d = self.head_joints76_rot6d(h)
            sparse_vel = self.head_sparse_vel(h)
            foot_contact_logits = self.head_foot_contact(h)
            features_rec = torch.cat(
                [root_traj, root_rot6d, joints76_rot6d, sparse_vel, foot_contact_logits],
                dim=1,
            )
        return {
            "features": features_rec,
            "root_traj": root_traj,
            "root_rot6d": root_rot6d,
            "joints76_rot6d": joints76_rot6d,
            "sparse_vel": sparse_vel,
            "foot_contact_logits": foot_contact_logits,
        }


__all__ = [
    "DilatedResBlock1d",
    "NearestUpsampleBlock1d",
    "StructuredDecoder",
    "StructuredDecoderConfig",
    "StructuredEncoder",
    "StructuredEncoderConfig",
]
