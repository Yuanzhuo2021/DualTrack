"""Structured VQ tokenizer (quantizer-agnostic, registry-driven).

Single 499D stream through one backbone encoder, one pluggable quantizer
(``ema_rvq`` / ``split_rvq``), and one backbone decoder. Design axes are
swappable via :class:`CodecConfig`: backbone family (``conv``; ``transformer``
reserved), quantizer kind (see :mod:`models.quantizers`), and causal/streaming
(``decode_indices`` supports prefix-truncated decode). The conv backbone reuses
:class:`StructuredEncoder` / :class:`StructuredDecoder` from
:mod:`models.umr.conv_backbone`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lcm

import torch
import torch.nn as nn
from torch import Tensor

from data.umr_schema import DIM_FEATURES
from models.causal_layers import CausalConv1d, DownsampleBlock1d, UpsampleBlock1d
from models.quantizers import build_quantizer
from models.umr.conv_backbone import (
    NearestUpsampleBlock1d,
    StructuredDecoder,
    StructuredDecoderConfig,
    StructuredEncoder,
    StructuredEncoderConfig,
    _factor_stride,
    _norm_layer,
    _resnet_stage,
)


@dataclass
class BackboneConfig:
    family: str = "conv"                 # conv | transformer (reserved)
    attn_mode: str = "self"              # transformer only: self | cross (reserved)
    latent_dim: int = 512
    temporal_stride: int = 2
    num_layers: int = 1                  # number of stride-2 down/up stages
    residual_blocks_per_stage: int = 3
    dilation_growth_rate: int = 3
    residual_dropout: float = 0.2
    residual_norm: str = "none"          # none | channel
    kernel_size: int = 3
    causal: bool = False
    upsample_mode: str = "nearest"       # nearest | transpose
    decoder_head_mode: str = "single"    # single | split


@dataclass
class QuantizerConfig:
    kind: str = "ema_rvq"                # ema_rvq | split_rvq
    num_quantizers: int = 8              # split_rvq: KINEMATIC layers (semantic is +1)
    codebook_size: int = 1024
    semantic_codebook_size: int = 1024  # split_rvq only: the depth-0 semantic VQ
    mu: float = 0.99
    quantize_dropout: bool = True
    quantize_dropout_cutoff_index: int = 0
    quantize_dropout_prob: float = 0.2
    sample_codebook_temp: float = 0.5

    def build_kwargs(self) -> dict:
        return {
            "num_quantizers": self.num_quantizers,
            "codebook_size": self.codebook_size,
            "semantic_codebook_size": self.semantic_codebook_size,
            "mu": self.mu,
            "quantize_dropout": self.quantize_dropout,
            "quantize_dropout_cutoff_index": self.quantize_dropout_cutoff_index,
            "quantize_dropout_prob": self.quantize_dropout_prob,
            "sample_codebook_temp": self.sample_codebook_temp,
        }


@dataclass
class GroupConfig:
    """One part-wise branch: a contiguous channel slice with its own codec.

    ``[start, stop)`` indexes the packed 499D feature; each group owns a
    backbone (its own ``temporal_stride`` -> token rate) and a quantizer. Groups
    must tile ``[0, input_dim)`` exactly (sorted, disjoint, gap-free) so the
    merged decoder output reproduces the 499D layout.
    """

    name: str
    start: int
    stop: int
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)


@dataclass
class CodecConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    input_dim: int = DIM_FEATURES
    groups: list[GroupConfig] | None = None   # None -> single-stream; else part-wise


@dataclass
class StructuredCodecOutput:
    features_rec: Tensor                          # [B, 499, T]
    z: Tensor | None                              # [B, latent, T_token] (single-stream only)
    z_q: Tensor | None                            # [B, latent, T_token] (single-stream only)
    indices: Tensor | dict[str, Tensor]           # single: [B, T_tok, Q]; part-wise: {name: idx}
    vq_loss: Tensor
    metrics: dict[str, Tensor]
    # Per-layer differentiable RVQ outputs [B, T_token, Q, latent], populated only
    # during training (or return_layer_codes) for the semantic head's early-prefix
    # sum(q0..q_{k-1}); None otherwise.
    layer_z_q: Tensor | None = None
    # Dedicated semantic-branch output [B, T_token, latent] from split_rvq,
    # distilled directly by the TMR SemanticHead; None for single-chain RVQ.
    z_sem: Tensor | None = None


class SemanticHead(nn.Module):
    """Project the RVQ early-prefix code onto the TMR teacher embedding space.

    ``per_token=False``: temporal mean-pool -> one unit-norm ``[B, dsem]`` clip
    vector. ``per_token=True``: no pool -> unit-norm ``[B, T_token, dsem]``,
    paired with a per-token teacher cache. Gradient flows back into the early
    RVQ layers and encoder; the teacher is stop-grad'd by the caller.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        dsem: int = 256,
        hidden_dim: int | None = None,
        per_token: bool = False,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or latent_dim
        self.per_token = bool(per_token)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dsem),
        )

    def forward(self, z_prefix: Tensor) -> Tensor:
        """``z_prefix`` ``[B, T_token, latent]`` -> unit-norm ``h_sem``.

        Returns ``[B, dsem]`` when ``per_token=False`` (temporal mean-pool) or
        ``[B, T_token, dsem]`` when ``per_token=True`` (no pool).
        """
        h = self.proj(z_prefix)               # [B, T_token, dsem]
        if not self.per_token:
            h = h.mean(dim=1)                  # temporal pool -> [B, dsem]
        return torch.nn.functional.normalize(h, dim=-1)


def build_backbone(cfg: BackboneConfig, role: str) -> nn.Module:
    """Build an encoder or decoder for the given backbone family.

    Contract: encoder maps ``[B, 499, T] -> [B, latent, T/stride]``; decoder maps
    ``[B, latent, T/stride] -> {"features": [B, 499, T], ...named fields}``.
    """
    if role not in {"encoder", "decoder"}:
        raise ValueError(f"role must be 'encoder' or 'decoder'; got {role!r}")

    if cfg.family == "conv":
        if role == "encoder":
            return StructuredEncoder(
                StructuredEncoderConfig(
                    latent_dim=cfg.latent_dim,
                    temporal_stride=cfg.temporal_stride,
                    kernel_size=cfg.kernel_size,
                    causal=cfg.causal,
                    num_downsample_layers=cfg.num_layers,
                    residual_blocks_per_stage=cfg.residual_blocks_per_stage,
                    dilation_growth_rate=cfg.dilation_growth_rate,
                    residual_dropout=cfg.residual_dropout,
                    residual_norm=cfg.residual_norm,
                )
            )
        return StructuredDecoder(
            StructuredDecoderConfig(
                latent_dim=cfg.latent_dim,
                temporal_stride=cfg.temporal_stride,
                kernel_size=cfg.kernel_size,
                causal=cfg.causal,
                num_upsample_layers=cfg.num_layers,
                residual_blocks_per_stage=cfg.residual_blocks_per_stage,
                dilation_growth_rate=cfg.dilation_growth_rate,
                residual_dropout=cfg.residual_dropout,
                upsample_mode=cfg.upsample_mode,
                residual_norm=cfg.residual_norm,
                decoder_head_mode=cfg.decoder_head_mode,
            )
        )

    if cfg.family == "transformer":
        raise NotImplementedError(
            f"transformer backbone (attn_mode={cfg.attn_mode!r}) is reserved; "
            "implement strided tokenization + symmetric upsample + causal masking "
            "to honour the encoder/decoder contract."
        )

    raise ValueError(f"unknown backbone family={cfg.family!r}; expected 'conv' or 'transformer'")


class StructuredVQTokenizer(nn.Module):
    """Single-stream 499D backbone encoder + pluggable quantizer + decoder."""

    def __init__(self, cfg: CodecConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or CodecConfig()
        self.cfg = cfg
        self.temporal_stride = cfg.backbone.temporal_stride

        self.encoder = build_backbone(cfg.backbone, "encoder")
        self.quantizer = build_quantizer(
            cfg.quantizer.kind, dim=cfg.backbone.latent_dim, **cfg.quantizer.build_kwargs()
        )
        self.decoder = build_backbone(cfg.backbone, "decoder")
        self._residual_decode = hasattr(self.quantizer, "get_layer_embeddings")

    def forward(self, features: Tensor) -> StructuredCodecOutput:
        if features.shape[1] != DIM_FEATURES:
            raise ValueError(f"features channels {features.shape[1]} != {DIM_FEATURES}")
        z = self.encoder(features)                       # [B, latent, T_token]
        z_btc = z.transpose(1, 2).contiguous()           # [B, T_token, latent]
        z_q_btc, indices, losses = self.quantizer(z_btc)
        # Move per-layer codes / semantic branch out of the losses dict into
        # typed fields so scalar metric logging stays clean.
        layer_z_q = losses.pop("layer_quantized", None)
        z_sem = losses.pop("z_sem", None)
        z_q = z_q_btc.transpose(1, 2).contiguous()
        out = self.decoder(z_q)
        return StructuredCodecOutput(
            features_rec=out["features"],
            z=z,
            z_q=z_q,
            indices=indices,
            vq_loss=losses["commit_loss"],
            metrics=losses,
            layer_z_q=layer_z_q,
            z_sem=z_sem,
        )

    @torch.no_grad()
    def encode_indices(self, features: Tensor) -> Tensor:
        return self.forward(features).indices

    @torch.no_grad()
    def decode_indices(self, indices: Tensor) -> Tensor:
        """Decode from token indices (residual stacks support prefix truncation)."""
        if not self._residual_decode:
            raise NotImplementedError(
                f"decode_indices requires a residual quantizer with get_layer_embeddings; "
                f"got {type(self.quantizer).__name__}"
            )
        layer_embeds = self.quantizer.get_layer_embeddings(indices)   # [B, T, K, latent]
        z_q_btc = layer_embeds.sum(dim=2)
        z_q = z_q_btc.transpose(1, 2).contiguous()
        return self.decoder(z_q)["features"]


class _GroupConvEncoder(nn.Module):
    """Generic conv encoder over an arbitrary channel count: ``[B, C, T] -> [B, latent, T/stride]``."""

    def __init__(self, in_dim: int, cfg: BackboneConfig) -> None:
        super().__init__()
        latent = cfg.latent_dim
        self.temporal_stride = cfg.temporal_stride
        self.latent_dim = latent
        strides = _factor_stride(cfg.temporal_stride, cfg.num_layers)
        self.input_proj = nn.Sequential(
            CausalConv1d(in_dim, latent, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        stages: list[nn.Module] = []
        for stride in strides:
            if stride > 1:
                stages.append(DownsampleBlock1d(latent, stride=stride, causal=cfg.causal))
            stages.append(
                _resnet_stage(
                    latent,
                    depth=cfg.residual_blocks_per_stage,
                    kernel_size=cfg.kernel_size,
                    dilation_growth_rate=cfg.dilation_growth_rate,
                    causal=cfg.causal,
                    dropout=cfg.residual_dropout,
                    norm=cfg.residual_norm,
                )
            )
        self.stages = nn.Sequential(*stages)
        self.out_proj = nn.Sequential(
            _norm_layer(latent, cfg.residual_norm),
            nn.ReLU(),
            CausalConv1d(latent, latent, kernel_size=3, causal=cfg.causal),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(self.stages(self.input_proj(x)))


class _GroupConvDecoder(nn.Module):
    """Generic conv decoder: ``[B, latent, T/stride] -> [B, C, T]`` (single head)."""

    def __init__(self, out_dim: int, cfg: BackboneConfig) -> None:
        super().__init__()
        latent = cfg.latent_dim
        strides = _factor_stride(cfg.temporal_stride, cfg.num_layers)
        self.input_proj = nn.Sequential(
            CausalConv1d(latent, latent, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        stages: list[nn.Module] = []
        for stride in reversed(strides):
            stages.append(
                _resnet_stage(
                    latent,
                    depth=cfg.residual_blocks_per_stage,
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
                    stages.append(NearestUpsampleBlock1d(latent, stride=stride, causal=cfg.causal))
                elif cfg.upsample_mode == "transpose":
                    stages.append(UpsampleBlock1d(latent, stride=stride))
                else:
                    raise ValueError(f"unknown upsample_mode={cfg.upsample_mode!r}")
        self.stages = nn.Sequential(*stages)
        self.refine = nn.Sequential(
            CausalConv1d(latent, latent, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        self.head = CausalConv1d(latent, out_dim, kernel_size=3, causal=cfg.causal)

    def forward(self, z_q: Tensor) -> Tensor:
        return self.head(self.refine(self.stages(self.input_proj(z_q))))


def _validate_groups(groups: list[GroupConfig], input_dim: int) -> list[GroupConfig]:
    ordered = sorted(groups, key=lambda g: g.start)
    cursor = 0
    for g in ordered:
        if g.start != cursor:
            raise ValueError(
                f"groups must tile [0, {input_dim}) with no gaps/overlap; "
                f"group {g.name!r} starts at {g.start}, expected {cursor}"
            )
        if g.stop <= g.start:
            raise ValueError(f"group {g.name!r} has empty/negative slice [{g.start}, {g.stop})")
        cursor = g.stop
    if cursor != input_dim:
        raise ValueError(f"groups cover [0, {cursor}); must cover [0, {input_dim})")
    return ordered


class StructuredMultiBranchTokenizer(nn.Module):
    """Part-wise codec: per-group encoder/quantizer/decoder, merged 499D output.

    Each group encodes its channel slice at its own token rate, quantizes
    independently, and decodes back to its slice. The exposed ``temporal_stride``
    is the LCM of group strides so eval can trim the clip to a common multiple.
    """

    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        if not cfg.groups:
            raise ValueError("StructuredMultiBranchTokenizer requires cfg.groups")
        groups = _validate_groups(cfg.groups, cfg.input_dim)
        self.cfg = cfg
        self.group_specs: list[tuple[str, int, int]] = [(g.name, g.start, g.stop) for g in groups]
        self.encoders = nn.ModuleList(
            [_GroupConvEncoder(g.stop - g.start, g.backbone) for g in groups]
        )
        self.quantizers = nn.ModuleList(
            [build_quantizer(g.quantizer.kind, dim=g.backbone.latent_dim, **g.quantizer.build_kwargs()) for g in groups]
        )
        self.decoders = nn.ModuleList(
            [_GroupConvDecoder(g.stop - g.start, g.backbone) for g in groups]
        )
        self._residual_decode = [hasattr(q, "get_layer_embeddings") for q in self.quantizers]
        self.temporal_stride = lcm(*[g.backbone.temporal_stride for g in groups]) if len(groups) > 1 else groups[0].backbone.temporal_stride

    def forward(self, features: Tensor) -> StructuredCodecOutput:
        if features.shape[1] != self.cfg.input_dim:
            raise ValueError(f"features channels {features.shape[1]} != {self.cfg.input_dim}")
        recs: list[Tensor] = []
        idx_map: dict[str, Tensor] = {}
        commits: list[Tensor] = []
        ppls: list[Tensor] = []
        metrics: dict[str, Tensor] = {}
        for i, (name, start, stop) in enumerate(self.group_specs):
            z = self.encoders[i](features[:, start:stop])
            z_btc = z.transpose(1, 2).contiguous()
            z_q_btc, idx, losses = self.quantizers[i](z_btc)
            recs.append(self.decoders[i](z_q_btc.transpose(1, 2).contiguous()))
            idx_map[name] = idx
            commits.append(losses["commit_loss"])
            for k, v in losses.items():
                metrics[f"{name}/{k}"] = v
            if "perplexity" in losses:
                ppls.append(losses["perplexity"])
        features_rec = torch.cat(recs, dim=1)
        vq_loss = torch.stack(commits).mean()
        if ppls:
            metrics["perplexity"] = torch.stack(ppls).mean()
        return StructuredCodecOutput(
            features_rec=features_rec, z=None, z_q=None,
            indices=idx_map, vq_loss=vq_loss, metrics=metrics,
        )

    @torch.no_grad()
    def encode_indices(self, features: Tensor) -> dict[str, Tensor]:
        return self.forward(features).indices  # type: ignore[return-value]

    @torch.no_grad()
    def decode_indices(self, indices: dict[str, Tensor]) -> Tensor:
        recs: list[Tensor] = []
        for i, (name, _start, _stop) in enumerate(self.group_specs):
            idx = indices[name]
            if not self._residual_decode[i]:
                raise NotImplementedError(
                    f"decode_indices requires a residual quantizer with get_layer_embeddings; "
                    f"group {name!r} has {type(self.quantizers[i]).__name__}"
                )
            z_q_btc = self.quantizers[i].get_layer_embeddings(idx).sum(dim=2)
            recs.append(self.decoders[i](z_q_btc.transpose(1, 2).contiguous()))
        return torch.cat(recs, dim=1)


def build_structured_vq(cfg: CodecConfig | None = None) -> nn.Module:
    """Build the codec; dispatches to multi-branch when ``cfg.groups`` is set."""
    cfg = cfg or CodecConfig()
    if cfg.groups:
        return StructuredMultiBranchTokenizer(cfg)
    return StructuredVQTokenizer(cfg)


__all__ = [
    "BackboneConfig",
    "CodecConfig",
    "GroupConfig",
    "QuantizerConfig",
    "SemanticHead",
    "StructuredCodecOutput",
    "StructuredMultiBranchTokenizer",
    "StructuredVQTokenizer",
    "build_backbone",
    "build_structured_vq",
]
