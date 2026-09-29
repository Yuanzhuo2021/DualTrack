"""Explicit masked losses. No Hugging Face label shifting is used here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class PacketLoss:
    total: torch.Tensor
    q0: torch.Tensor
    depth: torch.Tensor
    per_codebook: tuple[torch.Tensor, ...]


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits prefix {tuple(logits.shape[:-1])} != targets {tuple(targets.shape)}"
        )
    if mask.shape != targets.shape:
        raise ValueError(f"mask {tuple(mask.shape)} != targets {tuple(targets.shape)}")
    selected = mask.to(dtype=torch.bool)
    if not torch.any(selected):
        return logits.sum() * 0.0
    selected_targets = targets[selected]
    if torch.any(selected_targets < 0) or torch.any(selected_targets >= logits.shape[-1]):
        raise ValueError("a selected target id is outside its logit vocabulary")
    return F.cross_entropy(logits[selected].float(), selected_targets)


def motion_packet_loss(
    logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    packet_mask: torch.Tensor,
    weights: Sequence[float],
    depth_scale: float,
) -> PacketLoss:
    if targets.ndim != 3:
        raise ValueError("motion targets must have shape [B, L, Q]")
    q = targets.shape[-1]
    if len(logits) != q or len(weights) != q:
        raise ValueError("motion logits, weights, and target codebooks must have equal length")
    if packet_mask.shape != targets.shape[:2]:
        raise ValueError("motion packet mask must have shape [B, L]")

    per_codebook = tuple(
        masked_cross_entropy(logits[i], targets[..., i], packet_mask)
        for i in range(q)
    )
    q0_loss = per_codebook[0]
    residual_weights = torch.as_tensor(
        weights[1:],
        device=q0_loss.device,
        dtype=q0_loss.dtype,
    )
    residual = torch.stack(per_codebook[1:])
    depth_loss = (residual * residual_weights).sum() / residual_weights.sum()
    total = float(weights[0]) * q0_loss + float(depth_scale) * depth_loss
    return PacketLoss(total=total, q0=q0_loss, depth=depth_loss, per_codebook=per_codebook)


def speech_packet_loss(
    q0_logits: torch.Tensor,
    q0_targets: torch.Tensor,
    q0_mask: torch.Tensor,
    depth_logits: torch.Tensor,
    packet_targets: torch.Tensor,
    depth_scale: float,
) -> PacketLoss:
    if packet_targets.ndim != 2:
        raise ValueError("selected speech packet targets must have shape [N, Q]")
    if depth_logits.ndim != 3:
        raise ValueError("speech depth logits must have shape [N, Q-1, V]")
    if depth_logits.shape[:2] != (
        packet_targets.shape[0],
        packet_targets.shape[1] - 1,
    ):
        raise ValueError("speech depth logits do not match selected packet targets")

    q0_loss = masked_cross_entropy(q0_logits, q0_targets, q0_mask)
    residual_losses: list[torch.Tensor] = []
    for index in range(depth_logits.shape[1]):
        target = packet_targets[:, index + 1]
        if torch.any(target < 0) or torch.any(target >= depth_logits.shape[-1]):
            raise ValueError("a speech depth target id is outside the vocabulary")
        residual_losses.append(F.cross_entropy(depth_logits[:, index], target))
    depth_loss = torch.stack(residual_losses).mean()
    total = q0_loss + float(depth_scale) * depth_loss
    return PacketLoss(
        total=total,
        q0=q0_loss,
        depth=depth_loss,
        per_codebook=(q0_loss, *residual_losses),
    )
