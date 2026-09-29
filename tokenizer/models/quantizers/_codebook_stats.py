"""Sync-free codebook health helpers shared by the quantizers.

Kept dependency-free (pure torch) so they stay on the CUDA stream and never
trigger a GPU->CPU sync, which would otherwise break DDP/NCCL overlap.
"""

from __future__ import annotations

import torch
from torch import Tensor

@torch.no_grad()
def _layer_counts(flat: Tensor, codebook_size: int) -> Tensor:
    """Sync-free per-code counts over a flat ``[N]`` index tensor.

    ``quantize_dropout`` puts ``-1`` placeholders in dropped layers; these are
    routed into bin 0 via ``torch.where`` and the dummy count is subtracted back
    out, so the returned histogram has the true unigram distribution and a
    full ``[codebook_size]`` shape regardless of which codes were actually
    used in this batch. Uses zero ``.item()``/``.numel()``/``int(...)`` calls,
    so all ops stay on the CUDA stream and DDP backward can overlap.
    """
    flat = flat.reshape(-1).long()
    neg_mask = flat < 0
    flat_safe = torch.where(neg_mask, flat.new_zeros(()), flat)
    counts = torch.bincount(flat_safe, minlength=codebook_size).float()
    counts[0] = counts[0] - neg_mask.sum().float()
    return counts.clamp_(min=0)

@torch.no_grad()
def _perplexity_from_counts(counts: Tensor) -> Tensor:
    total = counts.sum().clamp_(min=1)
    probs = counts / total
    entropy = -(probs * (probs + 1e-10).log()).sum()
    return torch.exp(entropy)

__all__ = ["_layer_counts", "_perplexity_from_counts"]
