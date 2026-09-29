"""Quantizer registry for the structured codec.

``build_quantizer(kind, dim, **kw)`` returns a module with the codec's quantizer
contract ``forward(z[B,T,C]) -> (z_q[B,T,C], indices, losses{commit_loss,...})``
plus a decode path (``get_layer_embeddings`` for the residual stack).

Kinds: ``ema_rvq`` (default, EMA residual VQ); ``split_rvq`` (semantic VQ at
depth 0 in parallel with an n-layer kinematic RVQ, outputs summed).
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .ema_rvq import EMAResidualVQ, QuantizeEMAReset
from .split_semantic_rvq import SplitSemanticRVQ

def build_quantizer(kind: str, dim: int, **kw: Any) -> nn.Module:
    """Construct a quantizer by registry name.

    ``kw`` keys are quantizer-specific; unknown keys raise via the constructor.
    """
    if kind == "ema_rvq":
        return EMAResidualVQ(
            dim=dim,
            num_quantizers=int(kw.get("num_quantizers", 8)),
            codebook_size=int(kw.get("codebook_size", 1024)),
            mu=float(kw.get("mu", 0.99)),
            quantize_dropout=bool(kw.get("quantize_dropout", True)),
            quantize_dropout_cutoff_index=int(kw.get("quantize_dropout_cutoff_index", 0)),
            quantize_dropout_prob=float(kw.get("quantize_dropout_prob", 0.2)),
            sample_codebook_temp=float(kw.get("sample_codebook_temp", 0.5)),
        )
    if kind == "split_rvq":
        return SplitSemanticRVQ(
            dim=dim,
            num_kinematic_quantizers=int(kw.get("num_quantizers", 7)),
            codebook_size=int(kw.get("codebook_size", 1024)),
            semantic_codebook_size=int(kw.get("semantic_codebook_size", kw.get("codebook_size", 1024))),
            mu=float(kw.get("mu", 0.99)),
            quantize_dropout=bool(kw.get("quantize_dropout", True)),
            quantize_dropout_cutoff_index=int(kw.get("quantize_dropout_cutoff_index", 0)),
            quantize_dropout_prob=float(kw.get("quantize_dropout_prob", 0.2)),
            sample_codebook_temp=float(kw.get("sample_codebook_temp", 0.5)),
        )
    raise ValueError(f"unknown quantizer kind={kind!r}; expected 'ema_rvq' or 'split_rvq'")

__all__ = [
    "EMAResidualVQ",
    "QuantizeEMAReset",
    "SplitSemanticRVQ",
    "build_quantizer",
]
