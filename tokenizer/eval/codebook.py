"""L3 — Codebook health. Only meaningful for ``codec_mode = vq_vae``.

Aggregates usage / perplexity / dead-code ratio over an RVQ token corpus of
integer codes ``[N, T, R]`` (R residual layers)::

    _codebook_usage             unique_codes / codebook_size
    _codebook_perplexity        exp(unigram_entropy)
    _codebook_dead_code_ratio   1.0 - usage
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class CodebookMetrics:
    usage: float
    perplexity: float
    dead_code_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "_codebook_usage": self.usage,
            "_codebook_perplexity": self.perplexity,
            "_codebook_dead_code_ratio": self.dead_code_ratio,
        }

def _unigram_perplexity(flat: np.ndarray) -> float:
    if flat.size == 0:
        return 0.0
    valid = flat[flat >= 0]
    if valid.size == 0:
        return 0.0
    counts = np.bincount(valid)
    probs = counts.astype(np.float64) / counts.sum()
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return float(np.exp(entropy))

def codebook_metrics_single_layer(indices: np.ndarray, codebook_size: int) -> CodebookMetrics:
    """Aggregate over a single ``[N, T]`` token tensor."""
    flat = np.asarray(indices).reshape(-1)
    valid = flat[flat >= 0]
    used = int(np.unique(valid).size) if valid.size > 0 else 0
    usage = used / max(codebook_size, 1)
    perplexity = _unigram_perplexity(valid)
    return CodebookMetrics(
        usage=usage,
        perplexity=perplexity,
        dead_code_ratio=1.0 - usage,
    )

def codebook_metrics_per_layer(
    indices: np.ndarray, codebook_size: int
) -> list[CodebookMetrics]:
    """Aggregate per residual layer over ``[N, T, R]`` token tensor."""
    if indices.ndim != 3:
        raise ValueError(f"RVQ indices must be [N, T, R]; got {indices.shape}")
    R = indices.shape[-1]
    return [codebook_metrics_single_layer(indices[..., r], codebook_size) for r in range(R)]

__all__ = [
    "CodebookMetrics",
    "codebook_metrics_per_layer",
    "codebook_metrics_single_layer",
]
