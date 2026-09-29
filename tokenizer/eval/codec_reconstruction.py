"""L1 — UMR499 reconstruction metrics.

Compares ``umr499`` (GT) against ``umr499-rec`` in unnormalized feature space:

    feat_mse / feat_mae   over all channels EXCEPT the 4 foot-contact channels
    foot_contact_f1       binary F1 of ``logit > 0`` vs target ``>= 0.5``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.umr_schema import DIM_FEATURES, SLICE_FOOT_CONTACT

@dataclass
class CodecReconstructionMetrics:
    feat_mse: float
    feat_mae: float
    foot_contact_f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "feat_mse": self.feat_mse,
            "feat_mae": self.feat_mae,
            "foot_contact_f1": self.foot_contact_f1,
        }

def _non_foot_mask() -> np.ndarray:
    mask = np.ones(DIM_FEATURES, dtype=bool)
    mask[SLICE_FOOT_CONTACT] = False
    return mask

_NON_FOOT_MASK = _non_foot_mask()

def codec_reconstruction_metrics(
    rec: np.ndarray,                # [..., 499]
    gt: np.ndarray,                 # [..., 499]
) -> CodecReconstructionMetrics:
    """Compute L1 metrics over arbitrary leading dims (e.g. ``[N, T, 499]``)."""
    rec = np.asarray(rec, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if rec.shape != gt.shape:
        raise ValueError(f"rec / gt shape mismatch: {rec.shape} vs {gt.shape}")
    if rec.shape[-1] != DIM_FEATURES:
        raise ValueError(f"feature last dim must be {DIM_FEATURES}; got {rec.shape}")

    diff = rec[..., _NON_FOOT_MASK] - gt[..., _NON_FOOT_MASK]
    feat_mse = float((diff ** 2).mean())
    feat_mae = float(np.abs(diff).mean())

    foot_logits = rec[..., SLICE_FOOT_CONTACT]
    foot_target = gt[..., SLICE_FOOT_CONTACT]
    pred_pos = foot_logits > 0.0
    target_pos = foot_target > 0.5
    tp = float(np.logical_and(pred_pos, target_pos).sum())
    fp = float(np.logical_and(pred_pos, ~target_pos).sum())
    fn = float(np.logical_and(~pred_pos, target_pos).sum())
    if (tp + fp + fn) == 0.0:
        foot_contact_f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        foot_contact_f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

    return CodecReconstructionMetrics(
        feat_mse=feat_mse,
        feat_mae=feat_mae,
        foot_contact_f1=foot_contact_f1,
    )

__all__ = ["CodecReconstructionMetrics", "codec_reconstruction_metrics"]
