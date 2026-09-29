"""L4 — Motion validity.

Sanity over a materialized motion: NaN/Inf fraction and root path length drift
vs ground truth. Metric names::

    _validity_nan_fraction
    _validity_root_drift_m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.umr_schema import UMR_NUM_JOINTS

@dataclass
class MotionValidityMetrics:
    validity_nan_fraction: float
    validity_root_drift_m: float

    def to_dict(self) -> dict[str, float]:
        return {
            "_validity_nan_fraction": self.validity_nan_fraction,
            "_validity_root_drift_m": self.validity_root_drift_m,
        }

def motion_validity_metrics(
    joints77_rec: np.ndarray,         # [T, 77, 3] canonical FK joints (materialized)
    joints77_gt: np.ndarray | None = None,
) -> MotionValidityMetrics:
    """Compute L4 metrics from materialized joint positions.

    With ``joints77_gt`` provided, root drift is reported as the delta vs
    ground truth, factoring out clip-intrinsic motion.
    """
    if joints77_rec.shape[-2:] != (UMR_NUM_JOINTS, 3):
        raise ValueError(
            f"joints77_rec trailing shape must be ({UMR_NUM_JOINTS}, 3); got {joints77_rec.shape}"
        )

    nan_fraction = float(np.mean(~np.isfinite(joints77_rec)))

    root_xz = joints77_rec[:, 0, [0, 2]]
    root_path = float(np.linalg.norm(np.diff(root_xz, axis=0), axis=-1).sum())
    if joints77_gt is not None and joints77_gt.shape == joints77_rec.shape:
        gt_xz = joints77_gt[:, 0, [0, 2]]
        gt_path = float(np.linalg.norm(np.diff(gt_xz, axis=0), axis=-1).sum())
        root_drift_m = abs(root_path - gt_path)
    else:
        root_drift_m = root_path

    return MotionValidityMetrics(
        validity_nan_fraction=nan_fraction,
        validity_root_drift_m=root_drift_m,
    )

__all__ = ["MotionValidityMetrics", "motion_validity_metrics"]
