"""L2 — SOMA canonical metrics.

Compares materialized ``soma77_can-rec`` against ground-truth ``soma77_can``:
``soma_root_trans_rmse_mm``, ``soma_root_rot_rmse_rad``,
``soma_joints76_rot_rmse_rad``, and (when FK positions are passed)
``soma_mpjpe_mm``. Inputs are axis-angle rotations ``[T, 77, 3]`` (root =
index 0) and translations ``[T, 3]``; rotation error is geodesic on SO(3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from data.umr_schema import UMR_NUM_JOINTS

@dataclass
class SomaCanonicalMetrics:
    soma_root_trans_rmse_mm: float
    soma_root_rot_rmse_rad: float
    soma_joints76_rot_rmse_rad: float
    soma_mpjpe_mm: float | None = None

    def to_dict(self) -> dict[str, float]:
        out = {
            "soma_root_trans_rmse_mm": self.soma_root_trans_rmse_mm,
            "soma_root_rot_rmse_rad": self.soma_root_rot_rmse_rad,
            "soma_joints76_rot_rmse_rad": self.soma_joints76_rot_rmse_rad,
        }
        if self.soma_mpjpe_mm is not None:
            out["soma_mpjpe_mm"] = self.soma_mpjpe_mm
        return out

def _so3_geodesic(aa_rec: np.ndarray, aa_gt: np.ndarray) -> np.ndarray:
    """Geodesic distance in radians for a batch of axis-angle pairs ``[..., 3]``."""
    flat_rec = aa_rec.reshape(-1, 3)
    flat_gt = aa_gt.reshape(-1, 3)
    R_rec = Rotation.from_rotvec(flat_rec)
    R_gt = Rotation.from_rotvec(flat_gt)
    R_rel = R_gt.inv() * R_rec
    return np.linalg.norm(R_rel.as_rotvec(), axis=-1).reshape(aa_rec.shape[:-1])

def soma_canonical_metrics(
    transl_rec: np.ndarray,
    transl_gt: np.ndarray,
    rotvec77_rec: np.ndarray,
    rotvec77_gt: np.ndarray,
    *,
    joints77_pos_rec: np.ndarray | None = None,
    joints77_pos_gt: np.ndarray | None = None,
) -> SomaCanonicalMetrics:
    if transl_rec.shape != transl_gt.shape:
        raise ValueError(f"transl shape mismatch: {transl_rec.shape} vs {transl_gt.shape}")
    if transl_rec.shape[-1] != 3:
        raise ValueError(f"transl last dim must be 3; got {transl_rec.shape}")
    if rotvec77_rec.shape != rotvec77_gt.shape:
        raise ValueError(f"rotvec77 shape mismatch: {rotvec77_rec.shape} vs {rotvec77_gt.shape}")
    if rotvec77_rec.shape[-2:] != (UMR_NUM_JOINTS, 3):
        raise ValueError(
            f"rotvec77 trailing shape must be ({UMR_NUM_JOINTS}, 3); got {rotvec77_rec.shape}"
        )

    trans_diff = transl_rec - transl_gt
    trans_rmse_m = float(np.sqrt((trans_diff ** 2).mean()))
    trans_rmse_mm = trans_rmse_m * 1000.0

    root_geo = _so3_geodesic(rotvec77_rec[..., 0, :], rotvec77_gt[..., 0, :])
    root_rmse = float(np.sqrt((root_geo ** 2).mean()))

    joints_geo = _so3_geodesic(
        rotvec77_rec[..., 1:UMR_NUM_JOINTS, :],
        rotvec77_gt[..., 1:UMR_NUM_JOINTS, :],
    )
    joints_rmse = float(np.sqrt((joints_geo ** 2).mean()))
    mpjpe_mm: float | None = None
    if joints77_pos_rec is not None or joints77_pos_gt is not None:
        if joints77_pos_rec is None or joints77_pos_gt is None:
            raise ValueError("joints77_pos_rec and joints77_pos_gt must be provided together")
        if joints77_pos_rec.shape != joints77_pos_gt.shape:
            raise ValueError(
                f"joints77_pos shape mismatch: {joints77_pos_rec.shape} vs {joints77_pos_gt.shape}"
            )
        if joints77_pos_rec.shape[-2:] != (UMR_NUM_JOINTS, 3):
            raise ValueError(
                f"joints77_pos trailing shape must be ({UMR_NUM_JOINTS}, 3); got {joints77_pos_rec.shape}"
            )
        per_joint = np.linalg.norm(joints77_pos_rec - joints77_pos_gt, axis=-1)
        mpjpe_mm = float(per_joint.mean() * 1000.0)

    return SomaCanonicalMetrics(
        soma_root_trans_rmse_mm=trans_rmse_mm,
        soma_root_rot_rmse_rad=root_rmse,
        soma_joints76_rot_rmse_rad=joints_rmse,
        soma_mpjpe_mm=mpjpe_mm,
    )

__all__ = ["SomaCanonicalMetrics", "soma_canonical_metrics"]
