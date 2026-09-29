"""L5 — Streaming / causal equivalence.

Verifies that token-by-token decoding via
:class:`~data.umr_to_soma77.StreamingMaterializer` produces the same
materialized rotvec77 / translation as the offline one-shot
:func:`~data.umr_to_soma77.materialize_features`. Metrics:
``_streaming_feat_rmse``, ``_streaming_root_trans_rmse_mm``,
``_streaming_root_rot_rmse_rad``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.umr_schema import CanonicalAnchor, DIM_FEATURES
from data.umr_to_soma77 import (
    StreamingMaterializer,
    materialize_features,
)
from eval.geometry import _so3_geodesic

@dataclass
class StreamingMetrics:
    streaming_feat_rmse: float
    streaming_root_trans_rmse_mm: float
    streaming_root_rot_rmse_rad: float

    def to_dict(self) -> dict[str, float]:
        return {
            "_streaming_feat_rmse": self.streaming_feat_rmse,
            "_streaming_root_trans_rmse_mm": self.streaming_root_trans_rmse_mm,
            "_streaming_root_rot_rmse_rad": self.streaming_root_rot_rmse_rad,
        }

def streaming_equivalence_metrics(
    features: np.ndarray,             # [T-1, 499]
    anchor: CanonicalAnchor,
) -> StreamingMetrics:
    if features.ndim != 2 or features.shape[-1] != DIM_FEATURES:
        raise ValueError(f"features shape {features.shape} must be (T-1, {DIM_FEATURES})")

    offline = materialize_features(features, anchor)

    stream = StreamingMaterializer(anchor)
    rotvec_stream = [stream.anchor_rotvec77.copy()]
    transl_stream = [anchor.init_root_pos.copy()]
    for t in range(features.shape[0]):
        rvc, txl, _fc = stream.decode_step(features[t])
        rotvec_stream.append(rvc)
        transl_stream.append(txl)
    rotvec_stream_arr = np.stack(rotvec_stream, axis=0)
    transl_stream_arr = np.stack(transl_stream, axis=0)

    # rotvec RMSE (streaming vs offline)
    if rotvec_stream_arr.shape == offline.rotvec77.shape:
        diff = rotvec_stream_arr - offline.rotvec77
        feat_rmse = float(np.sqrt((diff ** 2).mean()))
    else:
        feat_rmse = float("inf")

    # RMSE in translation space (mm)
    trans_diff = transl_stream_arr - offline.transl
    trans_rmse_mm = float(np.sqrt((trans_diff ** 2).mean())) * 1000.0

    # Geodesic RMSE (radians)
    geo = _so3_geodesic(rotvec_stream_arr, offline.rotvec77)
    rot_rmse_rad = float(np.sqrt((geo ** 2).mean()))

    return StreamingMetrics(
        streaming_feat_rmse=feat_rmse,
        streaming_root_trans_rmse_mm=trans_rmse_mm,
        streaming_root_rot_rmse_rad=rot_rmse_rad,
    )

__all__ = ["StreamingMetrics", "streaming_equivalence_metrics"]
