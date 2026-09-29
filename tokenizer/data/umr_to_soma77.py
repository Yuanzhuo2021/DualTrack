"""Materialize ``UMR499`` (or ``umr499-rec``) back into canonical SOMA77 motion.

Inverse of :func:`data.soma77_to_umr.soma77_to_umr499`; produces
``DecodedSoma77Can`` (axis-angle ``rotvec77``, world ``transl``, foot contacts)
in the encoder's canonical frame. :func:`materialize_features` decodes a full
block ``[T-1, 499]`` one-shot; :class:`StreamingMaterializer` decodes
record-by-record with a state cache. Pure numpy + scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from data.umr_schema import (
    DIM_FEATURES,
    IDX_ROOT_Y,
    ROOT_DECODE_LOCAL_WEIGHT,
    SLICE_FOOT_CONTACT,
    SLICE_ROOT_LOCAL_DXZ,
    SLICE_ROOT_TRAJ,
    SLICE_ROOT_WORLD_DXZ,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    CanonicalAnchor,
    UMR499,
    unpack_features,
)

@dataclass
class DecodedSoma77Can:
    """Output of :func:`materialize_features` (canonical-frame motion)."""

    rotvec77: np.ndarray
    transl: np.ndarray
    foot_contacts: np.ndarray

@dataclass
class DecodedSoma77CanMatrices:
    """Decoded canonical motion with rotations kept as matrices."""

    rotmat77: np.ndarray
    transl: np.ndarray
    foot_contacts: np.ndarray

def rot6d_to_rotmat(d6: np.ndarray) -> np.ndarray:
    """Inverse of ``rotmat_to_rot6d`` via Gram-Schmidt."""
    if d6.shape[-1] != 6:
        raise ValueError(f"rot6d last dim must be 6; got {d6.shape}")
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)

def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrices ``[..., 3, 3]`` → axis-angle ``[..., 3]``."""
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"rotmat last two dims must be (3, 3); got {R.shape}")
    lead = R.shape[:-2]
    flat = R.reshape(-1, 3, 3).astype(np.float32, copy=False)
    aa = Rotation.from_matrix(flat).as_rotvec().astype(np.float32)
    return aa.reshape(*lead, 3)

def _root_yaw_inv_mats(root_mats: np.ndarray) -> np.ndarray:
    """Yaw-only matrices that rotate canonical vectors → per-frame root-yaw frame."""
    forward = root_mats[:, :, 2].astype(np.float32)
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    mats = np.zeros((root_mats.shape[0], 3, 3), dtype=np.float32)
    mats[:, 0, 0] = c
    mats[:, 0, 2] = -s
    mats[:, 1, 1] = 1.0
    mats[:, 2, 0] = s
    mats[:, 2, 2] = c
    return mats

def materialize_features(
    features: np.ndarray,
    anchor: CanonicalAnchor,
) -> DecodedSoma77Can:
    """Decode a feature block ``[T-1, 499]`` into canonical SOMA77 motion.

    Output layout::

        rotvec77      [T, 77, 3]    canonical-frame axis-angle (frame 0 from anchor)
        transl        [T, 3]        canonical-frame root xyz
        foot_contacts [T, 4]        clipped to [0, 1]; frame 0 replicates frame 1

    Decoding rules::

        R'_0.root        = Rot6DToRotMat(init_root_rot6d)
        R_local_{0,1..76} = Rot6DToRotMat(init_joints76_rot6d)
        p'_0.root        = init_root_pos

        for t = 1 .. T-1:
          root_rot_t       = Rot6DToRotMat(record_{t-1}.root_rot6d)
          joints76_rot_t,j = Rot6DToRotMat(record_{t-1}.joints76_rot6d[j])
          local_delta_xz   = record_{t-1}.root_traj[0:2]
          world_delta_dir  = record_{t-1}.root_traj[2:4]
          world_delta_from_local = yaw_inv(root_rot_t).T @ local_delta_xz
          world_delta      = 0.5 * world_delta_from_local + 0.5 * world_delta_dir
          p'_t.root.xz     = p'_{t-1}.root.xz + world_delta.xz
          p'_t.root.y      = record_{t-1}.root_traj[4]
    """
    decoded = materialize_features_matrices(features, anchor)
    rotvec77 = rotmat_to_axis_angle(decoded.rotmat77.reshape(-1, 3, 3)).reshape(
        decoded.rotmat77.shape[:-2] + (3,)
    )
    return DecodedSoma77Can(
        rotvec77=rotvec77.astype(np.float32, copy=False),
        transl=decoded.transl,
        foot_contacts=decoded.foot_contacts,
    )

def materialize_features_matrices(
    features: np.ndarray,
    anchor: CanonicalAnchor,
) -> DecodedSoma77CanMatrices:
    """Decode a feature block while preserving rotation matrices for FK."""
    if features.ndim != 2 or features.shape[-1] != DIM_FEATURES:
        raise ValueError(f"features shape {features.shape} must be (T-1, {DIM_FEATURES})")

    N_minus_1 = int(features.shape[0])
    N = N_minus_1 + 1
    fields = unpack_features(features)

    rotmat77 = np.empty((N, UMR_NUM_JOINTS, 3, 3), dtype=np.float32)
    rotmat77[0, 0] = rot6d_to_rotmat(anchor.init_root_rot6d)
    rotmat77[0, 1:] = rot6d_to_rotmat(anchor.init_joints76_rot6d)
    rotmat77[1:, 0] = rot6d_to_rotmat(fields.root_rot6d)
    rotmat77[1:, 1:] = rot6d_to_rotmat(
        fields.joints76_rot6d.reshape(N_minus_1, UMR_NUM_JOINTS76, 6)
    )
    root_mats_target = rotmat77[1:, 0]
    yaw_inv = _root_yaw_inv_mats(root_mats_target)
    local_delta = np.zeros((N_minus_1, 3), dtype=np.float32)
    local_delta[:, 0] = features[:, SLICE_ROOT_LOCAL_DXZ.start]
    local_delta[:, 2] = features[:, SLICE_ROOT_LOCAL_DXZ.start + 1]
    world_delta_from_local = np.einsum(
        "tji,tj->ti", yaw_inv, local_delta, optimize=True
    )
    world_delta_direct = np.zeros((N_minus_1, 3), dtype=np.float32)
    world_delta_direct[:, 0] = features[:, SLICE_ROOT_WORLD_DXZ.start]
    world_delta_direct[:, 2] = features[:, SLICE_ROOT_WORLD_DXZ.start + 1]
    world_delta = (
        ROOT_DECODE_LOCAL_WEIGHT * world_delta_from_local
        + (1.0 - ROOT_DECODE_LOCAL_WEIGHT) * world_delta_direct
    )
    transl = np.empty((N, 3), dtype=np.float32)
    transl[0] = anchor.init_root_pos
    xz_cum = np.cumsum(world_delta[:, [0, 2]], axis=0)
    transl[1:, 0] = anchor.init_root_pos[0] + xz_cum[:, 0]
    transl[1:, 2] = anchor.init_root_pos[2] + xz_cum[:, 1]
    transl[1:, 1] = features[:, SLICE_ROOT_TRAJ.start + IDX_ROOT_Y]

    contacts_records = np.clip(features[:, SLICE_FOOT_CONTACT], 0.0, 1.0).astype(np.float32)
    foot_contacts = np.empty((N, 4), dtype=np.float32)
    foot_contacts[1:] = contacts_records
    foot_contacts[0] = contacts_records[0]

    return DecodedSoma77CanMatrices(
        rotmat77=rotmat77.astype(np.float32, copy=False),
        transl=transl,
        foot_contacts=foot_contacts,
    )

def materialize_umr499(umr: UMR499) -> DecodedSoma77Can:
    """Convenience wrapper: :class:`UMR499` → :class:`DecodedSoma77Can`."""
    return materialize_features(umr.features, umr.canonical_anchor)

@dataclass
class StreamingState:
    """Mutable streaming-decode state cache."""

    root_xy_pos: np.ndarray
    """The last decoded root world position in the canonical frame."""

    last_root_mat: np.ndarray
    """Last decoded root rotation matrix; retained for debugging only."""

class StreamingMaterializer:
    """Materialize records one frame at a time against a fixed anchor.

    Equivalent to :func:`materialize_features` given the same anchor and all
    records in order.
    """

    def __init__(self, anchor: CanonicalAnchor) -> None:
        self.anchor = anchor
        root_mat0 = rot6d_to_rotmat(anchor.init_root_rot6d.reshape(1, 6))[0]
        self.state = StreamingState(
            root_xy_pos=anchor.init_root_pos.astype(np.float32, copy=True),
            last_root_mat=root_mat0.astype(np.float32, copy=False),
        )

    @property
    def anchor_rotvec77(self) -> np.ndarray:
        """``[77, 3]`` axis-angle for frame 0 (root + joints76)."""
        root_mat = rot6d_to_rotmat(self.anchor.init_root_rot6d.reshape(1, 6))[0]
        joints_mat = rot6d_to_rotmat(self.anchor.init_joints76_rot6d)
        rot = np.empty((UMR_NUM_JOINTS, 3, 3), dtype=np.float32)
        rot[0] = root_mat
        rot[1:] = joints_mat
        return rotmat_to_axis_angle(rot)

    def decode_step(
        self, feature: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode one 499D record into ``(rotvec77 [77, 3], transl [3], foot_contact [4])`` for frame ``t+1``."""
        if feature.shape != (DIM_FEATURES,):
            raise ValueError(f"feature shape {feature.shape} must be ({DIM_FEATURES},)")
        block = materialize_features(feature[None, :], self.anchor)
        # Bootstrap a fresh anchor from the produced frame so subsequent
        # decode_step calls don't accumulate root xz inside the same anchor.
        self.anchor = CanonicalAnchor(
            init_root_pos=block.transl[1].astype(np.float32, copy=False),
            init_root_rot6d=feature[SLICE_ROOT_TRAJ.stop : SLICE_ROOT_TRAJ.stop + 6].astype(
                np.float32, copy=False
            ),
            init_joints76_rot6d=feature[
                SLICE_ROOT_TRAJ.stop + 6 : SLICE_ROOT_TRAJ.stop + 6 + UMR_NUM_JOINTS76 * 6
            ].reshape(UMR_NUM_JOINTS76, 6).astype(np.float32, copy=False),
        )
        self.state = StreamingState(
            root_xy_pos=block.transl[1].astype(np.float32, copy=False),
            last_root_mat=rot6d_to_rotmat(self.anchor.init_root_rot6d.reshape(1, 6))[0],
        )
        return block.rotvec77[1], block.transl[1], block.foot_contacts[1]

__all__ = [
    "DecodedSoma77Can",
    "DecodedSoma77CanMatrices",
    "StreamingMaterializer",
    "StreamingState",
    "materialize_features",
    "materialize_features_matrices",
    "materialize_umr499",
    "rot6d_to_rotmat",
    "rotmat_to_axis_angle",
]
