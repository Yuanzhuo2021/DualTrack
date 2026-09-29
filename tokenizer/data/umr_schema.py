"""UMR (Unified Motion Representation) schema.

Source of truth for the ``umr499.npz`` artifact and the in-memory
:class:`UMR499` view. Logical structure::

    UMR499 =
      canonical_anchor (frame-0 absolute decode seed; not a motion token)
        init_root_pos          [3]              root xyz
        init_root_rot6d        [6]              root world-aligned rot6d
        init_joints76_rot6d    [76, 6]          non-root parent-local rot6d
      features                  [N-1, 499]      packed UMR records
      joints77_pos              [N, 77, 3]      canonical SOMA-X FK positions (eval only)
      identity_coeffs           [1, C]          static body identity for FK eval
      joint_orient              [78, 3, 3]      SOMA-X rig rest orientations for FK export/eval
      fps                       scalar = 50.0
      feature_variant           "delta_root_multiscale_sparsevel"

Per-row 499D layout::

    root_traj          [9]   = local_dxz(2) + world_dxz(2) + root_y(1)
                              + world_vel_0p2s_xz(2) + world_vel_0p4s_xz(2)
    root_rot6d         [6]   absolute target-frame root rot6d
    joints76_rot6d     [456] = 76 * 6 absolute parent-local rot6d
    sparse_vel         [24]  = 8 sparse joints * 3D root-yaw-local velocity
    foot_contact       [4]   target-frame foot contact

Field slices below are the only legitimate way to address the 499D vector;
callers must not hard-code offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Skeleton / variant constants
# ---------------------------------------------------------------------------

UMR_VERSION = "UMRv1.1"
UMR_FPS = 50.0
UMR_UNIT = "meter"
UMR_SKELETON = "SOMA77"
UMR_NUM_JOINTS = 77
UMR_NUM_JOINTS76 = 76
UMR_CANONICALIZATION = "soma77_can__floor_y__remove_initial_xz_yaw"
FEATURE_VARIANT = "delta_root_multiscale_sparsevel"

# Sparse velocity key joints.
SPARSE_VEL_JOINTS: tuple[int, int, int, int, int, int, int, int] = (
    3, 6, 14, 42, 69, 70, 74, 75,
)
NUM_SPARSE_VEL_JOINTS = len(SPARSE_VEL_JOINTS)

# Foot contact convention from the data pipeline.
FOOT_CONTACT_NAMES: tuple[str, str, str, str] = (
    "LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase",
)
FOOT_CONTACT_SOMA77_INDICES: tuple[int, int, int, int] = (69, 70, 74, 75)

# Multiscale root velocity horizons (seconds); frame offsets scale with fps.
ROOT_MULTISCALE_HORIZONS_SEC: tuple[float, float] = (0.2, 0.4)

def root_multiscale_windows_for_fps(fps: float) -> tuple[int, int]:
    """Frame offsets for the 0.2s / 0.4s root velocity auxiliary channels."""
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    short, long = (
        max(1, int(round(float(fps) * h)))
        for h in ROOT_MULTISCALE_HORIZONS_SEC
    )
    return short, long

ROOT_MULTISCALE_WINDOWS: tuple[int, int] = root_multiscale_windows_for_fps(UMR_FPS)

ROOT_DECODE_LOCAL_WEIGHT = 0.5

DIM_ROOT_TRAJ = 9
DIM_ROOT_ROT6D = 6
DIM_JOINTS76_ROT6D = UMR_NUM_JOINTS76 * 6
DIM_SPARSE_VEL = NUM_SPARSE_VEL_JOINTS * 3
DIM_FOOT_CONTACT = 4
DIM_FEATURES = (
    DIM_ROOT_TRAJ
    + DIM_ROOT_ROT6D
    + DIM_JOINTS76_ROT6D
    + DIM_SPARSE_VEL
    + DIM_FOOT_CONTACT
)

DIM_TRAJ_CONTACT = DIM_ROOT_TRAJ + DIM_ROOT_ROT6D + DIM_SPARSE_VEL + DIM_FOOT_CONTACT

SLICE_ROOT_TRAJ = slice(0, DIM_ROOT_TRAJ)
SLICE_ROOT_ROT6D = slice(DIM_ROOT_TRAJ, DIM_ROOT_TRAJ + DIM_ROOT_ROT6D)
SLICE_JOINTS76_ROT6D = slice(
    DIM_ROOT_TRAJ + DIM_ROOT_ROT6D,
    DIM_ROOT_TRAJ + DIM_ROOT_ROT6D + DIM_JOINTS76_ROT6D,
)
SLICE_SPARSE_VEL = slice(
    DIM_ROOT_TRAJ + DIM_ROOT_ROT6D + DIM_JOINTS76_ROT6D,
    DIM_ROOT_TRAJ + DIM_ROOT_ROT6D + DIM_JOINTS76_ROT6D + DIM_SPARSE_VEL,
)
SLICE_FOOT_CONTACT = slice(DIM_FEATURES - DIM_FOOT_CONTACT, DIM_FEATURES)

SLICE_ROOT_LOCAL_DXZ = slice(0, 2)
SLICE_ROOT_WORLD_DXZ = slice(2, 4)
IDX_ROOT_Y = 4
SLICE_ROOT_MULTISCALE_0P2S = slice(5, 7)
SLICE_ROOT_MULTISCALE_0P4S = slice(7, 9)

WINDOW_DEFAULT = 180

@dataclass
class FeatureFields:
    """Typed view of one or many 499D records; trailing dims match each field."""

    root_traj: np.ndarray            # [..., 9]
    root_rot6d: np.ndarray           # [..., 6]
    joints76_rot6d: np.ndarray       # [..., 76, 6]
    sparse_vel: np.ndarray           # [..., 8, 3]
    foot_contact: np.ndarray         # [..., 4]

@dataclass
class CanonicalAnchor:
    """Frame-0 absolute decode seed (boundary condition, not a motion token)."""

    init_root_pos: np.ndarray        # [3]
    init_root_rot6d: np.ndarray      # [6]
    init_joints76_rot6d: np.ndarray  # [76, 6]

# ---------------------------------------------------------------------------
# Pack / unpack helpers
# ---------------------------------------------------------------------------

def pack_features(fields: FeatureFields) -> np.ndarray:
    """Concatenate :class:`FeatureFields` into a ``[..., 499]`` array."""
    if fields.root_traj.shape[-1] != DIM_ROOT_TRAJ:
        raise ValueError(f"root_traj last dim must be {DIM_ROOT_TRAJ}; got {fields.root_traj.shape}")
    if fields.root_rot6d.shape[-1] != DIM_ROOT_ROT6D:
        raise ValueError(f"root_rot6d last dim must be {DIM_ROOT_ROT6D}; got {fields.root_rot6d.shape}")
    if fields.joints76_rot6d.shape[-2:] != (UMR_NUM_JOINTS76, 6):
        raise ValueError(
            f"joints76_rot6d trailing shape must be ({UMR_NUM_JOINTS76}, 6); got {fields.joints76_rot6d.shape}"
        )
    if fields.sparse_vel.shape[-2:] != (NUM_SPARSE_VEL_JOINTS, 3):
        raise ValueError(
            f"sparse_vel trailing shape must be ({NUM_SPARSE_VEL_JOINTS}, 3); got {fields.sparse_vel.shape}"
        )
    if fields.foot_contact.shape[-1] != DIM_FOOT_CONTACT:
        raise ValueError(f"foot_contact last dim must be {DIM_FOOT_CONTACT}; got {fields.foot_contact.shape}")

    lead = fields.root_traj.shape[:-1]
    j_flat = fields.joints76_rot6d.reshape(*lead, DIM_JOINTS76_ROT6D)
    s_flat = fields.sparse_vel.reshape(*lead, DIM_SPARSE_VEL)
    return np.concatenate(
        [fields.root_traj, fields.root_rot6d, j_flat, s_flat, fields.foot_contact],
        axis=-1,
    ).astype(np.float32, copy=False)

def unpack_features(features: np.ndarray) -> FeatureFields:
    """Split ``[..., 499]`` into :class:`FeatureFields`."""
    if features.shape[-1] != DIM_FEATURES:
        raise ValueError(f"features last dim must be {DIM_FEATURES}; got {features.shape}")
    lead = features.shape[:-1]
    return FeatureFields(
        root_traj=features[..., SLICE_ROOT_TRAJ],
        root_rot6d=features[..., SLICE_ROOT_ROT6D],
        joints76_rot6d=features[..., SLICE_JOINTS76_ROT6D].reshape(*lead, UMR_NUM_JOINTS76, 6),
        sparse_vel=features[..., SLICE_SPARSE_VEL].reshape(*lead, NUM_SPARSE_VEL_JOINTS, 3),
        foot_contact=features[..., SLICE_FOOT_CONTACT],
    )

def split_streams(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split packed 499D into ``(traj_contact [.., 43], joints76_rot6d [.., 456])``.

    ``traj_contact = [root_traj(9), root_rot6d(6), sparse_vel(24), foot_contact(4)]``.
    """
    fields = unpack_features(features)
    lead = features.shape[:-1]
    j_flat = fields.joints76_rot6d.reshape(*lead, DIM_JOINTS76_ROT6D)
    s_flat = fields.sparse_vel.reshape(*lead, DIM_SPARSE_VEL)
    traj_contact = np.concatenate(
        [fields.root_traj, fields.root_rot6d, s_flat, fields.foot_contact],
        axis=-1,
    ).astype(np.float32, copy=False)
    return traj_contact, j_flat.astype(np.float32, copy=False)

def join_streams(traj_contact: np.ndarray, joints76_rot6d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`split_streams`."""
    if traj_contact.shape[-1] != DIM_TRAJ_CONTACT:
        raise ValueError(f"traj_contact last dim must be {DIM_TRAJ_CONTACT}; got {traj_contact.shape}")
    if joints76_rot6d.shape[-1] != DIM_JOINTS76_ROT6D:
        raise ValueError(f"joints76_rot6d last dim must be {DIM_JOINTS76_ROT6D}; got {joints76_rot6d.shape}")
    lead = traj_contact.shape[:-1]
    root_traj = traj_contact[..., :DIM_ROOT_TRAJ]
    cursor = DIM_ROOT_TRAJ
    root_rot6d = traj_contact[..., cursor : cursor + DIM_ROOT_ROT6D]
    cursor += DIM_ROOT_ROT6D
    sparse_vel_flat = traj_contact[..., cursor : cursor + DIM_SPARSE_VEL]
    cursor += DIM_SPARSE_VEL
    foot_contact = traj_contact[..., cursor : cursor + DIM_FOOT_CONTACT]
    return np.concatenate(
        [root_traj, root_rot6d, joints76_rot6d, sparse_vel_flat, foot_contact],
        axis=-1,
    ).astype(np.float32, copy=False)

@dataclass
class UMR499:
    """One UMR499 clip / window (features + frame-0 anchor + eval-only FK reference)."""

    canonical_anchor: CanonicalAnchor
    features: np.ndarray              # [N-1, 499]
    joints77_pos: np.ndarray          # [N, 77, 3]
    identity_coeffs: np.ndarray       # [1, C]
    joint_orient: np.ndarray          # [78, 3, 3]
    fps: float = UMR_FPS
    feature_variant: str = FEATURE_VARIANT
    source_path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[-1] != DIM_FEATURES:
            raise ValueError(
                f"features shape {self.features.shape} must be (T-1, {DIM_FEATURES})"
            )
        N_minus_1 = int(self.features.shape[0])
        if self.joints77_pos.shape != (N_minus_1 + 1, UMR_NUM_JOINTS, 3):
            raise ValueError(
                f"joints77_pos shape {self.joints77_pos.shape} must be "
                f"({N_minus_1 + 1}, {UMR_NUM_JOINTS}, 3)"
            )
        if self.identity_coeffs.ndim != 2 or self.identity_coeffs.shape[0] != 1:
            raise ValueError(
                f"identity_coeffs shape {self.identity_coeffs.shape} must be (1, C)"
            )
        if self.joint_orient.shape != (78, 3, 3):
            raise ValueError(
                f"joint_orient shape {self.joint_orient.shape} must be (78, 3, 3)"
            )
        if self.canonical_anchor.init_root_pos.shape != (3,):
            raise ValueError(
                f"init_root_pos shape {self.canonical_anchor.init_root_pos.shape} must be (3,)"
            )
        if self.canonical_anchor.init_root_rot6d.shape != (DIM_ROOT_ROT6D,):
            raise ValueError(
                f"init_root_rot6d shape {self.canonical_anchor.init_root_rot6d.shape} must be ({DIM_ROOT_ROT6D},)"
            )
        if self.canonical_anchor.init_joints76_rot6d.shape != (UMR_NUM_JOINTS76, 6):
            raise ValueError(
                f"init_joints76_rot6d shape {self.canonical_anchor.init_joints76_rot6d.shape} "
                f"must be ({UMR_NUM_JOINTS76}, 6)"
            )

    @property
    def num_frames(self) -> int:
        """``N`` = ``features`` rows + 1 anchor frame."""
        return int(self.features.shape[0]) + 1

    @property
    def num_records(self) -> int:
        """``N - 1`` = number of UMR records (``features`` rows)."""
        return int(self.features.shape[0])

    def feature_fields(self) -> FeatureFields:
        """Lazy field view of ``features``."""
        return unpack_features(self.features)

    def to_npz(self, out_path: str | Path, *, compressed: bool = True) -> Path:
        """Persist as ``umr499.npz``.

        NPZ keys::

            features              [N-1, 499]
            init_root_pos         [3]
            init_root_rot6d       [6]
            init_joints76_rot6d   [76, 6]
            joints77_pos          [N, 77, 3]
            identity_coeffs       [1, C]
            joint_orient          [78, 3, 3]
            fps                   scalar f32
            feature_variant       string
            root_multiscale_horizons_sec [2]
            root_multiscale_windows      [2]
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save = np.savez_compressed if compressed else np.savez
        save(
            out,
            features=self.features.astype(np.float32, copy=False),
            init_root_pos=self.canonical_anchor.init_root_pos.astype(np.float32, copy=False),
            init_root_rot6d=self.canonical_anchor.init_root_rot6d.astype(np.float32, copy=False),
            init_joints76_rot6d=self.canonical_anchor.init_joints76_rot6d.astype(np.float32, copy=False),
            joints77_pos=self.joints77_pos.astype(np.float32, copy=False),
            identity_coeffs=self.identity_coeffs.astype(np.float32, copy=False),
            joint_orient=self.joint_orient.astype(np.float32, copy=False),
            fps=np.asarray(float(self.fps), dtype=np.float32),
            feature_variant=np.asarray(self.feature_variant),
            root_multiscale_horizons_sec=np.asarray(ROOT_MULTISCALE_HORIZONS_SEC, dtype=np.float32),
            root_multiscale_windows=np.asarray(root_multiscale_windows_for_fps(self.fps), dtype=np.int32),
        )
        return out

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> "UMR499":
        p = Path(npz_path)
        data = np.load(p, allow_pickle=False)
        required = {
            "features", "init_root_pos", "init_root_rot6d",
            "init_joints76_rot6d", "joints77_pos", "identity_coeffs", "joint_orient",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{p}: umr499.npz missing keys {sorted(missing)}")

        fps = float(data["fps"].item()) if "fps" in data.files else UMR_FPS
        variant = (
            str(data["feature_variant"].item())
            if "feature_variant" in data.files
            else FEATURE_VARIANT
        )
        if variant != FEATURE_VARIANT:
            raise ValueError(
                f"{p}: feature_variant={variant!r} != expected {FEATURE_VARIANT!r}; "
                f"this loader only supports UMRv1.1 delta_root_multiscale_sparsevel."
            )

        anchor = CanonicalAnchor(
            init_root_pos=np.asarray(data["init_root_pos"], dtype=np.float32),
            init_root_rot6d=np.asarray(data["init_root_rot6d"], dtype=np.float32),
            init_joints76_rot6d=np.asarray(data["init_joints76_rot6d"], dtype=np.float32),
        )
        return cls(
            canonical_anchor=anchor,
            features=np.asarray(data["features"], dtype=np.float32),
            joints77_pos=np.asarray(data["joints77_pos"], dtype=np.float32),
            identity_coeffs=np.asarray(data["identity_coeffs"], dtype=np.float32),
            joint_orient=np.asarray(data["joint_orient"], dtype=np.float32),
            fps=fps,
            feature_variant=variant,
            source_path=p,
        )

__all__ = [
    "CanonicalAnchor",
    "DIM_FEATURES",
    "DIM_FOOT_CONTACT",
    "DIM_JOINTS76_ROT6D",
    "DIM_ROOT_ROT6D",
    "DIM_ROOT_TRAJ",
    "DIM_SPARSE_VEL",
    "DIM_TRAJ_CONTACT",
    "FEATURE_VARIANT",
    "FOOT_CONTACT_NAMES",
    "FOOT_CONTACT_SOMA77_INDICES",
    "FeatureFields",
    "IDX_ROOT_Y",
    "NUM_SPARSE_VEL_JOINTS",
    "ROOT_DECODE_LOCAL_WEIGHT",
    "ROOT_MULTISCALE_WINDOWS",
    "SLICE_FOOT_CONTACT",
    "SLICE_JOINTS76_ROT6D",
    "SLICE_ROOT_LOCAL_DXZ",
    "SLICE_ROOT_MULTISCALE_0P2S",
    "SLICE_ROOT_MULTISCALE_0P4S",
    "SLICE_ROOT_ROT6D",
    "SLICE_ROOT_TRAJ",
    "SLICE_ROOT_WORLD_DXZ",
    "SLICE_SPARSE_VEL",
    "SPARSE_VEL_JOINTS",
    "UMR_CANONICALIZATION",
    "UMR_FPS",
    "UMR_NUM_JOINTS",
    "UMR_NUM_JOINTS76",
    "UMR_SKELETON",
    "UMR_UNIT",
    "UMR_VERSION",
    "UMR499",
    "WINDOW_DEFAULT",
    "join_streams",
    "pack_features",
    "split_streams",
    "unpack_features",
]
