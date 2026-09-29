"""Minimal SOMA-X FK wrapper for soma77.npz inputs.

Thin glue over the SOMA-X submodule (``third_party/SOMA-X``): path setup,
SOMALayer caching, and the SOMA77 → 77-joint world-XYZ FK call. SOMA-X /
smplx / warp-lang imports are deferred to the first call; tests exercising
this module must be marked ``pytest.mark.soma_x``.
"""

from __future__ import annotations

import contextlib
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Joint index maps (vendored constants)
# ---------------------------------------------------------------------------

# Foot contact column → SOMA77 joint index.
FOOT_CONTACT_SOMA77_INDICES: tuple[int, int, int, int] = (69, 70, 74, 75)
"""``[LeftFoot, LeftToeBase, RightFoot, RightToeBase]``"""

SMPL24_TO_SOMA77_INDEX: tuple[int, ...] = (
    0,
    67,
    72,
    1,
    68,
    73,
    2,
    69,
    74,
    3,
    70,
    75,
    4,
    11,
    39,
    6,
    12,
    40,
    13,
    41,
    14,
    42,
    14,
    42,
)

SMPL_BODY_POSE_SOMA77_INDICES: tuple[int, ...] = SMPL24_TO_SOMA77_INDEX[1:22]

LEFT_HAND_SOMA77_INDICES: tuple[int, ...] = tuple(range(15, 39))
RIGHT_HAND_SOMA77_INDICES: tuple[int, ...] = tuple(range(43, 67))
HEAD_SOMA77_INDICES: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10)

def _soma_x_path() -> Path:
    """Return the on-disk root of the SOMA-X submodule."""
    here = Path(__file__).resolve()
    return here.parent.parent / "third_party" / "SOMA-X"

def _ensure_soma_on_path() -> None:
    """Insert ``third_party/SOMA-X`` into ``sys.path`` and quiet warp-lang."""
    soma_root = str(_soma_x_path())
    if not Path(soma_root).is_dir():
        raise RuntimeError(
            f"SOMA-X submodule not found at {soma_root}. "
            f"Run: git submodule update --init third_party/SOMA-X"
        )
    if soma_root not in sys.path:
        sys.path.insert(0, soma_root)
    try:
        import warp as wp

        wp.config.quiet = True
    except Exception:
        pass

@contextlib.contextmanager
def _soma_init_context():
    """Suppress SMPL-X / trimesh / sparse-CSR warnings during SOMALayer build."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Sparse CSR tensor support is in beta.*", category=UserWarning
        )
        try:
            from numpy.exceptions import VisibleDeprecationWarning as _NVis
        except ImportError:
            _NVis = DeprecationWarning
        warnings.filterwarnings(
            "ignore",
            message=".*align should be passed as Python or NumPy boolean.*",
            category=_NVis,
        )
        warnings.filterwarnings("ignore", message=".*All-NaN slice encountered.*", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", message=".*invalid value encountered in cast.*", category=RuntimeWarning
        )
        yield

class _CachedSomaLayer:
    """Holds a SOMALayer + a per-instance reentrant lock for thread-safe FK."""

    def __init__(self, soma: Any) -> None:
        self.soma = soma
        self.lock = RLock()

@lru_cache(maxsize=8)
def cached_soma_layer(
    soma_data_root: str,
    *,
    identity_model_type: str = "smplx",
    device: str = "cpu",
    low_lod: bool = True,
) -> _CachedSomaLayer:
    """Reuse SOMALayer per process; SMPL-X load is heavy.

    ``soma_data_root`` points at ``third_party/SOMA-X/assets`` and is a string
    so :func:`functools.lru_cache` can hash it.
    """
    _ensure_soma_on_path()

    import torch

    from soma.soma import SOMALayer  # type: ignore[import-not-found]

    with _soma_init_context():
        soma = SOMALayer(
            Path(soma_data_root),
            low_lod=low_lod,
            identity_model_type=identity_model_type,
            device=torch.device(device),
            mode="warp",
        )
    return _CachedSomaLayer(soma)

def default_soma_data_root() -> Path:
    """Convenience: ``third_party/SOMA-X/assets``."""
    return _soma_x_path() / "assets"

def soma77_fk_npz(
    soma77_npz: str | Path,
    *,
    soma_data_root: Path | None = None,
    device: str = "cpu",
    soma_layer: _CachedSomaLayer | None = None,
    low_lod: bool = True,
    frame_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run SOMA-X FK on one ``<rec_id>/soma77.npz``.

    Returns ``joints`` ``[T, 77, 3]`` f32, ``vertices`` ``[T, V, 3]`` f32,
    ``faces`` ``[F, 3]`` i32. Optional 1-D ``frame_indices`` selects a frame
    subset before FK (e.g. sparse FK joints for a metric).
    """
    _ensure_soma_on_path()

    import torch

    p = Path(soma77_npz)
    payload = np.load(p, allow_pickle=False)
    for k in ("poses", "transl", "identity_coeffs"):
        if k not in payload.files:
            raise KeyError(f"{p} missing required field '{k}' for SOMA-X FK")

    poses_np = np.asarray(payload["poses"], dtype=np.float32)
    transl_np = np.asarray(payload["transl"], dtype=np.float32)
    identity_coeffs = np.asarray(payload["identity_coeffs"], dtype=np.float32)
    identity_model_type = (
        str(np.asarray(payload["identity_model_type"]).item()).strip().lower()
        if "identity_model_type" in payload.files
        else "smplx"
    )
    scale_params = (
        np.asarray(payload["scale_params"], dtype=np.float32)
        if "scale_params" in payload.files
        else None
    )

    T_full = int(poses_np.shape[0])
    if frame_indices is not None:
        idx = np.asarray(frame_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("frame_indices must be 1-D")
        if int(idx.min()) < 0 or int(idx.max()) >= T_full:
            raise IndexError("frame_indices out of range")
        poses_np = poses_np[idx]
        transl_np = transl_np[idx]

    if soma_data_root is None:
        soma_data_root = default_soma_data_root()
    dev = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    if soma_layer is None:
        soma_layer = cached_soma_layer(
            str(Path(soma_data_root).expanduser().resolve()),
            identity_model_type=identity_model_type,
            device=str(dev),
            low_lod=low_lod,
        )
    soma = soma_layer.soma
    soma_lock = soma_layer.lock

    betas_t = torch.tensor(identity_coeffs, dtype=torch.float32, device=dev)
    if betas_t.ndim != 2:
        raise ValueError(f"{p}: identity_coeffs shape {betas_t.shape} must be (N, C)")
    if betas_t.shape[0] not in (1, T_full):
        raise ValueError(
            f"{p}: identity_coeffs first dim must be 1 or T={T_full}; got {betas_t.shape[0]}"
        )
    if betas_t.shape[0] == T_full:
        betas_t = betas_t.mean(dim=0, keepdim=True)

    global_scale: Any = 1.0
    scale_t = None
    if scale_params is not None:
        scale_t = torch.tensor(scale_params, dtype=torch.float32, device=dev)
        if scale_t.ndim != 2:
            raise ValueError(f"{p}: scale_params shape {scale_t.shape} must be (N, S)")
        if scale_t.shape[0] not in (1, T_full):
            raise ValueError(
                f"{p}: scale_params first dim must be 1 or T={T_full}; got {scale_t.shape[0]}"
            )
        if scale_t.shape[0] == T_full:
            scale_t = scale_t.mean(dim=0, keepdim=True)
        if identity_model_type == "mhr":
            expected_scale_params = soma.identity_model.num_scale_params
            if callable(expected_scale_params):
                expected_scale_params = expected_scale_params()
            expected_scale_params = int(expected_scale_params)
            if scale_t.shape[1] == expected_scale_params + 1:
                global_scale = scale_t[:, :1]
                scale_t = scale_t[:, 1:]
            elif scale_t.shape[1] != expected_scale_params:
                raise ValueError(
                    f"{p}: MHR scale_params width {scale_t.shape[1]} must be "
                    f"{expected_scale_params} or {expected_scale_params + 1}"
                )

    poses_t = torch.tensor(poses_np, dtype=torch.float32, device=dev).contiguous()
    transl_t = torch.tensor(transl_np, dtype=torch.float32, device=dev)

    with soma_lock, torch.no_grad():
        soma.prepare_identity(betas_t, scale_t, global_scale=global_scale)
        fk_out = soma.pose(poses_t, transl=transl_t, pose2rot=True)
    return {
        "joints": fk_out["joints"].detach().cpu().numpy().astype(np.float32, copy=False),
        "vertices": fk_out["vertices"].detach().cpu().numpy().astype(np.float32, copy=False),
        "faces": soma.faces.detach().cpu().numpy().astype(np.int32, copy=False),
    }

def soma77_joints_world_xyz_from_arrays(
    poses: np.ndarray,
    transl: np.ndarray,
    identity_coeffs: np.ndarray,
    *,
    joint_orient: np.ndarray | None = None,
    soma_data_root: Path | None = None,
    device: str = "cpu",
    soma_layer: _CachedSomaLayer | None = None,
    low_lod: bool = True,
) -> np.ndarray:
    """Run SOMA-X FK from in-memory canonical SOMA77 arrays.

    ``joint_orient`` is accepted (part of the UMR499 static context) but not
    consumed by current SOMA-X FK.
    """
    _ensure_soma_on_path()

    import torch

    poses_np = np.asarray(poses, dtype=np.float32)
    transl_np = np.asarray(transl, dtype=np.float32)
    identity_np = np.asarray(identity_coeffs, dtype=np.float32)
    if poses_np.ndim != 3 or poses_np.shape[1:] != (77, 3):
        raise ValueError(f"poses must be [T, 77, 3], got {poses_np.shape}")
    if transl_np.shape != (poses_np.shape[0], 3):
        raise ValueError(f"transl must be [{poses_np.shape[0]}, 3], got {transl_np.shape}")
    if identity_np.ndim != 2:
        raise ValueError(f"identity_coeffs must be [1, C] or [T, C], got {identity_np.shape}")
    if joint_orient is not None and np.asarray(joint_orient).shape != (78, 3, 3):
        raise ValueError(f"joint_orient must be [78, 3, 3], got {np.asarray(joint_orient).shape}")

    T_full = int(poses_np.shape[0])
    if soma_data_root is None:
        soma_data_root = default_soma_data_root()
    dev = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    if soma_layer is None:
        soma_layer = cached_soma_layer(
            str(Path(soma_data_root).expanduser().resolve()),
            identity_model_type="smplx",
            device=str(dev),
            low_lod=low_lod,
        )
    soma = soma_layer.soma
    soma_lock = soma_layer.lock

    betas_t = torch.tensor(identity_np, dtype=torch.float32, device=dev)
    if betas_t.shape[0] not in (1, T_full):
        raise ValueError(
            f"identity_coeffs first dim must be 1 or T={T_full}; got {betas_t.shape[0]}"
        )
    if betas_t.shape[0] == T_full:
        betas_t = betas_t.mean(dim=0, keepdim=True)

    poses_t = torch.tensor(poses_np, dtype=torch.float32, device=dev).contiguous()
    transl_t = torch.tensor(transl_np, dtype=torch.float32, device=dev)
    with soma_lock, torch.no_grad():
        soma.prepare_identity(betas_t, None, global_scale=1.0)
        fk_out = soma.pose(poses_t, transl=transl_t, pose2rot=True)
    return fk_out["joints"].detach().cpu().numpy().astype(np.float32, copy=False)

def soma77_joints_world_xyz_from_matrices(
    pose_matrices: np.ndarray,
    transl: np.ndarray,
    identity_coeffs: np.ndarray,
    *,
    scale_params: np.ndarray | None = None,
    global_scale: float | np.ndarray = 1.0,
    soma_data_root: Path | None = None,
    device: str = "cpu",
    soma_layer: _CachedSomaLayer | None = None,
    low_lod: bool = True,
    identity_model_type: str = "smplx",
) -> np.ndarray:
    """Run SOMA-X FK from rotation matrices (``pose2rot=False``).

    ``identity_model_type`` selects the identity model the coefficients belong
    to. MHR clips must also provide their 68-D ``scale_params``. The defaults
    preserve the legacy SMPL-X call contract.
    """
    _ensure_soma_on_path()

    import torch

    poses_np = np.asarray(pose_matrices, dtype=np.float32)
    transl_np = np.asarray(transl, dtype=np.float32)
    identity_np = np.asarray(identity_coeffs, dtype=np.float32)
    if poses_np.ndim != 4 or poses_np.shape[1:] != (77, 3, 3):
        raise ValueError(f"pose_matrices must be [T, 77, 3, 3], got {poses_np.shape}")
    if transl_np.shape != (poses_np.shape[0], 3):
        raise ValueError(f"transl must be [{poses_np.shape[0]}, 3], got {transl_np.shape}")
    if identity_np.ndim != 2:
        raise ValueError(f"identity_coeffs must be [1, C] or [T, C], got {identity_np.shape}")

    T_full = int(poses_np.shape[0])
    if soma_data_root is None:
        soma_data_root = default_soma_data_root()
    dev = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    if soma_layer is None:
        soma_layer = cached_soma_layer(
            str(Path(soma_data_root).expanduser().resolve()),
            identity_model_type=identity_model_type,
            device=str(dev),
            low_lod=low_lod,
        )
    soma = soma_layer.soma
    soma_lock = soma_layer.lock

    betas_t = torch.tensor(identity_np, dtype=torch.float32, device=dev)
    if betas_t.shape[0] not in (1, T_full):
        raise ValueError(
            f"identity_coeffs first dim must be 1 or T={T_full}; got {betas_t.shape[0]}"
        )
    if betas_t.shape[0] == T_full:
        betas_t = betas_t.mean(dim=0, keepdim=True)

    scale_t = None
    global_scale_arg: Any = global_scale
    if scale_params is not None:
        scale_np = np.asarray(scale_params, dtype=np.float32)
        if scale_np.ndim != 2 or scale_np.shape[0] not in (1, T_full):
            raise ValueError(
                f"scale_params must be [1, S] or [T, S], got {scale_np.shape}"
            )
        if scale_np.shape[0] == T_full:
            scale_np = scale_np.mean(axis=0, keepdims=True)
        scale_t = torch.tensor(scale_np, dtype=torch.float32, device=dev)
        if identity_model_type == "mhr":
            expected = soma.identity_model.num_scale_params
            if callable(expected):
                expected = expected()
            expected = int(expected)
            if scale_t.shape[1] == expected + 1:
                global_scale_arg = scale_t[:, :1]
                scale_t = scale_t[:, 1:]
            elif scale_t.shape[1] != expected:
                raise ValueError(
                    f"MHR scale_params width {scale_t.shape[1]} must be "
                    f"{expected} or {expected + 1}"
                )
    if isinstance(global_scale_arg, np.ndarray):
        global_scale_arg = torch.tensor(
            global_scale_arg, dtype=torch.float32, device=dev
        )

    poses_t = torch.tensor(poses_np, dtype=torch.float32, device=dev).contiguous()
    transl_t = torch.tensor(transl_np, dtype=torch.float32, device=dev)
    with soma_lock, torch.no_grad():
        soma.prepare_identity(betas_t, scale_t, global_scale=global_scale_arg)
        fk_out = soma.pose(poses_t, transl=transl_t, pose2rot=False)
    joints = fk_out["joints"].detach().cpu().numpy().astype(np.float32, copy=False)
    if joints.shape[1] == 78:
        joints = joints[:, 1:]
    return joints

def soma77_joints_world_xyz(
    soma77_npz: str | Path,
    *,
    soma_data_root: Path | None = None,
    device: str = "cpu",
    soma_layer: _CachedSomaLayer | None = None,
    low_lod: bool = True,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience: return only ``[T, 77, 3]`` world-frame joint positions."""
    out = soma77_fk_npz(
        soma77_npz,
        soma_data_root=soma_data_root,
        device=device,
        soma_layer=soma_layer,
        low_lod=low_lod,
        frame_indices=frame_indices,
    )
    return out["joints"]

@lru_cache(maxsize=1)
def soma77_parent_indices(
    soma_data_root: str | None = None,
) -> tuple[int, ...]:
    """Return a 77-long tuple where ``parent[j]`` is the parent joint id or -1 for the root.

    Built from the cached :class:`SOMALayer`'s ``joint_parent_ids``. The
    kinematic tree is identity-independent, so the ``"soma"`` identity model
    (ships with the SOMA-X assets) avoids the licensed SMPL-X download.
    """
    _ensure_soma_on_path()
    import torch  # noqa: WPS433

    root = soma_data_root or str(default_soma_data_root())
    layer = cached_soma_layer(root, device="cpu", identity_model_type="soma")
    parents_t = layer.soma.joint_parent_ids
    if isinstance(parents_t, torch.Tensor):
        parents = parents_t.detach().cpu().tolist()
    else:
        parents = list(parents_t)
    raw_n = len(parents)
    if raw_n == 78:
        # Rig includes a virtual root at index 0; viewer/FK use 77 body joints.
        parents = parents[1:]
    elif raw_n != 77:
        raise RuntimeError(
            f"expected SOMA77 (77 or 78 rig joints); SOMALayer reported {raw_n}"
        )
    # SOMA-X stores parent ids 1-indexed with 0 == "no parent". Convert to
    # 0-indexed with -1 == root for downstream tooling.
    return tuple(int(p) - 1 for p in parents)

def soma77_edges(soma_data_root: str | None = None) -> list[list[int]]:
    """Return ``[[parent, child], ...]`` for all 76 non-root SOMA77 joints."""
    parents = soma77_parent_indices(soma_data_root)
    edges: list[list[int]] = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        edges.append([int(parent), int(child)])
    return edges

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SOMA-X FK wrapper sanity hooks")
    parser.add_argument("--check-import", action="store_true", help="verify SOMA-X imports cleanly")
    parser.add_argument("--fk", type=str, default=None, help="run FK on this soma77.npz")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.check_import:
        _ensure_soma_on_path()
        from soma.soma import SOMALayer

        print(f"OK: SOMA-X loaded from {_soma_x_path()}")
    if args.fk:
        out = soma77_fk_npz(args.fk, device=args.device)
        print(
            f"OK: FK ran on {args.fk}; joints={out['joints'].shape} "
            f"vertices={out['vertices'].shape} faces={out['faces'].shape}"
        )

if __name__ == "__main__":
    _cli()
