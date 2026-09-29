"""PyTorch ``UMRDataset`` over pre-materialized ``umr499.npz`` files.

Reads lazily and yields 181-frame canonical crops (1 anchor + 180 records)
as a dict of torch tensors::

    features                  [180, 499]    tokenizer input
    traj_contact              [180, 43]     factorized stream
    joints76_rot6d            [180, 456]    factorized stream
    init_root_pos             [3]           crop-anchor seed (root xyz)
    init_root_rot6d           [6]           crop-anchor seed (root rot6d)
    init_joints76_rot6d       [76, 6]       crop-anchor seed (joints76 rot6d)
    joints77_pos              [181, 77, 3]  canonical FK reference (L2 eval)
    identity_coeffs           [1, C]         static FK context
    joint_orient              [78, 3, 3]     static FK context
    foot_contact              [180, 4]      target-frame contact view
    source_path               str           diagnostics

Manifest entries are either absolute ``.npz`` paths, or bare
``recording_id``s resolved as ``<recordings_root>/<recording_id>/umr499.npz``;
``#``-prefixed lines are comments.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.local_uri import LOCAL_URI_PREFIX, default_data_root, resolve_local_uri
from data.umr_schema import (
    DIM_FEATURES,
    SLICE_FOOT_CONTACT,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    WINDOW_DEFAULT,
    CanonicalAnchor,
    UMR499,
    split_streams,
    unpack_features,
)

DEFAULT_TARGET_FPS = 50.0
"""UMR499 fps must match this constant; the loader does not resample at load time."""

DEFAULT_UMR499_FILENAME = "umr499.npz"

def _crop_anchor(
    umr: UMR499, start: int
) -> CanonicalAnchor:
    """Build the canonical anchor at crop start ``s = start``.

    For ``s == 0`` the clip's own anchor is returned; for ``s > 0`` it is
    reconstructed from ``features[s - 1]`` and ``joints77_pos[s, 0]``.
    """
    if start == 0:
        return CanonicalAnchor(
            init_root_pos=umr.canonical_anchor.init_root_pos.astype(np.float32, copy=True),
            init_root_rot6d=umr.canonical_anchor.init_root_rot6d.astype(np.float32, copy=True),
            init_joints76_rot6d=umr.canonical_anchor.init_joints76_rot6d.astype(np.float32, copy=True),
        )

    prev = unpack_features(umr.features[start - 1 : start])         # [1, ...]
    return CanonicalAnchor(
        init_root_pos=umr.joints77_pos[start, 0].astype(np.float32, copy=True),
        init_root_rot6d=prev.root_rot6d[0].astype(np.float32, copy=True),
        init_joints76_rot6d=prev.joints76_rot6d[0].astype(np.float32, copy=True),
    )

class UMRDataset(Dataset):
    """``umr499.npz`` → 180-record canonical crops as torch tensors."""

    def __init__(
        self,
        list_files: str | Path | Iterable[str | Path],
        recordings_root: Optional[str | Path] = None,
        *,
        data_root: Optional[str | Path] = None,
        window: int = WINDOW_DEFAULT,
        target_fps: float = DEFAULT_TARGET_FPS,
        umr499_filename: str = DEFAULT_UMR499_FILENAME,
        seed: int = 0,
        drop_short_at_init: bool = False,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else default_data_root()
        if recordings_root is None:
            self.recordings_root: Path | None = None
        else:
            self.recordings_root = resolve_local_uri(recordings_root, self.data_root)
        self.window = int(window)
        if self.window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        self.target_fps = float(target_fps)
        self.umr499_filename = str(umr499_filename)
        self.seed = int(seed)

        if isinstance(list_files, (str, Path)):
            manifest_inputs: list[str | Path] = [list_files]
        else:
            manifest_inputs = list(list_files)
        if not manifest_inputs:
            raise ValueError("UMRDataset requires at least one manifest path")

        manifest_paths = [resolve_local_uri(p, self.data_root) for p in manifest_inputs]
        self.manifest_paths = [str(p) for p in manifest_paths]
        all_entries: list[str] = []
        for mp in manifest_paths:
            for ln in mp.read_text().splitlines():
                s = ln.strip()
                if s and not s.startswith("#"):
                    all_entries.append(s)
        if not all_entries:
            raise ValueError(f"No valid entries in manifest(s) {self.manifest_paths!r}")
        self.npz_paths: list[str] = [self._resolve_entry(s) for s in all_entries]

        if drop_short_at_init:
            kept: list[str] = []
            for p in self.npz_paths:
                try:
                    n_records = self._peek_num_records(p)
                except Exception:
                    continue
                if n_records >= self.window:
                    kept.append(p)
            if not kept:
                raise ValueError(
                    f"All manifest entries shorter than {self.window} records"
                )
            self.npz_paths = kept

    def _resolve_entry(self, entry: str) -> str:
        if entry.startswith(LOCAL_URI_PREFIX):
            return str(resolve_local_uri(entry, self.data_root))
        if entry.endswith(".npz") or "/" in entry:
            return entry
        if self.recordings_root is None:
            raise ValueError(
                f"manifest entry {entry!r} looks like a bare recording_id but "
                f"recordings_root is None; pass recordings_root=... or use full paths."
            )
        return str(self.recordings_root / entry / self.umr499_filename)

    def _peek_num_records(self, path: str) -> int:
        with np.load(path, allow_pickle=False) as data:
            return int(data["features"].shape[0])

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = torch.Generator().manual_seed(self.seed ^ int(idx) ^ 0x9E3779B1)
        attempts = 0
        last_err: Exception | None = None
        cur = idx
        while attempts < 8:
            try:
                return self._load_one(self.npz_paths[cur], rng)
            except (ValueError, KeyError, FileNotFoundError, RuntimeError, EOFError, zipfile.BadZipFile) as e:
                last_err = e
                attempts += 1
                cur = int(torch.randint(0, len(self.npz_paths), (1,), generator=rng).item())
        raise RuntimeError(
            f"UMRDataset.__getitem__: 8 retries failed; last error: {last_err}"
        )

    def _load_one(self, path: str, rng: torch.Generator) -> dict[str, Any]:
        umr = UMR499.from_npz(path)
        if abs(umr.fps - self.target_fps) > 1e-3:
            raise ValueError(
                f"{path}: umr499.fps={umr.fps} != target_fps={self.target_fps}"
            )

        N = umr.num_records
        if N < self.window:
            raise ValueError(
                f"{path}: clip has {N} records < window={self.window}"
            )

        if N == self.window:
            start = 0
        else:
            start = int(
                torch.randint(0, N - self.window + 1, (1,), generator=rng).item()
            )

        anchor = _crop_anchor(umr, start)
        feat_crop = umr.features[start : start + self.window]
        joints_crop = umr.joints77_pos[start : start + self.window + 1]

        if feat_crop.shape != (self.window, DIM_FEATURES):
            raise ValueError(
                f"{path}: feature crop shape {feat_crop.shape} != "
                f"({self.window}, {DIM_FEATURES})"
            )
        if joints_crop.shape != (self.window + 1, UMR_NUM_JOINTS, 3):
            raise ValueError(
                f"{path}: joints77_pos crop shape {joints_crop.shape} != "
                f"({self.window + 1}, {UMR_NUM_JOINTS}, 3)"
            )

        traj_contact, joints76_rot6d = split_streams(feat_crop)
        foot_contact = feat_crop[..., SLICE_FOOT_CONTACT]
        return {
            "source_path": path,
            "features": torch.from_numpy(feat_crop.astype(np.float32, copy=False)),
            "traj_contact": torch.from_numpy(traj_contact),
            "joints76_rot6d": torch.from_numpy(joints76_rot6d),
            "foot_contact": torch.from_numpy(foot_contact.astype(np.float32, copy=False)),
            "init_root_pos": torch.from_numpy(anchor.init_root_pos),
            "init_root_rot6d": torch.from_numpy(anchor.init_root_rot6d),
            "init_joints76_rot6d": torch.from_numpy(anchor.init_joints76_rot6d),
            "joints77_pos": torch.from_numpy(joints_crop.astype(np.float32, copy=False)),
            "identity_coeffs": torch.from_numpy(umr.identity_coeffs.astype(np.float32, copy=False)),
            "joint_orient": torch.from_numpy(umr.joint_orient.astype(np.float32, copy=False)),
        }

def collate_umr(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor keys on dim 0; preserve metadata as lists."""
    out: dict[str, Any] = {}
    for k, v in batch[0].items():
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([s[k] for s in batch])
        else:
            out[k] = [s[k] for s in batch]
    return out

__all__ = ["UMRDataset", "collate_umr"]
