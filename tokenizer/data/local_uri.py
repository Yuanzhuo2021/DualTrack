"""``local://`` URI helpers for dataset artifacts under one ``data_root``.

URIs like ``local://recordings/<recording_id>/soma77.npz`` or
``local://manifests/<scale>/train.txt`` resolve against, in priority order:
an explicit ``data_root`` argument, the ``MOTIONVERSE_DATA_ROOT`` environment
variable, or the relative default ``../omega-MotionVerse``. Plain filesystem
paths pass through unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

LOCAL_URI_PREFIX = "local://"
ENV_DATA_ROOT = "MOTIONVERSE_DATA_ROOT"
DEFAULT_DATA_ROOT = Path("../omega-MotionVerse")

def default_data_root() -> Path:
    """Return the ambient ``data_root`` (env override, else ``../omega-MotionVerse``)."""
    env = os.environ.get(ENV_DATA_ROOT)
    return Path(env) if env else DEFAULT_DATA_ROOT

def resolve_local_uri(
    uri: str | Path, data_root: str | Path | None = None
) -> Path:
    """Resolve ``local://X`` → ``<data_root>/X``; pass-through otherwise."""
    s = str(uri)
    if s.startswith(LOCAL_URI_PREFIX):
        rel = s[len(LOCAL_URI_PREFIX):]
        root = Path(data_root) if data_root is not None else default_data_root()
        return root / rel
    return Path(s)

def to_local_uri(
    path: str | Path, data_root: str | Path | None = None
) -> str:
    """Inverse: ``<data_root>/X`` → ``local://X`` when ``path`` is under root."""
    p = Path(path)
    root = Path(data_root) if data_root is not None else default_data_root()
    try:
        rel = p.resolve().relative_to(root.resolve())
        return f"{LOCAL_URI_PREFIX}{rel}"
    except (ValueError, OSError):
        return str(p)

def scale_label(count: int) -> str:
    """Derive ``<scale>`` from a recording count: ``{count // 1000}k`` when >= 1000, else the integer."""
    if count < 0:
        raise ValueError(f"count must be >= 0; got {count}")
    if count >= 1000:
        return f"{count // 1000}k"
    return str(count)

__all__ = [
    "DEFAULT_DATA_ROOT",
    "ENV_DATA_ROOT",
    "LOCAL_URI_PREFIX",
    "default_data_root",
    "resolve_local_uri",
    "scale_label",
    "to_local_uri",
]
