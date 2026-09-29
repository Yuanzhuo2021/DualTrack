"""Evaluation CLI for the codec.

Runs the requested gates over a manifest of ``umr499.npz`` files using a
self-contained ``experiments.train_native`` checkpoint (``codec_config`` +
``net`` + ``norm``) — or ``codec_mode = identity`` for the no-codec baseline —
and writes a JSON report keyed by metric name.

Example::

    python -m tools.eval --codec-mode vq_vae --checkpoint runs/<run>/model/best.pt \
        --manifest local://manifests/val.txt --recordings-root local://raw/recordings \
        --gates L1,L2,L3,L4,L5 --out logs/eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from data.local_uri import default_data_root, resolve_local_uri
from data.umr_dataset import UMRDataset
from data.soma77_fk import (
    soma77_joints_world_xyz_from_matrices,
)
from data.umr_schema import (
    SLICE_FOOT_CONTACT,
    UMR499,
)
from data.umr_to_soma77 import materialize_features, materialize_features_matrices
from eval.codebook import codebook_metrics_per_layer, codebook_metrics_single_layer
from eval.codec_reconstruction import codec_reconstruction_metrics
from eval.geometry import soma_canonical_metrics
from eval.motion_validity import motion_validity_metrics
from eval.streaming import streaming_equivalence_metrics

log = logging.getLogger("eval")

CodecMode = str            # "identity" | "vae" | "vq_vae"

def _default_num_gpus() -> int:
    """Worker→GPU pinning count: ``EVAL_NUM_GPUS`` override, else autodetect."""
    env = os.environ.get("EVAL_NUM_GPUS")
    if env:
        return max(1, int(env))
    try:
        import torch  # deferred: identity workers may not need torch otherwise
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1

def _load_codec_adapter(args: argparse.Namespace):
    """Return an object with ``.reconstruct(features)`` and optionally
    ``.token_indices(features)``; ``identity`` mode reconstructs the input itself."""
    if args.codec_mode == "identity":
        return _IdentityCodec()
    return _load_native_codec(args)

class _IdentityCodec:
    name = "identity"
    is_vq_vae = False

    def reconstruct(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return features, {}

    def token_indices(self, features: np.ndarray) -> dict[str, np.ndarray]:
        return {}

def _cfg_kwargs(cls: type, raw: dict) -> dict:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in allowed}

def _load_native_codec(args: argparse.Namespace):
    """Load a plain native checkpoint (``experiments.train_native``): ``codec_config``
    rebuilds the tokenizer, ``net`` its state_dict, ``norm`` the packed mean/std
    (reconstruction returns physical-unit features)."""
    import torch                                              # noqa: WPS433

    from models.umr.structured_vq import (
        BackboneConfig,
        CodecConfig,
        GroupConfig,
        QuantizerConfig,
        build_structured_vq,
    )

    if args.checkpoint is None:
        raise ValueError(f"codec_mode={args.codec_mode!r} requires --checkpoint")

    from checkpoint_io import load_checkpoint
    ckpt = load_checkpoint(args.checkpoint)
    cc = ckpt["codec_config"]
    groups = None
    if cc.get("groups"):
        groups = [
            GroupConfig(
                name=g["name"], start=int(g["start"]), stop=int(g["stop"]),
                backbone=BackboneConfig(**_cfg_kwargs(BackboneConfig, g["backbone"])),
                quantizer=QuantizerConfig(**_cfg_kwargs(QuantizerConfig, g["quantizer"])),
            )
            for g in cc["groups"]
        ]
    cfg = CodecConfig(
        backbone=BackboneConfig(**_cfg_kwargs(BackboneConfig, cc["backbone"])),
        quantizer=QuantizerConfig(**_cfg_kwargs(QuantizerConfig, cc["quantizer"])),
        input_dim=int(cc.get("input_dim", 499)),
        groups=groups,
    )
    model = build_structured_vq(cfg)
    # Checkpoints may carry an extra ``semantic_head.*`` (TMR distillation head)
    # unused for geometry eval: allow via strict=False, fail on genuine mismatch.
    load_result = model.load_state_dict(ckpt["net"], strict=False)
    unexpected = [k for k in load_result.unexpected_keys if not k.startswith("semantic_head.")]
    if load_result.missing_keys or unexpected:
        raise RuntimeError(
            f"checkpoint/codec param mismatch: missing={load_result.missing_keys} "
            f"unexpected={unexpected}"
        )
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    mean = torch.from_numpy(np.asarray(ckpt["norm"]["mean"], dtype=np.float32)).view(1, -1, 1).to(device)
    std = torch.from_numpy(np.asarray(ckpt["norm"]["std"], dtype=np.float32)).view(1, -1, 1).to(device)
    stride = int(model.temporal_stride)

    _q = cfg.quantizer
    _cb_size = int(_q.codebook_size)

    def _trim(arr: np.ndarray) -> np.ndarray:
        if stride <= 1:
            return arr
        keep = (arr.shape[0] // stride) * stride
        return arr[:keep] if keep > 0 else arr

    class _NativeCodec:
        def __init__(self) -> None:
            self.name = args.codec_mode
            self.is_vq_vae = args.codec_mode == "vq_vae"
            self.codebook_size = _cb_size
            self.quantizer_kind = getattr(_q, "kind", "")
            self.temporal_stride = stride

        @torch.no_grad()
        def reconstruct(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            trimmed = _trim(features)
            t = torch.from_numpy(trimmed).to(device).unsqueeze(0).transpose(1, 2)   # [1, 499, T]
            t_norm = (t - mean) / std
            rec_norm = model(t_norm).features_rec
            rec = rec_norm * std + mean
            # foot_contact is identity under norm; keep raw logits from the decoder.
            rec[:, SLICE_FOOT_CONTACT] = rec_norm[:, SLICE_FOOT_CONTACT]
            rec_np = rec.transpose(1, 2).squeeze(0).cpu().numpy()
            return rec_np, {}

        @torch.no_grad()
        def token_indices(self, features: np.ndarray) -> dict[str, np.ndarray]:
            if not self.is_vq_vae:
                return {}
            trimmed = _trim(features)
            t = torch.from_numpy(trimmed).to(device).unsqueeze(0).transpose(1, 2)
            t_norm = (t - mean) / std
            indices = model.encode_indices(t_norm)
            if isinstance(indices, dict):  # part-wise: one stream per group
                return {name: idx.squeeze(0).cpu().numpy() for name, idx in indices.items()}
            return {"structured": indices.squeeze(0).cpu().numpy()}

    return _NativeCodec()

def _load_fk_identity_context(
    source_path: str | Path | None,
    fallback_identity_coeffs: np.ndarray,
) -> tuple[np.ndarray, str, np.ndarray | None, float | np.ndarray]:
    """Recover identity metadata omitted by legacy UMR-499 artifacts.

    BONES UMR files sit beside their canonical ``soma77.npz`` source. That
    file records whether the 45-D coefficients belong to MHR and carries the
    required scale parameters. Non-BONES callers retain the legacy SMPL-X
    fallback.
    """
    if source_path is None:
        return fallback_identity_coeffs, "smplx", None, 1.0
    canonical_path = Path(source_path).with_name("soma77.npz")
    if not canonical_path.is_file():
        return fallback_identity_coeffs, "smplx", None, 1.0
    with np.load(canonical_path, allow_pickle=False) as payload:
        identity = np.asarray(
            payload["identity_coeffs"]
            if "identity_coeffs" in payload.files
            else fallback_identity_coeffs,
            dtype=np.float32,
        )
        identity_type = (
            str(np.asarray(payload["identity_model_type"]).item()).strip().lower()
            if "identity_model_type" in payload.files
            else "smplx"
        )
        scale = (
            np.asarray(payload["scale_params"], dtype=np.float32)
            if "scale_params" in payload.files
            else None
        )
        global_scale: float | np.ndarray = (
            np.asarray(payload["global_scale"], dtype=np.float32)
            if "global_scale" in payload.files
            else 1.0
        )
    return identity, identity_type, scale, global_scale

def _evaluate_clip(
    umr: UMR499,
    codec,
    *,
    gates: set[str],
    accum: dict[str, list[float]],
    codebook_indices: dict[str, list[np.ndarray]],
    device: str = "cpu",
    source_path: str | Path | None = None,
) -> None:
    features_rec, _info = codec.reconstruct(umr.features)
    # Trained codecs round T down to a multiple of their encoder stride; align
    # the GT side to the reconstructed length so per-frame metrics line up.
    T_rec = features_rec.shape[0]
    features_gt = umr.features[:T_rec]
    joints77_gt = umr.joints77_pos[: T_rec + 1] if umr.joints77_pos.shape[0] >= T_rec + 1 else umr.joints77_pos

    if "L1" in gates:
        m1 = codec_reconstruction_metrics(features_rec, features_gt)
        for k, v in m1.to_dict().items():
            accum[k].append(v)

    if "L2" in gates or "L4" in gates or "L5" in gates:
        decoded = materialize_features(features_rec, umr.canonical_anchor)
        gt_decoded = materialize_features(features_gt, umr.canonical_anchor)
        decoded_mats = None
        joints77_rec = None
        if "L2" in gates or "L4" in gates:
            decoded_mats = materialize_features_matrices(features_rec, umr.canonical_anchor)
            identity, identity_type, scale, global_scale = _load_fk_identity_context(
                source_path, umr.identity_coeffs
            )
            joints77_rec = soma77_joints_world_xyz_from_matrices(
                decoded_mats.rotmat77,
                decoded_mats.transl,
                identity,
                identity_model_type=identity_type,
                scale_params=scale,
                global_scale=global_scale,
                device=device,
            )
        if "L2" in gates:
            m2 = soma_canonical_metrics(
                decoded.transl, gt_decoded.transl,
                decoded.rotvec77, gt_decoded.rotvec77,
                joints77_pos_rec=joints77_rec,
                joints77_pos_gt=joints77_gt,
            )
            for k, v in m2.to_dict().items():
                accum[k].append(v)
        if "L4" in gates:
            assert joints77_rec is not None
            m4 = motion_validity_metrics(joints77_rec, joints77_gt)
            for k, v in m4.to_dict().items():
                accum[k].append(v)
        if "L5" in gates:
            m5 = streaming_equivalence_metrics(features_rec, umr.canonical_anchor)
            for k, v in m5.to_dict().items():
                accum[k].append(v)

    if "L3" in gates and codec.is_vq_vae:
        for stream, idx in codec.token_indices(umr.features).items():
            codebook_indices[stream].append(idx)

def _identity_clip_worker(args_tuple: tuple[str, set[str], str, int, int]) -> dict[str, Any]:
    """Process-pool worker: evaluate one UMR clip with the identity codec.

    Args are plain python objects (pickle-friendly); ``worker_id`` pins
    ``CUDA_VISIBLE_DEVICES`` so workers spread across GPUs.
    """
    import os  # noqa: WPS433 (inside worker)
    path, gates, device, window, worker_id = args_tuple
    if device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id % _default_num_gpus())

    # deferred imports so worker startup only pays what it uses
    from data.umr_schema import UMR499  # noqa: WPS433 (inside worker)
    from data.umr_to_soma77 import materialize_features, materialize_features_matrices  # noqa: WPS433
    from eval.codec_reconstruction import codec_reconstruction_metrics  # noqa: WPS433
    from eval.geometry import soma_canonical_metrics  # noqa: WPS433
    from eval.motion_validity import motion_validity_metrics  # noqa: WPS433
    from eval.streaming import streaming_equivalence_metrics  # noqa: WPS433

    try:
        umr = UMR499.from_npz(path)
    except Exception as exc:  # noqa: BLE001
        return {"_skip": str(exc), "_path": path}
    if umr.num_records < window:
        return {"_skip": "too_short", "_path": path}

    features_rec = umr.features
    metrics: dict[str, float] = {}

    if "L1" in gates:
        m = codec_reconstruction_metrics(features_rec, umr.features)
        metrics.update(m.to_dict())

    if "L2" in gates or "L4" in gates or "L5" in gates:
        decoded = materialize_features(features_rec, umr.canonical_anchor)
        gt_decoded = materialize_features(umr.features, umr.canonical_anchor)
        joints77_rec = None
        if "L2" in gates or "L4" in gates:
            from data.soma77_fk import soma77_joints_world_xyz_from_matrices  # noqa: WPS433
            decoded_mats = materialize_features_matrices(features_rec, umr.canonical_anchor)
            identity, identity_type, scale, global_scale = _load_fk_identity_context(
                path, umr.identity_coeffs
            )
            joints77_rec = soma77_joints_world_xyz_from_matrices(
                decoded_mats.rotmat77,
                decoded_mats.transl,
                identity,
                identity_model_type=identity_type,
                scale_params=scale,
                global_scale=global_scale,
                device=device,
            )
        if "L2" in gates:
            m2 = soma_canonical_metrics(
                decoded.transl, gt_decoded.transl,
                decoded.rotvec77, gt_decoded.rotvec77,
                joints77_pos_rec=joints77_rec,
                joints77_pos_gt=umr.joints77_pos,
            )
            metrics.update(m2.to_dict())
        if "L4" in gates and joints77_rec is not None:
            m4 = motion_validity_metrics(joints77_rec, umr.joints77_pos)
            metrics.update(m4.to_dict())
        if "L5" in gates:
            m5 = streaming_equivalence_metrics(features_rec, umr.canonical_anchor)
            metrics.update(m5.to_dict())

    return metrics

class _CodecMeta:
    """Stand-in consulted by the parent in parallel trained-codec eval; only
    ``is_vq_vae`` is read (the workers own the real codec)."""

    name = "trained"

    def __init__(self, *, is_vq_vae: bool) -> None:
        self.is_vq_vae = is_vq_vae

# Module-level codec cache: each worker process loads the codec once on its
# assigned GPU and reuses it for every clip it receives.
_TRAINED_CODEC: Any = None

def _trained_codec_init_worker(
    num_gpus: int,
    counter,
    codec_kwargs: dict[str, Any],
) -> None:
    """Spawn-time init: pin ``CUDA_VISIBLE_DEVICES`` (unique rank per worker via
    the shared ``counter``) and lazy-load the codec."""
    with counter.get_lock():
        rank = int(counter.value)
        counter.value += 1
    if num_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_gpus)
    os.environ["EVAL_WORKER_RANK"] = str(rank)

    global _TRAINED_CODEC
    if _TRAINED_CODEC is None:
        ns = argparse.Namespace(**codec_kwargs)
        _TRAINED_CODEC = _load_codec_adapter(ns)

def _trained_clip_worker(args_tuple: tuple[str, list[str], str, int]) -> dict[str, Any]:
    """Worker: evaluate one clip using the per-process cached trained codec."""
    path, gates_list, device, window = args_tuple
    if _TRAINED_CODEC is None:
        return {"_skip": "codec not initialized", "_path": path}
    try:
        umr = UMR499.from_npz(path)
    except Exception as exc:
        return {"_skip": str(exc), "_path": path}
    if umr.num_records < window:
        return {"_skip": "too_short", "_path": path}

    accum: dict[str, list[float]] = defaultdict(list)
    codebook: dict[str, list[np.ndarray]] = defaultdict(list)
    try:
        _evaluate_clip(
            umr,
            _TRAINED_CODEC,
            gates=set(gates_list),
            accum=accum,
            codebook_indices=codebook,
            device=device,
            source_path=path,
        )
    except Exception as exc:
        return {"_skip": f"eval error: {exc}", "_path": path}

    metrics = {k: float(vs[-1]) for k, vs in accum.items() if vs}
    cb = {stream: list(arrs) for stream, arrs in codebook.items() if arrs}
    return {"_metrics": metrics, "_codebook": cb}

def _aggregate(
    accum: dict[str, list[float]],
    codebook_indices: dict[str, list[np.ndarray]],
    codec,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for k, vals in accum.items():
        report[k] = {
            "mean": float(mean(vals)),
            "median": float(median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n": len(vals),
        }

    if codebook_indices and codec.is_vq_vae:
        codebook = {}
        for stream, idx_list in codebook_indices.items():
            if not idx_list:
                continue
            sample = idx_list[0]
            if sample.ndim == 2:
                merged = np.concatenate(idx_list, axis=0)[None, :, :]
                cbsize = _guess_codebook_size(merged)
                metrics = codebook_metrics_per_layer(merged, cbsize)
                codebook[stream] = [asdict(m) for m in metrics]
            else:
                merged = np.concatenate(idx_list, axis=0)[None, :]
                cbsize = _guess_codebook_size(merged)
                metrics = codebook_metrics_single_layer(merged, cbsize)
                codebook[stream] = asdict(metrics)
        report["codebook"] = codebook
    return report

def _guess_codebook_size(indices: np.ndarray) -> int:
    """Best-effort codebook size from observed indices (next power-of-two)."""
    max_idx = int(indices[indices >= 0].max()) if (indices >= 0).any() else 0
    size = 1
    while size < max_idx + 1:
        size <<= 1
    return size or 1

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--codec-mode", required=True, choices=("identity", "vae", "vq_vae"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Self-contained native checkpoint from experiments.train_native "
                             "(codec_config + net + norm). Required for vae/vq_vae.")
    parser.add_argument("--manifest", action="append", required=True,
                        help="One or more manifest files (path or local:// URI).")
    parser.add_argument("--recordings-root", default=None,
                        help="Accepts local:// URIs.")
    parser.add_argument("--data-root", default=None,
                        help="Resolve local:// URIs against this root (default: $MOTIONVERSE_DATA_ROOT or ../omega-MotionVerse).")
    parser.add_argument("--gates", default="L1,L2,L4",
                        help="Comma-separated list from L1,L2,L3,L4,L5.")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--min-records", type=int, default=1,
                        help="Skip clips with fewer than this many feature records. "
                             "Default 1 = evaluate every clip. Set 180 to match the "
                             "training window.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Process workers (default 1 = serial). Each worker pins to a "
                             "distinct GPU via CUDA_VISIBLE_DEVICES (autodetected; set "
                             "EVAL_NUM_GPUS to cap how many GPUs are used).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True,
                        help="Output JSON path (accepts local:// URIs).")
    parser.add_argument("--save-per-clip", action=argparse.BooleanOptionalAction, default=True,
                        help="Also dump per-clip metric arrays alongside --out. Defaults to on. "
                             "Disable with --no-save-per-clip.")
    parser.add_argument("--save-per-clip-path", default=None,
                        help="Override location of the per-clip JSON. Defaults to a sibling of "
                             "--out named ``<stem>.per_clip.json`` (accepts local:// URIs).")
    args = parser.parse_args(argv)
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    out_path = resolve_local_uri(args.out, data_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    gates = set(args.gates.split(","))
    valid_gates = {"L1", "L2", "L3", "L4", "L5"}
    unknown = gates - valid_gates
    if unknown:
        raise ValueError(f"unknown gates {sorted(unknown)}; expected subset of {sorted(valid_gates)}")

    workers = max(1, int(args.workers)) if hasattr(args, "workers") else 1
    is_trained_codec = args.codec_mode != "identity"

    codec = None
    if not (workers > 1 and is_trained_codec):
        codec = _load_codec_adapter(args)
    if "L3" in gates and codec is not None and not codec.is_vq_vae:
        log.warning("gate L3 requested but codec_mode=%s; skipping L3.", codec.name)
        gates.discard("L3")
    if "L3" in gates and codec is None and args.codec_mode == "vae":
        gates.discard("L3")

    ds = UMRDataset(
        args.manifest if isinstance(args.manifest, list) else [args.manifest],
        recordings_root=args.recordings_root,
        data_root=data_root,
        window=max(1, int(args.min_records)),
    )
    if args.max_clips is not None:
        max_n = min(args.max_clips, len(ds))
    else:
        max_n = len(ds)

    accum: dict[str, list[float]] = defaultdict(list)
    codebook_indices: dict[str, list[np.ndarray]] = defaultdict(list)
    per_clip_ids: list[str] = []
    per_clip_metrics: list[dict[str, float]] = []

    def _record_clip(npz_path: str, metrics: dict[str, float]) -> None:
        per_clip_ids.append(Path(npz_path).parent.name)
        per_clip_metrics.append({k: float(v) for k, v in metrics.items()})

    use_parallel = workers > 1
    paths = ds.npz_paths[:max_n]
    num_gpus = _default_num_gpus()

    if use_parallel and not is_trained_codec:
        log.info("evaluating %d clips with %d parallel identity workers", len(paths), workers)
        tasks = [(p, gates, args.device, ds.window, i % workers) for i, p in enumerate(paths)]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_identity_clip_worker, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                result = fut.result()
                path = futures[fut]
                done += 1
                if "_skip" in result:
                    log.warning("skip %s: %s", result.get("_path"), result["_skip"])
                else:
                    for k, v in result.items():
                        accum[k].append(v)
                    _record_clip(path, result)
                if done % 500 == 0:
                    log.info("processed %d / %d clips", done, len(paths))
    elif use_parallel and is_trained_codec:
        log.info(
            "evaluating %d clips with %d parallel trained-codec workers (1 GPU per worker)",
            len(paths), workers,
        )
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        codec_kwargs = dict(
            codec_mode=args.codec_mode,
            checkpoint=args.checkpoint,
            device=args.device,
            data_root=args.data_root,
        )
        tasks = [(p, list(gates), args.device, ds.window) for p in paths]
        codec = _CodecMeta(is_vq_vae=(args.codec_mode == "vq_vae"))
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_trained_codec_init_worker,
            initargs=(num_gpus, counter, codec_kwargs),
        ) as pool:
            futures = {pool.submit(_trained_clip_worker, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                result = fut.result()
                path = futures[fut]
                done += 1
                if "_skip" in result:
                    log.warning("skip %s: %s", result.get("_path"), result["_skip"])
                    continue
                for k, v in result["_metrics"].items():
                    accum[k].append(v)
                for stream, arrs in result["_codebook"].items():
                    codebook_indices[stream].extend(arrs)
                _record_clip(path, result["_metrics"])
                if done % 500 == 0:
                    log.info("processed %d / %d clips", done, len(paths))
    else:
        assert codec is not None
        for i, path in enumerate(paths):
            try:
                umr = UMR499.from_npz(path)
            except Exception as e:
                log.warning("skip %s: %s", path, e)
                continue
            if umr.num_records < ds.window:
                log.warning("skip %s (too short)", path)
                continue
            local_accum: dict[str, list[float]] = defaultdict(list)
            _evaluate_clip(
                umr,
                codec,
                gates=gates,
                accum=local_accum,
                codebook_indices=codebook_indices,
                device=args.device,
                source_path=path,
            )
            clip_metrics: dict[str, float] = {}
            for k, vs in local_accum.items():
                if not vs:
                    continue
                v = float(vs[-1])
                accum[k].append(v)
                clip_metrics[k] = v
            if clip_metrics:
                _record_clip(path, clip_metrics)
            if (i + 1) % 25 == 0:
                log.info("processed %d / %d clips", i + 1, len(paths))

    report = _aggregate(accum, codebook_indices, codec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    log.info("wrote %s", out_path)

    if args.save_per_clip:
        if args.save_per_clip_path:
            per_clip_path = resolve_local_uri(args.save_per_clip_path, data_root)
        else:
            per_clip_path = out_path.with_name(out_path.stem + ".per_clip.json")
        all_keys: set[str] = set()
        for d in per_clip_metrics:
            all_keys.update(d.keys())
        keys = sorted(all_keys)
        metric_arrays: dict[str, list[float]] = {k: [] for k in keys}
        for d in per_clip_metrics:
            for k in keys:
                metric_arrays[k].append(float(d.get(k, float("nan"))))
        payload = {
            "schema_version": 1,
            "manifest": [str(m) for m in (args.manifest if isinstance(args.manifest, list) else [args.manifest])],
            "gates": sorted(gates),
            "codec_mode": args.codec_mode,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "n_clips": len(per_clip_ids),
            "recording_ids": per_clip_ids,
            "metrics": metric_arrays,
        }
        per_clip_path.parent.mkdir(parents=True, exist_ok=True)
        per_clip_path.write_text(json.dumps(payload))
        log.info("wrote per-clip metrics to %s", per_clip_path)

    return 0

if __name__ == "__main__":
    sys.exit(main())
