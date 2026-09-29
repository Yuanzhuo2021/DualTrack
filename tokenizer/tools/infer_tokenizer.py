"""Run a trained SeMoCo tokenizer on one UMR-499 clip.

The output NPZ contains the aligned ground-truth/reconstructed features and
every discrete token stream returned by the checkpoint.  A sibling JSON file
records the input/output shapes and timebase for quick auditing.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from data.umr_schema import UMR499
from tools.eval import _load_codec_adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Input umr499*.npz clip")
    parser.add_argument("--out", type=Path, required=True, help="Output inference NPZ")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    umr = UMR499.from_npz(args.input)
    codec = _load_codec_adapter(
        Namespace(
            codec_mode="vq_vae",
            checkpoint=args.checkpoint,
            device=args.device,
            data_root=None,
        )
    )
    features_rec, _ = codec.reconstruct(umr.features)
    token_streams = codec.token_indices(umr.features)
    features_gt = np.asarray(umr.features[: features_rec.shape[0]], dtype=np.float32)

    payload: dict[str, np.ndarray] = {
        "source_path": np.asarray(str(args.input)),
        "checkpoint": np.asarray(str(args.checkpoint)),
        "fps": np.asarray(float(umr.fps), dtype=np.float32),
        "temporal_stride": np.asarray(int(codec.temporal_stride), dtype=np.int32),
        "features_gt": features_gt,
        "features_rec": np.asarray(features_rec, dtype=np.float32),
    }
    payload.update({f"tokens_{name}": np.asarray(value) for name, value in token_streams.items()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    report = {
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "output": str(args.out),
        "fps": float(umr.fps),
        "temporal_stride": int(codec.temporal_stride),
        "input_shape": list(umr.features.shape),
        "reconstruction_shape": list(features_rec.shape),
        "token_shapes": {name: list(value.shape) for name, value in token_streams.items()},
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
