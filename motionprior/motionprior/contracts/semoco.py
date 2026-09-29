"""Fail-closed contract and loader for the one accepted SeMoCo tokenizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SeMoCoContract:
    root: Path
    checkpoint: Path
    checkpoint_sha256: str
    input_dim: int = 499
    latent_dim: int = 512
    input_fps: float = 25.0
    temporal_stride: int = 2
    token_hz: float = 12.5
    num_codebooks: int = 16
    semantic_codebooks: int = 1
    kinematic_codebooks: int = 15
    semantic_vocab_size: int = 1024
    kinematic_vocab_size: int = 1024

    @property
    def vocab_sizes(self) -> tuple[int, ...]:
        return (self.semantic_vocab_size,) + (self.kinematic_vocab_size,) * self.kinematic_codebooks

    def snapshot(self) -> dict[str, Any]:
        result = asdict(self)
        result["root"] = str(self.root)
        result["checkpoint"] = str(self.checkpoint)
        result["vocab_sizes"] = list(self.vocab_sizes)
        return result


SEMOCO_ROOT = Path(__file__).resolve().parents[3] / "tokenizer"
SEMOCO_CONTRACT = SeMoCoContract(
    root=SEMOCO_ROOT,
    checkpoint=SEMOCO_ROOT / "checkpoints/best.pt",
    checkpoint_sha256="c381a0db2ef794f93808169bb420dc0695b67788609acd0b999a651ea0f6f1f7",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filtered_kwargs(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in raw.items() if key in allowed}


def _validate_payload(payload: dict[str, Any], contract: SeMoCoContract) -> None:
    if "codec_config" not in payload or "net" not in payload or "norm" not in payload:
        raise ValueError("SeMoCo checkpoint lacks codec_config/net/norm")
    raw = payload["codec_config"]
    if raw.get("groups"):
        raise ValueError("MotionPrior requires the single-stream split RVQ checkpoint")
    backbone = raw["backbone"]
    quantizer = raw["quantizer"]
    observed = {
        "input_dim": int(raw.get("input_dim", -1)),
        "latent_dim": int(backbone.get("latent_dim", -1)),
        "temporal_stride": int(backbone.get("temporal_stride", -1)),
        "quantizer_kind": str(quantizer.get("kind", "")),
        "kinematic_codebooks": int(quantizer.get("num_quantizers", -1)),
        "semantic_vocab_size": int(quantizer.get("semantic_codebook_size", -1)),
        "kinematic_vocab_size": int(quantizer.get("codebook_size", -1)),
    }
    expected = {
        "input_dim": contract.input_dim,
        "latent_dim": contract.latent_dim,
        "temporal_stride": contract.temporal_stride,
        "quantizer_kind": "split_rvq",
        "kinematic_codebooks": contract.kinematic_codebooks,
        "semantic_vocab_size": contract.semantic_vocab_size,
        "kinematic_vocab_size": contract.kinematic_vocab_size,
    }
    if observed != expected:
        raise ValueError(f"SeMoCo codec contract mismatch: observed={observed}, expected={expected}")
    mean = torch.as_tensor(payload["norm"].get("mean"))
    std = torch.as_tensor(payload["norm"].get("std"))
    if mean.numel() != contract.input_dim or std.numel() != contract.input_dim:
        raise ValueError("SeMoCo normalization shape mismatch")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("SeMoCo normalization contains invalid values")


def load_frozen_semoco(
    contract: SeMoCoContract = SEMOCO_CONTRACT,
    *,
    device: torch.device | str = "cpu",
    verify_sha: bool = True,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Load, validate and freeze the exact codec declared by ``contract``."""

    if not contract.checkpoint.is_file():
        raise FileNotFoundError(contract.checkpoint)
    if verify_sha:
        observed_sha = sha256_file(contract.checkpoint)
        if observed_sha != contract.checkpoint_sha256:
            raise ValueError(
                f"SeMoCo checkpoint SHA mismatch: {observed_sha} != {contract.checkpoint_sha256}"
            )
    root = str(contract.root.resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    from models.umr.structured_vq import (  # noqa: PLC0415
        BackboneConfig,
        CodecConfig,
        QuantizerConfig,
        build_structured_vq,
    )

    from checkpoint_io import load_checkpoint
    payload = load_checkpoint(contract.checkpoint)
    _validate_payload(payload, contract)
    raw = payload["codec_config"]
    config = CodecConfig(
        backbone=BackboneConfig(**_filtered_kwargs(BackboneConfig, raw["backbone"])),
        quantizer=QuantizerConfig(**_filtered_kwargs(QuantizerConfig, raw["quantizer"])),
        input_dim=int(raw["input_dim"]),
        groups=None,
    )
    codec = build_structured_vq(config)
    loaded = codec.load_state_dict(payload["net"], strict=False)
    unexpected = [key for key in loaded.unexpected_keys if not key.startswith("semantic_head.")]
    if loaded.missing_keys or unexpected:
        raise RuntimeError(
            f"SeMoCo state mismatch: missing={loaded.missing_keys}, unexpected={unexpected}"
        )
    codec.eval().to(device)
    codec.requires_grad_(False)
    mean = torch.as_tensor(payload["norm"]["mean"], dtype=torch.float32, device=device).view(1, -1, 1)
    std = torch.as_tensor(payload["norm"]["std"], dtype=torch.float32, device=device).view(1, -1, 1)
    return codec, mean, std
