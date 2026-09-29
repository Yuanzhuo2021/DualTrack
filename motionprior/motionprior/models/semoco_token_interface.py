"""Frozen, BOS-safe adapter around the exact SeMoCo tokenizer."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from motionprior.contracts.semoco import SEMOCO_CONTRACT, SeMoCoContract, load_frozen_semoco


class FrozenSeMoCoTokenInterface(nn.Module):
    num_codebooks: Final[int] = 16
    vocab_size: Final[int] = 1024
    bos_id: Final[int] = 1024

    def __init__(
        self,
        codec: nn.Module,
        mean: Tensor,
        std: Tensor,
        *,
        input_dim: int = 499,
        latent_dim: int = 512,
        temporal_stride: int = 2,
    ) -> None:
        super().__init__()
        self.codec = codec.eval()
        self.codec.requires_grad_(False)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.temporal_stride = int(temporal_stride)
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1)
        if mean.shape != (1, self.input_dim, 1) or std.shape != mean.shape:
            raise ValueError("normalization must have shape [1,input_dim,1]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("normalization must be finite with strictly positive std")
        self.register_buffer("norm_mean", mean, persistent=True)
        self.register_buffer("norm_std", std, persistent=True)

    @classmethod
    def from_contract(
        cls,
        contract: SeMoCoContract = SEMOCO_CONTRACT,
        *,
        device: torch.device | str = "cpu",
        verify_sha: bool = True,
    ) -> "FrozenSeMoCoTokenInterface":
        codec, mean, std = load_frozen_semoco(contract, device=device, verify_sha=verify_sha)
        return cls(
            codec,
            mean,
            std,
            input_dim=contract.input_dim,
            latent_dim=contract.latent_dim,
            temporal_stride=contract.temporal_stride,
        )

    def train(self, mode: bool = True) -> "FrozenSeMoCoTokenInterface":
        super().train(False)
        self.codec.eval()
        return self

    def _validate_tokens(self, tokens: Tensor, *, allow_prefix: bool) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B,T,K]")
        expected = range(1, self.num_codebooks + 1) if allow_prefix else (self.num_codebooks,)
        if tokens.shape[-1] not in expected:
            raise ValueError(f"token depth must be {'1..16' if allow_prefix else '16'}")
        values = tokens.long()
        if torch.any((values < 0) | (values >= self.vocab_size)):
            raise ValueError("codec tokens must be in [0,1023]; BOS/padding cannot enter SeMoCo")
        return values

    @torch.no_grad()
    def encode(self, umr499_25hz: Tensor) -> Tensor:
        if umr499_25hz.ndim != 3 or umr499_25hz.shape[1] != self.input_dim:
            raise ValueError(f"UMR input must be [B,{self.input_dim},T25]")
        keep = (umr499_25hz.shape[-1] // self.temporal_stride) * self.temporal_stride
        if keep < self.temporal_stride:
            raise ValueError("UMR sequence is shorter than one codec stride")
        features = umr499_25hz[..., :keep].to(device=self.norm_mean.device, dtype=torch.float32)
        indices = self.codec.encode_indices((features - self.norm_mean) / self.norm_std)
        if not isinstance(indices, Tensor):
            raise TypeError("expected single-stream tensor indices from SeMoCo")
        indices = self._validate_tokens(indices, allow_prefix=False)
        return indices

    def layer_embeddings(self, tokens: Tensor) -> Tensor:
        values = self._validate_tokens(tokens, allow_prefix=True)
        # The codec is frozen, but autograd through the returned values is
        # intentionally retained so MotionPrior inputs behave like constants.
        embeddings = self.codec.quantizer.get_layer_embeddings(values)
        expected = (*values.shape, self.latent_dim)
        if embeddings.shape != expected:
            raise RuntimeError(f"SeMoCo layer embedding shape {embeddings.shape} != {expected}")
        return embeddings

    def frame_embedding(self, tokens: Tensor) -> Tensor:
        values = self._validate_tokens(tokens, allow_prefix=False)
        return self.layer_embeddings(values).sum(dim=2)

    @torch.no_grad()
    def decode(self, tokens: Tensor) -> Tensor:
        values = self._validate_tokens(tokens, allow_prefix=True)
        decoded = self.codec.decode_indices(values)
        if decoded.ndim != 3 or decoded.shape[1] != self.input_dim:
            raise RuntimeError("SeMoCo decode returned an invalid shape")
        if not torch.isfinite(decoded).all():
            raise RuntimeError("SeMoCo decode produced non-finite values")
        return decoded
