"""EMA residual vector quantizer with dead-code reset.

``forward(z[B,T,C]) -> (z_q[B,T,C], indices[B,T,Q], losses)`` where ``losses``
carries ``commit_loss`` plus sync-free ``perplexity`` / per-layer usage metrics;
``get_layer_embeddings(indices) -> [B, T, K, C]`` for prefix-truncated decode.

The EMA codebook lives in registered buffers, so under DDP run with
``broadcast_buffers=True`` (rank-0 codebook broadcast each forward).
"""

from __future__ import annotations

import random
from random import randrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.quantizers._codebook_stats import _layer_counts, _perplexity_from_counts

def _log(t: Tensor, eps: float = 1e-20) -> Tensor:
    return torch.log(t.clamp(min=eps))

def _gumbel_noise(t: Tensor) -> Tensor:
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -_log(-_log(noise))

def _gumbel_sample(logits: Tensor, *, temperature: float, stochastic: bool, training: bool) -> Tensor:
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + _gumbel_noise(logits)
    else:
        sampling_logits = logits
    return sampling_logits.argmax(dim=-1)

class QuantizeEMAReset(nn.Module):
    """Single EMA codebook with dead-code reset.

    Operates on channels-first ``[N, C, T]`` to mirror the upstream module.
    """

    codebook: Tensor

    def __init__(self, nb_code: int, code_dim: int, *, mu: float = 0.99) -> None:
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.init = False
        self.code_sum: Tensor | None = None
        self.code_count: Tensor | None = None
        self.register_buffer("codebook", torch.zeros(nb_code, code_dim))

    def _tile(self, x: Tensor) -> Tensor:
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    @torch.no_grad()
    def init_codebook(self, x: Tensor) -> None:
        out = self._tile(x)
        self.codebook = out[: self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args, **kwargs) -> None:
        # EMA state (init / code_sum / code_count) is not serialised; rebuild it
        # from the loaded codebook so a resumed run keeps the trained codebook.
        ckpt_codebook = state_dict.get(prefix + "codebook")
        if ckpt_codebook is not None:
            self.init = True
            self.code_sum = ckpt_codebook.clone()
            self.code_count = torch.ones(self.nb_code, device=ckpt_codebook.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def quantize(self, x: Tensor, sample_codebook_temp: float = 0.0) -> Tensor:
        k_w = self.codebook.t()
        distance = (
            torch.sum(x ** 2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w ** 2, dim=0, keepdim=True)
        )
        return _gumbel_sample(
            -distance,
            temperature=sample_codebook_temp,
            stochastic=True,
            training=self.training,
        )

    def dequantize(self, code_idx: Tensor) -> Tensor:
        return F.embedding(code_idx, self.codebook)

    @torch.no_grad()
    def _perplexity(self, code_idx: Tensor) -> Tensor:
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        return torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

    @torch.no_grad()
    def update_codebook(self, x: Tensor, code_idx: Tensor) -> Tensor:
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[: self.nb_code]

        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1 - usage) * code_rand

        prob = code_count / torch.sum(code_count)
        return torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

    @staticmethod
    def _preprocess(x: Tensor) -> Tensor:
        # [N, C, T] -> [N*T, C]
        return x.permute(0, 2, 1).reshape(-1, x.shape[1])

    def forward(self, x: Tensor, *, temperature: float = 0.0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Codebook math runs in fp32 (the codebook buffer dtype) so the module
        # is safe and numerically stable under bf16 autocast.
        n, _width, t = x.shape
        in_dtype = x.dtype
        x_flat = self._preprocess(x).float()
        if self.training and not self.init:
            self.init_codebook(x_flat)

        code_idx = self.quantize(x_flat, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x_flat, code_idx)
        else:
            perplexity = self._perplexity(code_idx)

        commit_loss = F.mse_loss(x_flat, x_d.detach())
        # straight-through
        x_d = x_flat + (x_d - x_flat).detach()
        x_d = x_d.view(n, t, -1).permute(0, 2, 1).contiguous().to(in_dtype)
        code_idx = code_idx.view(n, t).contiguous()
        return x_d, code_idx, commit_loss.to(in_dtype), perplexity

class EMAResidualVQ(nn.Module):
    """Residual stack of :class:`QuantizeEMAReset` on the ``[B, T, C]`` contract."""

    def __init__(
        self,
        dim: int,
        num_quantizers: int,
        codebook_size: int,
        *,
        mu: float = 0.99,
        quantize_dropout: bool = True,
        quantize_dropout_cutoff_index: int = 0,
        quantize_dropout_prob: float = 0.2,
        sample_codebook_temp: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.quantize_dropout = bool(quantize_dropout) and num_quantizers > 1
        self.quantize_dropout_cutoff_index = int(quantize_dropout_cutoff_index)
        self.quantize_dropout_prob = float(quantize_dropout_prob)
        self.sample_codebook_temp = float(sample_codebook_temp)
        self.return_layer_codes = False
        self.layers = nn.ModuleList(
            [QuantizeEMAReset(codebook_size, dim, mu=mu) for _ in range(num_quantizers)]
        )

    def _metrics(self, indices: Tensor) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}
        with torch.no_grad():
            ppl_parts: list[Tensor] = []
            for r in range(self.num_quantizers):
                counts = _layer_counts(indices[..., r], self.codebook_size)
                used = (counts > 0).sum().to(torch.float32) / float(self.codebook_size)
                losses[f"codebook_usage_r{r}"] = used
                losses[f"dead_code_ratio_r{r}"] = 1.0 - used
                ppl_parts.append(_perplexity_from_counts(counts))
            losses["perplexity"] = torch.stack(ppl_parts).mean()
        return losses

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        x = z.transpose(1, 2).contiguous()
        device = x.device

        quantized_out = torch.zeros_like(x)
        residual = x
        all_indices: list[Tensor] = []
        commit_losses: list[Tensor] = []
        emit_layers = self.training or self.return_layer_codes
        layer_quantized: list[Tensor] = []

        should_dropout = self.training and self.quantize_dropout and (random.random() < self.quantize_dropout_prob)
        start_drop = self.num_quantizers
        null_indices = None
        if should_dropout:
            start_drop = randrange(self.quantize_dropout_cutoff_index, self.num_quantizers)
            null_indices = torch.full((x.shape[0], x.shape[-1]), -1, device=device, dtype=torch.long)

        for q_idx, layer in enumerate(self.layers):
            if should_dropout and q_idx > start_drop:
                all_indices.append(null_indices)
                if emit_layers:
                    layer_quantized.append(torch.zeros_like(x))
                continue
            quantized, embed_idx, commit_loss, _perplexity = layer(residual, temperature=self.sample_codebook_temp)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            if emit_layers:
                layer_quantized.append(quantized)
            all_indices.append(embed_idx)
            commit_losses.append(commit_loss)

        indices = torch.stack(all_indices, dim=-1)
        commit = torch.stack(commit_losses).mean() if commit_losses else x.new_zeros(())
        losses: dict[str, Tensor] = {"commit_loss": commit}
        losses.update(self._metrics(indices))

        if emit_layers and layer_quantized:
            layers_cf = torch.stack(layer_quantized, dim=1)
            losses["layer_quantized"] = layers_cf.permute(0, 3, 1, 2).contiguous()

        z_q = quantized_out.transpose(1, 2).contiguous()
        return z_q, indices, losses

    def get_layer_embeddings(self, indices: Tensor) -> Tensor:
        """Per-layer codebook embeddings for prefix-truncated decode.

        ``indices`` is ``[B, T, K]`` with ``K <= num_quantizers``; returns
        ``[B, T, K, C]`` (dropped ``-1`` entries map to a zero embedding).
        """
        n_used = int(indices.shape[-1])
        if n_used > self.num_quantizers:
            raise ValueError(f"indices last dim {n_used} exceeds num_quantizers {self.num_quantizers}")
        embeds = []
        for layer_idx in range(n_used):
            idx_r = indices[..., layer_idx]
            mask = idx_r < 0
            safe = idx_r.clamp(min=0)
            emb = self.layers[layer_idx].dequantize(safe)
            emb = emb.masked_fill(mask.unsqueeze(-1), 0.0)
            embeds.append(emb)
        return torch.stack(embeds, dim=2)

__all__ = ["EMAResidualVQ", "QuantizeEMAReset"]
