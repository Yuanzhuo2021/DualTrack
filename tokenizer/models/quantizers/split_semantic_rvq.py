"""Split semantic / kinematic quantizer.

A semantic VQ runs in parallel with an ``n``-layer kinematic residual RVQ; both
quantize the full latent ``z`` (NOT ``z - z_sem``), each with its own learnable
in/out projection, and the two outputs are summed::

    z_q = out_proj_sem(VQ(in_proj_sem(z))) + out_proj_kin(RVQ_n(in_proj_kin(z)))

``indices`` is ``[B, T, 1 + n]`` with depth 0 = semantic; the semantic branch is
never dropped by ``quantize_dropout`` and is routed out as ``losses["z_sem"]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.quantizers._codebook_stats import _layer_counts, _perplexity_from_counts
from models.quantizers.ema_rvq import EMAResidualVQ, QuantizeEMAReset

class SplitSemanticRVQ(nn.Module):
    """Semantic VQ ∥ kinematic residual RVQ, outputs summed.

    Contract mirrors :class:`EMAResidualVQ`: ``forward(z[B,T,C]) -> (z_q[B,T,C],
    indices[B,T,1+n], losses)`` and ``get_layer_embeddings(indices) -> [B,T,K,C]``
    for prefix-truncated / summed decode.
    """

    def __init__(
        self,
        dim: int,
        num_kinematic_quantizers: int,
        codebook_size: int,
        *,
        semantic_codebook_size: int | None = None,
        mu: float = 0.99,
        quantize_dropout: bool = True,
        quantize_dropout_cutoff_index: int = 0,
        quantize_dropout_prob: float = 0.2,
        sample_codebook_temp: float = 0.5,
    ) -> None:
        super().__init__()
        if num_kinematic_quantizers < 1:
            raise ValueError(f"num_kinematic_quantizers must be >= 1; got {num_kinematic_quantizers}")
        self.codebook_size = codebook_size
        self.semantic_codebook_size = semantic_codebook_size or codebook_size
        self.num_kinematic_quantizers = int(num_kinematic_quantizers)
        # Total codes/token exposed to the AR = 1 semantic (depth0) + n kinematic.
        self.num_quantizers = 1 + self.num_kinematic_quantizers
        self.sample_codebook_temp = float(sample_codebook_temp)
        # Opt-in mirror of EMAResidualVQ: emit the dedicated semantic branch even
        # in eval (off by default; training mode emits it implicitly).
        self.return_layer_codes = False

        # Bias-free projections so a zero/dropped code still maps to a zero
        # contribution (needed by get_layer_embeddings).
        self.in_proj_sem = nn.Linear(dim, dim, bias=False)
        self.out_proj_sem = nn.Linear(dim, dim, bias=False)
        self.in_proj_kin = nn.Linear(dim, dim, bias=False)
        self.out_proj_kin = nn.Linear(dim, dim, bias=False)

        # Semantic branch: a single EMA VQ over the FULL latent (never dropped).
        self.semantic_vq = QuantizeEMAReset(self.semantic_codebook_size, dim, mu=mu)
        # Kinematic branch: standard n-layer residual RVQ over the FULL latent.
        self.kinematic_rvq = EMAResidualVQ(
            dim=dim,
            num_quantizers=self.num_kinematic_quantizers,
            codebook_size=codebook_size,
            mu=mu,
            quantize_dropout=quantize_dropout,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            quantize_dropout_prob=quantize_dropout_prob,
            sample_codebook_temp=sample_codebook_temp,
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # --- semantic branch: out_proj(VQ(in_proj(z))) on the FULL latent ---
        z_in_sem = self.in_proj_sem(z)                             # [B, T, C]
        x_cf = z_in_sem.transpose(1, 2).contiguous()              # [B, C, T]
        z_sem_cf, idx_sem, commit_sem, ppl_sem = self.semantic_vq(
            x_cf, temperature=self.sample_codebook_temp
        )
        z_sem = self.out_proj_sem(z_sem_cf.transpose(1, 2).contiguous())   # [B, T, C]

        # --- kinematic branch: out_proj(RVQ(in_proj(z))) on the FULL latent ---
        # (NOT z - z_sem; that is the whole point of the parallel split.)
        z_in_kin = self.in_proj_kin(z)                            # [B, T, C]
        z_kin_q, idx_kin, losses_kin = self.kinematic_rvq(z_in_kin)
        z_kin = self.out_proj_kin(z_kin_q)                        # [B, T, C]

        z_q = z_sem + z_kin
        indices = torch.cat([idx_sem.unsqueeze(-1), idx_kin], dim=-1)   # [B, T, 1+n] depth0=sem

        commit = 0.5 * (commit_sem + losses_kin["commit_loss"])
        losses: dict[str, Tensor] = {"commit_loss": commit}

        # Metrics: semantic usage/ppl + prefixed kinematic per-layer stats + mean perplexity.
        with torch.no_grad():
            sem_counts = _layer_counts(idx_sem, self.semantic_codebook_size)
            sem_used = (sem_counts > 0).sum().to(torch.float32) / float(self.semantic_codebook_size)
            losses["sem_codebook_usage"] = sem_used
            losses["sem_dead_code_ratio"] = 1.0 - sem_used
            losses["sem_perplexity"] = _perplexity_from_counts(sem_counts)
            kin_ppl = losses_kin.get("perplexity")
            if kin_ppl is not None:
                losses["perplexity"] = torch.stack([losses["sem_perplexity"], kin_ppl]).mean()
            else:
                losses["perplexity"] = losses["sem_perplexity"]
        for k, v in losses_kin.items():
            if k in ("commit_loss", "perplexity", "layer_quantized"):
                continue
            losses[f"kin_{k}"] = v

        # Route the dedicated semantic branch out for the TMR SemanticHead.
        if self.training or self.return_layer_codes:
            losses["z_sem"] = z_sem

        return z_q, indices, losses

    def get_layer_embeddings(self, indices: Tensor) -> Tensor:
        """Per-depth embeddings for summed / prefix-truncated decode.

        ``indices`` is ``[B, T, K]`` with ``K <= 1 + n``; depth 0 maps to the
        semantic codebook, depth 1..K-1 to the kinematic RVQ layers. Returns
        ``[B, T, K, C]`` so the caller's ``.sum(dim=2)`` reproduces
        ``z_sem + sum(kinematic_prefix)``.
        """
        n_used = int(indices.shape[-1])
        if n_used > self.num_quantizers:
            raise ValueError(f"indices last dim {n_used} exceeds num_quantizers {self.num_quantizers}")
        embeds: list[Tensor] = []
        # depth 0: semantic. The bias-free proj keeps masked (idx<0) entries at
        # zero, so masking after the projection is exact.
        idx_sem = indices[..., 0]
        mask = idx_sem < 0
        sem_emb = self.out_proj_sem(self.semantic_vq.dequantize(idx_sem.clamp(min=0)))
        sem_emb = sem_emb.masked_fill(mask.unsqueeze(-1), 0.0)
        embeds.append(sem_emb)
        # depth 1..: kinematic. out_proj_kin is linear, so per-layer projection
        # then sum == out_proj_kin(sum(prefix)) (matches the forward sum);
        # dropped layers are zero in -> zero out (bias-free).
        if n_used > 1:
            kin_embeds = self.kinematic_rvq.get_layer_embeddings(indices[..., 1:])  # [B,T,K-1,C]
            kin_embeds = self.out_proj_kin(kin_embeds)
            for j in range(kin_embeds.shape[2]):
                embeds.append(kin_embeds[:, :, j, :])
        return torch.stack(embeds, dim=2)

__all__ = ["SplitSemanticRVQ"]
