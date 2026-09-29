"""Small pre-norm causal Transformer blocks with RMSNorm, SwiGLU and RoPE."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _apply_rope(x: Tensor, positions: Tensor, base: float = 10_000.0) -> Tensor:
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions.to(torch.float32), inv_freq)
    cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, *, rope: bool) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        if rope and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.rope = bool(rope)
        self.dropout = float(dropout)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.output = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor, positions: Tensor) -> Tensor:
        batch, steps, _ = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.rope:
            query = _apply_rope(query, positions)
            key = _apply_rope(key, positions)
        causal = torch.ones(steps, steps, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & mask[:, None, None, :] & mask[:, None, :, None]
        hidden = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        hidden = hidden.transpose(1, 2).contiguous().view(batch, steps, self.d_model)
        return self.output(hidden).masked_fill(~mask.unsqueeze(-1), 0.0)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_size, bias=False)
        self.value = nn.Linear(d_model, ffn_size, bias=False)
        self.output = nn.Linear(ffn_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(self.dropout(F.silu(self.gate(x)) * self.value(x)))


class CausalBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_size: int, dropout: float, *, rope: bool) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, dropout, rope=rope)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor, positions: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x), mask, positions))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        *,
        rope: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            CausalBlock(d_model, num_heads, ffn_size, dropout, rope=rope)
            for _ in range(num_layers)
        )
        self.norm = nn.RMSNorm(d_model)
        self.last_sequence_length = 0

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if x.ndim != 3 or mask.shape != x.shape[:2]:
            raise ValueError("transformer expects x[B,T,D] and mask[B,T]")
        if not mask.dtype == torch.bool:
            raise TypeError("attention mask must be boolean")
        self.last_sequence_length = int(x.shape[1])
        positions = torch.arange(x.shape[1], device=x.device)
        for block in self.blocks:
            x = block(x, mask, positions)
        return self.norm(x).masked_fill(~mask.unsqueeze(-1), 0.0)
