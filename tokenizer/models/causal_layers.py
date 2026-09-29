"""Causal 1D convolutional and residual building blocks (channels-first [B, C, T])."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class CausalConv1d(nn.Module):
    """Conv1d with optional causal (left-only) padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self._pad = (kernel_size - stride) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.causal:
            x = F.pad(x, (self._pad, 0))
        else:
            half = self._pad // 2
            extra = self._pad - half
            x = F.pad(x, (half, extra))
        return self.conv(x)

class _ChannelNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

class DownsampleBlock1d(nn.Module):
    def __init__(self, channels: int, stride: int, causal: bool = False) -> None:
        super().__init__()
        self.conv = CausalConv1d(
            channels, channels, kernel_size=2 * stride, stride=stride, causal=causal
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class UpsampleBlock1d(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            channels, channels, kernel_size=2 * stride, stride=stride
        )
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return out[..., : x.shape[-1] * self.stride]
