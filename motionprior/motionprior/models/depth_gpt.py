"""Causal q1-q15 predictor within one frozen SeMoCo frame."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .semoco_token_interface import FrozenSeMoCoTokenInterface
from .transformer import CausalTransformer


class DepthGPT(nn.Module):
    def __init__(
        self,
        token_interface: FrozenSeMoCoTokenInterface,
        *,
        d_model: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_size: int = 1536,
        dropout: float = 0.1,
        vocab_size: int = 1024,
    ) -> None:
        super().__init__()
        self.token_interface = token_interface
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.token_input_projection = (
            nn.Identity()
            if token_interface.latent_dim == self.d_model
            else nn.Linear(token_interface.latent_dim, self.d_model, bias=False)
        )
        self.depth_position = nn.Embedding(16, self.d_model)
        self.context_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.transformer = CausalTransformer(
            self.d_model,
            int(num_layers),
            int(num_heads),
            int(ffn_size),
            float(dropout),
            rope=False,
        )
        self.heads = nn.ModuleList(
            nn.Linear(self.d_model, self.vocab_size, bias=False) for _ in range(15)
        )

    def _inputs(self, context: Tensor, prefixes: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[-1] != self.d_model:
            raise ValueError("depth context must be [N,D]")
        if prefixes.ndim != 2 or prefixes.shape[0] != context.shape[0]:
            raise ValueError("depth prefixes must be [N,K]")
        if not 1 <= prefixes.shape[1] <= 15:
            raise ValueError("depth prefix must contain q0 through at most q14")
        embedded = self.token_input_projection(
            self.token_interface.layer_embeddings(prefixes[:, None, :]).squeeze(1)
        )
        inputs = torch.cat((self.context_projection(context)[:, None, :], embedded), dim=1)
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        return inputs + self.depth_position(positions)[None]

    def forward_parallel(self, context: Tensor, targets: Tensor) -> tuple[Tensor, ...]:
        if context.ndim != 3 or targets.shape != (*context.shape[:2], 16):
            raise ValueError("parallel depth expects context[B,T,D], targets[B,T,16]")
        batch, steps, _ = context.shape
        flat_context = context.reshape(batch * steps, self.d_model)
        flat_prefix = targets[..., :15].reshape(batch * steps, 15)
        inputs = self._inputs(flat_context, flat_prefix)
        mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        hidden = self.transformer(inputs, mask)
        return tuple(
            head(hidden[:, depth + 1]).reshape(batch, steps, self.vocab_size)
            for depth, head in enumerate(self.heads)
        )

    def forward_mixed_prefix(
        self,
        context: Tensor,
        targets: Tensor,
        predicted_rows: Tensor,
        predicted_q0: Tensor,
    ) -> tuple[Tensor, ...]:
        """Use greedy predicted q0..q14 prefixes only on selected frame rows."""
        if predicted_rows.shape != context.shape[:2] or predicted_rows.dtype != torch.bool:
            raise ValueError("predicted depth rows must be bool[B,T]")
        if predicted_q0.shape != context.shape[:2]:
            raise ValueError("predicted q0 must be [B,T]")
        parallel = list(self.forward_parallel(context, targets))
        if not predicted_rows.any():
            return tuple(parallel)
        selected_context = context[predicted_rows]
        prefix = predicted_q0[predicted_rows].unsqueeze(-1)
        for depth in range(1, 16):
            selected_logits = self.next_logits(selected_context, prefix, depth)
            parallel[depth - 1] = parallel[depth - 1].clone()
            parallel[depth - 1][predicted_rows] = selected_logits
            prefix = torch.cat((prefix, selected_logits.argmax(dim=-1, keepdim=True)), dim=-1)
        return tuple(parallel)

    def next_logits(self, context: Tensor, prefix: Tensor, depth: int) -> Tensor:
        if not 1 <= depth <= 15:
            raise ValueError("depth must be in [1,15]")
        if prefix.shape[-1] != depth:
            raise ValueError(f"q{depth} requires prefix q0..q{depth - 1}")
        inputs = self._inputs(context, prefix)
        mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        hidden = self.transformer(inputs, mask)
        return self.heads[depth - 1](hidden[:, -1])
