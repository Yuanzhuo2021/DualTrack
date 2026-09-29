from __future__ import annotations

from torch import Tensor, nn

from .transformer import CausalTransformer


class TemporalGPT(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_size: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.transformer = CausalTransformer(
            self.d_model,
            int(num_layers),
            int(num_heads),
            int(ffn_size),
            float(dropout),
            rope=True,
        )

    def forward(self, held_frame_embeddings: Tensor, mask: Tensor) -> Tensor:
        return self.transformer(held_frame_embeddings, mask)
