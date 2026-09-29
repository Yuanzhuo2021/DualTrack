"""Hierarchical motion prior: full-frame temporal GPT plus q1-q15 Depth GPT."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .depth_gpt import DepthGPT
from .frame_embedding import AxisSeparatedFrameEmbedding, FullFrameEmbedding
from .q0_factorization import factorized_q0_log_probs, factorized_q0_sample
from .semoco_token_interface import FrozenSeMoCoTokenInterface
from .temporal_gpt import TemporalGPT


def sample_logits(logits: Tensor, *, temperature: float, top_k: int | None) -> Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    scores = logits.float() / float(temperature)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k < scores.shape[-1]:
            threshold = scores.topk(top_k, dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1).reshape(
        probabilities.shape[:-1]
    )


@dataclass(frozen=True)
class MotionPriorConfig:
    d_model: int = 512
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_ffn: int = 1536
    depth_layers: int = 3
    depth_heads: int = 8
    depth_ffn: int = 1536
    dropout: float = 0.1
    context_frames: int = 150
    vocab_size: int = 1024
    num_codebooks: int = 16
    detail_dim: int = 64
    history_architecture: str = "legacy_sum_detail"
    semantic_temporal_layers: int = 2
    q0_prediction_mode: str = "flat_softmax"
    parallel_depths: int = 0

    def validate(self) -> None:
        if self.num_codebooks != 16 or self.vocab_size != 1024:
            raise ValueError("the immutable SeMoCo contract is exactly 16 x 1024")
        if self.d_model <= 0 or self.d_model % self.temporal_heads or self.d_model % self.depth_heads:
            raise ValueError("d_model must be positive and divisible by both head counts")
        if min(self.temporal_layers, self.depth_layers, self.temporal_ffn, self.depth_ffn) <= 0:
            raise ValueError("layer counts and FFN sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.context_frames <= 0 or self.detail_dim <= 0:
            raise ValueError("context_frames and detail_dim must be positive")
        if self.history_architecture not in {"legacy_sum_detail", "axis_separated_q0_v2"}:
            raise ValueError("unsupported history_architecture")
        if self.semantic_temporal_layers <= 0:
            raise ValueError("semantic_temporal_layers must be positive")
        if self.q0_prediction_mode not in {"flat_softmax", "stay_change_factorized_v1"}:
            raise ValueError("unsupported q0_prediction_mode")
        if not 0 <= self.parallel_depths <= 15:
            raise ValueError("parallel_depths must be in [0,15]")


@dataclass
class MotionPriorOutput:
    logits: tuple[Tensor, ...]
    temporal_context: Tensor
    main_history_norm: Tensor
    detail_residual_norm: Tensor
    detail_gate: Tensor
    q0_temporal_context: Tensor

    q0_transition_logits: Tensor | None
    q0_nonstay_logits: Tensor | None

class MotionPrior(nn.Module):
    def __init__(
        self,
        token_interface: FrozenSeMoCoTokenInterface,
        config: MotionPriorConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MotionPriorConfig()
        self.config.validate()
        self.token_interface = token_interface
        if self.config.history_architecture == "axis_separated_q0_v2":
            self.frame_embedding = AxisSeparatedFrameEmbedding(
                token_interface,
                d_model=self.config.d_model,
                detail_dim=self.config.detail_dim,
            )
        else:
            self.frame_embedding = FullFrameEmbedding(
                token_interface,
                d_model=self.config.d_model,
                detail_dim=self.config.detail_dim,
            )
        self.temporal = TemporalGPT(
            d_model=self.config.d_model,
            num_layers=self.config.temporal_layers,
            num_heads=self.config.temporal_heads,
            ffn_size=self.config.temporal_ffn,
            dropout=self.config.dropout,
        )
        if self.config.q0_prediction_mode == "stay_change_factorized_v1":
            self.q0_head = None
            self.q0_transition_head = nn.Linear(self.config.d_model, 2, bias=True)
            self.q0_nonstay_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        else:
            self.q0_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            self.q0_transition_head = None
            self.q0_nonstay_head = None
        if self.config.history_architecture == "axis_separated_q0_v2":
            self.semantic_temporal = TemporalGPT(
                d_model=self.config.d_model,
                num_layers=self.config.semantic_temporal_layers,
                num_heads=self.config.temporal_heads,
                ffn_size=self.config.temporal_ffn,
                dropout=self.config.dropout,
            )
            self.semantic_to_depth = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
            self.depth_context_norm = nn.RMSNorm(self.config.d_model)
        else:
            self.semantic_temporal = None
            self.semantic_to_depth = None
            self.depth_context_norm = None
        self.depth = DepthGPT(
            token_interface,
            d_model=self.config.d_model,
            num_layers=self.config.depth_layers,
            num_heads=self.config.depth_heads,
            ffn_size=self.config.depth_ffn,
            dropout=self.config.dropout,
            vocab_size=self.config.vocab_size,
        )
        self.parallel_depth_heads = nn.ModuleList(
            nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            for _ in range(self.config.parallel_depths)
        )
        self.max_generation_context_seen = 0

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def initialize_parallel_depth_heads_from_autoregressive(self) -> None:
        """Bootstrap independent q1..qK classifiers from trained depth heads."""
        with torch.no_grad():
            for parallel, autoregressive in zip(
                self.parallel_depth_heads,
                self.depth.heads,
                strict=False,
            ):
                parallel.weight.copy_(autoregressive.weight)

    def parallel_depth_logits(self, context: Tensor) -> tuple[Tensor, ...]:
        if context.ndim not in {2, 3} or context.shape[-1] != self.config.d_model:
            raise ValueError("parallel depth context must be [N,D] or [B,T,D]")
        return tuple(head(context) for head in self.parallel_depth_heads)

    def _q0_outputs(
        self,
        q0_context: Tensor,
        previous_q0: Tensor,
        *,
        transition_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if not q0_context.shape[:-1] == previous_q0.shape:
            raise ValueError("q0 context/previous shape mismatch")
        if transition_context is None:
            transition_context = q0_context
        elif transition_context.shape != q0_context.shape:
            raise ValueError("q0 destination/transition context shape mismatch")
        if self.config.q0_prediction_mode == "flat_softmax":
            if self.q0_head is None:
                raise RuntimeError("flat q0 head is missing")
            return self.q0_head(q0_context), None, None
        if self.q0_transition_head is None or self.q0_nonstay_head is None:
            raise RuntimeError("factorized q0 heads are missing")
        transition_logits = self.q0_transition_head(transition_context)
        nonstay_logits = self.q0_nonstay_head(q0_context)
        effective = factorized_q0_log_probs(
            transition_logits,
            nonstay_logits,
            previous_q0,
            vocab_size=self.config.vocab_size,
        )
        return effective, transition_logits, nonstay_logits

    def q0_logits(self, q0_context: Tensor, previous_q0: Tensor) -> Tensor:
        return self._q0_outputs(q0_context, previous_q0)[0]

    def sample_q0(
        self,
        q0_context: Tensor,
        previous_q0: Tensor,
        *,
        temperature: float,
        top_k: int | None,
        transition_context: Tensor | None = None,
    ) -> Tensor:
        effective, transition_logits, nonstay_logits = self._q0_outputs(
            q0_context,
            previous_q0,
            transition_context=transition_context,
        )
        if transition_logits is None or nonstay_logits is None:
            return sample_logits(effective, temperature=temperature, top_k=top_k)
        # transition_temperature is intentionally left at the legacy default
        # (== temperature) so previously published MotionPrior results stay
        # reproducible; FusedMotionPrior exposes it as a config knob.
        return factorized_q0_sample(
            transition_logits,
            nonstay_logits,
            previous_q0,
            vocab_size=self.config.vocab_size,
            sampler=sample_logits,
            temperature=temperature,
            top_k=top_k,
        )

    def _validate_forward_inputs(
        self,
        held_tokens: Tensor,
        mask: Tensor,
        targets: Tensor | None,
        condition: Tensor | None,
    ) -> None:
        if held_tokens.ndim != 3 or held_tokens.shape[-1] != 16:
            raise ValueError("held_tokens must be [B,T,16]")
        if mask.shape != held_tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool[B,T]")
        if targets is not None and targets.shape != held_tokens.shape:
            raise ValueError("targets must match held_tokens [B,T,16]")
        if condition is not None and condition.shape != (*held_tokens.shape[:2], self.config.d_model):
            raise ValueError("condition must be [B,T,d_model]")

    def forward(
        self,
        held_tokens: Tensor,
        mask: Tensor,
        *,
        targets: Tensor | None = None,
        condition: Tensor | None = None,
        semantic_condition: Tensor | None = None,
        kinematic_condition: Tensor | None = None,
        transition_condition: Tensor | None = None,
        predicted_depth_rows: Tensor | None = None,
    ) -> MotionPriorOutput:
        self._validate_forward_inputs(held_tokens, mask, targets, condition)
        structured = (
            semantic_condition,
            kinematic_condition,
            transition_condition,
        )
        supplied = tuple(value is not None for value in structured)
        if any(supplied) and not all(supplied):
            raise ValueError("all structured speech conditions must be supplied together")
        if condition is not None and any(supplied):
            raise ValueError("condition and structured speech conditions are mutually exclusive")
        if all(supplied):
            if self.semantic_temporal is None:
                raise ValueError("structured speech conditions require axis-separated history")
            expected = (*held_tokens.shape[:2], self.config.d_model)
            if any(value.shape != expected for value in structured if value is not None):
                raise ValueError("structured speech conditions must be [B,T,d_model]")
        if predicted_depth_rows is not None and (
            predicted_depth_rows.shape != mask.shape
            or predicted_depth_rows.dtype != torch.bool
        ):
            raise ValueError("predicted_depth_rows must be bool[B,T]")
        embedded = self.frame_embedding(held_tokens)
        transition_context: Tensor | None = None
        if all(supplied):
            assert semantic_condition is not None and kinematic_condition is not None and transition_condition is not None
            assert self.semantic_temporal is not None and self.semantic_to_depth is not None and self.depth_context_norm is not None
            temporal_context = self.temporal(embedded.hidden, mask)
            q0_context = self.semantic_temporal(embedded.semantic, mask) + semantic_condition
            transition_context = q0_context + transition_condition
            depth_context = self.depth_context_norm(
                temporal_context + self.semantic_to_depth(q0_context) + kinematic_condition
            )
        else:
            # Motion-only conditioning is an exact zero tensor, not a learned null.
            if condition is None:
                condition = torch.zeros_like(embedded.hidden)
            temporal_context = self.temporal(embedded.hidden + condition, mask)
            if self.semantic_temporal is None:
                q0_context = temporal_context
                depth_context = temporal_context
            else:
                q0_context = self.semantic_temporal(embedded.semantic + condition, mask)
                depth_context = self.depth_context_norm(
                    temporal_context + self.semantic_to_depth(q0_context)
                )
        q0_logits, q0_transition_logits, q0_nonstay_logits = self._q0_outputs(
            q0_context,
            held_tokens[..., 0],
            transition_context=transition_context,
        )
        logits: tuple[Tensor, ...] = (q0_logits,)
        if targets is not None:
            if torch.any((targets < 0) | (targets >= self.config.vocab_size)):
                raise ValueError("targets must be frozen SeMoCo ids in [0,1023]")
            if predicted_depth_rows is None:
                depth_logits = self.depth.forward_parallel(depth_context, targets)
            else:
                with torch.no_grad():
                    predicted_q0 = self.sample_q0(
                        q0_context,
                        held_tokens[..., 0],
                        temperature=0.0,
                        top_k=None,
                        transition_context=transition_context,
                    )
                depth_logits = self.depth.forward_mixed_prefix(
                    depth_context,
                    targets,
                    predicted_depth_rows & mask,
                    predicted_q0,
                )
            if self.parallel_depth_heads:
                independent = self.parallel_depth_logits(depth_context)
                depth_logits = independent + depth_logits[len(independent) :]
            logits += depth_logits
        valid = mask.unsqueeze(-1)
        denom = valid.sum().clamp_min(1)
        main_norm = (embedded.main.norm(dim=-1, keepdim=True) * valid).sum() / denom
        residual_norm = (embedded.residual.norm(dim=-1, keepdim=True) * valid).sum() / denom
        return MotionPriorOutput(
            logits=logits,
            temporal_context=depth_context,
            main_history_norm=main_norm,
            detail_residual_norm=residual_norm,
            detail_gate=(
                self.frame_embedding.detail_gate
                if hasattr(self.frame_embedding, "detail_gate")
                else torch.ones((), device=held_tokens.device)
            ),
            q0_temporal_context=q0_context,
            q0_transition_logits=q0_transition_logits,
            q0_nonstay_logits=q0_nonstay_logits,
        )

    @torch.no_grad()
    def generate(
        self,
        initial_tokens: Tensor,
        num_frames: int,
        *,
        condition: Tensor | None = None,
        semantic_condition: Tensor | None = None,
        kinematic_condition: Tensor | None = None,
        transition_condition: Tensor | None = None,
        context_frames: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        detail_temperature: float | None = None,
        detail_sampling_start_depth: int = 5,
    ) -> Tensor:
        if initial_tokens.ndim not in {2, 3} or initial_tokens.shape[-1] != 16:
            raise ValueError("initial_tokens must be [B,16] or [B,T_prefix,16]")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not 1 <= detail_sampling_start_depth <= 15:
            raise ValueError("detail_sampling_start_depth must be in [1,15]")
        if detail_temperature is not None and detail_temperature < 0.0:
            raise ValueError("detail_temperature must be non-negative")
        resolved_detail_temperature = (
            temperature if detail_temperature is None else float(detail_temperature)
        )
        context_limit = self.config.context_frames if context_frames is None else int(context_frames)
        if context_limit <= 0 or context_limit > self.config.context_frames:
            raise ValueError(f"context_frames must be in [1,{self.config.context_frames}]")
        batch = initial_tokens.shape[0]
        structured = (
            semantic_condition,
            kinematic_condition,
            transition_condition,
        )
        supplied = tuple(value is not None for value in structured)
        if any(supplied) and not all(supplied):
            raise ValueError("all structured generation conditions must be supplied together")
        if condition is not None and any(supplied):
            raise ValueError("condition and structured generation conditions are mutually exclusive")
        if all(supplied) and self.semantic_temporal is None:
            raise ValueError("structured generation requires axis-separated history")
        expected = (batch, num_frames, self.config.d_model)
        if all(supplied) and any(value.shape != expected for value in structured if value is not None):
            raise ValueError("structured generation conditions must be [B,num_frames,d_model]")
        if condition is not None and condition.shape != (batch, num_frames, self.config.d_model):
            raise ValueError("generation condition must be [B,num_frames,d_model]")
        history = initial_tokens.long()
        if history.ndim == 2:
            history = history[:, None, :]
        if (condition is not None or any(supplied)) and history.shape[1] != 1:
            pass
        generated: list[Tensor] = []
        self.max_generation_context_seen = 0
        for step in range(num_frames):
            held = history[:, -context_limit:]
            self.max_generation_context_seen = max(self.max_generation_context_seen, held.shape[1])
            mask = torch.ones(held.shape[:2], dtype=torch.bool, device=held.device)
            embedded = self.frame_embedding(held)
            transition_context: Tensor | None = None
            if all(supplied):
                assert semantic_condition is not None and kinematic_condition is not None and transition_condition is not None
                assert self.semantic_temporal is not None and self.semantic_to_depth is not None and self.depth_context_norm is not None
                full_context = self.temporal(embedded.hidden, mask)[:, -1]
                semantic_history = self.semantic_temporal(embedded.semantic, mask)[:, -1]
                q0_context = semantic_history + semantic_condition[:, step]
                transition_context = q0_context + transition_condition[:, step]
                context = self.depth_context_norm(
                    full_context + self.semantic_to_depth(q0_context) + kinematic_condition[:, step]
                )
            else:
                if condition is None:
                    aligned_condition = torch.zeros_like(embedded.hidden)
                else:
                    prefix_padding = history.shape[1] - (step + 1)
                    condition_history = torch.cat(
                        (
                            condition.new_zeros(batch, prefix_padding, condition.shape[-1]),
                            condition[:, : step + 1],
                        ),
                        dim=1,
                    )
                    aligned_condition = condition_history[:, -held.shape[1] :]
                    if aligned_condition.shape[1] != held.shape[1]:
                        raise RuntimeError("condition/history alignment failure")
                full_context = self.temporal(embedded.hidden + aligned_condition, mask)[:, -1]
                if self.semantic_temporal is None:
                    q0_context = full_context
                    context = full_context
                else:
                    q0_context = self.semantic_temporal(
                        embedded.semantic + aligned_condition, mask
                    )[:, -1]
                    context = self.depth_context_norm(
                        full_context + self.semantic_to_depth(q0_context)
                    )
            q0 = self.sample_q0(
                q0_context,
                held[:, -1, 0],
                temperature=temperature,
                top_k=top_k,
                transition_context=transition_context,
            )
            prefix = q0[:, None]
            codes = [q0]
            for depth, parallel_logits in enumerate(
                self.parallel_depth_logits(context), start=1
            ):
                token = sample_logits(
                    parallel_logits,
                    temperature=(
                        resolved_detail_temperature
                        if depth >= detail_sampling_start_depth
                        else temperature
                    ),
                    top_k=top_k,
                )
                codes.append(token)
                prefix = torch.cat((prefix, token[:, None]), dim=-1)
            for depth in range(1 + len(self.parallel_depth_heads), 16):
                logits = self.depth.next_logits(context, prefix, depth)
                token = sample_logits(
                    logits,
                    temperature=(
                        resolved_detail_temperature
                        if depth >= detail_sampling_start_depth
                        else temperature
                    ),
                    top_k=top_k,
                )
                codes.append(token)
                prefix = torch.cat((prefix, token[:, None]), dim=-1)
            frame = torch.stack(codes, dim=-1)
            if torch.any((frame < 0) | (frame >= self.config.vocab_size)):
                raise RuntimeError("generated an invalid SeMoCo frame")
            generated.append(frame)
            history = torch.cat((history, frame[:, None, :]), dim=1)
        return torch.stack(generated, dim=1)
