"""Shared HOLD/CHANGE q0 factorization.

Both :class:`~motionprior.models.motion_prior.MotionPrior` and the fused
backbone use this single implementation so the ``-inf`` masking, the
previous-code scatter and the invalid-previous fallback can never drift
apart between the two architectures.

The factorization is::

    P(q0_t = v) = P(stay) * [v == q0_{t-1}] + P(change) * P_nonstay(v)

where ``P_nonstay`` is a softmax over the 1024 codes with the previous code
removed. ``effective`` is returned as normalized log-probabilities, so it can
be consumed anywhere raw logits are expected (``softmax(log p) == p``).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _masked_nonstay(
    nonstay_logits: Tensor,
    previous: Tensor,
    vocab_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(masked_logits, valid_previous, clamped_previous)`` in fp32."""
    valid_previous = previous.ge(0) & previous.lt(vocab_size)
    clamped_previous = previous.clamp(0, vocab_size - 1)
    masked = nonstay_logits.float().clone()
    masked.scatter_(-1, clamped_previous[..., None], float("-inf"))
    masked = torch.where(valid_previous[..., None], masked, nonstay_logits.float())
    return masked, valid_previous, clamped_previous


def factorized_q0_log_probs(
    transition_logits: Tensor,
    nonstay_logits: Tensor,
    previous: Tensor,
    *,
    vocab_size: int,
) -> Tensor:
    """Effective 1024-way log-probabilities from the two factorized heads.

    ``transition_logits`` is ``[..., 2]`` (index 0 = HOLD, 1 = CHANGE),
    ``nonstay_logits`` is ``[..., vocab_size]`` and ``previous`` is ``[...]``.
    Rows whose previous code is outside ``[0, vocab_size)`` (BOS, padding)
    fall back to a plain softmax over ``nonstay_logits``.
    """
    masked_nonstay, valid_previous, clamped_previous = _masked_nonstay(
        nonstay_logits, previous, vocab_size
    )
    nonstay_log_probs = F.log_softmax(masked_nonstay, dim=-1)
    transition_log_probs = F.log_softmax(transition_logits.float(), dim=-1)
    effective = transition_log_probs[..., 1, None] + nonstay_log_probs
    effective.scatter_(
        -1,
        clamped_previous[..., None],
        transition_log_probs[..., 0, None],
    )
    return torch.where(
        valid_previous[..., None],
        effective,
        F.log_softmax(nonstay_logits.float(), dim=-1),
    )


def factorized_q0_sample(
    transition_logits: Tensor,
    nonstay_logits: Tensor,
    previous: Tensor,
    *,
    vocab_size: int,
    sampler,
    temperature: float,
    top_k: int | None,
    transition_temperature: float | None = None,
) -> Tensor:
    """Hierarchical sample: decide HOLD/CHANGE first, then the destination.

    ``transition_temperature`` controls only the binary head. It defaults to
    ``temperature`` for backward compatibility, but callers should normally
    pass ``1.0`` (or ``0.0`` for greedy) so that the sampling temperature
    reshapes the destination distribution without silently rescaling the
    calibrated HOLD/CHANGE prior.
    """
    masked_nonstay, valid_previous, _ = _masked_nonstay(
        nonstay_logits, previous, vocab_size
    )
    binary_temperature = (
        temperature if transition_temperature is None else float(transition_temperature)
    )
    transition = sampler(transition_logits, temperature=binary_temperature, top_k=None)
    changed_code = sampler(masked_nonstay, temperature=temperature, top_k=top_k)
    return torch.where(valid_previous & transition.eq(0), previous, changed_code)
