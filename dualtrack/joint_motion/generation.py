from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True, slots=True)
class SamplingConfig:
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    min_speech_ticks: int = 2

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.min_speech_ticks < 0:
            raise ValueError("min_speech_ticks must be non-negative")


def _sample(logits: torch.Tensor, config: SamplingConfig, generator=None) -> torch.Tensor:
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any(dim=-1).all():
        raise FloatingPointError("invalid sampling logits")
    if not config.do_sample:
        return logits.argmax(dim=-1)
    scores = logits.float() / config.temperature
    if 0 < config.top_k < scores.shape[-1]:
        threshold = torch.topk(scores, config.top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if config.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        remove = probabilities.cumsum(dim=-1) > config.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, sorted_indices, sorted_scores)
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1, generator=generator).squeeze(-1)


def _speech_depth_packet(
    model,
    hidden: torch.Tensor,
    q0: torch.Tensor,
    sampling: SamplingConfig,
) -> torch.Tensor:
    talker = model.talker
    q0 = q0.unsqueeze(-1)
    # Use the native cached depth generation path. Recomputing each prefix can
    # change bf16 logits and therefore even greedy packet ids.
    result = talker.code_predictor.generate(
        inputs_embeds=torch.cat((hidden.unsqueeze(1), talker.get_input_embeddings()(q0)), dim=1),
        max_new_tokens=model.joint_config.speech_num_codebooks - 1,
        do_sample=sampling.do_sample, top_p=sampling.top_p, top_k=sampling.top_k,
        temperature=sampling.temperature, output_hidden_states=True, return_dict_in_generate=True,
    )
    return torch.cat((q0, result.sequences), dim=-1)


def _speech_packet_embedding(
    model,
    packet: torch.Tensor,
) -> torch.Tensor:
    talker = model.talker
    parts = [talker.get_input_embeddings()(packet[:, 0])]
    for codebook in range(1, model.joint_config.speech_num_codebooks):
        embedding = talker.code_predictor.get_input_embeddings()[codebook - 1]
        parts.append(embedding(packet[:, codebook]))
    return torch.stack(parts, dim=1).sum(dim=1, keepdim=True)
