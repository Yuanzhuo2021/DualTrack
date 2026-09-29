"""Native Qwen prompt/generation with teacher-forced hidden replay."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import nn
from .io import locked_path


@contextmanager
def isolated_rng(device, seed):
    device = torch.device(device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        yield


@dataclass
class SpeechTrajectory:
    codes: torch.Tensor
    hidden: torch.Tensor
    prompt: dict
    stop_reason: str


class QwenSpeechTower(nn.Module):
    def __init__(self, api):
        super().__init__()
        self.api = api
        self.model = api.model.requires_grad_(False).eval()

    @classmethod
    def load(cls, device="cuda:0", dtype=torch.bfloat16):
        from qwen_tts import Qwen3TTSModel
        path = locked_path("qwen_model").parent
        locked_path("qwen_speech_tokenizer")
        api = Qwen3TTSModel.from_pretrained(str(path), device_map=device, dtype=dtype,
                                          attn_implementation="sdpa", local_files_only=True)
        return cls(api)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def reference(self, audio, sample_rate):
        items = self.api.create_voice_clone_prompt(ref_audio=(audio, sample_rate), x_vector_only_mode=True)
        return self.api._prompt_items_to_voice_clone_prompt(items)

    def generation_arguments(self, text, reference, language="english", max_ticks=300, sample=False):
        if not 1 <= max_ticks <= 300:
            raise ValueError("V2.2 maximum motion duration is 300 ticks")
        ids = self.api._tokenize_texts([self.api._build_assistant_text(text)])
        return dict(input_ids=ids, voice_clone_prompt=reference, languages=[language],
                    non_streaming_mode=True, max_new_tokens=max_ticks+1,
                    do_sample=sample, subtalker_dosample=sample,
                    top_k=50, top_p=1., temperature=.9,
                    subtalker_top_k=50, subtalker_top_p=1., subtalker_temperature=.9,
                    repetition_penalty=1.05)

    @torch.no_grad()
    def generate(self, text, reference, *, language="english", max_ticks=300, seed=0, sample=False):
        prompt = {}
        def capture(module, args, kwargs):
            if not prompt and kwargs.get("inputs_embeds") is not None:
                for key in ("inputs_embeds", "attention_mask", "trailing_text_hidden", "tts_pad_embed"):
                    value = kwargs.get(key)
                    prompt[key] = value.detach().clone() if torch.is_tensor(value) else value
        handle = self.model.talker.register_forward_pre_hook(capture, with_kwargs=True)
        # Actual stop status is also checked by the returned effective length.
        try:
            with isolated_rng(self.device, seed):
                codes, hidden = self.model.generate(**self.generation_arguments(
                    text, reference, language, max_ticks, sample))
        finally:
            handle.remove()
        if len(codes) != 1 or codes[0].shape != (hidden[0].shape[0], 16):
            raise ValueError("native Qwen returned mismatched hidden/code lengths")
        return SpeechTrajectory(codes[0], hidden[0], prompt,
                                "eos" if len(codes[0]) < max_ticks else "max_ticks_or_eos_at_limit")

    @torch.no_grad()
    def packet_embedding(self, codes):
        talker = self.model.talker
        parts = [talker.get_input_embeddings()(codes[..., 0])]
        parts.extend(talker.code_predictor.get_input_embeddings()[i-1](codes[..., i]) for i in range(1, 16))
        # Match native torch.cat(...).sum(...): sequential bf16 additions round differently.
        return torch.stack(parts, dim=-2).sum(dim=-2)

    @torch.no_grad()
    def teacher_forced_hidden(self, codes, prompt):
        if codes.ndim != 2 or codes.shape[1] != 16 or len(codes) == 0:
            raise ValueError("codes must have shape [T,16], T>0")
        if codes.min() < 0 or codes.max() >= 2048:
            raise ValueError("speech codec ID outside 0..2047")
        codes = codes.to(self.device)
        prefill = prompt["inputs_embeds"]
        if prefill.shape[0] != 1 or not prompt["attention_mask"].bool().all():
            raise ValueError("hidden cache uses one unpadded prompt per utterance")
        if prompt["trailing_text_hidden"].shape[1] != 1:
            raise ValueError("cache protocol requires non_streaming_mode=True")
        history = self.packet_embedding(codes[:-1])[None]
        history = history + prompt["tts_pad_embed"]
        x = torch.cat((prefill, history), 1)
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        positions, _ = self.model.talker.get_rope_index(mask)
        h = self.model.talker.model(inputs_embeds=x, attention_mask=mask, position_ids=positions,
                                   use_cache=False, return_dict=True).last_hidden_state
        return h[0, prefill.shape[1]-1:]

    @torch.no_grad()
    def decode(self, codes):
        waves, sample_rate = self.model.speech_tokenizer.decode([{"audio_codes": codes}])
        return waves[0], sample_rate
