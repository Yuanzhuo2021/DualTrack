from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class JointMotionConfig:
    talker_hidden_size: int = 1024
    talker_num_hidden_layers: int = 28
    talker_num_attention_heads: int = 16
    talker_num_key_value_heads: int = 8
    text_hidden_size: int = 2048
    speech_num_codebooks: int = 16
    speech_temporal_vocab_size: int = 3072
    speech_depth_vocab_size: int = 2048
    speech_codec_bos_token_id: int = 2149
    speech_codec_eos_token_id: int = 2150
    speech_code_predictor_hidden_size: int = 1024
    speech_code_predictor_layers: int = 5
    speech_code_predictor_heads: int = 16
    speech_code_predictor_kv_heads: int = 8
    motion_embedding_size: int = 768
    motion_num_codebooks: int = 16
    motion_codebook_size: int = 1024
    motion_code_predictor_hidden_size: int = 1024
    motion_code_predictor_layers: int = 5
    motion_code_predictor_heads: int = 8
    motion_code_predictor_kv_heads: int = 4
    motion_code_predictor_ffn_size: int = 3072
    motion_loss_weights: tuple[float, ...] = (
        1.5, 1.2,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        0.7, 0.7, 0.7, 0.7,
    )
    speech_depth_loss_scale: float = 0.3
    motion_depth_loss_scale: float = 0.3
    speech_loss_scale: float = 1.0
    motion_loss_scale: float = 1.0
    token_rate_hz: float = 12.5
    freeze_talker: bool = True
    freeze_motion_branch: bool = True
    strict_qwen_0_6b: bool = True

    def __post_init__(self) -> None:
        if self.motion_num_codebooks != len(self.motion_loss_weights):
            raise ValueError("motion_loss_weights length must equal motion_num_codebooks")
        if self.speech_num_codebooks != 16 or self.motion_num_codebooks != 16:
            raise ValueError("DualTrack V1 requires exactly 16 speech and motion codebooks")
        if self.token_rate_hz != 12.5:
            raise ValueError("DualTrack V1 requires a shared 12.5 Hz token clock")


def _read_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def validate_qwen_0_6b_talker(talker: Any, config: JointMotionConfig) -> None:
    """Fail early when a 1.7B or otherwise incompatible talker is supplied."""
    if not config.strict_qwen_0_6b:
        return
    talker_config = getattr(talker, "config", None)
    if talker_config is None:
        raise TypeError("talker must expose its Hugging Face config")

    checks = {
        "hidden_size": config.talker_hidden_size,
        "num_hidden_layers": config.talker_num_hidden_layers,
        "num_attention_heads": config.talker_num_attention_heads,
        "num_key_value_heads": config.talker_num_key_value_heads,
        "text_hidden_size": config.text_hidden_size,
        "num_code_groups": config.speech_num_codebooks,
        "codec_bos_id": config.speech_codec_bos_token_id,
        "codec_eos_token_id": config.speech_codec_eos_token_id,
        "vocab_size": config.speech_temporal_vocab_size,
    }
    errors: list[str] = []
    for field, expected in checks.items():
        actual = _read_attr(talker_config, field)
        if actual != expected:
            errors.append(f"{field}: expected {expected}, got {actual}")

    code_config = _read_attr(talker_config, "code_predictor_config")
    if code_config is None:
        errors.append("missing talker.config.code_predictor_config")
    else:
        code_checks = {
            "hidden_size": config.speech_code_predictor_hidden_size,
            "num_hidden_layers": config.speech_code_predictor_layers,
            "num_attention_heads": config.speech_code_predictor_heads,
            "num_key_value_heads": config.speech_code_predictor_kv_heads,
            "vocab_size": config.speech_depth_vocab_size,
        }
        for field, expected in code_checks.items():
            actual = _read_attr(code_config, field)
            if actual != expected:
                errors.append(f"code_predictor.{field}: expected {expected}, got {actual}")

    for name in ("model", "codec_head", "code_predictor", "text_projection"):
        if not hasattr(talker, name):
            errors.append(f"missing talker.{name}")
    if errors:
        raise ValueError("incompatible Qwen3-TTS 0.6B talker: " + "; ".join(errors))
