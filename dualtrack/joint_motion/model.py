from __future__ import annotations
from dataclasses import dataclass
import torch
from .config import JointMotionConfig
from .contracts import next_packet_layout

@dataclass(slots=True)
class MotionTeacherForcing:
    input_codes: torch.Tensor
    input_mask: torch.Tensor
    bos_mask: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    eos_position: int


def build_motion_teacher_forcing(
    motion_packets: torch.Tensor,
    sequence_length: int,
    target_start: int,
) -> MotionTeacherForcing:
    """Place previous motion packets beside explicitly aligned next-packet targets."""
    if motion_packets.ndim != 3 or motion_packets.shape[-1] != 16:
        raise ValueError("motion_packets must have shape [B, T, 16]")
    batch, packet_count, codebooks = motion_packets.shape
    layout = next_packet_layout(sequence_length, target_start, packet_count)
    input_codes = motion_packets.new_zeros((batch, sequence_length, codebooks))
    targets = motion_packets.new_zeros((batch, sequence_length, codebooks))
    input_mask = torch.zeros((batch, sequence_length), dtype=torch.bool, device=motion_packets.device)
    bos_mask = torch.zeros_like(input_mask)
    target_mask = torch.zeros_like(input_mask)

    input_codes[:, layout.input_start:layout.input_stop] = motion_packets
    targets[:, layout.target_start:layout.target_stop] = motion_packets
    input_mask[:, layout.input_start:layout.input_stop] = True
    bos_mask[:, target_start] = True
    target_mask[:, layout.target_start:layout.target_stop] = True
    return MotionTeacherForcing(
        input_codes=input_codes,
        input_mask=input_mask,
        bos_mask=bos_mask,
        targets=targets,
        target_mask=target_mask,
        eos_position=layout.eos_position,
    )


@dataclass(slots=True)
class JointBatch:
    base_input_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    motion_input_codes: torch.Tensor
    motion_input_mask: torch.Tensor
    motion_bos_mask: torch.Tensor
    speech_q0_targets: torch.Tensor
    speech_q0_mask: torch.Tensor
    speech_packet_targets: torch.Tensor
    speech_packet_mask: torch.Tensor
    motion_packet_targets: torch.Tensor
    motion_packet_mask: torch.Tensor
    position_ids: torch.Tensor | None = None
    generated_history_mask: torch.Tensor | None = None

    def validate(self, config: JointMotionConfig) -> None:
        if self.base_input_embeddings.ndim != 3:
            raise ValueError("base_input_embeddings must have shape [B, L, H]")
        batch, length, hidden = self.base_input_embeddings.shape
        if hidden != config.talker_hidden_size:
            raise ValueError(f"expected talker hidden size {config.talker_hidden_size}; got {hidden}")
        shape_2d = (batch, length)
        shape_speech = (batch, length, config.speech_num_codebooks)
        shape_motion = (batch, length, config.motion_num_codebooks)
        for name in (
            "attention_mask",
            "motion_input_mask",
            "motion_bos_mask",
            "speech_q0_targets",
            "speech_q0_mask",
            "speech_packet_mask",
            "motion_packet_mask",
        ):
            if getattr(self, name).shape != shape_2d:
                raise ValueError(f"{name} must have shape {shape_2d}")
        if self.motion_input_codes.shape != shape_motion:
            raise ValueError(f"motion_input_codes must have shape {shape_motion}")
        if self.motion_packet_targets.shape != shape_motion:
            raise ValueError(f"motion_packet_targets must have shape {shape_motion}")
        if self.speech_packet_targets.shape != shape_speech:
            raise ValueError(f"speech_packet_targets must have shape {shape_speech}")

        motion_input_mask = self.motion_input_mask.bool()
        motion_bos_mask = self.motion_bos_mask.bool()
        speech_packet_mask = self.speech_packet_mask.bool()
        motion_packet_mask = self.motion_packet_mask.bool()
        if torch.any(motion_input_mask & motion_bos_mask):
            raise ValueError("motion input and BOS masks must be disjoint")
        if not torch.equal(speech_packet_mask, motion_packet_mask):
            raise ValueError("speech and motion packet target masks must be exactly aligned")
        if not torch.equal(
            motion_input_mask.sum(dim=1),
            motion_packet_mask.sum(dim=1),
        ):
            raise ValueError("each sample needs one previous-motion input per motion target")
        if not torch.all(motion_bos_mask.sum(dim=1) == 1):
            raise ValueError("each sample needs exactly one motion BOS")
        for values in (self.motion_input_codes, self.motion_packet_targets,
                       self.speech_packet_targets, self.speech_q0_targets):
            if values.dtype != torch.long:
                raise ValueError("batch token ids must be torch.long")
        generated = self.generated_history_mask
        if generated is None:
            generated = torch.zeros_like(motion_input_mask)
        elif (generated.shape != shape_2d or generated.dtype != torch.bool
              or torch.any(generated & ~motion_input_mask)):
            raise ValueError("generated history mask must select shifted motion inputs only")
        for row in range(batch):
            start = int(motion_bos_mask[row].nonzero()[0])
            count = int(motion_packet_mask[row].sum())
            layout = next_packet_layout(length, start, count)
            expected = torch.zeros(length, dtype=torch.bool, device=motion_packet_mask.device)
            expected[layout.target_start:layout.target_stop] = True
            if not torch.equal(motion_packet_mask[row], expected):
                raise ValueError("packet targets must be contiguous starting at BOS")
            shifted = torch.zeros_like(expected)
            shifted[layout.input_start:layout.input_stop] = True
            if not torch.equal(motion_input_mask[row], shifted):
                raise ValueError("motion inputs must be shifted exactly one tick")
            unchanged = ~generated[row, shifted]
            if not torch.equal(self.motion_input_codes[row, shifted][unchanged], self.motion_packet_targets[row, expected][unchanged]):
                raise ValueError("previous motion inputs do not match shifted targets")
            qmask = self.speech_q0_mask[row].bool().clone()
            qmask[layout.eos_position] = False
            if not torch.equal(qmask, expected):
                raise ValueError("q0 mask must cover packets and optionally their terminal EOS")
        active = motion_input_mask | motion_bos_mask | self.speech_q0_mask.bool()
        if torch.any(active & ~self.attention_mask.bool()):
            raise ValueError("active packet positions cannot be attention padding")
        if not torch.equal(self.speech_q0_targets[speech_packet_mask], self.speech_packet_targets[..., 0][speech_packet_mask]):
            raise ValueError("q0 labels must match speech packet q0")

        self._validate_selected_ids(
            self.motion_input_codes,
            motion_input_mask,
            config.motion_codebook_size,
            "motion input",
        )
        self._validate_selected_ids(
            self.motion_packet_targets,
            motion_packet_mask,
            config.motion_codebook_size,
            "motion target",
        )
        selected_speech = self.speech_packet_targets[speech_packet_mask]
        if selected_speech.numel() == 0:
            raise ValueError("batch contains no speech-motion packet target")
        if torch.any(selected_speech[:, 0] < 0) or torch.any(
            selected_speech[:, 0] >= config.speech_depth_vocab_size
        ):
            raise ValueError("speech q0 packet target is outside its vocabulary")
        if torch.any(selected_speech[:, 1:] < 0) or torch.any(
            selected_speech[:, 1:] >= config.speech_depth_vocab_size
        ):
            raise ValueError("speech depth packet target is outside its vocabulary")
        selected_q0 = self.speech_q0_targets[self.speech_q0_mask.bool()]
        if torch.any(selected_q0 < 0) or torch.any(
            selected_q0 >= config.speech_temporal_vocab_size
        ):
            raise ValueError("speech q0 target is outside its vocabulary")
        eos_only = self.speech_q0_mask.bool() & ~speech_packet_mask
        if not torch.all(self.speech_q0_targets[eos_only] == config.speech_codec_eos_token_id):
            raise ValueError("the only non-packet speech q0 target must be EOS")

    @staticmethod
    def _validate_selected_ids(
        values: torch.Tensor,
        mask: torch.Tensor,
        vocab_size: int,
        label: str,
    ) -> None:
        selected = values[mask]
        if selected.numel() == 0:
            raise ValueError(f"batch contains no {label} ids")
        if torch.any(selected < 0) or torch.any(selected >= vocab_size):
            raise ValueError(f"{label} id is outside 0..{vocab_size - 1}")
