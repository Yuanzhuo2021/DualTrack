"""Pure shape, trim, alignment, and next-packet contracts."""

from __future__ import annotations

from dataclasses import dataclass


MOTION_INPUT_FPS = 25.0
MOTION_TOKENIZER_STRIDE = 2
TOKEN_RATE_HZ = 12.5
TICK_SECONDS = 1.0 / TOKEN_RATE_HZ


@dataclass(frozen=True, slots=True)
class MotionTrim:
    raw_frames: int
    trimmed_frames: int
    dropped_tail_frames: int
    token_count: int


@dataclass(frozen=True, slots=True)
class AlignedWindow:
    speech_start: int
    speech_stop: int
    motion_start: int
    motion_stop: int

    @property
    def token_count(self) -> int:
        return self.speech_stop - self.speech_start


@dataclass(frozen=True, slots=True)
class NextPacketLayout:
    input_start: int
    input_stop: int
    target_start: int
    target_stop: int
    eos_position: int


def motion_trim(raw_frames: int) -> MotionTrim:
    if raw_frames < MOTION_TOKENIZER_STRIDE:
        raise ValueError("motion clip must contain at least 2 frames")
    trimmed = raw_frames - (raw_frames % MOTION_TOKENIZER_STRIDE)
    return MotionTrim(
        raw_frames=raw_frames,
        trimmed_frames=trimmed,
        dropped_tail_frames=raw_frames - trimmed,
        token_count=trimmed // MOTION_TOKENIZER_STRIDE,
    )


def align_equal_rate_tokens(
    speech_tokens: int,
    motion_tokens: int,
    alignment_offset_ticks: int = 0,
) -> AlignedWindow:
    """Return equal slices where speech i pairs with motion i plus offset."""
    if speech_tokens <= 0 or motion_tokens <= 0:
        raise ValueError("both streams must contain at least one token")
    speech_start = max(0, -alignment_offset_ticks)
    motion_start = max(0, alignment_offset_ticks)
    count = min(speech_tokens - speech_start, motion_tokens - motion_start)
    if count <= 0:
        raise ValueError("alignment offset leaves no shared token window")
    return AlignedWindow(
        speech_start=speech_start,
        speech_stop=speech_start + count,
        motion_start=motion_start,
        motion_stop=motion_start + count,
    )


def next_packet_layout(
    sequence_length: int,
    bos_position: int,
    packet_count: int,
) -> NextPacketLayout:
    """Place BOS/input packets one position after their prediction targets."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if bos_position < 0 or packet_count <= 0:
        raise ValueError("bos_position must be non-negative and packet_count positive")
    input_start = bos_position + 1
    input_stop = input_start + packet_count
    target_start = bos_position
    target_stop = target_start + packet_count
    eos_position = target_stop
    if eos_position >= sequence_length:
        raise ValueError("sequence has no position reserved for speech EOS")
    return NextPacketLayout(
        input_start=input_start,
        input_stop=input_stop,
        target_start=target_start,
        target_stop=target_stop,
        eos_position=eos_position,
    )
