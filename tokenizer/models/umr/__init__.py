"""Models for the UMR tokenizer family.

The active codec is the config-driven
:class:`~models.umr.structured_vq.StructuredVQTokenizer` (single-stream) and
:class:`~models.umr.structured_vq.StructuredMultiBranchTokenizer` (part-wise),
built via :func:`~models.umr.structured_vq.build_structured_vq`.
"""

from models.umr.conv_backbone import (
    StructuredDecoder,
    StructuredDecoderConfig,
    StructuredEncoder,
    StructuredEncoderConfig,
)
from models.umr.structured_vq import (
    BackboneConfig,
    CodecConfig,
    GroupConfig,
    QuantizerConfig,
    StructuredCodecOutput,
    StructuredMultiBranchTokenizer,
    StructuredVQTokenizer,
    build_backbone,
    build_structured_vq,
)

__all__ = [
    "BackboneConfig",
    "CodecConfig",
    "GroupConfig",
    "QuantizerConfig",
    "StructuredCodecOutput",
    "StructuredDecoder",
    "StructuredDecoderConfig",
    "StructuredEncoder",
    "StructuredEncoderConfig",
    "StructuredMultiBranchTokenizer",
    "StructuredVQTokenizer",
    "build_backbone",
    "build_structured_vq",
]
