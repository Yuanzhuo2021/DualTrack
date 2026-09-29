# Motion tokenizer

The tokenizer maps **25 Hz UMR499** motion to **12.5 Hz, 16-code packets** and reconstructs continuous motion. Its noncausal convolutional backbone has width 512 and temporal stride 2. The semantic VQ and 15-layer kinematic RVQ independently quantize projections of the full latent, then their projected outputs are summed.

| Path | Responsibility |
| --- | --- |
| `models/umr/structured_vq.py` | Codec configuration, encode/decode interfaces |
| `models/umr/conv_backbone.py` | Convolutional encoder and decoder |
| `models/quantizers/split_semantic_rvq.py` | Parallel semantic/kinematic quantization |
| `models/quantizers/ema_rvq.py` | Individual codebooks and residual quantization |
| `data/umr_schema.py` | 499D feature layout and canonical-anchor format |
| `eval/` | Reconstruction, codebook, geometry, and motion-validity metrics |
| `checkpoint_io.py` | Restricted tensor/NumPy checkpoint loading |

Download `checkpoints/best.pt` using `python download_checkpoints.py --model tokenizer` at the repository root. Its immutable contract is `balanced_v2_s2`: 16 codebooks, each with 1,024 entries.

```bash
# Run from tokenizer/
python inference.py --checkpoint checkpoints/best.pt \
  --input /path/to/umr499.npz --out outputs/reconstruction.npz --device cpu
python evaluate.py --codec-mode vq_vae --checkpoint checkpoints/best.pt \
  --manifest /path/to/eval_manifest.txt --gates L1,L3 \
  --device cpu --out outputs/eval.json
```

The manifest contains one UMR NPZ path per line. `L1` evaluates continuous-feature reconstruction and foot-contact F1; `L3` reports codebook usage. `L2`/`L4` need external geometry assets. Input features are `[T,499]`; reconstructed features use the same units, while foot-contact outputs remain logits. Input lengths are trimmed to the codec stride.

See the [main architecture description](../README.md#1-motion-tokenizer). Training code and preprocessing pipelines are not included.
