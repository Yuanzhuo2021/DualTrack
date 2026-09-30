# Motion tokenizer

The **SeMoCo-based tokenizer** converts 25 Hz UMR499 body and finger motion into 12.5 Hz packets with one semantic code and fifteen kinematic codes. Parallel semantic VQ and kinematic RVQ branches share an encoder; their quantized representations are combined to reconstruct motion.

The main implementation is in [structured_vq.py](models/umr/structured_vq.py), with [semantic/kinematic quantization](models/quantizers/split_semantic_rvq.py) and the [UMR499 input format](data/umr_schema.py).

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
