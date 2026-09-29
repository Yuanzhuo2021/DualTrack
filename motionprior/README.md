# MotionPrior

The released **step 861853** model predicts semantic and kinematic motion codes on a 12.5 Hz clock. It uses a 768D full-motion temporal stack (4 layers), a 768D semantic stack (2 layers), and a depth stack (3 layers), all with 12 attention heads. Its temporal context is 150 ticks.

The semantic stream predicts `q0` through stay/change factorization. The full-motion and semantic states form the context for autoregressive `q1…q15` prediction. All 16 frozen codebook embeddings are taken from the matching tokenizer checkpoint.

| Path | Responsibility |
| --- | --- |
| `motionprior/models/motion_prior.py` | Complete prior, forward and generation |
| `motionprior/models/frame_embedding.py` | Semantic and kinematic packet embeddings |
| `motionprior/models/temporal_gpt.py` | Temporal modeling |
| `motionprior/models/depth_gpt.py` | Within-packet autoregression |
| `motionprior/models/q0_factorization.py` | Semantic stay/change distribution |
| `motionprior/contracts/semoco.py` | Tokenizer checkpoint and shape contract |
| `runtime.py` | Exact weight loading and physical-unit decoding |

Download both the tokenizer and MotionPrior assets from the repository root. The runtime reads `../tokenizer/checkpoints/best.pt` and verifies every embedded tokenizer tensor against the separately loaded codec.

```bash
# Run from motionprior/
python inference.py --ticks 96 --device cpu --out outputs/motion.npz
python inference.py --prefix /path/to/prefix.npz --ticks 96 \
  --temperature 0.6 --top-k 20 --device cuda:0 --out outputs/continuation.npz
python evaluate.py --input /path/to/reference_tokens.npz \
  --device cpu --out outputs/token_metrics.json
```

A prefix NPZ may use `motion_codes`, `motion_tokens`, `semoco_motion_tokens`, `tokens_structured`, or `tokens`, with integer shape `[T,16]` and values 0–1023. Without a prefix, generation starts from the dedicated BOS packet. Output contains the newly generated `motion_codes` and `features [2*ticks,499]`, with `fps=25`.

Evaluation reports teacher-forced per-codebook NLL/accuracy over disjoint windows. It does not report perceptual quality or FGD. See the [main architecture description](../README.md#2-motionprior).
