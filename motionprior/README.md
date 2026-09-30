# MotionPrior

**MotionPrior** is pretrained on motion-only data to generate packets from the frozen tokenizer. Separate semantic and full-packet temporal models capture motion history. A semantic predictor chooses whether to keep or change the semantic code, then a depth predictor completes the fifteen kinematic codes.

See [motion_prior.py](motionprior/models/motion_prior.py) for the model and [depth_gpt.py](motionprior/models/depth_gpt.py) for within-packet prediction. The released checkpoint is step 861853.

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
