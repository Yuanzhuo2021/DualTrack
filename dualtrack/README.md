# DualTrack

**DualTrack** couples Qwen3-TTS and MotionPrior on a shared 12.5 Hz clock. Residual adapters exchange previous packets in both directions, while **Tick Fusion** coordinates the current speech and motion-semantic states. Each stream then completes its own packet in parallel with the other stream.

See [dual_tick.py](DualTrack/dual_tick.py) for coupling and generation, [motionprior.py](DualTrack/motionprior.py) for the motion stream, and [qwen.py](DualTrack/qwen.py) for speech conditioning and decoding.

The delta checkpoint does not contain the frozen Qwen base. Download all three repository checkpoints, and supply a complete Qwen3-TTS-12Hz-0.6B-Base directory containing its `speech_tokenizer/` subdirectory. The original parameter names and checkpoint bytes are preserved while the public Python namespaces are `DualTrack` and `motionprior`.

```bash
# Run from dualtrack/
python inference.py --text "Hello, how are you?" --language english \
  --reference-audio /path/to/reference.wav \
  --qwen-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --device cuda:0 --ticks 150 --out-dir outputs/example
```

Supported languages: `english`, `spanish`, `chinese`, `japanese`. This entry point conditions on the reference speaker embedding. Use `--greedy` to disable sampling. The maximum requested budget is 300 ticks; speech EOS can stop earlier.

Outputs: `speech.wav`, `motion.npz`, `result.json`. The NPZ contains 25 Hz UMR499 features and both modalities’ codes, not SMPL-X poses. No Flan model is needed by this stream, and waveform/motion decoding is performed after token generation.

```bash
python evaluate.py --condition /path/to/condition.pt \
  --tokens /path/to/paired_tokens.npz \
  --qwen-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --device cpu --out outputs/eval.json
```

Conditioning PT fields: `prompt` (`inputs_embeds`, `attention_mask`, `trailing_text_hidden`, `tts_pad_embed`), `flan_text`, `text_valid`. Paired NPZ fields: `speech_codes` and `motion_codes`, matching `[T,16]` arrays with 1≤T≤149. This reports teacher-forced speech/motion losses. See the [main architecture description](../README.md#3-dualtrack-coupling).
