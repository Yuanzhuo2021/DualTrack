# Interactive demo

**[Open the hosted website](https://yuanzhuo2021.github.io/DualTrack/demo/)**

`index.html` is a self-contained static gallery using local CSS, JavaScript, an SVG architecture diagram, and the archived media. It has language/model filters, transcript search, pagination, and one-at-a-time playback.

- `comparison/`: 71 original videos (DualTrack: EN22 / ES6 / ZH9 / JA6; GELINA: EN22 / ES6).
- `perceptual_test/`: 28 matched triplets, each with ground truth, GELINA, and DualTrack. Each version includes combined video, motion-only video, and speech-only audio.
- `assets/`: site presentation and the media index. The original full galleries remain available as fallbacks.

Open index.html directly, or serve the repository root with `python -m http.server 8000 --bind 127.0.0.1` and visit `http://127.0.0.1:8000/demo/`.

Media retain their original bytes and timing. No ratings database or administrator credentials are included.
