# Interactive skeleton comparison

**[Open the hosted demo](https://yuanzhuo2021.github.io/DualTrack/demo/)**

Compare **GT (ground truth), GELINA, and DualTrack** with front and side skeleton views and corresponding audio. Language/source filters, transcript search, pagination, and one-at-a-time playback are available.

- 43 samples: 22 English, 6 Spanish, 9 Chinese, and 6 Japanese.
- GT and DualTrack: all 43 samples. GELINA: 28 English/Spanish samples.
- 114 skeleton videos, using archived FP32 outputs. DualTrack uses reference speaker embeddings.

`assets/` contains the page styles, scripts, sample index, and the paper's main architecture figure. `skeleton/` contains the displayed MP4s and posters; [manifest.json](skeleton/manifest.json) records source hashes and rendering details. DualTrack uses SOMA77; GT and GELINA use SMPL-X55. Original durations and corresponding audio are preserved, with a common scale within each sample.

Open `index.html` directly, or serve the repository root with `python -m http.server 8000 --bind 127.0.0.1` and visit `http://127.0.0.1:8000/demo/`. The [complete gallery](comparison/index.html) also works without JavaScript. Legacy gallery links now lead to skeleton comparisons; previously archived study/mesh media remain in the repository but are not displayed.
