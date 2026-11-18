# Codec Attribute Control

**Question:** What speech attributes are accessible in a frozen neural speech codec, and can we eventually manipulate them independently?

The first experiment probes **pitch (F0)** in the **post-quantization prosody representation** of pretrained Amphion NaturalSpeech 3 FACodec. We freeze FACodec and train a single linear projection to predict framewise pitch. A successful probe shows that pitch information is *linearly accessible*; it does **not** establish causal control or disentanglement.

**Pipeline:** 16 kHz speech → frozen FACodec prosody features → linear F0 head. In parallel, pretrained JDC produces aligned 80 Hz F0 labels in Hz. We normalize voiced log2 F0 per recording, then optimize masked mean squared error over voiced frames.

## Quick start

```bash
uv sync --group dev --extra train --extra facodec
git submodule update --init
uv run --extra train --extra facodec train-probe data_root=/path/to/collections
```

`data_root` is the parent of `librispeech/LibriSpeech`. The JDC checkpoint is read from the Amphion source tree (`models/codec/facodec/modules/JDC/bst.t7`). FACodec pretrained weights are also required. The default training split is `train-clean-100`, with `dev-clean` for validation.

For faster repeated training, precompute JDC targets with Quick Convert's existing precompute pipeline, then switch one config group:

```bash
uv run --extra train --extra facodec precompute-f0 data_root=/path/to/collections
uv run --extra train --extra facodec train-probe data_root=/path/to/collections f0=cached
```

The default cache is **`./outputs/precomputed/f0/jdc` after PR #15**; until that fix is merged, pass `f0_cache_root=./outputs/precomputed/f0/jdc` to both commands. See [the experiment guide](docs/f0-probe.md) for exact cache behavior, configuration, metrics, and limitations.

## Validation and scope

```bash
uv run --extra train --extra facodec pytest -q
FACODEC_PARITY=1 uv run --extra train --extra facodec pytest -q tests/test_facodec_parity.py
```

FACodec supports mixed-length batches; set `system.encoder.batched=false` to compare per-utterance inference. YAAPT is an optional alternative pitch backend, not the default JDC supervision.

**Current scope:** frozen representation probing, not pitch intervention or waveform editing. The JDC checkpoint matches the upstream Plachtaa/Amphion artifact, but that alone does not prove exact parity with FACodec's original training labels.

More detail: [F0 probe methodology and reproducibility](docs/f0-probe.md) · [Quick Convert reuse and gaps](docs/quick-convert-gaps.md).
