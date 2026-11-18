# Codec Attribute Control

Minimal frozen-FACodec post-quantization prosody F0 probe and intervention.

## Setup

```bash
uv sync --extra dev
# FACodec is loaded from the upstream Amphion Python source.
git clone https://github.com/open-mmlab/Amphion.git ../Amphion
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
```

The package imports quick-convert for audio loading/resampling, and optionally
its YAAPT extractor. FACodec checkpoints are downloaded from
`amphion/naturalspeech3_facodec` on first use.

## Run

```bash
uv run codec-attribute-control train --audio-dir /path/to/speech --output outputs/probe.joblib
uv run codec-attribute-control intervene --audio /path/to/heldout.wav --probe outputs/probe.joblib --output-dir outputs/interventions
uv run pytest
```

The training command fits a ridge probe to **post-quantization prosody only**.
The intervention adds a continuous delta to the summed decoder input, leaving
content, residual and speaker conditioning fixed.

**F0 normalization:** the released FACodec training code uses voiced log2 F0,
per-segment mean and sample standard deviation (z-score), and an unvoiced
sentinel of -10. The probe trains on voiced frames only. The default pYIN
extractor is **not** the upstream pitch extractor; see the limitations below.

**Interpretation:** the optional semitone sweep holds the source's normalized
statistics fixed. Absolute uniform shifts are removed by per-segment z-scoring,
so the decoder's response is an empirical question, not a guarantee.

See [quick-convert reuse and gaps](docs/quick-convert-gaps.md).

## Status

This is a scaffold, not a verified pretrained-model run. Before relying on
results, verify upstream FACodec import/constructor/checkpoint/tensor contracts
and F0 extraction parity. No model-backed results are claimed.
