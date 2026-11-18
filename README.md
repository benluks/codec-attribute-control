# Codec Attribute Control

FACodec post-quantization prosody F0 probing and continuous intervention.

## quick-convert integration

The experiment directly uses `quick_convert.data.BaseDataset` to discover,
load, downmix, and resample speech. All F0 labels come from the same
`quick_convert.components.features.f0.base.F0Extractor` contract:

- `JDCF0Extractor(F0Extractor)`: default pretrained JDC, using the
  Plachtaa/FAcodec 24kHz mel preprocessing and JDC checkpoint.
- `quick_convert.components.features.f0.yaapt.YAAPTF0Extractor`:
  existing, optional fallback.

No pYIN backend is included. F0 extractors return physical Hz; voiced log2
normalization is an independent label transform.

## Setup

```bash
uv sync --extra dev
git clone https://github.com/open-mmlab/Amphion.git ../Amphion
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
```

The default JDC checkpoint is loaded from the Amphion source tree at
`models/codec/facodec/modules/JDC/bst.t7`.
You can override it with `--pitch-checkpoint /path/to/bst.t7`.

## Run

```bash
uv run codec-attribute-control train --audio-dir /path/to/speech
uv run codec-attribute-control intervene --audio /path/to/heldout.wav --probe outputs/probe.joblib
uv run --extra dev pytest -q
```

Use the installed quick-convert YAAPT fallback:

```bash
uv run --extra yaapt codec-attribute-control train --audio-dir /path/to/speech --f0-backend yaapt
```

Probe target: voiced, log2 F0, recording-level sample z-score (a current
approximation of the Plachtaa per-training-crop recipe). The predictor reads
the **post-quantization prosody contribution** of the frozen Amphion
NaturalSpeech 3 FACodec. Content, residual and speaker conditioning stay fixed.

JDC checkpoint weights are identical in Plachtaa and Amphion's facodec
subtree, but parity with NS3's training labels is not established.

Absolute semitone shifting is exploratory: normalized pitch contours do
not retain constant log-pitch offsets.

See [quick-convert reuse and gaps](docs/quick-convert-gaps.md).

## Status

Draft PR, not yet verified against downloaded FACodec checkpoints or
model-backed generated audio.
