# Codec Attribute Control

FACodec post-quantization prosody F0 probing and continuous intervention.

## quick-convert integration

The experiment uses `quick_convert.data.BaseDataset` for discovery,
loading, mono downmix and resampling. Pitch backends follow the
`quick_convert.components.features.f0.base.F0Extractor` contract:

- `JDCF0Extractor(F0Extractor)`: default, pretrained JDC, using the
  Plachtaa/FAcodec 24 kHz mel preprocessing and checkpoint.
- `quick_convert.components.features.f0.yaapt.YAAPTF0Extractor`:
  existing optional fallback, imported directly.

No pYIN implementation or dependency. F0 extractors return physical Hz;
normalizing voiced log2 F0 is a separate label transform.

## Setup

```bash
uv sync --group dev
git submodule update --init
uv sync --extra train --extra facodec
```

The default JDC checkpoint is read from the Amphion source tree at
`models/codec/facodec/modules/JDC/bst.t7`.
Override using `f0.checkpoint_path=/path/to/bst.t7` when needed.

## Run

```bash
uv run train-probe dataset.root=/path/to/speech
# Intervention CLI deferred until the QC waveform generator is available
uv run pytest -q
```

FACodec encodes mixed-length batches by default. Set
`system.encoder.batched=false` to use per-utterance inference.
Run the codec parity checks with the official pretrained weights:

```bash
FACODEC_PARITY=1 uv run --extra train --extra facodec pytest -q tests/test_facodec_parity.py
```

For quick-convert's existing YAAPT fallback:

```bash
uv run --extra yaapt train-probe dataset.root=/path/to/speech f0.backend=yaapt
```

The linear probe predicts a voiced log2 F0 recording-z-score from the
**post-quantization prosody contribution** of frozen Amphion NaturalSpeech 3
FACodec. Content, residual and speaker conditioning stay unchanged.
Recording normalization is currently an approximation to Plachtaa's
per-training-segment target.

The JDC checkpoint Git hash matches between Amphion and Plachtaa, but this
does not establish training-label parity with the NS3 codec. Absolute pitch
shifts remain exploratory because z-score normalization erases constant
log-pitch offsets.

See [quick-convert reuse and gaps](docs/quick-convert-gaps.md).

## Status

Training enters QC's `TrainingPipeline.run()` via `LightningTrainer` and
a project-specific `BaseTrainingModule`. The pretrained FACodec parity
tests cover quantized streams, code IDs and speaker embeddings.
