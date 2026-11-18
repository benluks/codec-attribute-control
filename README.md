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

## Installation

```bash
git clone https://github.com/benluks/codec-attribute-control.git
cd codec-attribute-control
uv sync --group dev
git clone https://github.com/open-mmlab/Amphion.git ../Amphion
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
```

`uv sync` installs this project's declared Quick Convert dependency. The
direct codec wrapper uses PyTorch plus Amphion's FACodec implementation;
Quick Convert is only needed for the `FACodecContentEncoder` pipeline adapter
and the training workflow.

The default JDC checkpoint is read from the Amphion source tree at
`models/codec/facodec/modules/JDC/bst.t7`.
Override using `f0.checkpoint_path=/path/to/bst.t7` when needed.

## Use the pretrained FACodec wrapper

`codec_attribute_control.facodec.FACodec` constructs the published 16 kHz
NaturalSpeech 3 FACodec architecture and downloads its official checkpoint
files from `amphion/naturalspeech3_facodec` on first use. It exposes separate
prosody, content and residual contributions, a speaker embedding, and six RVQ
codebooks.

```python
import torchaudio
from codec_attribute_control.facodec import FACodec, SAMPLE_RATE

waveform, rate = torchaudio.load("speech.wav")
waveform = waveform.mean(dim=0)
if rate != SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)

codec = FACodec(device="cpu")
frame = codec.encode(waveform)
reconstruction = codec.decode(frame)

print(frame.codes.shape)     # [6, 1, frames]
print(frame.prosody.shape)   # [1, 256, frames]
print(frame.content.shape)   # [1, 256, frames]
print(frame.residual.shape)  # [1, 256, frames]
print(reconstruction.shape)  # [samples]
```

`decode(frame, delta=...)` supports the experiment's post-quantization
prosody intervention. `delta` must match `frame.prosody`; content, residual
detail, speaker embedding, and discrete IDs are retained unchanged.

## Use it with Quick Convert

`FACodecContentEncoder` adapts the frozen codec to Quick Convert's
`ContentEncoder` contract. It trims each padded waveform to its valid length
and returns a combined `[batch, frames, 256]` representation with exact frame
lengths and retained factor streams.

```python
import torch
from codec_attribute_control.facodec_encoder import FACodecContentEncoder

encoder = FACodecContentEncoder(device="cpu")
features = encoder.encode_waveforms(
    torch.randn(2, 16_000),
    lengths=torch.tensor([16_000, 12_000]),
    sample_rate=16_000,
)
print(features.values.shape)      # [2, 80, 256]
print(features.lengths.tolist())  # [80, 60]
```

Quick Convert is not needed for direct `FACodec` use. It is required for
`FACodecContentEncoder`, dataset abstractions, and the training command.

## Run

```bash
uv run train-probe dataset.root=/path/to/speech
# Intervention CLI deferred until the QC waveform generator is available
uv run pytest -q
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

Training now enters QC's `TrainingPipeline.run()` via `LightningTrainer` and a project-specific `BaseTrainingModule`. FACodec integration remains provisional and has not been model-tested.\n\n
The draft has not been verified against real FACodec/JDC checkpoints or
model-backed generated audio; no synthesis results are claimed.
