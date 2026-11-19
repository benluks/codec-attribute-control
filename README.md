# Codec Attribute Control

**Question:** What speech attributes are accessible in a frozen neural speech codec, and can we eventually manipulate them independently?

The first experiment probes **pitch (F0)** in the **post-quantization prosody representation** of pretrained Amphion NaturalSpeech 3 FACodec. We freeze FACodec and train a single linear projection to predict framewise pitch. A successful probe shows that pitch information is *linearly accessible*; it does **not** establish causal control or disentanglement.

**Pipeline:** 16 kHz speech → frozen FACodec prosody features → linear F0 head. The selected pitch backend produces frame-aligned F0 targets at 80 Hz. We normalize voiced log2 F0 per recording, then optimize masked mean squared error over voiced frames.

## Quick start

```bash
uv sync --group dev --extra train --extra facodec
git submodule update --init
uv run --extra train --extra facodec train-probe data_root=/path/to/collections
```

`data_root` is the parent of `librispeech/LibriSpeech`. The JDC checkpoint is read from the Amphion source tree (`models/codec/facodec/modules/JDC/bst.t7`). FACodec pretrained weights are also required. The default training split is `train-clean-100`, with `dev-clean` for validation.

For a fully cached F0 regression probe, precompute WORLD pitch and FACodec prosody
codes, then use the 17-frame contextual head:

```bash
uv run --extra train --extra facodec precompute-f0 data_root=/path/to/collections f0_backend=world
uv run --extra train --extra facodec precompute-codes data_root=/path/to/collections
uv run --extra train --extra facodec train-probe data_root=/path/to/collections \
  f0=cached f0_backend=world codes=cached head=contextual
```

WORLD runs on CPU and stores voiced pitch in Hz and unvoiced frames as `NaN`.
Use `f0_backend=jdc_corrected` for JDC with the calibrated mel filter bank or
`f0_backend=jdc` for the original JDC recipe and existing caches. Use the same
backend in precomputation and training. Each backend has its own directory under
`./outputs/precomputed/f0`. See [the experiment guide](docs/f0-probe.md).

## Validation and scope

```bash
uv run --extra train --extra facodec pytest -q
FACODEC_PARITY=1 uv run --extra train --extra facodec pytest -q tests/test_facodec_parity.py
```

FACodec supports mixed-length batches; set `system.encoder.batched=false` to compare per-utterance inference. YAAPT is an optional alternative pitch backend, not the default JDC supervision.

**Current scope:** frozen representation probing, not pitch intervention or waveform editing. The JDC checkpoint matches the upstream Plachtaa/Amphion artifact, but that alone does not prove exact parity with FACodec's original training labels.

More detail: [F0 probe methodology and reproducibility](docs/f0-probe.md) · [Quick Convert reuse and gaps](docs/quick-convert-gaps.md).

## Training diagnostics

Training logs batch loss and learning rate every optimizer step (`trainer.trainer_kwargs.log_every_n_steps: 1`), plus epoch-aggregated training/validation loss and voiced-frame F0 errors and correlations. Gradient-norm logging is enabled through Quick Convert's base training module. Validation pitch-curve examples are logged once per validation epoch. For lower W&B logging volume, override `trainer.trainer_kwargs.log_every_n_steps=10` (or another interval). These settings do not alter optimization or F0 targets.

## Probe capacity

The default is a single linear projection. Configure hidden layers and their activation through Hydra:

```bash
uv run --extra train --extra facodec train-probe data_root=/path/to/collections \
  'system.head.hidden_sizes=[128,64]' system.head.activation=gelu
```

`hidden_sizes=[]` retains the original linear probe. Supported activations: `gelu`, `relu`, `tanh`, `silu`. Compare nonlinear and linear results on the same split: an MLP probes nonlinear accessibility, not invertibility or demonstrated waveform control.

## Absolute log-pitch supervision

By default, the probe predicts within-recording z-scored log₂ F0 (`target_mode=normalized_log2`). For absolute pitch, use `target_mode=log2`: the target is `log2(F0 in Hz)` on voiced frames (>5 Hz), without recording-wise normalization. Unvoiced frames are excluded from both losses. A 1.0 log₂ difference is an octave; 1/12 is a semitone. Validation plots convert predictions back to Hz using `2**prediction` in this mode, rather than reference-derived normalization statistics. Metrics are named `log2_f0_*` in absolute mode, and `normalized_log2_f0_*` otherwise; their scales are not directly comparable. The run name and checkpoint monitor follow the selected mode.

```bash
uv run --extra train --extra facodec train-probe data_root=/cfs/collections target_mode=log2 'system.head.hidden_sizes=[128]' system.head.activation=gelu batch_size=128
```

## Interchangeable F0 heads

Use Hydra `head=mlp` (default), `head=contextual`, or `head=facodec_cnn`. The linear probe is the MLP with `head.hidden_sizes=[]`; nonempty hidden sizes add nonlinear layers. The `head.context_size` setting accepts positive odd frame counts (1, 3, 5, ...). Neighboring prosody frames are concatenated to predict the **center frame** F0. Sequence edges are replicated; outputs remain aligned at 80 Hz. Padded batch frames remain masked by the existing loss.

```bash
# Linear, one frame (default)
uv run --extra train --extra facodec train-probe data_root=/cfs/collections head=mlp

# Linear, centered three-frame context
uv run --extra train --extra facodec train-probe data_root=/cfs/collections head=mlp head.context_size=3

# MLP, centered three-frame context
uv run --extra train --extra facodec train-probe data_root=/cfs/collections head=mlp 'head.hidden_sizes=[128]' head.context_size=3

# Original FACodec temporal CNN (random initialization)
uv run --extra train --extra facodec train-probe data_root=/cfs/collections head=facodec_cnn
```

The CNN uses Amphion's original `CNNLSTM` class, including SnakeBeta and residual dilated convolutions; despite its name it has no LSTM. It trains its F0 output, not its voicing output. The frozen encoder and training pipeline are shared. Architecture appears in W&B run names; context and hidden widths appear in run config.

## Pitch-to-prosody-code baseline

Train a pitch-conditioned classifier for FACodec's **first (prosody) codebook** without changing the existing F0 probes. Frozen FACodec provides integer targets; the online or cached pitch resource provides frame-aligned F0. Input is **absolute log₂ F0 in Hz plus a voicing bit**, including unvoiced frames. A centered window is flattened and passed through a GELU MLP to predict one of 1024 code IDs per frame. Train with cross-entropy on valid frames (including unvoiced); log code accuracy and loss. This is a *code-reconstruction baseline*, not yet a pitch-editing decoder or a demonstration of control.

```bash
uv run --extra train --extra facodec train-pitch-codes data_root=/cfs/collections
uv run --extra train --extra facodec train-pitch-codes data_root=/cfs/collections system.head.context_size=9 'system.head.hidden_sizes=[256,128]'
```

Config: `configs/train_codes.yaml`. Existing QC datasets, F0 resources, optimization, W&B, and checkpointing are reused. Code indices are taken from the first of six FACodec quantizers; the other five remain untouched. The next experiment is to replace the original prosody codes with predicted ones and measure decoded waveform F0, retaining the original content/residual/speaker streams. High code accuracy alone does not establish controllability.

### Cache prosody code indices

The single prosody quantizer produces one integer index (0–1023) per 80 Hz frame. Cache only this index as a `uint16` tensor (2 bytes/frame; approximately 576 KB/hour before file overhead), using QC's existing precompute pipeline and manifest/skip-existing behavior:

```bash
uv run --extra facodec precompute-codes data_root=/cfs/collections
uv run --extra train --extra facodec train-pitch-codes data_root=/cfs/collections codes=cached
```

Both commands default to `./outputs/precomputed/prosody_codes/facodec`; override `codes_cache_root` in either config. The pitch-to-code classifier reads code IDs directly. With both caches selected, `train-pitch-codes codes=cached f0=cached f0_backend=world` trains without loading audio. The F0 regression probe projects the same cached IDs through the frozen prosody codebook.

## W2V-BERT reconstruction-latent pitch probe

This experiment reuses Quick Convert's existing `W2VBertContentEncoder` and
`LayerWeightedSum`. It loads the **learned 24-layer fusion weights** from a
W2V-BERT + CosyVoice DiT SSL-reconstruction Lightning checkpoint, freezes the
encoder and fusion, and trains only one of the existing F0 heads. W2V-BERT
features are 50 Hz, so the existing 80 Hz WORLD F0 labels are sampled onto
the centers of the 50 Hz feature frames. Unvoiced frames remain excluded.

```bash
uv run --extra train --extra w2vbert train-w2vbert-probe \
  data_root=/cfs/collections \
  reconstruction_checkpoint=/path/to/w2vbert_dit_cosyvoice_librispeech.ckpt \
  head=mlp 'head.hidden_sizes=[512]' head.context_size=3 \
  head.activation=gelu f0=cached f0_backend=world \
  target_mode=log2
```

Other supported heads: `head=facodec_cnn` (temporal CNN, requires `--extra facodec`
and the Amphion submodule) and
`head=contextual`. The default is absolute log₂ F0; the target normalization
is not identical to the original SSL-probe experiment's global z-score, so
compare correlation and R² only after matching metric definitions. The
reconstruction checkpoint must contain `system.feature_transform.weights`.
Pass a trusted checkpoint path. When encoder weights are present, they are
restored along with the fusion weights. `reconstruction_revision` selects the
Hugging Face backbone and processor revision from the reconstruction config.
For Praat labels, add `--extra praat` and set `f0_backend=praat`. Cached Praat
contours use the same 80 Hz label grid as WORLD.

## Praat (Amphion-style) F0

The `praat` backend uses `praat-parselmouth` with Praat autocorrelation
(`to_pitch_ac`), a 12.5 ms time step, 0.6 voicing threshold, and 60–640 Hz
pitch bounds. The algorithm and threshold match Amphion's
`get_f0_features_using_parselmouth`; the pitch bounds are explicit defaults
and can be adjusted in code. Unlike treating the first Praat pitch frame as
time zero, we align by Praat's actual timestamps.

```bash
uv sync --extra train --extra praat
uv run --extra train --extra praat precompute-f0 \
  data_root=/cfs/collections f0_backend=praat \
  splits='[train-clean-100,dev-clean]' \
  batch_size=16 num_workers=2 pitch_workers=4
uv run --extra train --extra praat train-probe \
  data_root=/cfs/collections f0=cached f0_backend=praat
```

Outputs are isolated under `outputs/precomputed/f0/praat`, with the same
QC provenance checks, resumability, and optional multi-machine sharding as
WORLD. The saved tensors contain Hz on voiced frames and NaN otherwise.
