# F0 probe: methodology and reproduction

## 1. Research question and interpretation

We ask whether pitch is **linearly decodable** from a pretrained speech codec's post-quantization prosody contribution. FACodec remains frozen; only a framewise linear head is optimized. This is a *diagnostic* of accessible information, not proof that the representation contains pitch exclusively, that the factors are disentangled, or that editing the representation will reliably change F0 while preserving everything else.

The immediate experiment predicts **relative pitch movement within a recording**. Per-recording normalization removes absolute log-pitch offsets, so this setup cannot by itself validate absolute pitch shifts.

## 2. Signal and target path

1. Quick Convert's `BaseDataset` discovers LibriSpeech recordings and loads mono audio at 16 kHz.
2. Frozen pretrained FACodec produces post-quantization prosody features at approximately 80 frames/s; the head maps `[B,T,256]` to `[B,T]`.
3. WORLD or JDC computes raw F0 in Hz. The JDC mel front end operates at 24 kHz, with a 300-sample hop (12.5 ms). Online F0 resamples the 16 kHz batch, performs batched mel extraction and JDC inference, then aligns to FACodec's 200-sample hop at 16 kHz (also 12.5 ms).
4. Invalid/unvoiced frames are represented as NaN in raw F0. `prepare_f0_targets` applies voiced log2-F0 recording-wise z-score normalization and excludes unvoiced, invalid, or insufficiently varying segments.
5. The linear head is trained with masked MSE on valid voiced frames only. The training module checks that the F0 and codec valid frame lengths agree exactly.

**Important:** matching nominal 80 Hz frame rates does not guarantee every boundary frame is identical. Run parity checks when changing the frontend, padding, or frame alignment.

## 3. Online versus cached labels

Both modes provide the same F0 resource (`[B,T,1]` Hz plus valid lengths) to the training module. The probe, loss, and metrics are unchanged.

| Mode | Configuration | What happens |
| --- | --- | --- |
| Online | `f0=online` (default) | `OnlineF0` runs the selected backend during training |
| Cached | `f0=cached` | QC `PathResourceProvider` loads `{split}/{utt_id}.pt` |

To precompute:

```bash
uv run --extra train --extra facodec precompute-f0 \
  data_root=/path/to/collections \
  f0_backend=world \
  splits='[train-clean-100,dev-clean]' \
  batch_size=16 num_workers=2 pitch_workers=4
```

This invokes Quick Convert's `PrecomputeFeaturesPipeline` through its existing `execute_pipeline` entry point. The project-specific `F0PrecomputeExtractor` is a thin `extract_batch` adapter over `OnlineF0`. QC handles dataset iteration, provenance checking, atomic writes, skip-existing resumption, and `manifest.jsonl`. Each `.pt` file stores the unpadded `[T,1]` Hz contour.

For WORLD, `pitch_workers` sets the number of CPU processes extracting pitch.
`num_workers` separately controls audio loading. Set `batch_size` at least as
large as `pitch_workers` to keep the pool busy. Changing the worker count
can resume the same cache.

Select the backend with `f0_backend` for precomputation and both trainers:

| Backend | Target recipe |
| --- | --- |
| `world` | WORLD DIO + StoneMask at 80 Hz, with WORLD voiced/unvoiced decisions. Runs on CPU. |
| `jdc_corrected` | JDC on 24 kHz audio with the checkpoint-matched 16 kHz mel filter bank. |
| `jdc` (default) | Original JDC preprocessing for comparison and existing caches. |

Caches default to `./outputs/precomputed/f0/<backend>`. Each contour stores Hz
for voiced frames and `NaN` for unvoiced frames. The regression loss normalizes
only voiced frames. The code classifier uses the same contour and voicing mask.

```bash
uv run --extra train --extra facodec precompute-codes data_root=/path/to/collections
uv run --extra train --extra facodec train-probe \
  data_root=/path/to/collections f0=cached f0_backend=world codes=cached head=contextual
```

The 17-frame contextual head sees roughly 100 ms on either side of each frame.
With both caches selected, the probe reads F0 and code IDs without loading audio.
It reconstructs the same post-VQ prosody vectors with a frozen codebook lookup.

The cache is tied to the dataset and extractor identity. Do not reuse it for changed weights or audio revisions without an appropriate new cache identity/output directory. QC supports `cache_key=my-revision` for additional provenance when needed.

## 4. Training and evaluation

Default splits: `train-clean-100` → `dev-clean`. Default model: frozen FACodec and `LinearF0Head(input_dim=256)`. Optimization is selected through the `optimization` Hydra group; training runs via QC's `TrainingPipeline`, `LightningTrainer`, and a project-specific training module.

Reported metrics are computed **only on valid voiced frames**:

- `normalized_log2_f0_mse` and `normalized_log2_f0_mae`: frame-weighted errors in normalized log2 pitch, **not Hz**.
- `normalized_log2_f0_pearson`: average utterance-level Pearson correlation for utterances with sufficient voiced variation.
- `voiced_frames` and `pearson_utterances`: denominators for interpreting those metrics.

Validation also logs example pitch curves when a W&B logger is available. Lower error and higher correlation support pitch decodability; they do not measure identity privacy or perceptual quality.

## 5. Reproduction checks and limitations

```bash
uv run --extra train --extra facodec pytest -q
FACODEC_PARITY=1 uv run --extra train --extra facodec pytest -q tests/test_facodec_parity.py
```

Check that cached targets match online targets for the same recordings, particularly at padded boundaries; verify `[T,1]` tensors, valid lengths, and consistent cache paths. The pretrained JDC checkpoint's upstream match does **not** establish exact identity with the labels used to train FACodec. Recording-wise normalization is an approximation of segment-wise target normalization used in related implementations. LibriSpeech is not a clinical speech dataset, so these results cannot establish preservation of disease cues.

**Next scientific step:** intervention tests that manipulate the codec's prosody representation, synthesize audio, and measure whether F0 changes while speaker identity, content, and other attributes remain stable. This has not been established by the linear probe.
