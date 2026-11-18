# quick-convert reuse and missing abstractions

This experiment intentionally imports quick-convert as a dependency; it does not
copy its audio utilities or its YAAPT F0 implementation.

## Reused

- `quick_convert.utils.audio.load_audio`: mono decoding and 16 kHz resampling.
- `quick_convert.components.features.f0.yaapt.YAAPTF0Extractor`: optional
  F0 backend, if `amfm-decompy` is installed.

## Implemented here, and why

- FACodec pretrained encoder/decoder adapter: no stable, published FACodec
  post-quantization decomposition interface was identified in quick-convert.
  This needs upstream checkpoint and shape tests before promoting it.
- `f0_target`: FACodec-specific log2/voiced/per-segment normalization and
  unvoiced sentinel; not appropriate as a generic F0 extractor default.
- Ridge probe, recording-disjoint evaluation, and minimum-norm interventions:
  experimental analysis rather than a reusable training system. Consider
  upstream generic frozen-feature probe and intervention protocols later.
- Frame timestamp alignment: pitch and codec features may have different hop
  sizes. A generic typed feature/timebase alignment utility would be valuable.
- CLI and result JSON: intentionally small until a stable quick-convert
  experiment/evaluation runner is suitable.

## Potential upstream improvements

1. A common codec adapter contract exposing quantized factor contributions,
   combined decoder input, speaker conditioning, sample rate, and hop.
2. A generic F0-target transform with explicit statistics scope, voiced mask,
   normalization provenance, and timestamp alignment.
3. Frozen feature extraction with recording-disjoint split and cache provenance.
4. Generic latent-direction intervention and resynthesis evaluation.

## Known limitations

- No model/checkpoint integration test has run yet.
- F0 extraction uses pYIN by default, not the exact upstream FACodec training
  pitch extractor; target normalization matches the released training recipe,
  but extractor parity and crop/window parity are not established.
- Per-recording normalization approximates per-crop training normalization
  only when each recording is treated as one segment.
- A constant absolute pitch shift cancels under per-segment log-F0 z-scoring.
  The semitone intervention deliberately holds source statistics fixed and
  must not be presented as a learned absolute-pitch control.
- Random direction controls, disentanglement metrics, and calibration are
  follow-up work.
