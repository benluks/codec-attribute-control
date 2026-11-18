# quick-convert reuse and remaining gaps

This experiment follows the downstream-library pattern of
[speech-semantic-vectors](https://github.com/benluks/speech-semantic-vectors):
compose quick-convert's published interfaces instead of duplicating them.

## Reused directly

- `quick_convert.data.BaseDataset`: discovery, lazy audio loading, mono
  downmix and 16kHz resampling.
- `quick_convert.utils.audio.load_audio`: loading at the JDC native 24kHz
  when required, avoiding duplicate decoding logic.
- `quick_convert.components.features.f0.base.F0Extractor`: shared abstract
  base; local `JDCF0Extractor` subclasses this.
- `quick_convert.components.features.f0.yaapt.YAAPTF0Extractor`:
  fallback directly instantiated, without local wrapper subclass.
- `quick_convert.components.resources.base.BaseResourceExtractor`: inherited
  feature_name, extract_sample, extract_batch and provide_sample contracts.

## Local code and reasons

- JDC implementation: original pretrained model architecture, checkpoint,
  mel preprocessing, and sample rate are backend-specific. The backend is a
  proper QC feature extractor; no new generic interface is introduced.
- FACodec encoder/decoder adapter: NS3 post-VQ factor decomposition and
  speaker-conditioned decoding have no matching published QC class.
- Codec↔pitch timebase alignment: currently per-backend; QC could provide
  an explicit timebase alignment primitive.
- Voiced log2-F0 normalization: belongs in experimental label transforms,
  not in an F0 extractor that returns physical Hz.
- Frozen ridge probe and minimum-norm latent edit: currently a small
  experiment; generic QC support might help if repeated across codecs.

## Upstream candidate improvements

1. Typed codec-factor contributions and codec decoder inputs with timebase.
2. Timebase-aware feature alignment (target frame centers and valid lengths).
3. Label normalization provenance, including extractor/checkpoint, normalization
   scope, and crop/window length.
4. A generalized frozen linear probe/evaluator.

## Validation and caveats

- FACodec NS3 and Plachtaa's FAcodec are different implementations. The JDC
  checkpoint Git SHA is identical between Plachtaa and Amphion's facodec assets.
  This does not prove it was used to supervise the pretrained NS3 codec.
- The Plachtaa one-second training crop does not establish the NS3 crop size.
  This scaffold currently normalizes each full recording rather than crops.
- Ordinary unvoiced frames are zero in Plachtaa's training normalization;
  -10 is reserved for invalid values. Our regression excludes UV entirely.
- JDC pitch output is not independently evaluated here; YAAPT can be selected
  for an alternative label source.
- Pretrained FACodec model loading and end-to-end audio synthesis have not
  yet been runtime-validated.
- Absolute semitone control is extrapolative after per-segment normalization.
