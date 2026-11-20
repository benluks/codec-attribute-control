# SSL reconstruction identity smoke test

This is the first inference milestone: restore the **entire** SSL reconstruction
system from its saved Quick Convert configuration and checkpoint, then synthesize
audio without modifying its latent features.

```bash
uv run --extra w2vbert reconstruct-sslr \
  --checkpoint /path/to/quick-convert/outputs/ssl_reconstruction/hf8jdifd \
  --source /path/to/utterance.wav \
  --output /tmp/identity.wav \
  --device cuda
```

Use `--config /path/to/saved/config.yaml` if the checkpoint does not embed its
inference composition. The decoder and vocoder must be configured in the saved
system; this command does not construct a substitute vocoder.

**Important:** This does not yet load the separately trained CNN pitch probe,
modify the latent, or measure synthesized F0. Check the identity waveform
before proceeding. The native loader may require model assets and optional
dependencies from the original Quick Convert configuration.
