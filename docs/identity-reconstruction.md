# SSLR reconstruction and pitch editing

Restore a trusted Quick Convert SSL reconstruction checkpoint and save the
source reconstruction as a WAV:

```bash
uv run --extra sslr reconstruct-sslr \
  --checkpoint "$HOME/quick-convert/outputs/ssl_reconstruction/hf8jdifd" \
  --source /path/to/utterance.flac \
  --output /tmp/identity.wav --device cuda
```

Use `--config /path/to/config.yaml` to override the saved system configuration.
The command translates the legacy W2V-BERT, CAMPPlus and CosyVoice class paths
found in older run configs. Quick Convert restores the saved decoder and
vocoder; its model assets must be available locally or downloadable.

To run inversion and render the edited latent, supply a trained
`head=facodec_cnn target_mode=log2` probe from the same SSLR representation.
Initialize the Amphion submodule before using the CNN head:

```bash
git submodule update --init external/Amphion
uv run --extra sslr --extra facodec reconstruct-sslr \
  --checkpoint "$HOME/quick-convert/outputs/ssl_reconstruction/hf8jdifd" \
  --probe-checkpoint /path/to/probe/checkpoints/best.ckpt \
  --source /path/to/utterance.flac \
  --semitones 3 --output /tmp/edited.wav --device cuda
```

`--semitones` shifts the frozen probe's prediction on WORLD-voiced frames.
The decoded waveform may realize a different shift. Use a separate F0
measurement on the generated WAV to evaluate the edit.
