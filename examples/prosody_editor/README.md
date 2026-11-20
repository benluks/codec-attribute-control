# Cluster-hosted WORLD editor

This is a minimal proof-of-concept: the browser edits the F0 contour, and
the server analyzes the uploaded waveform with WORLD (DIO + StoneMask,
CheapTrick, D4C), substitutes the target F0, and resynthesizes. No learned
pitch-to-code model or FACodec decoder is involved.

On the cluster (in a login or interactive allocation where permitted):

```bash
uv sync --extra facodec
uv run --extra facodec python examples/prosody_editor/world_server.py
```

On your laptop:

```bash
ssh -N -L 8765:127.0.0.1:8765 YOUR_CLUSTER_HOST
```

Open http://127.0.0.1:8765, load the bundled OGG example (or another short
audio file), choose **WORLD (cluster backend)**, edit the blue contour, and
press **Convert utterance**. The output is playable with **Converted**.

The server binds to 127.0.0.1 by default; use SSH forwarding rather than
exposing the endpoint to the public Internet. For compute nodes, forward via
the allocated node (or a two-hop SSH tunnel) as required by the cluster.

Limits: mono first-channel input, 30 seconds maximum, 20 MB request maximum.
WORLD's original voicing decision is preserved: editing cannot create voiced
frames in originally unvoiced regions. The orange realized contour is still
the browser's lightweight estimator, not a scientific evaluation metric.

## W2v-BERT latent conversion (experimental)

The same editor can now submit its **edited absolute F0 contour** to the frozen
CNN probe and SSL reconstruction decoder. Set both checkpoint locations on the
cluster before starting the server:

```bash
export SSLR_CHECKPOINT=/path/to/outputs/ssl_reconstruction/hf8jdifd
export F0_PROBE_CHECKPOINT=/path/to/hmts1hb9/checkpoints/best.ckpt
export SSLR_DEVICE=cuda
export SSLR_INVERSION_STEPS=80
uv run --extra sslr --extra facodec --extra train python examples/prosody_editor/world_server.py
```

Use the same SSH forwarding as above. Select **W2v-BERT + CNN probe (cluster)**,
edit the blue F0 curve, then explicitly click **Convert utterance**. The server
resamples the browser's approximately 80 Hz contour to the 50 Hz SSL feature
clock, masks unvoiced frames, inverts the continuous latents, and
runs the original SSLR decoder/vocoder. Nothing runs automatically on drag.

Choose **Adam** or **Jacobian** under **Inversion**. Adam is the default.
Jacobian uses one regularized local step and is intended for short utterances.
If the server reports that the Jacobian is too large, shorten the clip or
select Adam. `SSLR_INVERSION_STEPS` controls Adam.

The probe's saved run config selects the edit objective. For `hmts1hb9`
(`normalized_log2`), the server standardizes the edited contour over frames
voiced in both source and target, using the probe's training normalization.
Change the relative pitch trajectory, for example by adding a local rise.
Uniform shifts and global scaling in log-pitch leave the standardized
contour unchanged. Absolute `log2` probes still receive targets in log2 Hz.

The server loads the two models for each conversion request; this first version
prioritizes correctness over response time. The orange realized F0 remains a
browser estimate. Model quality and checkpoint compatibility must be validated
on the cluster. Do not expose this endpoint publicly.

## PitchFlower pretrained codec (experimental)

PitchFlower uses an RVQ autoencoder, flow-based mel decoder conditioned on
**absolute log2 F0**, and a pretrained Vocos vocoder. The adapter follows the
[upstream inference notebook](https://github.com/diegotg2000/PitchFlower/blob/master/f0_control.ipynb),
including its 24 kHz synthesis clock and unvoiced log-F0 value of zero.

Quick Convert pins PyTorch 2.6 while upstream PitchFlower pins 2.8.
To avoid mixing them, create a separate PitchFlower environment **once**,
and point the regular editor server at that Python executable:

```bash
uv venv --python 3.11 .venv-pitchflower
uv pip install --python .venv-pitchflower/bin/python \
  'git+https://github.com/diegotg2000/PitchFlower.git'
export PITCHFLOWER_PYTHON="$(pwd)/.venv-pitchflower/bin/python"

# Editor dependencies (separate main environment):
uv sync --extra facodec
uv run --extra facodec python examples/prosody_editor/world_server.py
```

Use the same SSH tunnel as for WORLD. Select **PitchFlower (pretrained codec)**,
load audio, edit the blue curve, then click **Convert utterance**. The editor
starts a separate persistent PitchFlower worker on first use. PitchFlower
and Vocos are downloaded from Hugging Face on first conversion and cached in
memory for subsequent requests. No manually provided checkpoints are needed.

The optional configuration is:

```bash
export PITCHFLOWER_DEVICE=cuda  # cpu is also supported
export PITCHFLOWER_MODEL=diegotg343/PitchFlower
export PITCHFLOWER_VOCODER=charactr/vocos-mel-24khz
export PITCHFLOWER_STEPS=10
export PITCHFLOWER_GUIDANCE=3.0
```

The browser's F0 is an approximate 80 Hz estimate, interpolated only between
voiced points to PitchFlower's mel clock; unvoiced frames remain zero. The orange
curve is likewise an approximate browser estimate, not a validated pitch metric.
The server runs synthesis only on explicit conversion, never on edit release.

**Licensing:** Upstream PitchFlower is licensed under CC BY-NC-SA 4.0; check its
terms before commercial usage or redistribution.
