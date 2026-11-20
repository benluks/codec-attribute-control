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
uv run --extra w2vbert --extra facodec python examples/prosody_editor/world_server.py
```

Use the same SSH forwarding as above. Select **W2v-BERT + CNN probe (cluster)**,
edit the blue F0 curve, then explicitly click **Convert utterance**. The server
resamples the browser's approximately 80 Hz contour to the 50 Hz SSL feature
clock, masks unvoiced frames, optimizes the continuous latents with Adam, and
runs the original SSLR decoder/vocoder. Nothing runs automatically on drag.

The server loads the two models for each conversion request; this first version
prioritizes correctness over response time. The orange realized F0 remains a
browser estimate. Model quality and checkpoint compatibility must be validated
on the cluster. Do not expose this endpoint publicly.
