"""SSLR conversion backend for the browser pitch editor."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf


def synthesize_sslr(payload):
    import torch
    from codec_attribute_control.reconstruct import reconstruct_identity

    checkpoint = os.environ.get("SSLR_CHECKPOINT")
    probe = os.environ.get("F0_PROBE_CHECKPOINT")
    if not checkpoint or not probe:
        raise ValueError("Set SSLR_CHECKPOINT and F0_PROBE_CHECKPOINT")
    sr = int(payload["sample_rate"])
    audio = np.asarray(payload["audio"], dtype=np.float32)
    if not 8000 <= sr <= 96000 or audio.ndim != 1 or not 0 < audio.size <= sr * 30:
        raise ValueError("Invalid source audio")
    if not np.isfinite(audio).all():
        raise ValueError("Nonfinite source audio")
    edits = payload["target_f0"]
    if not isinstance(edits, list) or not 2 <= len(edits) <= 10000:
        raise ValueError("Invalid target contour")
    target = np.asarray([np.nan if x is None else float(x) for x in edits])
    if np.any(np.isfinite(target) & ((target <= 0) | (target > 1500))):
        raise ValueError("Voiced F0 must be in (0, 1500] Hz")
    frame_period = float(payload.get("target_frame_period", 0.0125))
    if not np.isfinite(frame_period) or not 0 < frame_period <= 0.1:
        raise ValueError("Invalid target frame period")
    method = payload.get("inversion_method", "adam")
    if method not in ("adam", "jacobian"):
        raise ValueError("Choose Adam or Jacobian inversion")
    steps = int(os.environ.get("SSLR_INVERSION_STEPS", "80"))
    if not 1 <= steps <= 1000:
        raise ValueError("Invalid SSLR_INVERSION_STEPS")
    device = os.environ.get("SSLR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.wav"
        output = Path(tmp) / "converted.wav"
        sf.write(source, audio, sr)
        reconstruct_identity(
            source, checkpoint, output,
            device=device, probe_checkpoint=probe,
            target_f0=target, target_frame_period=frame_period, steps=steps, inversion_method=method,
        )
        return output.read_bytes()
