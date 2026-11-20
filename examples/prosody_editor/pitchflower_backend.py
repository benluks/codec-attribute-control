"""Optional PitchFlower backend for the shared browser F0 editor.

PitchFlower's public notebook conditions its flow decoder on absolute log2 F0
at the 24 kHz mel clock; unvoiced frames are represented by zero.
"""
from functools import lru_cache
from io import BytesIO
import os
from threading import Lock

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
MAX_SECONDS = 30
_MODEL_LOCK = Lock()


def _request(payload):
    """Validate input before importing or downloading a model."""
    sr = int(payload["sample_rate"])
    audio = np.asarray(payload["audio"], dtype=np.float32)
    if not 8000 <= sr <= 96000 or audio.ndim != 1 or not 0 < audio.size <= sr * MAX_SECONDS:
        raise ValueError("Expected mono audio of at most 30 seconds at 8–96 kHz")
    if not np.isfinite(audio).all():
        raise ValueError("Nonfinite source audio")
    edits = payload["target_f0"]
    if not isinstance(edits, list) or not 2 <= len(edits) <= 10000:
        raise ValueError("Invalid target F0 contour")
    contour = np.asarray([np.nan if value is None else float(value) for value in edits], dtype=np.float64)
    if np.any(np.isinf(contour)) or np.any(np.isfinite(contour) & ((contour <= 0) | (contour > 1500))):
        raise ValueError("Voiced target F0 must be in (0, 1500] Hz; unvoiced must be null")
    frame_period = float(payload.get("target_frame_period", 0.0125))
    if not np.isfinite(frame_period) or not 0 < frame_period <= 0.1:
        raise ValueError("Invalid target frame period")
    return sr, audio, contour, frame_period


def target_log2_f0(contour, frame_period, frames, hop_length, sample_rate=SAMPLE_RATE):
    """Sample an editor F0 contour on the model's mel clock.

    Interpolate only between two voiced browser frames. Null or nonpositive
    values represent unvoiced frames (log2 F0 = 0 in PitchFlower).
    """
    if frames <= 0 or hop_length <= 0 or sample_rate <= 0:
        raise ValueError("Invalid mel frame grid")
    t = np.arange(frames, dtype=np.float64) * hop_length / sample_rate
    index = t / frame_period
    # Beyond the editor contour is unvoiced, not a held-last F0.
    inside = index <= len(contour) - 1
    left = np.clip(np.floor(index).astype(np.int64), 0, len(contour) - 1)
    right = np.minimum(left + 1, len(contour) - 1)
    weight = np.clip(index - left, 0.0, 1.0)
    a, b = contour[left], contour[right]
    valid = inside & np.isfinite(a) & (a > 0) & (
        (np.isclose(weight, 0.0, atol=1e-9))
        | (np.isfinite(b) & (b > 0))
    )
    hz = np.where(valid, np.nan_to_num(a) * (1 - weight) + np.nan_to_num(b) * weight, 1.0)
    return np.where(valid, np.log2(hz), 0.0).astype(np.float32)


def _model_device():
    import torch

    configured = os.environ.get("PITCHFLOWER_DEVICE")
    device = configured or ("cuda" if torch.cuda.is_available() else "cpu")
    return device


@lru_cache(maxsize=2)
def _models(model_id, vocoder_id, device):
    try:
        from pitchflower.synthesizer import PitchFlowerSynthesizer
        from vocos import Vocos
    except ImportError as exc:
        raise RuntimeError(
            "PitchFlower is not installed. Install the optional 'pitchflower' extra."
        ) from exc
    synth = PitchFlowerSynthesizer.from_pretrained(model_id).to(device).eval()
    vocos = Vocos.from_pretrained(vocoder_id).to(device).eval()
    return synth, vocos


def _synthesize_in_worker(payload):
    """Run the pretrained codec and vocoder inside the dedicated worker environment."""
    import torch
    from scipy.signal import resample_poly
    from math import gcd

    sr, audio, contour, frame_period = _request(payload)
    steps = int(os.environ.get("PITCHFLOWER_STEPS", "10"))
    guidance = float(os.environ.get("PITCHFLOWER_GUIDANCE", "3.0"))
    if not 1 <= steps <= 100 or not np.isfinite(guidance) or not 0 <= guidance <= 10:
        raise ValueError("Invalid PitchFlower steps or guidance")
    model_id = os.environ.get("PITCHFLOWER_MODEL", "diegotg343/PitchFlower")
    vocoder_id = os.environ.get("PITCHFLOWER_VOCODER", "charactr/vocos-mel-24khz")
    device = _model_device()
    divisor = gcd(sr, SAMPLE_RATE)
    samples = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor)
    waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0).to(device)

    # Shared models (especially CUDA ones) are cached and run serially.
    with _MODEL_LOCK, torch.inference_mode():
        synth, vocos = _models(model_id, vocoder_id, device)
        mel = synth.get_mels(waveform)
        hop = synth.mel.spectrogram.hop_length
        model_sr = synth.mel.sample_rate
        if model_sr != SAMPLE_RATE:
            raise ValueError(f"Expected PitchFlower model at 24 kHz, got {model_sr}")
        log_f0 = target_log2_f0(contour, frame_period, mel.shape[-1], hop, model_sr)
        pitches = torch.from_numpy(log_f0).unsqueeze(0).to(device)
        # Match the official notebook's synthesis entrypoint and vocoder.
        rendered = vocos.decode(synth.reconstruct_mels(waveform, pitches, steps=steps, w=guidance))
        if rendered.ndim != 2 or rendered.shape[0] != 1:
            raise ValueError(f"Unexpected PitchFlower audio shape {tuple(rendered.shape)}")
        output = rendered[0].detach().cpu().numpy()

    if output.size == 0 or not np.isfinite(output).all():
        raise ValueError("PitchFlower returned invalid audio")
    # Clip to the original duration; mel/vocoder padding may produce a short tail.
    output = output[:round(audio.size * SAMPLE_RATE / sr)]
    out = BytesIO()
    sf.write(out, np.clip(output, -1, 1), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return out.getvalue()


# The main editor uses Quick Convert (torch==2.6) while PitchFlower pins
# torch==2.8. Keep them in independent Python environments instead of trying
# to resolve incompatible requirements into the same uv environment.
_WORKER_PROCESS = None
_WORKER_PROCESS_LOCK = Lock()


def _start_worker():
    import subprocess
    from pathlib import Path

    interpreter = os.environ.get("PITCHFLOWER_PYTHON")
    if not interpreter:
        raise ValueError(
            "Set PITCHFLOWER_PYTHON to the Python executable in a separate "
            "PitchFlower environment (PyTorch 2.8); see editor README"
        )
    executable = Path(interpreter).expanduser().resolve()
    if not executable.is_file():
        raise ValueError(f"PITCHFLOWER_PYTHON not found: {executable}")
    return subprocess.Popen(
        [str(executable), str(Path(__file__).with_name("pitchflower_worker.py"))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # Allow model logs and tracebacks to reach the editor server console,
        # never the JSON line protocol on stdout.
        stderr=None,
        text=True,
        bufsize=1,
    )


def synthesize_pitchflower(payload):
    """Send conversion to a persistent isolated worker, returning WAV bytes."""
    import base64
    import json

    global _WORKER_PROCESS
    _request(payload)  # validate before launching an expensive process
    with _WORKER_PROCESS_LOCK:
        if _WORKER_PROCESS is None or _WORKER_PROCESS.poll() is not None:
            _WORKER_PROCESS = _start_worker()
        try:
            _WORKER_PROCESS.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            _WORKER_PROCESS.stdin.flush()
            line = _WORKER_PROCESS.stdout.readline()
            if not line:
                raise RuntimeError("PitchFlower worker exited without a response; check server logs")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(response["error"])
            return base64.b64decode(response["wav"], validate=True)
        except (OSError, BrokenPipeError):
            _WORKER_PROCESS.kill()
            _WORKER_PROCESS = None
            raise RuntimeError("PitchFlower worker connection failed; check server logs") from None
