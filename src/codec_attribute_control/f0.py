"""Frame alignment and F0 extractor selection."""
from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as audio_functional

from .facodec import HOP, SAMPLE_RATE


def align_f0_to_feature_grid(
    f0_hz,
    f0_lengths,
    feature_lengths,
    n_frames,
    *,
    source_hz=80,
    target_hz=50,
    target_offset_seconds=0.0,
):
    """Nearest-neighbor resample F0 labels onto SSL feature timestamps.

    Invalid/padded frames stay NaN; source timestamps are measured from the
    start of the utterance. No interpolation across unvoiced regions.
    """
    if f0_hz.ndim != 2 or f0_lengths.ndim != 1 or feature_lengths.ndim != 1:
        raise ValueError("Expected F0 [B,T] and frame lengths [B]")
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("Frame rates must be positive")
    t = torch.arange(n_frames, device=f0_hz.device)
    if (
        f0_lengths.shape != feature_lengths.shape
        or f0_lengths.numel() != f0_hz.shape[0]
    ):
        raise ValueError("Expected one source and feature length per utterance")
    if torch.any(f0_lengths < 0) or torch.any(f0_lengths > f0_hz.shape[1]):
        raise ValueError("Invalid F0 lengths")
    if torch.any(feature_lengths < 0) or torch.any(feature_lengths > n_frames):
        raise ValueError("Invalid feature lengths")
    if f0_hz.shape[1] == 0:
        return f0_hz.new_full((f0_hz.shape[0], n_frames), float("nan")), feature_lengths
    index = torch.floor(
        (t / target_hz + target_offset_seconds) * source_hz + 0.5
    ).long()
    index = torch.minimum(
        index.clamp_min(0)[None, :], (f0_lengths[:, None] - 1).clamp_min(0)
    )
    values = f0_hz.gather(1, index)
    valid = (t[None, :] < feature_lengths[:, None]) & (f0_lengths[:, None] > 0)
    return values.masked_fill(~valid, float("nan")), feature_lengths


def aligned_pitch(extractor, waveform: torch.Tensor, sample_rate: int, n_frames: int) -> np.ndarray:
    """Nearest align Hz F0 from a quick-convert extractor to FACodec's 80Hz grid."""
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono [samples], got {tuple(waveform.shape)}")
    native_sr = getattr(extractor, "sample_rate", sample_rate)
    if sample_rate != native_sr:
        waveform = audio_functional.resample(waveform, sample_rate, native_sr)
    raw = extractor.extract(waveform, native_sr).detach().cpu().float().numpy()
    if not raw.size:
        return np.full(n_frames, np.nan, dtype=np.float32)
    hop_seconds = (extractor.hop_length / native_sr
                   if hasattr(extractor, "hop_length")
                   else extractor.frame_space / 1000.0)
    t = np.arange(n_frames) * HOP / SAMPLE_RATE
    index = np.clip(np.rint(t / hop_seconds).astype(int), 0, len(raw) - 1)
    pitch = raw[index].copy()
    pitch[~np.isfinite(pitch) | (pitch <= 5.0)] = np.nan
    return pitch


def make_extractor(backend: str, *, device: str = "cpu", checkpoint_path: str | None = None):
    if backend in ("jdc", "jdc_corrected"):
        from .extractors import JDCF0Extractor
        kwargs = {"mel_sample_rate": SAMPLE_RATE} if backend == "jdc_corrected" else {}
        return JDCF0Extractor(checkpoint_path=checkpoint_path, device=device, **kwargs)
    if backend == "praat":
        from .extractors import PraatF0Extractor
        return PraatF0Extractor()
    if backend == "world":
        from .extractors import WORLDF0Extractor

        return WORLDF0Extractor()
    if backend == "yaapt":
        from quick_convert.components.features.f0.yaapt import YAAPTF0Extractor
        return YAAPTF0Extractor(zero_on_failure=False)
    raise ValueError(f"F0 backend must be 'praat', 'world', 'jdc_corrected', 'jdc' or 'yaapt', got {backend!r}")
