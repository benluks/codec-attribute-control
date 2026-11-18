"""Pitch extraction and explicit timestamp alignment to codec frame centers."""
import numpy as np
import torch

from .facodec import SAMPLE_RATE, HOP


def extract_pitch(waveform: torch.Tensor, n_frames: int, backend: str = "pyin") -> np.ndarray:
    """Output Hz with NaN for unvoiced, [n_frames].

    pyin: 12.5ms hop / aligned nearest voiced mask; accurate but slow.
    yaapt: quick-convert's YAAPTF0Extractor with 10ms hop (optional extra).
    """
    y = waveform.detach().cpu().float().reshape(-1)
    if backend == "yaapt":
        from quick_convert.components.features.f0.yaapt import YAAPTF0Extractor
        extractor = YAAPTF0Extractor(zero_on_failure=False)
        f = extractor.extract(y, SAMPLE_RATE).numpy()
        hop = int(SAMPLE_RATE * extractor.frame_space / 1000)
        t_ref = np.arange(f.size) * hop / SAMPLE_RATE
        voiced = np.isfinite(f) & (f > 5)
    elif backend == "pyin":
        import librosa
        f, voiced, _ = librosa.pyin(y.numpy(), sr=SAMPLE_RATE, hop_length=HOP,
                                    frame_length=1024, fmin=65, fmax=900)
        t_ref = np.arange(len(f)) * HOP / SAMPLE_RATE
        voiced = np.asarray(voiced, bool) & np.isfinite(f)
    else:
        raise ValueError(f"Unknown F0 backend: {backend}")
    if len(f) == 0:
        return np.full(n_frames, np.nan, dtype=np.float32)
    t_codec = (np.arange(n_frames) + 0.5) * HOP / SAMPLE_RATE
    right = np.clip(np.searchsorted(t_ref, t_codec), 0, len(f) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(t_ref[left] - t_codec) <= np.abs(t_ref[right] - t_codec), left, right)
    aligned = np.asarray(f, dtype=np.float32)[nearest]
    aligned[~voiced[nearest]] = np.nan
    return aligned
