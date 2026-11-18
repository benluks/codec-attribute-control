"""FACodec F0 supervision targets and scientifically distinct intervention units.

Published FACodec training code (Plachtaa/FAcodec train.py, lines 218-256):
  - use voiced F0 > 5 Hz;
  - log2 transform voiced F0;
  - normalize per sampled training *segment* with torch.std (unbiased=True);
  - unvoiced frames = -10, used by the original F0 loss.
This probe fits only voiced frames, so it does not learn the -10 sentinel.
"""
from dataclasses import dataclass

import numpy as np

TARGET_ID = "facodec_segment_voiced_log2_f0_zscore_v1"


@dataclass(frozen=True)
class PitchStats:
    mean_log2_hz: float
    std_log2_hz: float
    count: int


def f0_target(f0_hz: np.ndarray) -> tuple[np.ndarray, PitchStats]:
    """Compute per-segment log2-Hz z-score; UV sentinel is -10.

    The upstream training script computes torch.std() with Bessel's correction,
    so ddof=1 here, not the NumPy default ddof=0.
    """
    f0 = np.asarray(f0_hz, dtype=np.float64)
    mask = np.isfinite(f0) & (f0 > 5.0)
    if mask.sum() < 3:
        raise ValueError("Need >=3 voiced F0 frames to normalize segment")
    voiced = np.log2(f0[mask])
    mean = float(voiced.mean())
    std = float(voiced.std(ddof=1))
    if not np.isfinite(std) or std < 1e-6:
        raise ValueError("Near-constant F0 in segment")
    result = np.full(f0.shape, -10.0, dtype=np.float32)
    result[mask] = ((voiced - mean) / std).astype(np.float32)
    return result, PitchStats(mean, std, int(mask.sum()))


def contour_delta(f0_hz: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    """Desired change in *normalized contour* (a supervised FACodec target).

    No guarantee of absolute pitch shift: adding a constant to log2 F0 is
    annihilated by per-segment z-scoring.
    """
    observed, _ = f0_target(f0_hz)
    target_z = np.asarray(target_z, dtype=float)
    if target_z.shape != observed.shape:
        raise ValueError("target contour shape mismatch")
    valid = observed != -10.0
    delta = np.zeros_like(observed)
    delta[valid] = target_z[valid] - observed[valid]
    return delta


def semitone_delta_z(f0_hz: np.ndarray, semitones: float) -> np.ndarray:
    """Exploratory absolute shift in segment-z units, holding source stats fixed.

    A constant semitone shift would disappear if target segment stats were
    recomputed. This mapping is *not* what FACodec's normalized objective learned.
    """
    target, stats = f0_target(f0_hz)
    delta = np.zeros_like(target)
    delta[target != -10.0] = float(semitones) / (12.0 * stats.std_log2_hz)
    return delta


def ridge_intervention(coef: np.ndarray, delta_z: np.ndarray) -> np.ndarray:
    """[frames, channels] minimum-norm probe-space intervention."""
    w = np.asarray(coef, dtype=np.float32).reshape(-1)
    d = np.asarray(delta_z, dtype=np.float32).reshape(-1)
    norm2 = float(np.dot(w, w))
    if not np.isfinite(norm2) or norm2 <= 1e-12:
        raise ValueError("Degenerate probe direction")
    return (d[:, None] * w[None, :] / norm2).astype(np.float32)
