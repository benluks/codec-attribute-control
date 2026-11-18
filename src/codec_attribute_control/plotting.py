"""Validation pitch-contour visualization in absolute Hz."""
import numpy as np

from .targets import f0_target


def pitch_curves_hz(reference_hz, predicted_z, length):
    """Return masked reference and inverse-normalized prediction in Hz.

    Prediction is reconstructed with the reference segment's log2-F0 statistics,
    not independently estimated absolute-pitch statistics.
    """
    reference = np.asarray(reference_hz[:length], dtype=np.float64)
    predicted = np.asarray(predicted_z[:length], dtype=np.float64)
    normalized, stats = f0_target(reference)
    voiced = normalized != -10.0
    actual = np.where(voiced, reference, np.nan)
    reconstructed = np.full(reference.shape, np.nan, dtype=np.float64)
    reconstructed[voiced] = np.exp2(predicted[voiced] * stats.std_log2_hz + stats.mean_log2_hz)
    return actual, reconstructed


def make_f0_figure(reference_hz, predicted_z, lengths, max_items=8):
    """One subplot per utterance; NaNs break lines at unvoiced frames."""
    import matplotlib.pyplot as plt

    curves = []
    for reference, predicted, length in zip(reference_hz, predicted_z, lengths):
        try:
            curves.append(pitch_curves_hz(reference, predicted, int(length)))
        except ValueError:
            continue
        if len(curves) >= max_items:
            break
    if not curves:
        return None

    fig, axes = plt.subplots(len(curves), 1, figsize=(12, 2.3 * len(curves)), squeeze=False)
    for index, ((actual, predicted), axis) in enumerate(zip(curves, axes[:, 0])):
        time = np.arange(len(actual)) / 80.0
        axis.plot(time, actual, label="Reference F0", linewidth=1.4)
        axis.plot(time, predicted, label="Predicted F0 (reference-calibrated)", linewidth=1.2)
        axis.set_ylabel(f"#{index + 1} · Hz")
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
