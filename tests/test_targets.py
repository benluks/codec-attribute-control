import numpy as np
import pytest

from codec_attribute_control.targets import f0_target, ridge_intervention, semitone_delta_z


def test_normalization_uses_voiced_log2_sample_std():
    hz = np.array([100.0, 200.0, 400.0, np.nan, 0.0])
    target, stats = f0_target(hz)
    np.testing.assert_allclose(target[:3].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(target[:3].std(ddof=1), 1, atol=1e-6)
    assert target[3] == target[4] == -10
    assert stats.count == 3


def test_constant_pitch_shift_uses_fixed_source_stats():
    hz = np.array([100.0, 150.0, 200.0, 250.0])
    delta = semitone_delta_z(hz, 12)
    _, stats = f0_target(hz)
    np.testing.assert_allclose(delta, 1 / stats.std_log2_hz)


def test_ridge_direction_changes_probe_by_requested_delta():
    w = np.array([1.0, 2.0, -1.0])
    requested = np.array([-1.0, 0.0, 2.0])
    edit = ridge_intervention(w, requested)
    np.testing.assert_allclose(edit @ w, requested, atol=1e-6)


def test_degenerate_direction_rejected():
    with pytest.raises(ValueError):
        ridge_intervention(np.zeros(3), np.ones(2))
