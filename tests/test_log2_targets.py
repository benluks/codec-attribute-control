import numpy as np
import pytest
import torch

from codec_attribute_control.supervision import prepare_f0_targets
from codec_attribute_control.plotting import pitch_curves_hz


def test_absolute_log2_targets_keep_pitch_offset_and_mask_unvoiced():
    hz = torch.tensor([[100., 200., float("nan"), 0., 400.], [200., 400., 800., 0., 0.]])
    values, valid = prepare_f0_targets(hz, torch.tensor([5, 3]), target_mode="log2")
    assert valid.tolist() == [[True, True, False, False, True], [True, True, True, False, False]]
    assert torch.allclose(values[0, valid[0]], torch.log2(hz[0, valid[0]]))
    assert torch.allclose(values[1, valid[1]] - values[0, valid[0]], torch.ones(3))


def test_absolute_mode_accepts_constant_pitch():
    values, valid = prepare_f0_targets(torch.full((1, 3), 200.), torch.tensor([3]), target_mode="log2")
    assert valid.all()
    assert torch.allclose(values, torch.full((1, 3), np.log2(200.)))


def test_absolute_pitch_plot_inverse():
    hz = np.array([100., 200., np.nan, 400.])
    predicted = np.array([np.log2(200.), np.log2(400.), 0., np.log2(800.)])
    actual, reconstructed = pitch_curves_hz(hz, predicted, 4, target_mode="log2")
    np.testing.assert_allclose(reconstructed[[0, 1, 3]], [200., 400., 800.])
    assert np.isnan(actual[2]) and np.isnan(reconstructed[2])


def test_unknown_target_mode():
    with pytest.raises(ValueError):
        prepare_f0_targets(torch.ones(1, 2), torch.tensor([2]), target_mode="hz")
