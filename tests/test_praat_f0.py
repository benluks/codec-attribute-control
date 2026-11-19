"""Praat backend contract, using a fake parselmouth module."""

import sys
from types import SimpleNamespace

import numpy as np
import torch

from codec_attribute_control.extractors import PraatF0Extractor
from codec_attribute_control.f0 import make_extractor


def test_praat_uses_amphion_parameters_and_real_frame_timestamps(monkeypatch):
    calls = {}

    class FakePitch:
        selected_array = {"frequency": np.array([100., 0., 200.])}

        def xs(self):
            return np.array([0.0125, 0.025, 0.0375])

    class FakeSound:
        def __init__(self, audio, sampling_frequency):
            calls["sr"] = sampling_frequency

        def to_pitch_ac(self, **kwargs):
            calls.update(kwargs)
            return FakePitch()

    monkeypatch.setitem(sys.modules, "parselmouth", SimpleNamespace(Sound=FakeSound))
    extractor = make_extractor("praat")
    result = extractor.extract(torch.ones(800), 16000)
    assert calls["time_step"] == 0.0125
    assert calls["voicing_threshold"] == 0.6
    assert calls["pitch_floor"] == 60.0
    assert calls["pitch_ceiling"] == 640.0
    assert result.shape == (4,)
    assert torch.isnan(result[0])
    assert result[1] == 100
    assert torch.isnan(result[2])
    assert result[3] == 200


def test_praat_invalid_bounds():
    import pytest

    with pytest.raises(ValueError):
        PraatF0Extractor(f0_min=300, f0_max=100)
