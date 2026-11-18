"""No pretrained weights required: QC F0 contract and alignment tests."""
import numpy as np
import pytest
import torch

from quick_convert.components.features.f0.base import F0Extractor
from quick_convert.components.features.f0.yaapt import YAAPTF0Extractor

from codec_attribute_control.extractors import JDCF0Extractor
from codec_attribute_control.f0 import aligned_pitch, make_extractor


class SyntheticPitch(F0Extractor):
    @property
    def sample_rate(self):
        return 16_000

    @property
    def hop_length(self):
        return 200

    def extract(self, waveform, sample_rate):
        assert sample_rate == 16_000
        return torch.tensor([110.0, 0.0, 220.0, 440.0])


def test_jdc_subclasses_quick_convert_f0_contract():
    assert issubclass(JDCF0Extractor, F0Extractor)
    assert issubclass(YAAPTF0Extractor, F0Extractor)
    assert JDCF0Extractor.__abstractmethods__ == frozenset()


def test_quick_convert_yaapt_is_direct_fallback():
    extractor = make_extractor("yaapt")
    assert isinstance(extractor, YAAPTF0Extractor)


def test_alignment_and_unvoiced_mask():
    pitch = aligned_pitch(SyntheticPitch(), torch.zeros(800), 16_000, 4)
    np.testing.assert_allclose(pitch[[0, 2, 3]], [110, 220, 440])
    assert np.isnan(pitch[1])


def test_bad_backend_fails():
    with pytest.raises(ValueError):
        make_extractor("pyin")
