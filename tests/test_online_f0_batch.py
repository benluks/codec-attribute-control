"""Batched online F0 resource contract without pretrained JDC weights."""
from types import SimpleNamespace

import torch
from torch import nn

from codec_attribute_control.extractors import JDCF0Extractor
from codec_attribute_control.f0_resource import OnlineF0


class FakeJDC(JDCF0Extractor):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Linear(1, 1, bias=False)
        self.calls = 0

    def extract_batch(self, waveforms, lengths, sample_rate):
        assert sample_rate == 24_000
        assert waveforms.ndim == 2
        self.calls += 1
        frame_lengths = (lengths + 299) // 300
        frames = torch.arange(int(frame_lengths.max()), device=waveforms.device)
        values = (100 + frames).expand(waveforms.shape[0], -1).clone().float()
        return values, frame_lengths


def test_online_jdc_batches_and_masks(monkeypatch):
    import codec_attribute_control.f0_resource as resource
    extractor = FakeJDC()
    monkeypatch.setattr(resource, "make_extractor", lambda *args, **kwargs: extractor)
    online = OnlineF0()
    assert isinstance(online.extractor, nn.Module)
    batch = SimpleNamespace(
        waveforms=torch.randn(2, 800),
        lengths=torch.tensor([800, 400]),
        sample_rates=torch.tensor([16000, 16000]),
    )
    result = online(batch)
    assert extractor.calls == 1
    assert result.values.shape == (2, 4, 1)
    assert result.lengths.tolist() == [4, 2]
    assert torch.isfinite(result.values[0]).all()
    assert torch.isnan(result.values[1, 2:]).all()
