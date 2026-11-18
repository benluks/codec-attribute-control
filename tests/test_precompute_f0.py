"""Adapter contract without pretrained weights."""
from types import SimpleNamespace
from pathlib import Path
import torch
from quick_convert.data.types import AudioBatch, AudioSample
from codec_attribute_control.f0_precompute_extractor import JDCF0PrecomputeExtractor

def test_adapter_returns_unpadded_contours():
    adapter = JDCF0PrecomputeExtractor.__new__(JDCF0PrecomputeExtractor)
    torch.nn.Module.__init__(adapter)
    class FakeOnline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.extractor = torch.nn.Module()
            self.extractor.model = torch.nn.Linear(1, 1)
        def forward(self, batch):
            return SimpleNamespace(values=torch.ones(2, 3, 1), lengths=torch.tensor([3, 1]))
    adapter.online = FakeOnline()
    samples = [AudioSample(utt_id=str(i), path=Path(str(i)), split="test-clean",
                waveform=torch.ones(1, 400-i*200), sample_rate=16000) for i in range(2)]
    output = adapter.extract_batch(AudioBatch.from_samples(samples))
    assert [x.shape for x in output] == [(3, 1), (1, 1)]
