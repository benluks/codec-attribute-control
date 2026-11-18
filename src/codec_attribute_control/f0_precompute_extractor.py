"""JDC adapter for Quick Convert feature precomputation."""
import torch
from torch import nn
from .f0_resource import OnlineF0

class JDCF0PrecomputeExtractor(nn.Module):
    cache_format = "jdc-f0-hz-80hz-v1"

    def __init__(self, checkpoint_path=None, device="cpu"):
        super().__init__()
        self.online = OnlineF0(backend="jdc", checkpoint_path=checkpoint_path)
        self.to(device).eval()

    @torch.inference_mode()
    def extract_batch(self, batch):
        device = next(self.online.extractor.model.parameters()).device
        batch.waveforms = batch.waveforms.to(device)
        batch.lengths = batch.lengths.to(device)
        batch.sample_rates = batch.sample_rates.to(device)
        result = self.online(batch)
        return [value[:int(length)].detach().cpu().float().contiguous()
                for value, length in zip(result.values, result.lengths, strict=True)]
