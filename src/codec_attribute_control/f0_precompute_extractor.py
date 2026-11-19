"""Pitch adapter for Quick Convert feature precomputation."""

import torch
from torch import nn

from .f0_resource import OnlineF0


class F0PrecomputeExtractor(nn.Module):
    def __init__(self, checkpoint_path=None, device="cpu", backend="jdc"):
        super().__init__()
        self.cache_format = f"{backend}-f0-hz-80hz-v1"
        self.online = OnlineF0(
            backend=backend, checkpoint_path=checkpoint_path, device=device
        )
        self.eval()

    @torch.inference_mode()
    def extract_batch(self, batch):
        result = self.online(batch)
        return [
            value[: int(length)].detach().cpu().float().contiguous()
            for value, length in zip(
                result.values, result.lengths.tolist(), strict=True
            )
        ]


class JDCF0PrecomputeExtractor(F0PrecomputeExtractor):
    """Keep the extractor identity used by existing JDC caches."""
