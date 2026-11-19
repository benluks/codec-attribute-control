"""Pitch adapter for Quick Convert feature precomputation."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import torch
from quick_convert.pipelines.precompute_features import PrecomputeFeaturesPipeline
from torch import nn

from .extractors import WORLDF0Extractor
from .f0 import aligned_pitch
from .f0_resource import OnlineF0
from .facodec import HOP, SAMPLE_RATE


def _world_contour(waveform):
    frames = (len(waveform) + HOP - 1) // HOP
    return aligned_pitch(
        WORLDF0Extractor(), torch.from_numpy(waveform), SAMPLE_RATE, frames
    )


class PitchPrecomputePipeline(PrecomputeFeaturesPipeline):
    def run(self, pitch_workers=1):
        if type(pitch_workers) is not int or pitch_workers < 1:
            raise ValueError("pitch_workers must be a positive integer")
        if pitch_workers == 1:
            return super().run()
        if not isinstance(self.extractor.online.extractor, WORLDF0Extractor):
            raise TypeError("pitch_workers > 1 requires f0_backend=world")

        # Spawn independent processes instead of forking a Torch runtime that
        # may already have active threads or a CUDA context.
        with ProcessPoolExecutor(
            max_workers=pitch_workers, mp_context=get_context("spawn")
        ) as pool:
            self.extractor.pitch_pool = pool
            try:
                return super().run()
            finally:
                self.extractor.pitch_pool = None


class F0PrecomputeExtractor(nn.Module):
    pitch_pool = None

    def __init__(self, checkpoint_path=None, device="cpu", backend="jdc"):
        super().__init__()
        self.cache_format = f"{backend}-f0-hz-80hz-v1"
        self.online = OnlineF0(
            backend=backend, checkpoint_path=checkpoint_path, device=device
        )
        self.eval()

    @torch.inference_mode()
    def extract_batch(self, batch):
        if self.pitch_pool is not None:
            if batch.waveforms is None or batch.lengths is None:
                raise ValueError("WORLD precomputation requires loaded audio")
            if not torch.all(batch.sample_rates == SAMPLE_RATE):
                raise ValueError("WORLD precomputation requires 16 kHz audio")
            waveforms = batch.waveforms.detach().cpu().numpy()
            lengths = batch.lengths.tolist()
            contours = self.pitch_pool.map(
                _world_contour,
                (row[:n] for row, n in zip(waveforms, lengths, strict=True)),
            )
            return [torch.from_numpy(pitch).unsqueeze(-1) for pitch in contours]
        result = self.online(batch)
        return [
            value[: int(length)].detach().cpu().float().contiguous()
            for value, length in zip(
                result.values, result.lengths.tolist(), strict=True
            )
        ]


class JDCF0PrecomputeExtractor(F0PrecomputeExtractor):
    """Keep the extractor identity used by existing JDC caches."""
