"""Frozen FACodec prosody code adapter for QC's precompute pipeline."""
import torch
from torch import nn

from .facodec_encoder import FACodecContentEncoder


class ProsodyCodePrecomputeExtractor(nn.Module):
    cache_format = "facodec-prosody-code-index-80hz-uint16-v1"

    def __init__(self, device="cpu"):
        super().__init__()
        self.encoder = FACodecContentEncoder(device=device, batched=True)
        self.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def extract_batch(self, batch):
        device = next(self.encoder.parameters()).device
        batch.waveforms = batch.waveforms.to(device)
        batch.lengths = batch.lengths.to(device)
        batch.sample_rates = batch.sample_rates.to(device)
        features = self.encoder(batch)
        codes = features.codes
        if codes.ndim == 3 and codes.shape[0] == 6:
            prosody = codes[0]
        elif codes.ndim == 4 and codes.shape[0] == 6 and codes.shape[2] == 1:
            prosody = codes[0, :, 0, :]
        else:
            raise ValueError(f"Unexpected FACodec code layout: {tuple(codes.shape)}")
        if torch.any(prosody < 0) or torch.any(prosody > 1023):
            raise ValueError("Prosody code index outside [0,1023]")
        return [values[:int(length)].detach().cpu().to(torch.uint16).contiguous()
                for values, length in zip(prosody, features.lengths, strict=True)]
