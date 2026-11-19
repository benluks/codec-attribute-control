"""Shared pitch-to-code input transformation."""
import torch
from torch import nn

def pitch_inputs(f0_hz, lengths=None, normalization="none", global_mean=None, global_std=None):
    """Log2 pitch plus voicing; normalize only voiced, valid frames."""
    if normalization not in ("none", "utterance", "global"):
        raise ValueError(f"Unsupported pitch normalization: {normalization}")
    if f0_hz.ndim != 2:
        raise ValueError("Expected F0 [B,T] in Hz")
    b, t = f0_hz.shape
    if lengths is None:
        lengths = torch.full((b,), t, device=f0_hz.device, dtype=torch.long)
    if lengths.shape != (b,) or torch.any(lengths < 1) or torch.any(lengths > t):
        raise ValueError("Invalid F0 sequence lengths")
    valid = torch.arange(t, device=f0_hz.device)[None, :] < lengths[:, None]
    voiced = torch.isfinite(f0_hz) & (f0_hz > 5) & valid
    pitch = torch.where(voiced, f0_hz.clamp_min(5).log2(), 0.)
    if normalization == "utterance":
        count = voiced.sum(1, keepdim=True).clamp_min(1)
        mean = pitch.sum(1, keepdim=True) / count
        variance = ((pitch - mean) * voiced).square().sum(1, keepdim=True) / (count - 1).clamp_min(1)
        pitch = torch.where(voiced, (pitch - mean) / variance.clamp_min(1e-8).sqrt(), 0.)
    elif normalization == "global":
        if global_mean is None or global_std is None or global_std <= 0:
            raise ValueError("global normalization requires global_mean and positive global_std in log2 Hz")
        pitch = torch.where(voiced, (pitch - global_mean) / global_std, 0.)
    return pitch, voiced, valid



class PitchInputTransform(nn.Module):
    def __init__(self, normalization="none", global_mean=None, global_std=None):
        super().__init__()
        self.normalization = normalization
        self.global_mean = global_mean
        self.global_std = global_std

    def forward(self, f0_hz, lengths=None):
        pitch, voiced, _ = pitch_inputs(f0_hz, lengths, self.normalization, self.global_mean, self.global_std)
        return torch.stack([pitch, voiced.to(pitch.dtype)], dim=-1)
