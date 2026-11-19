"""F0 target preparation independent of how raw F0 was obtained.

Inputs may come from online extraction or QC tensor resources. Cropping and
alignment must already have selected the same audio segment as the encoder.
"""
import numpy as np
import torch

from .targets import f0_target


def prepare_f0_targets(f0_hz: torch.Tensor, lengths: torch.Tensor, target_mode: str = "normalized_log2"):
    """Normalize each valid segment, returning [B,T] targets and voiced mask.

    Unvoiced/invalid frames are excluded from loss. Segments with insufficient
    F0 variation contribute no supervised frames.
    """
    if target_mode not in ("normalized_log2", "log2"):
        raise ValueError(f"Unsupported F0 target mode: {target_mode}")
    if f0_hz.ndim != 2 or lengths.ndim != 1 or lengths.numel() != f0_hz.shape[0]:
        raise ValueError("Expected f0_hz [B,T] and lengths [B]")
    if torch.any(lengths < 0) or torch.any(lengths > f0_hz.shape[1]):
        raise ValueError("Invalid F0 lengths")
    targets = torch.zeros_like(f0_hz, dtype=torch.float32)
    voiced = torch.zeros_like(f0_hz, dtype=torch.bool)
    for i, length in enumerate(lengths.tolist()):
        if not length:
            continue
        hz = f0_hz[i, :length].detach().cpu().numpy()
        if target_mode == "log2":
            valid_np = np.isfinite(hz) & (hz > 5.0)
            values = np.zeros_like(hz, dtype=np.float32)
            values[valid_np] = np.log2(hz[valid_np]).astype(np.float32)
        else:
            try:
                normalized, _ = f0_target(hz)
            except ValueError:
                continue
            valid_np = normalized != -10.0
            values = np.where(valid_np, normalized, 0.0)
        target = torch.as_tensor(values, dtype=torch.float32, device=f0_hz.device)
        valid = torch.as_tensor(valid_np, dtype=torch.bool, device=f0_hz.device)
        targets[i, :length] = torch.where(valid, target, 0.0)
        voiced[i, :length] = valid
    return targets, voiced
