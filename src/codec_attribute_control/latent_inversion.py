"""Frozen-probe inversion of SSL latents for pitch control.

Works directly on the [B, T, D] representation emitted by
ReconstructionW2VBertEncoder. Does not instantiate another SSL encoder.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InversionResult:
    latents: torch.Tensor
    initial_prediction: torch.Tensor
    final_prediction: torch.Tensor
    pitch_loss: float
    displacement_rms: float


def invert_pitch(
    probe: torch.nn.Module,
    latents: torch.Tensor,
    target_log2_f0: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    steps: int = 150,
    learning_rate: float = 0.03,
    displacement_weight: float = 0.01,
    smoothness_weight: float = 0.0,
) -> InversionResult:
    """Optimize input latents, keeping all model parameters frozen.

    The target is absolute log2(Hz), matching train_w2vbert.yaml.
    Unvoiced/invalid targets may be NaN and are ignored. A separate
    valid_mask can restrict optimization to voiced, nonpadding frames.
    The caller must supply the exact reconstruction-weighted SSL features.
    """
    if latents.ndim != 3 or target_log2_f0.shape != latents.shape[:2]:
        raise ValueError("Expected latents [B,T,D] and target [B,T]")
    if steps < 0 or learning_rate <= 0:
        raise ValueError("steps must be nonnegative and learning_rate positive")
    if min(displacement_weight, smoothness_weight) < 0:
        raise ValueError("regularization weights must be nonnegative")

    valid = torch.isfinite(target_log2_f0)
    if valid_mask is not None:
        if valid_mask.shape != valid.shape:
            raise ValueError("valid_mask must have shape [B,T]")
        valid = valid & valid_mask.to(device=valid.device, dtype=torch.bool)
    if not valid.any().item():
        raise ValueError("No valid pitch target frames")
    target = torch.nan_to_num(target_log2_f0)
    source = latents.detach()
    # Freeze model parameters without changing their original requires_grad flags.
    params = tuple(probe.parameters())
    flags = tuple(p.requires_grad for p in params)
    training = probe.training
    try:
        probe.eval()
        for p in params:
            p.requires_grad_(False)
        with torch.no_grad():
            initial = probe(source).detach()
        if initial.shape != target.shape:
            raise ValueError(f"Probe returned {tuple(initial.shape)}, expected {tuple(target.shape)}")
        delta = torch.nn.Parameter(torch.zeros_like(source))
        optimizer = torch.optim.Adam([delta], lr=learning_rate)
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                prediction = probe(source + delta)
                pitch_loss = F.mse_loss(prediction[valid], target[valid])
                penalty = delta.square().mean() * displacement_weight
                if smoothness_weight and delta.shape[1] > 1:
                    penalty = penalty + smoothness_weight * (
                        delta[:, 1:] - delta[:, :-1]
                    ).square().mean()
                (pitch_loss + penalty).backward()
                optimizer.step()
        with torch.no_grad():
            edited = (source + delta).detach()
            final = probe(edited).detach()
            error = F.mse_loss(final[valid], target[valid]).item()
            rms = delta.detach().square().mean().sqrt().item()
        return InversionResult(edited, initial, final, error, rms)
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)
        probe.train(training)


def shift_log2_f0(log2_f0: torch.Tensor, semitones: float) -> torch.Tensor:
    """Preserve NaN unvoiced frames while shifting voiced pitch."""
    return log2_f0 + semitones / 12.0
