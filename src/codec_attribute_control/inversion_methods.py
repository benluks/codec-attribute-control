"""Pluggable latent inversion strategies and capability discovery.

A probe adapter declares supported methods. The inference layer chooses a
method from the adapter's advertised capabilities, never from class names.
"""
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .latent_inversion import InversionResult, invert_pitch


@dataclass(frozen=True)
class InversionCapability:
    name: str
    description: str


class ProbeAdapter:
    """Probe-specific mapping and optional exact linear operator.

    `linear_operator` returns (W, b), with W [T, T*D] or [T,D] for
    frame-independent probes. This is an explicit capability, not inferred
    from the probe class name.
    """

    def __init__(
        self,
        probe: nn.Module,
        *,
        linear_operator: Callable | None = None,
    ):
        self.probe = probe
        if linear_operator is None:
            head = getattr(probe, 'head', probe)
            getter = getattr(head, 'inversion_linear_operator', None)
            if getter is not None and getter() is not None:
                linear_operator = lambda latents: (getter(), None)
        self.linear_operator = linear_operator

    def available_methods(self):
        methods = ["adam", "jacobian"]
        if self.linear_operator is not None:
            methods.insert(0, "linear")
        return tuple(methods)


def available_methods(adapter: ProbeAdapter) -> tuple[str, ...]:
    return adapter.available_methods()


def invert(
    adapter: ProbeAdapter,
    latents: torch.Tensor,
    target: torch.Tensor,
    *,
    method: str = "adam",
    valid_mask: torch.Tensor | None = None,
    regularization: float = 1e-3,
    **kwargs,
) -> InversionResult:
    """Dispatch inversion through a capability-checked strategy registry."""
    if method not in adapter.available_methods():
        raise ValueError(
            f"Method {method!r} unavailable; supported: {adapter.available_methods()}"
        )
    if method == "adam":
        return invert_pitch(adapter.probe, latents, target, valid_mask=valid_mask, **kwargs)
    if method == "linear":
        return _linear(adapter, latents, target, valid_mask, regularization)
    if method == "jacobian":
        return _jacobian(adapter, latents, target, valid_mask, regularization)
    raise ValueError(f"Unknown inversion method: {method}")


def _mask(target, valid_mask):
    valid = torch.isfinite(target)
    if valid_mask is not None:
        valid &= valid_mask.to(device=target.device, dtype=torch.bool)
    if not valid.any().item():
        raise ValueError("No valid pitch target frames")
    return valid


def _solve_rows(A, residual, ridge):
    """Minimum-norm ridge update: A.T @ (A @ A.T + ridge I)^-1 @ residual."""
    if A.numel() == 0:
        return A.new_zeros(A.shape[1])
    gram = A @ A.T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return A.T @ torch.linalg.solve(gram + ridge * identity, residual)


def _linear(adapter, latents, target, valid_mask, ridge):
    valid = _mask(target, valid_mask)
    if ridge <= 0:
        raise ValueError("regularization must be positive")
    with torch.no_grad():
        original = adapter.probe(latents)
        W, _ = adapter.linear_operator(latents)
        # Exact minimum-displacement solution for independent linear frames.
        if W.ndim != 1 or W.shape[0] != latents.shape[-1]:
            raise ValueError("Automatic linear inversion expects a framewise [D] operator")
        residual = torch.where(valid, target - original, 0.0)
        scale = W.square().sum() + ridge
        delta = residual.unsqueeze(-1) * W / scale
        edited = latents.detach() + delta
        final = adapter.probe(edited)
        return InversionResult(
            edited, original, final,
            (final[valid] - target[valid]).square().mean().item(),
            delta.square().mean().sqrt().item(),
        )

def _jacobian(adapter, latents, target, valid_mask, ridge):
    """Local regularized Jacobian step; suitable for small utterances.

    Full Jacobian materialization can be expensive; replace with matrix-free
    CG/JVP-VJP for long utterances before production deployment.
    """
    valid = _mask(target, valid_mask)
    if ridge <= 0:
        raise ValueError("regularization must be positive")
    source = latents.detach()
    edited = source.clone()
    originals = []
    for b in range(source.shape[0]):
        x = source[b].detach().requires_grad_(True)
        with torch.enable_grad():
            def f(z):
                return adapter.probe(z.unsqueeze(0)).squeeze(0)[valid[b]]
            pred = f(x)
            J = torch.autograd.functional.jacobian(f, x, vectorize=True).reshape(pred.numel(), -1)
        originals.append(adapter.probe(source[b:b+1]).detach())
        with torch.no_grad():
            update = _solve_rows(J.detach(), target[b][valid[b]] - pred.detach(), ridge)
            edited[b] += update.reshape_as(x)
    with torch.no_grad():
        initial = torch.cat(originals, 0)
        final = adapter.probe(edited)
        return InversionResult(edited, initial, final, (final[valid] - target[valid]).square().mean().item(), (edited - source).square().mean().sqrt().item())
