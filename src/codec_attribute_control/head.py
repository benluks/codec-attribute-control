"""Framewise normalized-F0 probe head."""
import torch
from torch import nn


class LinearF0Head(nn.Module):
    """Linear F0 probe, optionally extended with hidden MLP layers."""

    def __init__(self, input_dim=256, hidden_sizes=None, activation="gelu"):
        super().__init__()
        hidden_sizes = [] if hidden_sizes is None else list(hidden_sizes)
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation {activation!r}; choose from {sorted(activations)}")
        layers = []
        previous = input_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous, size), activations[activation.lower()]()])
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.projection = nn.Sequential(*layers) if hidden_sizes else layers[0]

    def forward(self, features):
        if features.ndim != 3:
            raise ValueError("Expected [B,T,C] features")
        return self.projection(features).squeeze(-1)

    def loss(self, predictions, targets, valid):
        if predictions.shape != targets.shape or valid.shape != targets.shape:
            raise ValueError("Predictions, targets and mask must have matching [B,T] shapes")
        if not torch.any(valid):
            return predictions.sum() * 0.0
        return (predictions[valid] - targets[valid]).square().mean()
