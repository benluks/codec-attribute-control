"""Epoch-level, voiced-frame F0 regression statistics."""
import torch
from torchmetrics import Metric


class F0ProbeMetrics(Metric):
    """Frame-weighted errors and utterance-weighted Pearson correlation."""

    full_state_update = False

    def __init__(self):
        super().__init__()
        for name in ("squared_error", "absolute_error", "voiced_frames", "pearson_sum", "pearson_utterances"):
            self.add_state(name, default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor):
        predictions = predictions.detach().double()
        targets = targets.detach().double()
        errors = predictions[valid] - targets[valid]
        self.squared_error += errors.square().sum()
        self.absolute_error += errors.abs().sum()
        self.voiced_frames += valid.sum()

        for pred, target, mask in zip(predictions, targets, valid):
            if mask.sum() < 2:
                continue
            x, y = pred[mask], target[mask]
            x, y = x - x.mean(), y - y.mean()
            denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
            if denominator > 1e-12:
                self.pearson_sum += (x * y).sum() / denominator
                self.pearson_utterances += 1

    def compute(self):
        count = self.voiced_frames.clamp_min(1)
        return {
            "normalized_log2_f0_mse": self.squared_error / count,
            "normalized_log2_f0_mae": self.absolute_error / count,
            "normalized_log2_f0_pearson": self.pearson_sum / self.pearson_utterances.clamp_min(1),
            "voiced_frames": self.voiced_frames,
            "pearson_utterances": self.pearson_utterances,
        }
