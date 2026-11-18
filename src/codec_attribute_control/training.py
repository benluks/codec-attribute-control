"""Trainable linear F0 probe using quick-convert's Lightning module contract.

The frozen FACodec feature extraction and frame alignment are intentionally
separate from this module. The next step wires them into a QC dataset and
TrainingPipeline.
"""
from dataclasses import dataclass

import torch
from torch import nn
from quick_convert.training.lightning.modules.base import BaseTrainingModule


@dataclass
class ProbeStepOutput:
    loss: torch.Tensor


class F0ProbeTrainingModule(BaseTrainingModule):
    """Fit a linear mapping from frozen post-VQ prosody frames to voiced F0.

    Expected batches are dictionaries with:
      features: float tensor [B, T, C]
      targets: float tensor [B, T] (normalized voiced log2 F0)
      voiced: bool tensor [B, T] (valid target frames)
    This contract is local to the probe and does not pretend that QC's
    AudioBatch already exposes FACodec's attribute streams.
    """

    def __init__(self, feature_dim: int, optimization):
        super().__init__(optimization=optimization, enable_grad_norm_logging=False)
        self.probe = nn.Linear(feature_dim, 1)

    def _shared_step(self, batch, stage: str) -> ProbeStepOutput:
        features = batch["features"].float()
        targets = batch["targets"].float()
        voiced = batch["voiced"].bool()
        if features.ndim != 3 or targets.shape != features.shape[:2] or voiced.shape != targets.shape:
            raise ValueError("Expected features [B,T,C], targets and voiced [B,T]")
        predictions = self.probe(features).squeeze(-1)
        if not torch.any(voiced):
            # Keep the loss connected to the trainable parameters.
            loss = predictions.sum() * 0.0
            mae = predictions.detach().sum() * 0.0
        else:
            residual = predictions[voiced] - targets[voiced]
            loss = residual.square().mean()
            mae = residual.detach().abs().mean()
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True,
                 batch_size=features.size(0), prog_bar=True)
        self.log(f"{stage}/mae", mae, on_step=False, on_epoch=True,
                 batch_size=features.size(0), prog_bar=True)
        return ProbeStepOutput(loss=loss)
