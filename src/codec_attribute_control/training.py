"""F0 probe training using QC's BaseTrainingModule lifecycle."""
from dataclasses import dataclass

import torch
from quick_convert.training.lightning.modules.base import BaseTrainingModule

from .supervision import prepare_f0_targets


@dataclass
class ProbeStepOutput:
    loss: torch.Tensor


class F0ProbeTrainingModule(BaseTrainingModule):
    """Train the probe on aligned raw F0 supplied alongside an audio batch."""

    def __init__(self, system, optimization):
        super().__init__(optimization=optimization, enable_grad_norm_logging=False)
        self.system = system

    def _shared_step(self, batch, stage: str) -> ProbeStepOutput:
        # Target sourcing is intentionally separate from this module.
        # The dataset/adapter must provide F0 on the encoder's frame grid.
        audio = batch["audio"]
        f0_hz = batch["f0_hz"]
        f0_lengths = batch["f0_lengths"]
        predictions, feature_lengths = self.system(audio)
        if f0_hz.shape != predictions.shape:
            raise ValueError("Aligned F0 must match prediction shape [B,T]")
        if not torch.equal(f0_lengths.to(feature_lengths.device), feature_lengths):
            raise ValueError("F0 and encoder valid frame lengths differ")
        targets, voiced = prepare_f0_targets(
            f0_hz.to(predictions.device), f0_lengths.to(predictions.device)
        )
        frames = torch.arange(predictions.shape[1], device=predictions.device)
        valid = voiced & (frames[None, :] < feature_lengths[:, None])
        loss = self.system.head.loss(predictions, targets, valid)
        mae = ((predictions.detach() - targets).abs()[valid].mean()
               if valid.any() else loss.detach() * 0.0)
        self.log(f"{stage}/loss", loss, on_step=stage == "train",
                 on_epoch=True, batch_size=predictions.shape[0], prog_bar=True)
        self.log(f"{stage}/mae", mae, on_step=False,
                 on_epoch=True, batch_size=predictions.shape[0], prog_bar=True)
        return ProbeStepOutput(loss=loss)
