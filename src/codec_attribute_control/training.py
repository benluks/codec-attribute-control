"""QC F0 probe training from named, precomputed tensor resources."""
from dataclasses import dataclass

import torch
from quick_convert.training.lightning.modules.base import BaseTrainingModule

from .supervision import prepare_f0_targets


@dataclass
class ProbeStepOutput:
    loss: torch.Tensor


class F0ProbeTrainingModule(BaseTrainingModule):
    def __init__(self, system, optimization):
        super().__init__(optimization=optimization, enable_grad_norm_logging=False)
        self.system = system

    def _shared_step(self, batch, stage: str) -> ProbeStepOutput:
        predictions, feature_lengths = self.system(batch)
        if "f0" not in batch.resources:
            raise ValueError("F0 resource missing; configure a dataset with loaded F0 tensors")
        resource = batch.resources["f0"]
        f0_hz = resource.values
        if f0_hz.ndim != 3 or f0_hz.shape[-1] != 1:
            raise ValueError("F0 resource must be [B,T,1] (one Hz value per frame)")
        f0_hz = f0_hz.squeeze(-1)
        if f0_hz.shape != predictions.shape:
            raise ValueError("F0 and encoder frames must have matching [B,T] shapes")
        f0_lengths = resource.lengths.to(predictions.device)
        if not torch.equal(f0_lengths, feature_lengths.to(predictions.device)):
            raise ValueError("F0 and encoder valid lengths differ")
        targets, voiced = prepare_f0_targets(f0_hz.to(predictions.device), f0_lengths)
        valid = voiced & (torch.arange(predictions.shape[1], device=predictions.device)[None, :]
                          < f0_lengths[:, None])
        loss = self.system.head.loss(predictions, targets, valid)
        mae = ((predictions.detach() - targets).abs()[valid].mean()
               if valid.any() else loss.detach() * 0.0)
        self.log(f"{stage}/loss", loss, on_step=stage == "train",
                 on_epoch=True, batch_size=predictions.shape[0], prog_bar=True)
        self.log(f"{stage}/mae", mae, on_step=False,
                 on_epoch=True, batch_size=predictions.shape[0], prog_bar=True)
        return ProbeStepOutput(loss=loss)
