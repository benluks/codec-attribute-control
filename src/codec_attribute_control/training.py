"""QC F0 probe training, independent of online or cached target sourcing."""
from dataclasses import dataclass

import torch
from quick_convert.training.lightning.modules.base import BaseTrainingModule

from .metrics import F0ProbeMetrics
from .plotting import make_f0_figure
from .supervision import prepare_f0_targets


@dataclass
class ProbeStepOutput:
    loss: torch.Tensor


class F0ProbeTrainingModule(BaseTrainingModule):
    def __init__(self, system, optimization):
        super().__init__(optimization=optimization, enable_grad_norm_logging=False)
        self.system = system
        self.train_metrics = F0ProbeMetrics()
        self.val_metrics = F0ProbeMetrics()
        self._validation_curves_logged = False

    def _shared_step(self, batch, stage: str) -> ProbeStepOutput:
        predictions, feature_lengths = self.system(batch)
        resource = self.system.get_resource(batch, "f0")
        f0_hz = resource.values
        if f0_hz.ndim == 3 and f0_hz.shape[-1] == 1:
            f0_hz = f0_hz.squeeze(-1)
        if f0_hz.shape != predictions.shape:
            raise ValueError("Aligned F0 must match prediction shape [B,T]")
        if resource.lengths is None:
            raise ValueError("F0 resource must provide valid lengths")
        f0_lengths = resource.lengths.to(predictions.device)
        if not torch.equal(f0_lengths, feature_lengths.to(predictions.device)):
            raise ValueError("F0 and encoder valid frame lengths differ")
        targets, voiced = prepare_f0_targets(f0_hz.to(predictions.device), f0_lengths)
        valid = voiced & (torch.arange(predictions.shape[1], device=predictions.device)[None, :]
                          < f0_lengths[:, None])
        loss = self.system.head.loss(predictions, targets, valid)
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(predictions, targets, valid)
        if stage == "val" and not self._validation_curves_logged and not self.trainer.sanity_checking:
            self._log_validation_curves(f0_hz, predictions, f0_lengths)
            self._validation_curves_logged = True
        self.log(f"{stage}/loss", loss, on_step=stage == "train",
                 on_epoch=False, batch_size=predictions.shape[0], prog_bar=True)
        return ProbeStepOutput(loss=loss)

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train", self.train_metrics)

    def on_validation_epoch_start(self):
        self._validation_curves_logged = False

    def _log_validation_curves(self, f0_hz, predictions, lengths):
        logger = self.logger
        if logger is None or not hasattr(logger.experiment, "log"):
            return
        import matplotlib.pyplot as plt
        import wandb

        figure = make_f0_figure(
            f0_hz.detach().cpu().numpy(),
            predictions.detach().cpu().numpy(),
            lengths.detach().cpu().numpy(),
        )
        if figure is None:
            return
        try:
            logger.experiment.log({"val/f0_curves_hz": wandb.Image(figure)}, step=self.global_step)
        finally:
            plt.close(figure)

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val", self.val_metrics)

    def _log_epoch_metrics(self, stage: str, metrics: F0ProbeMetrics):
        for name, value in metrics.compute().items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True,
                     prog_bar=name == "normalized_log2_f0_mae", sync_dist=False)
        metrics.reset()
