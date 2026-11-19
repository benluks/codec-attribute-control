"""QC F0 probe training, independent of online or cached target sourcing."""
from dataclasses import dataclass

import torch
from quick_convert.training.lightning.modules.base import BaseTrainingModule

from .metrics import F0ProbeMetrics
from .plotting import make_f0_figure
from .supervision import prepare_f0_targets


def align_f0_to_feature_grid(f0_hz, f0_lengths, feature_lengths, n_frames, *, source_hz=80, target_hz=50):
    """Nearest-neighbor resample F0 labels onto SSL feature timestamps.

    Invalid/padded frames stay NaN; source timestamps are measured from the
    start of the utterance. No interpolation across unvoiced regions.
    """
    if f0_hz.ndim != 2 or f0_lengths.ndim != 1 or feature_lengths.ndim != 1:
        raise ValueError("Expected F0 [B,T] and frame lengths [B]")
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("Frame rates must be positive")
    t = torch.arange(n_frames, device=f0_hz.device)
    index = torch.floor(t.float() * source_hz / target_hz + 0.5).long()
    index = torch.minimum(index[None, :], (f0_lengths[:, None] - 1).clamp_min(0))
    values = f0_hz.gather(1, index)
    valid = t[None, :] < feature_lengths[:, None]
    return values.masked_fill(~valid, float('nan')), feature_lengths



@dataclass
class ProbeStepOutput:
    loss: torch.Tensor


class F0ProbeTrainingModule(BaseTrainingModule):
    def __init__(self, system, optimization, target_mode="normalized_log2", align_f0_to_features=False):
        super().__init__(optimization=optimization, enable_grad_norm_logging=True)
        if target_mode not in ("normalized_log2", "log2"):
            raise ValueError(f"Unsupported F0 target mode: {target_mode}")
        self.target_mode = target_mode
        self.align_f0_to_features = align_f0_to_features
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
        f0_lengths = aligned_lengths.to(predictions.device)
        if not torch.equal(f0_lengths, feature_lengths.to(predictions.device)):
            raise ValueError("F0 and encoder valid frame lengths differ")
        targets, voiced = prepare_f0_targets(f0_hz.to(predictions.device), f0_lengths, target_mode=self.target_mode)
        valid = voiced & (torch.arange(predictions.shape[1], device=predictions.device)[None, :]
                          < f0_lengths[:, None])
        loss = self.system.head.loss(predictions, targets, valid)
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(predictions, targets, valid)
        if stage == "val" and not self._validation_curves_logged and not self.trainer.sanity_checking:
            self._log_validation_curves(f0_hz, predictions, f0_lengths)
            self._validation_curves_logged = True
        self.log(f"{stage}/loss", loss, on_step=stage == "train",
                 on_epoch=True, batch_size=predictions.shape[0], prog_bar=True)
        return ProbeStepOutput(loss=loss)

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train", self.train_metrics)

    def on_validation_epoch_start(self):
        self._validation_curves_logged = False

    def _log_validation_curves(self, f0_hz, predictions, lengths):
        logger = self.logger
        from lightning.pytorch.loggers import WandbLogger

        if not isinstance(logger, WandbLogger):
            return
        import matplotlib.pyplot as plt

        figure = make_f0_figure(
            f0_hz.detach().cpu().numpy(),
            predictions.detach().cpu().numpy(),
            lengths.detach().cpu().numpy(),
            target_mode=self.target_mode,
        )
        if figure is None:
            return
        try:
            logger.log_image("val/f0_curves_hz", images=[figure], step=self.global_step)
        finally:
            plt.close(figure)

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val", self.val_metrics)

    def _log_epoch_metrics(self, stage: str, metrics: F0ProbeMetrics):
        for name, value in metrics.compute().items():
            metric_name = name.replace("normalized_log2_f0", "log2_f0") if self.target_mode == "log2" else name
            self.log(f"{stage}/{metric_name}", value, on_step=False, on_epoch=True,
                     prog_bar=metric_name.endswith("_mae"), sync_dist=False)
        metrics.reset()
