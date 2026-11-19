"""Classification training with frozen FACodec code labels and JDC F0."""
import torch
import torch.nn.functional as F
from quick_convert.training.lightning.modules.base import BaseTrainingModule


class PitchCodeTrainingModule(BaseTrainingModule):
    def __init__(self, system, optimization):
        super().__init__(optimization=optimization, enable_grad_norm_logging=True)
        self.system = system

    def _step(self, batch, stage):
        codes, lengths = self.system(batch)
        resource = self.system.get_resource(batch, "f0")
        f0 = resource.values
        if f0.ndim == 3 and f0.shape[-1] == 1:
            f0 = f0.squeeze(-1)
        if f0.shape != codes.shape:
            raise ValueError("F0 and code frames must align")
        if resource.lengths is None or not torch.equal(resource.lengths.to(lengths.device), lengths):
            raise ValueError("F0 and code lengths differ")
        logits = self.system.head(f0.to(codes.device), lengths=lengths)
        if logits.shape[:2] != codes.shape:
            raise ValueError("Classifier output and codes must align")
        valid = torch.arange(codes.shape[1], device=codes.device)[None, :] < lengths[:, None]
        if not valid.any():
            raise ValueError("No valid prosody code frames")
        targets = codes[valid]
        predictions = logits[valid]
        if torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
            raise ValueError("FACodec code ID outside configured codebook")
        loss = F.cross_entropy(predictions, targets)
        accuracy = (predictions.argmax(-1) == targets).float().mean()
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True,
                 batch_size=codes.shape[0], prog_bar=True)
        self.log(f"{stage}/code_accuracy", accuracy, on_step=False, on_epoch=True,
                 batch_size=codes.shape[0], prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
