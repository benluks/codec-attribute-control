"""FACodec prosody F0 probe, using QC's BaseTrainingModule lifecycle."""
from dataclasses import dataclass

import torch
from torch import nn
from quick_convert.training.lightning.modules.base import BaseTrainingModule

from .facodec import FACodec, SAMPLE_RATE
from .f0 import aligned_pitch, make_extractor
from .targets import f0_target


@dataclass
class ProbeStepOutput:
    loss: torch.Tensor


class F0ProbeTrainingModule(BaseTrainingModule):
    """Train a linear F0 head on frozen FACodec post-VQ prosody features.

    The FACodec wrapper is temporary pending the QC content-encoder interface.
    """

    def __init__(self, optimization, feature_dim=256, f0_backend="jdc",
                 f0_checkpoint_path=None, device="cpu"):
        super().__init__(optimization=optimization, enable_grad_norm_logging=False)
        self.probe = nn.Linear(feature_dim, 1)
        self.codec = FACodec(device=device)
        self.extractor = make_extractor(f0_backend, device=device,
                                        checkpoint_path=f0_checkpoint_path)

    def _shared_step(self, batch, stage: str) -> ProbeStepOutput:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("Expected QC AudioBatch with loaded waveforms")
        predictions, targets = [], []
        for i in range(len(batch)):
            if int(batch.sample_rates[i]) != SAMPLE_RATE:
                raise ValueError(f"Expected {SAMPLE_RATE} Hz input")
            waveform = batch.waveforms[i, :int(batch.lengths[i])]
            with torch.no_grad():
                frame = self.codec.encode(waveform)
                features = frame.prosody[0].transpose(0, 1).detach()
                rate = getattr(self.extractor, "sample_rate", SAMPLE_RATE)
                if rate != SAMPLE_RATE:
                    import torchaudio
                    pitch_wave = torchaudio.functional.resample(waveform.cpu(), SAMPLE_RATE, rate)
                else:
                    pitch_wave = waveform.cpu()
                pitch = aligned_pitch(self.extractor, pitch_wave, rate, features.shape[0])
                try:
                    target, _ = f0_target(pitch)
                except ValueError:
                    continue
            mask = torch.as_tensor(target != -10.0, device=self.probe.weight.device)
            features = features.to(self.probe.weight.device)
            predictions.append(self.probe(features).squeeze(-1)[mask])
            targets.append(torch.as_tensor(target, device=self.probe.weight.device)[mask])
        if predictions:
            pred = torch.cat(predictions)
            truth = torch.cat(targets)
            residual = pred - truth
            loss = residual.square().mean()
            mae = residual.detach().abs().mean()
        else:
            loss = self.probe.weight.sum() * 0.0
            mae = loss.detach()
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True,
                 batch_size=len(batch), prog_bar=True)
        self.log(f"{stage}/mae", mae, on_step=False, on_epoch=True,
                 batch_size=len(batch), prog_bar=True)
        return ProbeStepOutput(loss=loss)
