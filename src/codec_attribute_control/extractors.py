"""F0 extractors implementing quick-convert's public feature contract.

JDC mirrors the Plachtaa/FAcodec 24kHz mel recipe; it is not established
as the precise supervision extractor for the Amphion NS3 checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from quick_convert.components.features.f0.base import F0Extractor
from quick_convert.components.features.f0.yaapt import YAAPTF0Extractor

from .amphion_source import ensure_amphion_importable

JDC_SAMPLE_RATE = 24_000
JDC_HOP = 300
JDC_CHECKPOINT_RELATIVE = Path("models/codec/facodec/modules/JDC/bst.t7")


class JDCF0Extractor(torch.nn.Module, F0Extractor):
    """Pretrained JDC labels, with raw F0 in Hz at 80 Hz frame rate.

    An upstream Amphion checkout with bst.t7 is required. We intentionally
    avoid an unverified Hugging Face model identifier.
    """

    def __init__(
        self, checkpoint_path: str | None = None, device: str = "cpu",
        *, mel_sample_rate: int = JDC_SAMPLE_RATE,
    ):
        super().__init__()
        ensure_amphion_importable()
        from Amphion.models.codec.facodec.modules.JDC.model import JDCNet
        if checkpoint_path is None:
            try:
                import Amphion
                root = Path(next(iter(Amphion.__path__)))
                checkpoint_path = root / JDC_CHECKPOINT_RELATIVE
            except (ImportError, StopIteration) as error:
                raise RuntimeError(
                    "Supply --pitch-checkpoint pointing at upstream bst.t7"
                ) from error
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"JDC checkpoint missing: {checkpoint}. Pass --pitch-checkpoint."
            )
        model = JDCNet(num_class=1, seq_len=192)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["net"])
        self.model = model.to(device).eval().requires_grad_(False)
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_sample_rate, n_fft=2048, win_length=1200,
            hop_length=JDC_HOP, n_mels=80,
        ).to(device)

    @property
    def sample_rate(self) -> int:
        return JDC_SAMPLE_RATE

    @property
    def hop_length(self) -> int:
        return JDC_HOP

    @torch.inference_mode()
    def extract(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim != 1:
            raise ValueError("JDC expects mono waveform [samples]")
        if sample_rate != JDC_SAMPLE_RATE:
            raise ValueError(f"Expected {JDC_SAMPLE_RATE} Hz; got {sample_rate}")
        mel = self.to_mel(waveform.to(next(self.model.parameters()).device, dtype=torch.float32))
        mel = (torch.log(mel + 1e-5) + 4.0) / 4.0
        # Upstream JDC forward: (absolute F0, GAN_feature, poolblock_out).
        f0 = self.model(mel[None, None])[0][0]
        return f0.detach().cpu().float()

    def train(self, mode: bool = True):
        return super().train(False)

    def _batch_log_mel(self, waveforms, lengths):
        spec = self.to_mel.spectrogram
        half_fft = spec.n_fft // 2
        positions = torch.arange(
            -half_fft, waveforms.shape[-1] + half_fft, device=waveforms.device
        )
        # Reflect at each true end, including recordings shorter than half an FFT.
        period = (2 * (lengths[:, None] - 1)).clamp_min(1)
        indices = positions.remainder(period)
        indices = torch.minimum(indices, period - indices)
        reflected = waveforms.gather(1, indices)
        power = torchaudio.functional.spectrogram(
            reflected,
            pad=spec.pad,
            window=spec.window,
            n_fft=spec.n_fft,
            hop_length=spec.hop_length,
            win_length=spec.win_length,
            power=spec.power,
            normalized=spec.normalized,
            center=False,
        )
        return (torch.log(self.to_mel.mel_scale(power) + 1e-5) + 4.0) / 4.0

    def _batch_pitch(self, mel, frame_lengths):
        model = self.model
        if torch.all(frame_lengths == mel.shape[-1]):
            return model(mel[:, None])[0]

        x = mel.transpose(1, 2).unsqueeze(1)
        padding = (
            torch.arange(x.shape[2], device=x.device) >= frame_lengths[:, None]
        )[:, None, :, None]

        def convolve(layers, x):
            for layer in layers:
                # BatchNorm and activations can make padded positions nonzero.
                if isinstance(layer, torch.nn.Conv2d):
                    x.masked_fill_(padding, 0)
                x = layer(x)
            return x

        x = convolve(model.conv_block, x)
        for block in (model.res_block1, model.res_block2, model.res_block3):
            x = block.pre_conv(x)
            residual = block.conv1by1(x) if block.conv1by1 is not None else x
            x = convolve(block.conv, x) + residual
        x = model.pool_block(x).transpose(1, 2).flatten(2)

        # The backward LSTM must start at each utterance's last valid frame.
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, frame_lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed, _ = model.bilstm_classifier(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=mel.shape[-1],
        )
        return model.classifier(x).squeeze(-1).abs()

    @torch.inference_mode()
    def extract_batch(self, waveforms: torch.Tensor, lengths: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched JDC inference; returns padded Hz contours and native frame lengths."""
        if sample_rate != JDC_SAMPLE_RATE:
            raise ValueError(f"Expected {JDC_SAMPLE_RATE} Hz; got {sample_rate}")
        if waveforms.ndim != 2 or lengths.shape != (waveforms.shape[0],):
            raise ValueError("Expected waveforms [B,T] and lengths [B]")
        device = next(self.model.parameters()).device
        waveforms = waveforms.to(device=device, dtype=torch.float32)
        lengths = lengths.to(device=device, dtype=torch.long)
        if torch.any(lengths <= 0) or torch.any(lengths > waveforms.shape[1]):
            raise ValueError("Invalid waveform lengths")
        mel = self._batch_log_mel(waveforms, lengths)
        frame_lengths = torch.div(lengths, JDC_HOP, rounding_mode="floor") + 1
        return self._batch_pitch(mel, frame_lengths), frame_lengths

    def extract_sample(self, sample):
        if sample.waveform is None or sample.sample_rate is None:
            raise ValueError("JDC extraction requires loaded audio")
        return self.extract(sample.waveform, int(sample.sample_rate))


class WORLDF0Extractor(F0Extractor):
    """DIO + StoneMask pitch with WORLD's own voiced/unvoiced decision."""

    sample_rate = 16_000
    hop_length = 200

    def __init__(self):
        import pyworld

        self.world = pyworld

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim != 1 or sample_rate != self.sample_rate:
            raise ValueError("WORLD expects a mono waveform at 16 kHz")
        audio = waveform.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        f0, time = self.world.dio(
            audio, sample_rate, f0_floor=60.0, f0_ceil=640.0, frame_period=12.5,
        )
        return torch.from_numpy(self.world.stonemask(audio, f0, time, sample_rate)).float()


__all__ = ["JDCF0Extractor", "WORLDF0Extractor", "YAAPTF0Extractor"]
