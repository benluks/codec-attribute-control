"""F0 extractors implementing quick-convert's public feature contract.

JDC mirrors the Plachtaa/FAcodec 24kHz mel recipe; it is not established
as the precise supervision extractor for the Amphion NS3 checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from .amphion_source import ensure_amphion_importable

from quick_convert.components.features.f0.base import F0Extractor
from quick_convert.components.features.f0.yaapt import YAAPTF0Extractor

JDC_SAMPLE_RATE = 24_000
JDC_HOP = 300
JDC_CHECKPOINT_RELATIVE = Path("models/codec/facodec/modules/JDC/bst.t7")


class JDCF0Extractor(F0Extractor):
    """Pretrained JDC labels, with raw F0 in Hz at 80 Hz frame rate.

    An upstream Amphion checkout with bst.t7 is required. We intentionally
    avoid an unverified Hugging Face model identifier.
    """

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        ensure_amphion_importable()
        try:
            from Amphion.models.codec.facodec.modules.JDC.model import JDCNet
        except ImportError as error:
            raise ImportError(
                "Amphion source is required; initialize the Git submodule."
            ) from error
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
        self.device = torch.device(device)
        model = JDCNet(num_class=1, seq_len=192)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["net"])
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=JDC_SAMPLE_RATE, n_fft=2048, win_length=1200,
            hop_length=JDC_HOP, n_mels=80,
        ).to(self.device)

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
        mel = self.to_mel(waveform.to(self.device, dtype=torch.float32))
        mel = (torch.log(mel + 1e-5) + 4.0) / 4.0
        # Upstream JDC forward: (absolute F0, GAN_feature, poolblock_out).
        f0 = self.model(mel[None, None])[0][0]
        return f0.detach().cpu().float()

    def extract_sample(self, sample):
        if sample.waveform is None or sample.sample_rate is None:
            raise ValueError("JDC extraction requires loaded audio")
        return self.extract(sample.waveform, int(sample.sample_rate))


__all__ = ["JDCF0Extractor", "YAAPTF0Extractor"]
