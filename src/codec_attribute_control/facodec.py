"""Frozen NaturalSpeech 3 FACodec wrapper with named factor streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


SAMPLE_RATE = 16_000
HOP = 200
MODEL_ID = "amphion/naturalspeech3_facodec"


@dataclass(frozen=True)
class CodecFrame:
    """Continuous factors plus six RVQ codes for one mono waveform.

    Continuous tensors are ``[1, 256, frames]``; codes are
    ``[6, 1, frames]`` (prosody, two content, then three residual codebooks).
    """

    combined: torch.Tensor
    prosody: torch.Tensor
    content: torch.Tensor
    residual: torch.Tensor
    speaker: torch.Tensor
    codes: torch.LongTensor


def _model_classes():
    try:
        from Amphion.models.codec.ns3_codec import FACodecDecoder, FACodecEncoder
    except ImportError as error:
        raise ImportError(
            "FACodec requires an Amphion checkout; clone it beside this repository "
            "and add its parent directory to PYTHONPATH."
        ) from error
    return FACodecEncoder, FACodecDecoder


class FACodec(torch.nn.Module):
    """Official 16 kHz FACodec, frozen for reconstruction and conversion."""

    sample_rate, hop_length, frame_hz = SAMPLE_RATE, HOP, SAMPLE_RATE / HOP
    n_codebooks, feature_dim = 6, 256

    def __init__(self, device: str | torch.device = "cpu", checkpoint_dir: str | Path | None = None):
        super().__init__()
        encoder_class, decoder_class = _model_classes()
        self.encoder = encoder_class(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
        self.decoder = decoder_class(
            in_channels=256, upsample_initial_channel=1024, ngf=32, up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3, vq_dim=256, codebook_dim=8,
            codebook_size_prosody=10, codebook_size_content=10, codebook_size_residual=10,
            use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True,
        )
        from huggingface_hub import hf_hub_download

        directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
        for model, filename in ((self.encoder, "ns3_facodec_encoder.bin"), (self.decoder, "ns3_facodec_decoder.bin")):
            path = directory / filename if directory else Path(hf_hub_download(MODEL_ID, filename=filename))
            if not path.is_file():
                raise FileNotFoundError(f"FACodec checkpoint not found: {path}")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.to(device).eval().requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> CodecFrame:
        """Encode a finite 16 kHz mono waveform shaped ``[samples]``."""
        if waveform.ndim != 1 or not waveform.numel() or not waveform.is_floating_point() or not torch.isfinite(waveform).all():
            raise ValueError("Expected a nonempty finite mono floating-point waveform [samples].")
        padding = (-waveform.numel()) % self.hop_length
        if padding:
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        latent = self.encoder(waveform[None, None].to(self.device, dtype=torch.float32))
        combined, codes, _, streams, speaker = self.decoder(latent, vq=True, eval_vq=True)
        if len(streams) != 3 or codes.shape[0] != self.n_codebooks:
            raise RuntimeError("Unexpected FACodec quantizer contract.")
        return CodecFrame(combined, streams[0], streams[1], streams[2], speaker, codes)

    @torch.inference_mode()
    def decode(self, frame: CodecFrame, delta: torch.Tensor | None = None) -> torch.Tensor:
        """Decode, optionally adding a same-shaped post-VQ prosody delta."""
        if delta is not None and delta.shape != frame.prosody.shape:
            raise ValueError("delta must match prosody [1, 256, frames].")
        latent = frame.combined if delta is None else frame.combined + delta.to(self.device)
        return self.decoder.inference(latent.to(self.device), frame.speaker.to(self.device))[0, 0].detach().cpu()

    @torch.inference_mode()
    def convert(self, source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Use source prosody/content and reference speaker, omitting source residual detail."""
        source_frame, reference_frame = self.encode(source), self.encode(reference)
        latent = self.decoder.vq2emb(source_frame.codes.to(self.device), use_residual_code=False)
        return self.decoder.inference(latent, reference_frame.speaker.to(self.device))[0, 0].detach().cpu()
