"""Thin adapter around upstream frozen FACodec; no vendoring or retraining."""
from dataclasses import dataclass

from .amphion_source import ensure_amphion_importable

import torch

SAMPLE_RATE = 16000
HOP = 200
MODEL_ID = "amphion/naturalspeech3_facodec"


@dataclass
class CodecFrame:
    combined: torch.Tensor  # [1, C, T]
    prosody: torch.Tensor   # [1, C, T]
    content: torch.Tensor   # [1, C, T]
    residual: torch.Tensor  # [1, C, T]
    speaker: torch.Tensor
    codes: torch.Tensor


class FACodec(torch.nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        ensure_amphion_importable()
        from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
        from huggingface_hub import hf_hub_download

        self.device = torch.device(device)
        self.encoder = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
        self.decoder = FACodecDecoder(
            in_channels=256, upsample_initial_channel=1024, ngf=32,
            up_ratios=[5, 5, 4, 2], vq_num_q_c=2, vq_num_q_p=1,
            vq_num_q_r=3, vq_dim=256, codebook_dim=8,
            codebook_size_prosody=10, codebook_size_content=10,
            codebook_size_residual=10, use_gr_x_timbre=True,
            use_gr_residual_f0=True, use_gr_residual_phone=True,
        )
        for model, filename in (
            (self.encoder, "ns3_facodec_encoder.bin"),
            (self.decoder, "ns3_facodec_decoder.bin"),
        ):
            path = hf_hub_download(repo_id=MODEL_ID, filename=filename)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> CodecFrame:
        """Accept mono [samples] at exactly 16kHz; pad to hop multiple."""
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono [samples], got {tuple(waveform.shape)}")
        padding = (-waveform.numel()) % HOP
        if padding:
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        device = next(self.encoder.parameters()).device
        enc = self.encoder(waveform[None, None].to(device, dtype=torch.float32))
        total, ids, _, parts, speaker = self.decoder(enc, vq=True, eval_vq=True)
        if len(parts) != 3 or total.ndim != 3 or parts[0].shape != total.shape:
            raise RuntimeError("Unexpected FACodec quantizer contract")
        if not torch.allclose(total, sum(parts), rtol=1e-4, atol=1e-4):
            raise RuntimeError("FACodec quantizer sum contract changed")
        if ids.shape[0] != 6:
            raise RuntimeError(f"Expected six quantizer codebooks, got {tuple(ids.shape)}")
        return CodecFrame(total, parts[0], parts[1], parts[2], speaker, ids)


    @torch.inference_mode()
    def encode_batch(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> CodecFrame:
        """Encode [B, samples] in one GPU forward pass with masked timbre pooling.

        Right padding is excluded from timbre attention and pooling. Convolutional
        boundary effects near the end of shorter recordings can still differ from
        independently encoded, unpadded inputs.
        """
        if waveforms.ndim != 2 or lengths.shape != (waveforms.shape[0],):
            raise ValueError("Expected waveforms [B, samples] and lengths [B]")
        if torch.any(lengths <= 0) or torch.any(lengths > waveforms.shape[1]):
            raise ValueError("Invalid waveform lengths")
        device = next(self.encoder.parameters()).device
        lengths = lengths.to(device=device, dtype=torch.long)
        waveforms = waveforms.to(device=device, dtype=torch.float32)
        padded_samples = ((waveforms.shape[1] + HOP - 1) // HOP) * HOP
        waveforms = torch.nn.functional.pad(waveforms, (0, padded_samples - waveforms.shape[1]))
        frames = (lengths + HOP - 1) // HOP
        enc = self.encoder(waveforms.unsqueeze(1))
        if enc.shape[-1] != padded_samples // HOP:
            raise RuntimeError("Unexpected FACodec encoder frame count")
        total, ids, _, parts = self.decoder.quantize(enc)
        if len(parts) != 3 or ids.shape[0] != 6:
            raise RuntimeError("Unexpected FACodec quantizer contract")
        valid = torch.arange(enc.shape[-1], device=device)[None, :] < frames[:, None]
        timbre = self.decoder.timbre_encoder(enc.transpose(1, 2), valid, None)
        speaker = (timbre * valid.unsqueeze(-1)).sum(dim=1) / frames.unsqueeze(1)
        return CodecFrame(total, parts[0], parts[1], parts[2], speaker, ids)

    @torch.inference_mode()
    def decode(self, frame: CodecFrame, delta: torch.Tensor | None = None) -> torch.Tensor:
        """Resynthesize with only continuous prosody delta changed.

        Content, residual, speaker and integer codebook IDs are held fixed.
        """
        if delta is not None and delta.shape != frame.prosody.shape:
            raise ValueError("Delta shape must match prosody contribution [1,C,T]")
        combined = frame.combined if delta is None else frame.combined + delta
        return self.decoder.inference(combined, frame.speaker)[0, 0].detach().cpu()
