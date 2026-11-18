"""Thin adapter around upstream frozen FACodec; no vendoring or retraining."""

from dataclasses import dataclass

import torch

from .amphion_source import ensure_amphion_importable

SAMPLE_RATE = 16000
HOP = 200
MODEL_ID = "amphion/naturalspeech3_facodec"


@dataclass
class CodecFrame:
    combined: torch.Tensor  # [B, C, T]
    prosody: torch.Tensor  # [B, C, T]
    content: torch.Tensor  # [B, C, T]
    residual: torch.Tensor  # [B, C, T]
    speaker: torch.Tensor
    codes: torch.Tensor


class FACodec(torch.nn.Module):
    def __init__(self, device: str | None = None):
        super().__init__()
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        ensure_amphion_importable()
        from Amphion.models.codec.ns3_codec import FACodecDecoder, FACodecEncoder
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

    def train(self, mode: bool = True):
        return super().train(False)

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

    def _encode_padded(self, waveforms, lengths):
        if torch.all(lengths == waveforms.shape[-1]):
            return self.encoder(waveforms)

        from Amphion.models.codec.ns3_codec.facodec import (
            Activation1d,
            EncoderBlock,
            ResidualUnit,
        )

        def replicate_tail(x, sizes):
            edge = (sizes - 1).expand(-1, x.shape[1], 1)
            return torch.where(
                torch.arange(x.shape[-1], device=x.device) >= sizes,
                x.gather(-1, edge),
                x,
            )

        def apply(layer, x, sizes):
            if isinstance(layer, (torch.nn.Sequential, EncoderBlock, ResidualUnit)):
                children = (
                    layer if isinstance(layer, torch.nn.Sequential) else layer.block
                )
                residual = x
                for child in children:
                    x, sizes = apply(child, x, sizes)
                if isinstance(layer, ResidualUnit):
                    x = residual + x
                return x, sizes
            if isinstance(layer, Activation1d):
                # Both alias-free filters replicate their own last valid input.
                x = layer.act(layer.upsample(replicate_tail(x, sizes)))
                x = layer.downsample(replicate_tail(x, sizes * layer.up_ratio))
                return x, sizes
            if isinstance(layer, torch.nn.Conv1d):
                x = x.masked_fill(
                    torch.arange(x.shape[-1], device=x.device) >= sizes, 0
                )
                # Hop-aligned lengths are divisible by each encoder stride.
                return layer(x), sizes // layer.stride[0]
            raise TypeError(
                f"Unsupported FACodec encoder layer: {type(layer).__name__}"
            )

        return apply(self.encoder.block, waveforms, lengths[:, None, None])[0]

    def _batch_speaker(self, enc, valid, frames):
        timbre = self.decoder.timbre_encoder
        # Amphion indexes its positional table by batch row. Single inference
        # always uses row zero, so broadcast that row across every utterance.
        x = enc.transpose(1, 2) + timbre.position_emb.pe[:1]
        padding = ~valid.unsqueeze(-1)
        for layer in timbre.layers:
            normalized = layer.ln_1(x)
            attended, _ = layer.self_attn(
                normalized,
                normalized,
                normalized,
                key_padding_mask=~valid,
            )
            x = x + attended
            # Attention masking leaves padded queries active. Zero their values
            # after normalization, before the feed-forward temporal convolution.
            x = x + layer.ffn(layer.ln_2(x).masked_fill(padding, 0))
        x = timbre.last_ln(x).masked_fill(padding, 0)
        return x.sum(dim=1) / frames.unsqueeze(1)

    @torch.inference_mode()
    def encode_batch(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> CodecFrame:
        """Encode a mixed-length mono batch with single-utterance boundaries."""
        if waveforms.ndim != 2 or lengths.shape != (waveforms.shape[0],):
            raise ValueError("Expected waveforms [B, samples] and lengths [B]")
        if (
            not lengths.numel()
            or torch.any(lengths <= 0)
            or torch.any(lengths > waveforms.shape[1])
        ):
            raise ValueError("Invalid waveform lengths")
        device = next(self.encoder.parameters()).device
        lengths = lengths.to(device=device, dtype=torch.long)
        waveforms = waveforms.to(device=device, dtype=torch.float32)
        positions = torch.arange(waveforms.shape[1], device=device)
        waveforms = waveforms.masked_fill(positions >= lengths[:, None], 0)
        padding = (-waveforms.shape[1]) % HOP
        waveforms = torch.nn.functional.pad(waveforms, (0, padding)).unsqueeze(1)
        frames = (lengths + HOP - 1) // HOP
        enc = self._encode_padded(waveforms, frames * HOP)
        if enc.shape[-1] != waveforms.shape[-1] // HOP:
            raise RuntimeError("Unexpected FACodec encoder frame count")
        total, ids, _, parts = self.decoder.quantize(enc)
        if len(parts) != 3 or ids.shape[0] != 6:
            raise RuntimeError("Unexpected FACodec quantizer contract")
        valid = torch.arange(enc.shape[-1], device=device) < frames[:, None]
        speaker = self._batch_speaker(enc, valid, frames)
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
