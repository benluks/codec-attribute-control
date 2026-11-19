"""Frozen FACodec code targets, resolved online or from QC tensor resources."""
import torch
from quick_convert.systems.base import InferenceSystem
from quick_convert.components.resources.resolution import OnlineResourceMixin


class PitchToProsodyCodeSystem(OnlineResourceMixin, InferenceSystem):
    def __init__(self, head, encoder=None, online_encoders=None, code_source="online"):
        super().__init__()
        if code_source not in ("online", "cached"):
            raise ValueError(f"Unsupported code source: {code_source}")
        if code_source == "online" and encoder is None:
            raise ValueError("Online code targets require a FACodec encoder")
        self.encoder = encoder.requires_grad_(False) if encoder is not None else None
        self.head = head
        self.code_source = code_source
        self.configure_online_encoders(online_encoders)

    def forward(self, batch):
        if self.code_source == "cached":
            resource = self.get_resource(batch, "prosody_codes")
            codes = resource.values
            if codes.ndim == 3 and codes.shape[-1] == 1:
                codes = codes.squeeze(-1)
            if codes.ndim != 2:
                raise ValueError(f"Expected cached prosody codes [B,T], got {tuple(codes.shape)}")
            if resource.lengths is None:
                raise ValueError("Cached prosody codes require frame lengths")
            lengths = resource.lengths
        else:
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(batch)
            raw_codes = features.codes
            if raw_codes.ndim == 3 and raw_codes.shape[0] == 6:
                codes = raw_codes[0]
            elif raw_codes.ndim == 4 and raw_codes.shape[0] == 6 and raw_codes.shape[2] == 1:
                codes = raw_codes[0, :, 0, :]
            else:
                raise ValueError(f"Unexpected FACodec code layout: {tuple(raw_codes.shape)}")
            lengths = features.lengths
            if codes.shape != features.streams["prosody"].shape[:2]:
                raise ValueError("Prosody code and latent frame counts differ")
        return codes.long(), lengths
