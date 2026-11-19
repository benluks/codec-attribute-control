"""Frozen FACodec probe with QC online/static resource resolution."""

import torch
from quick_convert.components.resources.resolution import OnlineResourceMixin
from quick_convert.systems.base import InferenceSystem

from .facodec import FACodec


class F0ProbeSystem(OnlineResourceMixin, InferenceSystem):
    def __init__(self, encoder, head, freeze_encoder=True, online_encoders=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.freeze_encoder = freeze_encoder
        if encoder is None:
            with torch.no_grad():
                quantizer = FACodec().decoder.quantizer[0]
                codebook = quantizer.layers[0].codebook
                ids = torch.arange(
                    codebook.num_embeddings, device=codebook.weight.device
                )
                ids = ids.view(1, 1, -1)
                lookup = quantizer.vq2emb(ids)[0].T.contiguous()
            self.register_buffer("prosody_lookup", lookup)
        elif freeze_encoder:
            self.encoder.requires_grad_(False)
        self.configure_online_encoders(online_encoders)

    def encode(self, batch):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(batch)
        return self.encoder(batch)

    def forward(self, batch):
        if self.encoder is None:
            resource = self.get_resource(batch, "prosody_codes")
            codes = resource.values
            if codes.ndim == 3 and codes.shape[-1] == 1:
                codes = codes.squeeze(-1)
            if codes.ndim != 2 or resource.lengths is None:
                raise ValueError(
                    "Cached prosody codes require [B,T] indices and lengths"
                )
            lengths = resource.lengths.to(self.prosody_lookup.device)
            codes = codes.to(device=self.prosody_lookup.device, dtype=torch.long)
            values = torch.nn.functional.embedding(codes, self.prosody_lookup)
            valid = (
                torch.arange(values.shape[1], device=values.device) < lengths[:, None]
            )
            values = values.masked_fill(~valid.unsqueeze(-1), 0)
        else:
            features = self.encode(batch)
            values = (
                features.streams["prosody"]
                if hasattr(features, "streams")
                else features.values
            )
            lengths = features.lengths
        if values.ndim != 3:
            raise ValueError("Expected [B,T,C] prosody features")
        if getattr(self.head, "context_size", 1) > 1:
            positions = torch.arange(values.shape[1], device=values.device)
            indices = positions.clamp(max=(lengths[:, None] - 1).clamp_min(0))
            values = values.gather(1, indices.unsqueeze(-1).expand_as(values))
        return self.head(values), lengths
