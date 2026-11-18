"""Frozen FACodec probe with QC online/static resource resolution."""
import torch
from quick_convert.systems.base import InferenceSystem
from quick_convert.components.resources.resolution import OnlineResourceMixin


class F0ProbeSystem(OnlineResourceMixin, InferenceSystem):
    def __init__(self, encoder, head, freeze_encoder=True, online_encoders=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.configure_online_encoders(online_encoders)

    def encode(self, batch):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(batch)
        return self.encoder(batch)

    def forward(self, batch):
        features = self.encode(batch)
        values = features.streams["prosody"] if hasattr(features, "streams") else features.values
        if values.ndim != 3:
            raise ValueError("Expected [B,T,C] prosody features")
        return self.head(values), features.lengths
