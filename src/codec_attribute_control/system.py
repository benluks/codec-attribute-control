"""Probe inference system, following SSV's InferenceSystem composition."""
import torch
from quick_convert.systems.base import InferenceSystem


class F0ProbeSystem(InferenceSystem):
    """Predict F0 from the named prosody stream of a frozen content encoder."""

    def __init__(self, encoder, head, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)

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
        predictions = self.head(values)
        return predictions, features.lengths
