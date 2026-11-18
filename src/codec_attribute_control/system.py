"""Probe inference system, following SSV's InferenceSystem composition."""
import torch
from quick_convert.systems.base import InferenceSystem


class F0ProbeSystem(InferenceSystem):
    """Encode audio into post-VQ prosody features and predict normalized F0.

    The encoder must accept a QC AudioBatch and return a ContentFeatures-like
    object with .values [B,T,C] and .lengths [B]. FACodec's eventual QC
    content encoder will implement this interface.
    """

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
        if features.values.ndim != 3:
            raise ValueError("Expected [B,T,C] prosody features")
        predictions = self.head(features.values)
        return predictions, features.lengths
