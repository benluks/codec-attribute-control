"""Frozen FACodec code targets with pitch-conditioned classification."""
import torch
from quick_convert.systems.base import InferenceSystem
from quick_convert.components.resources.resolution import OnlineResourceMixin


class PitchToProsodyCodeSystem(OnlineResourceMixin, InferenceSystem):
    def __init__(self, encoder, head, online_encoders=None):
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        self.head = head
        self.configure_online_encoders(online_encoders)

    def forward(self, batch):
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(batch)
        if features.codes.ndim != 4 or features.codes.shape[0] != 6 or features.codes.shape[2] != 1:
            raise ValueError(f"Unexpected FACodec code layout: {tuple(features.codes.shape)}")
        # [6,B,1,T]: first quantizer is the single prosody codebook.
        codes = features.codes[0, :, 0, :].long()
        if codes.shape != features.streams["prosody"].shape[:2]:
            raise ValueError("Prosody code and latent frame counts differ")
        return codes, features.lengths
