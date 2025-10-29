import json
from pathlib import Path

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from pitchflower.modules import AutoEncoderRVQ
from pitchflower.flow_matcher import FlowMatcher


class PitchFlowerSynthesizer(nn.Module):

    def __init__(
        self,
        autoencoder: AutoEncoderRVQ,
        flow_matcher: FlowMatcher,
        mel_transform: MelSpectrogram,
    ):
        super(PitchFlowerSynthesizer, self).__init__()
        self.autoencoder = autoencoder
        self.flow_matcher = flow_matcher
        self.mel = mel_transform

    @torch.inference_mode()
    def reconstruct_mels(self, audios, f0s, *, steps=10, w=1.0):
        mels = self.get_mels(audios)
        rec_mels, _, _ = self.autoencoder(mels)
        flow_mels = self.flow_matcher.improve_mels(rec_mels, f0s, steps=steps, w=w)
        return flow_mels
    
    @torch.inference_mode()
    def synthesize_mels(self, codes, f0s, *, steps=10, w=1.0):
        features = self.autoencoder.quantizer.decode(codes)
        features = self.autoencoder.quantizer_out(features)
        rec_mel = self.autoencoder.decoder(features)
        flow_mel = self.flow_matcher.improve_mels(rec_mel, f0s, steps=steps, w=w)
        return flow_mel
    
    def analyze_audios(self, audios):
        with torch.no_grad():
            mels = self.get_mels(audios)
            z = self.autoencoder.encoder(mels)
            z = self.autoencoder.quantizer_in(z)
            codes = self.autoencoder.quantizer.encode(z)
        return codes
    
    def get_mels(self, audios):
        mels = self.mel(audios)
        mels = torch.log(torch.clamp(mels, min=1e-7))
        return mels

    @classmethod
    def from_pretrained(cls, ckpt_path):
        ckpt_path = Path(ckpt_path)
        autoencoder_path = ckpt_path/'autoencoder.pt'
        flow_path = ckpt_path/'flow_matcher.pt'
        config_path = ckpt_path/'config.json'

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load autoencoder
        autoencoder = AutoEncoderRVQ(**config['autoencoder'])
        autoencoder.load_state_dict(torch.load(autoencoder_path))

        # Load flow
        flow_matcher = FlowMatcher(**config['flow_matcher'])
        flow_matcher.load_state_dict(torch.load(flow_path))

        # Load Mel
        mel_transform = MelSpectrogram(**config['mel_transform'])

        return cls(autoencoder, flow_matcher, mel_transform)