import random

import torch
from torchaudio.transforms import MelSpectrogram

import pytorch_lightning as pl

from pitchflower.modules import AutoEncoderRVQ, MelNormalizer
from pitchflower.flow_matcher import FlowMatcher


_vocos = None

def _get_vocos(device: torch.device):
    global _vocos
    if _vocos is None:
        from vocos import Vocos
        _vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    return _vocos


class PitchFlower(pl.LightningModule):

    def __init__(
        self,
        autoencoder: AutoEncoderRVQ,
        flow_matcher: FlowMatcher,
        mel_transform: dict,
        loss_weights=dict(commit_loss=1.0, flow_loss=1.0),
        lr=1e-4,
        rvq_dropout=True,
        cfg_proba=0.1,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.flow_matcher = flow_matcher
        self.mel = MelSpectrogram(**mel_transform)

        self.lr = lr
        self.loss_weights = loss_weights
        self.rvq_dropout = rvq_dropout
        self.cfg_proba = cfg_proba

        self.save_hyperparameters()

    def preprocess(self, audios):
        mels = self.mel(audios)
        mels = torch.log(torch.clamp(mels, min=1e-7))
        return mels
    
    def step(self, audios, *, n_q=None, drop_cond_proba=0.0, give_mels=False):
        # Initialize loss dict
        loss_dict = {}

        # Extract from batch
        real_audios, flat_audios, f0s = audios

        # Convert to mels
        real_mels = self.preprocess(real_audios)
        flat_mels = self.preprocess(flat_audios)

        # Commitment loss
        rec_mels, commit_loss, feature = self.autoencoder(flat_mels, n_q=n_q)
        loss_dict['commit_loss'] = commit_loss

        # Flow-matching loss
        flow_loss = self.flow_matcher.loss(real_mels, rec_mels, f0s, drop_cond_proba=drop_cond_proba)
        loss_dict['flow_loss'] = flow_loss

        if give_mels:
            return loss_dict, real_mels, rec_mels, f0s
        return loss_dict
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr
        )
    
    def training_step(self, batch, batch_idx):
        # Determine how many RVQ levels to use
        if self.rvq_dropout:
            n_q = random.randint(1, self.autoencoder.n_q)
        else:
            n_q = None
        loss_dict = self.step(batch, n_q=n_q, drop_cond_proba=self.cfg_proba)
        for name, loss_value in loss_dict.items():
            self.log(f"Train/{name}", loss_value, prog_bar=True)
            loss_dict[name] *= self.loss_weights[name]
        loss = sum(loss_dict.values())
        self.log("Train/Loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            loss_dict, real_mels, rec_mels, f0s = self.step(batch, give_mels=True)
        else:
            loss_dict = self.step(batch)
        for name, loss_value in loss_dict.items():
            self.log(f"Val/{name}", loss_value, prog_bar=True)
            loss_dict[name] *= self.loss_weights[name]
        loss = sum(loss_dict.values())
        self.log("Val/Loss", loss, prog_bar=True)
        if batch_idx == 0:
            self.log_images(real_mels, rec_mels, f0s)
            
        return loss

    def log_images(self, mels, rec_mels, f0s):
        n_show = min(5, mels.size(0))
        epoch = self.current_epoch
        tensorboard = self.logger.experiment
        vocos = _get_vocos(self.device)
        for j in range(n_show):
            # real mel
            real_mel = mels[j].unsqueeze(0)
            rec_mel = rec_mels[j].unsqueeze(0)
            f0 = f0s[j].unsqueeze(0)
            flow_mel = self.flow_matcher.improve_mels(rec_mel, f0)
            normalizer = MelNormalizer(real_mel, rec_mel, flow_mel)
            if epoch == 0:
                real_mel_norm = normalizer(real_mel)
                tensorboard.add_image(
                    f'Real_Mel/Sample_{j}',
                    real_mel_norm.cpu(), 
                    epoch,
                    dataformats='CHW'
                )
                voc_audio = vocos.decode(real_mel)[0].cpu()
                tensorboard.add_audio(
                    f'Voc_Audio/Sample_{j}',
                    voc_audio,
                    epoch,
                    sample_rate=24000
                )
            # Reconstructed mel
            rec_mel_norm = normalizer(rec_mel)
            tensorboard.add_image(
                f'Recon_Mel/Sample_{j}',
                rec_mel_norm.cpu(),
                epoch,
                dataformats='CHW'
            )
            # Flow mel
            flow_mel_norm = normalizer(flow_mel)
            tensorboard.add_image(
                f'Flow_mel/Sample_{j}',
                flow_mel_norm.cpu(),
                epoch,
                dataformats='CHW'
            )
            # Flow audio
            flow_audio = vocos.decode(flow_mel)[0].cpu()
            tensorboard.add_audio(
                f'Flow_Audio/Sample_{j}',
                flow_audio,
                epoch,
                sample_rate=24000
            )


def main():
    from pytorch_lightning.cli import LightningCLI
    LightningCLI(
        PitchFlower,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    
if __name__ == "__main__":
    main()