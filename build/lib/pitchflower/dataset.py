import random

import numpy as np
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler

import pytorch_lightning as pl



class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path,
        batch_size,
        val_size,
        num_samples,
        num_workers,
    ):
        super().__init__()
        self.train_data = pd.read_csv(train_path) # col names are audio_path, pitch_path, flat_path
        self.val_data = pd.read_csv(val_path, nrows=val_size)
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.num_samples = num_samples

    def setup(self, stage):
        self.train_dataset = DatasetVC(self.train_data, num_samples=self.num_samples)
        self.val_dataset = DatasetVC(self.val_data, num_samples=self.num_samples, train=False)

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    

class AudioDataset(Dataset):

    def __init__(self, data, *, num_samples, train=True):
        self.data = data
        self.train = train
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        waveform, _ = torchaudio.load(row['audio_path'])
        waveform = self.repeat_or_sample(waveform)
        waveform = self.normalize_audio(waveform)[0]
        return waveform

    def repeat_or_sample(self, waveform):
        if waveform.size(1) < self.num_samples:
            pad_length = self.num_samples - waveform.size(1)
            n_repeat = pad_length//waveform.size(1) + 1
            repeat_tensor = waveform.repeat(1, n_repeat)
            waveform = torch.cat((waveform, repeat_tensor[:, :pad_length]), dim=1)

        else:
            waveform = self.sample_audio(waveform)

        return waveform
    
    def sample_audio(self, waveform):
        length = waveform.size(1)
        if not self.train:
            start = 0
        else:
            start = random.randint(0, length-self.num_samples)

        return waveform[:, start:start+self.num_samples]
    
    def normalize_audio(self, audio):
        max_amplitude = torch.max(torch.abs(audio))
        if max_amplitude > 0:
            audio = audio / max_amplitude
        return audio
    

class DatasetVC(AudioDataset):

    def __init__(self, data, *, num_samples, train=True):
        self.data = data
        self.train = train
        self.num_samples = num_samples
        self.num_frames = num_samples // 256 + 1 

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Load normal wav
        og_wav, _ = torchaudio.load(row['audio_path'])

        # Load flattened wav
        flat_wav, _ = torchaudio.load(row['flat_path'])

        # Load pitch
        f0 = np.load(row['pitch_path'])
        f0 = torch.tensor(f0.astype(np.float32)).view(1, -1)
        
        # Sample wavs and f0
        wav, flat_wav, f0 = self.repeat_or_sample(og_wav, flat_wav, f0)

        # Normalize wavs
        wav = self.normalize_audio(wav)
        flat_wav = self.normalize_audio(flat_wav)

        return wav.flatten(), flat_wav.flatten(), f0.flatten()

    def repeat_or_sample(self, wav, flat_wav, f0):
        if wav.size(1) < self.num_samples:
            wav = self.repeat(wav, self.num_samples)
            flat_wav = self.repeat(wav, self.num_samples)
            f0 = self.repeat(f0, self.num_frames)

        else:
            wav, flat_wav, f0 = self.sample_audio(wav, flat_wav, f0)

        return wav, flat_wav, f0
    
    def sample_audio(self, wav, flat_wav, f0):
        length = wav.size(1)
        len_f0 = f0.size(1)

        if not self.train:
            start = 0.0
        else:
            start = random.random() * (1-self.num_samples/length)

        start_waveform = int(start*length)
        start_f0 = int(start*len_f0)

        wav = wav[:, start_waveform:start_waveform+self.num_samples]
        flat_wav = flat_wav[:, start_waveform:start_waveform+self.num_samples]
        f0 = f0[:, start_f0:start_f0+self.num_frames]
        return wav, flat_wav, f0
    
    @staticmethod
    def repeat(tensor, samples):
        pad_length = samples - tensor.size(1)
        n_repeat = pad_length//tensor.size(1) + 1
        repeat_tensor = tensor.repeat(1, n_repeat)
        tensor = torch.cat((tensor, repeat_tensor[:, :pad_length]), dim=1)
        return tensor