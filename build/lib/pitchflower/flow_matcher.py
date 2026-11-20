import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdyn.core import NeuralODE
from pitchflower.modules import EncoderBlock, DecoderBlock, PitchEmbedder
from pitchflower.transformer import AttnBlock


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=128, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    
class FlowMatcher(nn.Module):

    def __init__(self, sigma_min=1e-4, mels=100, channels=256, n_convs=[5,5], strides=[1,2]):
        super().__init__()

        self.sigma_min = sigma_min

        self.time_emb = GaussianFourierProjection(channels//2)

        self.pitch_emb = PitchEmbedder(hidden_dim=64, n_layers=3)
        
        self.init_proj = nn.Conv1d(2*mels+64, 
                                   channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.encoder = nn.ModuleList()
        for i in range(len(n_convs)):
            self.encoder.append(
                EncoderBlock(
                    n_in=channels, 
                    n_out=channels, 
                    down_kernel_size=7,
                    stride=strides[i],
                    n_convs=n_convs[i]
                )
            )
            self.encoder.append(
                AttnBlock(in_channels=channels)
            )
        

        self.decoder = nn.ModuleList()
        for i in reversed(range(len(n_convs))):
            self.decoder.append(
                AttnBlock(in_channels=channels)
            )
            self.decoder.append(
                DecoderBlock(
                    n_in=channels,
                    n_out=channels,
                    up_kernel_size=7,
                    stride=strides[i],
                    n_convs=n_convs[i]
                )
            )

        self.out_proj = nn.Conv1d(channels, 
                                  mels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x, y, f0s, t, *, f0_mask=None):
        # x (B, mels, n_frames)
        # y (B, mels, n_frames)
        # f0s (B, n_frames)
        # t (B,)
        B, _, T = x.shape  
        temb = self.time_emb(t) # (B, 256)
        pemb = self.pitch_emb(f0s.unsqueeze(2), pitch_mask=f0_mask).transpose(1,2) # (B, 64, n_frames)
        x = torch.cat([x, y, pemb], dim=1) # (B, 2*mels+64, n_frames)
        x = self.init_proj(x) # (B, 256, n_frames)

        hidden_states = []
        for enc_block in self.encoder:
            x = enc_block(x+temb[:, :, None])
            hidden_states.append(x)

        for dec_block in self.decoder:
            hs = hidden_states.pop()
            x = dec_block(x+hs+temb[:, :, None])

        return self.out_proj(x)
    
    def loss(self, mels, rec_mels, f0s, *, drop_cond_proba=0.0):
        t = torch.rand(mels.shape[0], device=mels.device)

        x_0 = torch.randn_like(rec_mels)
        x_t = t[:, None, None] * mels + (1 - (1-self.sigma_min) * t[:, None, None]) * x_0
        u_t = mels - (1-self.sigma_min) * x_0

        # Drop some conditioning from the batch
        if drop_cond_proba > 0.0:
            mask = drop_cond_proba < torch.rand(f0s.shape[0], device=f0s.device)
        else:
            mask = None

        v_t = self(x_t, rec_mels, f0s, t, f0_mask=mask)
        loss = F.mse_loss(v_t, u_t)

        return loss
    
    def improve_mels(self, rec_mels, f0s, *, steps=10, w=1.0):
        def solver_fn(t, Xt, *args, **kwargs):
            f0_mask = torch.zeros(f0s.shape[0], dtype=torch.bool, device=f0s.device)
            v = w*(self(Xt, rec_mels, f0s, t.unsqueeze(0))) + (1-w)*(self(Xt, rec_mels, f0s, t.unsqueeze(0), f0_mask=f0_mask))
            return v
        
        dummy = nn.Parameter(torch.empty(0), requires_grad=False)

        with torch.no_grad():
            neural_od = NeuralODE(solver_fn, 
                                  solver='midpoint', 
                                  sensitivity='adjoint',
                                  optimizable_params=[dummy])
            initial_state = torch.randn_like(rec_mels)
            t_span = torch.linspace(0, 1, steps+1, device=rec_mels.device)
            traj = neural_od.trajectory(initial_state, t_span=t_span)

        return traj[-1]