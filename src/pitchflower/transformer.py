# Mostly taken from https://github.com/jishengpeng/WavTokenizer/blob/main/decoder/models.py
import torch
import torch.nn as nn

from rotary_embedding_torch import RotaryEmbedding


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        
        self.rope = RotaryEmbedding(dim=32)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_) # (b, c, t)
        k = self.k(h_)
        v = self.v(h_)

        q = self.rope.rotate_queries_or_keys(q.unsqueeze(3).permute(0, 3, 2, 1)) # (b, 1, t, c)
        k = self.rope.rotate_queries_or_keys(k.unsqueeze(3).permute(0, 3, 2, 1))

        q = q.permute(0, 3, 2, 1).squeeze(3)
        k = k.permute(0, 3, 2, 1).squeeze(3)

        # compute attention
        b, c, h = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_ = self.proj_out(h_)

        return x + h_