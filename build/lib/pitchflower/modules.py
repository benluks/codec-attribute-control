import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


from pitchflower.transformer import AttnBlock
from pitchflower.convnext import ConvNeXtBlock
from pitchflower.quantization  import ResidualVectorQuantizer



class AutoEncoderRVQ(nn.Module):
    
    def __init__(
        self, 
        *, 
        hidden_dim,
        n_convs,
        strides, 
        stride_kernel_sizes,
        mels=100,
        n_q=8,
        codebook_size=1024,
        codebook_dim=512,
        semantic_dim=768,
    ):
        super(AutoEncoderRVQ, self).__init__()

        self.encoder = AudioEncoder(
            mels,
            hidden_dim,
            strides=strides,
            down_kernels=stride_kernel_sizes,
            n_convs=n_convs
        )

        self.decoder = AudioDecoder(
            mels,
            hidden_dim,
            strides=strides[::-1],
            up_kernels=stride_kernel_sizes[::-1],
            n_convs=n_convs
        )

        self.quantizer = ResidualVectorQuantizer(
                            dimension=codebook_dim, 
                            n_q=n_q, 
                            bins=codebook_size
                        )
        
        self.quantizer_in = nn.Conv1d(hidden_dim, 
                                      codebook_dim, 
                                      kernel_size=1, 
                                      stride=1, 
                                      padding=0)
        self.quantizer_out = nn.Conv1d(codebook_dim, 
                                      hidden_dim, 
                                      kernel_size=1, 
                                      stride=1, 
                                      padding=0)

        if codebook_dim != semantic_dim:
            self.transform = nn.Linear(codebook_dim, semantic_dim)
        else:
            self.transform = nn.Identity()

        self.initialize_weights()
        self.n_q = n_q

    def forward(
        self, 
        audios,
        n_q=None,
    ):
        n_q = n_q if n_q else self.n_q
        e = self.encoder(audios)
        e = self.quantizer_in(e)
        quantized, codes, commit_loss, quantized_list = self.quantizer(e, n_q=n_q)
        feature = quantized.transpose(1,2)
        feature = self.transform(feature)
        quantized_proj = self.quantizer_out(quantized)   
        o = self.decoder(quantized_proj)
        return o, commit_loss, feature
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class PitchEmbedder(nn.Module):

    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim)]
        for _ in range(n_layers):
            layers.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)])

        self.mlp = nn.Sequential(*layers)

        self.no_pitch_emb = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)

        self.unvoiced_emb = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)

    def forward(self, pitch, *, pitch_mask=None):
        # pitch (B, T, 1)
        # pitch_mask (B,)
        embs = self.mlp(pitch) # embs (B, T, D)

        # Unvoiced regions masking
        unvoiced_mask = (pitch == 0).expand(-1, -1, embs.size(2))
        embs = torch.where(unvoiced_mask, self.unvoiced_emb, embs) 

        if pitch_mask is not None:
            mask = pitch_mask.view(-1, 1, 1).expand(-1, embs.size(1), embs.size(2))
            embs = torch.where(mask, embs, self.no_pitch_emb.view(1, 1, -1))

        return embs


class AudioEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, *, strides, down_kernels, n_convs):
        super(AudioEncoder, self).__init__()

        self.feature_extractor = CausalConv1d(in_channels=input_dim, 
                                              out_channels=hidden_dim, 
                                              kernel_size=7)

        self.encoder = nn.ModuleList()
        for i in range(len(strides)):
            self.encoder.append(
                EncoderBlock(
                    n_in=hidden_dim, 
                    n_out=hidden_dim,
                    stride=strides[i],
                    down_kernel_size=down_kernels[i],
                    n_convs=n_convs[i]
                )
            )
            self.encoder.append(
                AttnBlock(in_channels=hidden_dim)
            )

    def forward(self, x):
        x = self.feature_extractor(x)
        for block in self.encoder:
            x = block(x)
        return x


class AudioDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, *, strides, up_kernels, n_convs):
        super(AudioDecoder, self).__init__()

        self.decoder = nn.ModuleList()
        for i in range(len(strides)):
            self.decoder.append(
                AttnBlock(in_channels=hidden_dim)
            )
            self.decoder.append(
                DecoderBlock(
                    n_in=hidden_dim,
                    n_out=hidden_dim,
                    stride=strides[i],
                    up_kernel_size=up_kernels[i],
                    n_convs=n_convs[i]
                )
            )

        self.out_conv = CausalConv1d(in_channels=hidden_dim,
                                     out_channels=output_dim,
                                     kernel_size=7)

    def forward(self, x):
        for block in self.decoder:
            x = block(x)
        x = self.out_conv(x)
        return x
    

class EncoderBlock(nn.Module):

    def __init__(self, n_in, n_out, *, stride=1, down_kernel_size=None, n_convs=1):
        super(EncoderBlock, self).__init__()

        self.convs = nn.Sequential(*[
            ConvNeXtBlock(n_in) for _ in range(n_convs)
        ])
        
        self.down_conv = None
        if stride > 1:
            assert isinstance(down_kernel_size, int)
            self.down_conv = DownsampleLayer(
                n_in, n_out, down_kernel_size, stride=stride)
        else:
            assert n_in == n_out


    def forward(self, x):
        x = self.convs(x)
        if self.down_conv:
            x = self.down_conv(x)
        return x
    

class DecoderBlock(nn.Module):

    def __init__(self, n_in, n_out, *, stride=1, up_kernel_size=None, n_convs=1):
        super(DecoderBlock, self).__init__()

        self.convs = nn.Sequential(*[
            ConvNeXtBlock(n_out) for _ in range(n_convs)
        ])
        
        self.up_conv = None
        if stride > 1:
            assert isinstance(up_kernel_size, int)
            self.up_conv = UpsampleLayer(
                n_in, n_out, up_kernel_size, stride=stride)
        else:
            assert n_in == n_out


    def forward(self, x):
        if self.up_conv:
            x = self.up_conv(x)
        x = self.convs(x)
        return x
    

class DownsampleLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1):
        super(DownsampleLayer, self).__init__()
        self.stride = stride
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    

class UpsampleLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1):
        super(UpsampleLayer, self).__init__()
        self.stride = stride
        self.conv = CausalConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class CausalConv1d(nn.Conv1d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 ):
        self.left_padding = dilation * (kernel_size - 1)
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation)
        weight_norm(self)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return super(CausalConv1d, self).forward(x)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1
                 ):
        super(CausalConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride
        )
        self.stride = stride
        self.right_padding = kernel_size - stride

    def forward(self, x):
        x = super(CausalConvTranspose1d, self).forward(x)
        return F.pad(x, (0, -self.right_padding))
    
    
class MelNormalizer:

    def __init__(self, *mels):
        self.min = min(mel.min().item() for mel in mels)
        self.max = max(mel.max().item() for mel in mels)

    def __call__(self, mel):
        return (mel - self.min) / (self.max - self.min + 1e-8)