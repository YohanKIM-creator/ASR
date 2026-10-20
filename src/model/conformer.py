import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def length_mask(lengths, time):
    return torch.arange(time, device=lengths.device)[None, :] < lengths[:, None]


class FeedForward(nn.Module):
    def __init__(self, dim, expansion, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RotaryAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads or (dim // heads) % 2:
            raise ValueError("head dimension must be even and divide model dimension")
        self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        inv = 10000.0 ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("inv_freq", inv, persistent=False)

    def _rotate(self, x):
        phase = torch.outer(torch.arange(x.shape[-2], device=x.device).float(), self.inv_freq.float())
        cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x, valid):
        batch, time, dim = x.shape
        qkv = self.qkv(self.norm(x)).view(batch, time, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self._rotate(q), self._rotate(k)
        # SDPA bool mask uses True for allowed positions.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(batch, time, dim)
        return self.drop(self.out(y)).masked_fill(~valid[..., None], 0)


class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size, dropout):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("convolution kernel must be odd")
        self.pre_norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.post_norm = nn.LayerNorm(dim)
        self.pointwise_out = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        x = self.pre_norm(x).transpose(1, 2)
        x = F.glu(self.pointwise_in(x), dim=1)
        x = x.masked_fill(~valid[:, None, :], 0)
        x = self.depthwise(x).transpose(1, 2)
        x = F.silu(self.post_norm(x)).transpose(1, 2)
        x = self.pointwise_out(x).transpose(1, 2)
        return self.dropout(x).masked_fill(~valid[..., None], 0)


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, expansion, kernel_size, dropout):
        super().__init__()
        self.ff1 = FeedForward(dim, expansion, dropout)
        self.attention = RotaryAttention(dim, heads, dropout)
        self.convolution = ConvolutionModule(dim, kernel_size, dropout)
        self.ff2 = FeedForward(dim, expansion, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, valid):
        x = (x + 0.5 * self.ff1(x)).masked_fill(~valid[..., None], 0)
        x = x + self.attention(x, valid)
        x = x + self.convolution(x, valid)
        x = self.final_norm(x + 0.5 * self.ff2(x))
        return x.masked_fill(~valid[..., None], 0)


class ConvSubsampling(nn.Module):
    def __init__(self, n_feats, dim, channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        reduced_feats = (n_feats + 3) // 4
        self.projection = nn.Linear(channels * reduced_feats, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, lengths):
        x = features.transpose(1, 2)[:, None]
        x = x.masked_fill(~length_mask(lengths, x.shape[2])[:, None, :, None], 0)
        x = F.silu(self.conv1(x))
        lengths = (lengths + 1) // 2
        x = x.masked_fill(~length_mask(lengths, x.shape[2])[:, None, :, None], 0)
        x = F.silu(self.conv2(x))
        lengths = (lengths + 1) // 2
        valid = length_mask(lengths, x.shape[2])
        x = x.masked_fill(~valid[:, None, :, None], 0)
        x = x.transpose(1, 2).flatten(2)
        x = self.dropout(self.projection(x))
        return x.masked_fill(~valid[..., None], 0), lengths, valid


class ConformerCTC(nn.Module):
    def __init__(
        self,
        n_feats=80,
        n_tokens=29,
        dim=256,
        num_layers=16,
        heads=4,
        expansion=4,
        kernel_size=31,
        dropout=0.1,
        subsampling_channels=32,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.subsampling = ConvSubsampling(n_feats, dim, subsampling_channels, dropout)
        self.blocks = nn.ModuleList(
            [ConformerBlock(dim, heads, expansion, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.output = nn.Linear(dim, n_tokens)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, spectrogram, spectrogram_length, **batch):
        if spectrogram.ndim != 3:
            raise ValueError("spectrogram must be B,F,T")
        lengths = spectrogram_length.to(spectrogram.device, dtype=torch.long)
        if (lengths < 1).any() or (lengths > spectrogram.shape[-1]).any():
            raise ValueError("invalid spectrogram lengths")
        x, lengths, valid = self.subsampling(spectrogram, lengths)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, valid, use_reentrant=False)
            else:
                x = block(x, valid)
        # CTC normalization in float32, including under mixed precision.
        return {
            "log_probs": F.log_softmax(self.output(x).float(), dim=-1),
            "log_probs_length": lengths,
        }

    @staticmethod
    def transform_input_lengths(input_lengths):
        return (input_lengths + 3) // 4

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
