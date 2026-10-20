import torch
import torchaudio
from torch import nn


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=7600,
            power=2,
            center=True,
        )

    def forward(self, waveform):
        if waveform.shape[-1] < 400:
            waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))
        features = self.mel(waveform).clamp_min(1e-5).log().squeeze(0)
        mean = features.mean(-1, keepdim=True)
        std = features.std(-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (features - mean) / std


class SpecAugment(nn.Module):
    def __init__(self, time_masks=2, time_width=40, freq_masks=2, freq_width=12, max_time_fraction=0.1):
        super().__init__()
        self.time_masks, self.time_width = time_masks, time_width
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.max_time_fraction = max_time_fraction

    def forward(self, spectrogram, lengths):
        result = spectrogram.clone()
        for row, length in zip(result, lengths):
            time = int(length)
            for _ in range(self.time_masks):
                upper = min(self.time_width, int(time * self.max_time_fraction))
                width = int(torch.randint(upper + 1, (1,)).item())
                start = int(torch.randint(time - width + 1, (1,)).item())
                row[:, start : start + width] = 0
            for _ in range(self.freq_masks):
                upper = min(self.freq_width, row.shape[0])
                width = int(torch.randint(upper + 1, (1,)).item())
                start = int(torch.randint(row.shape[0] - width + 1, (1,)).item())
                row[start : start + width, :time] = 0
        return result
