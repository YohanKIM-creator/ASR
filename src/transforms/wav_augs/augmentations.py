import torch
import torchaudio
from torch import nn


class Gain(nn.Module):
    def __init__(self, min_db=-6, max_db=6):
        super().__init__()
        self.min_db, self.max_db = min_db, max_db

    def forward(self, audio):
        db = float(torch.empty(()).uniform_(self.min_db, self.max_db))
        return audio * (10 ** (db / 20))


class SpeedPerturb(nn.Module):
    def __init__(self, sample_rate=16000, speeds=(0.9, 1.0, 1.1)):
        super().__init__()
        self.sample_rate, self.speeds = sample_rate, tuple(speeds)

    def forward(self, audio):
        speed = self.speeds[int(torch.randint(len(self.speeds), ()).item())]
        if speed == 1.0:
            return audio.clone()
        return torchaudio.functional.resample(
            audio, int(self.sample_rate * speed), self.sample_rate
        )


class AdditiveNoise(nn.Module):
    def __init__(self, min_snr_db=15, max_snr_db=30):
        super().__init__()
        self.min_snr_db, self.max_snr_db = min_snr_db, max_snr_db

    def forward(self, audio):
        noise = torch.randn_like(audio)
        snr = float(torch.empty(()).uniform_(self.min_snr_db, self.max_snr_db))
        signal_rms = audio.square().mean().sqrt().clamp_min(1e-8)
        noise_rms = noise.square().mean().sqrt().clamp_min(1e-8)
        return audio + noise * (signal_rms / noise_rms) * (10 ** (-snr / 20))


class Reverberation(nn.Module):
    def __init__(self, sample_rate=16000, max_delay_ms=60, decay=0.35):
        super().__init__()
        self.sample_rate, self.max_delay_ms, self.decay = (
            sample_rate,
            max_delay_ms,
            decay,
        )

    def forward(self, audio):
        max_delay = max(2, int(self.sample_rate * self.max_delay_ms / 1000))
        delay = int(torch.randint(max(1, max_delay // 3), max_delay + 1, ()).item())
        output = audio.clone()
        for multiple, coefficient in [(1, self.decay), (2, self.decay**2)]:
            offset = delay * multiple
            if offset < audio.shape[-1]:
                output[..., offset:] += coefficient * audio[..., :-offset]
        return output / (1 + self.decay + self.decay**2)


class WaveAugment(nn.Module):
    def __init__(
        self, sample_rate=16000, gain_p=0.5, speed_p=0.6, noise_p=0.3, reverb_p=0.2
    ):
        super().__init__()
        self.transforms = nn.ModuleList(
            [
                SpeedPerturb(sample_rate),
                Gain(),
                AdditiveNoise(),
                Reverberation(sample_rate),
            ]
        )
        self.probabilities = [speed_p, gain_p, noise_p, reverb_p]

    def forward(self, audio):
        for transform, probability in zip(self.transforms, self.probabilities):
            if float(torch.rand(())) < probability:
                audio = transform(audio)
        # A single scale factor preserves the waveform instead of hard clipping.
        peak = audio.abs().max().clamp_min(1.0)
        return audio / peak
