import random

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from src.transforms.features import LogMelSpectrogram


class BaseDataset(Dataset):
    def __init__(
        self,
        index,
        text_encoder,
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        max_text_length=None,
        shuffle_index=False,
        instance_transforms=None,
        features=None,
        wave_augment=None,
        ctc_alignment_padding_frames=0,
    ):
        self._assert_index_is_valid(index)
        index = [
            item
            for item in index
            if (
                (max_audio_length is None or item["audio_len"] <= max_audio_length)
                and (max_text_length is None or len(item.get("text", "")) <= max_text_length)
            )
        ]
        if shuffle_index:
            index = list(index)
            random.Random(42).shuffle(index)
        if limit is not None:
            index = index[:limit]
        if not index:
            raise ValueError("dataset is empty after filtering")
        self._index = index
        self.text_encoder, self.target_sr = text_encoder, target_sr
        self.instance_transforms = instance_transforms or {}
        self.features = (
            features or self.instance_transforms.get("get_spectrogram") or LogMelSpectrogram(target_sr)
        )
        self.wave_augment = wave_augment or self.instance_transforms.get("audio")
        self.ctc_alignment_padding_frames = int(ctc_alignment_padding_frames)
        if self.ctc_alignment_padding_frames < 0:
            raise ValueError("CTC alignment padding budget must be nonnegative")
        self.epoch, self.seed = 0, 42

    def set_epoch(self, epoch, seed=42):
        self.epoch, self.seed = int(epoch), int(seed)

    def __getitem__(self, ind):
        data = self._index[ind]
        original = self.load_audio(data["path"])
        audio = original.clone()
        # Per-utterance RNG isolates worker/prefetch ordering for exact resume.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.epoch * 1000003 + ind)
            if self.wave_augment is not None:
                audio = self.wave_augment(audio)
            spectrogram = self.get_spectrogram(audio)
        text = self.text_encoder.normalize_text(data.get("text", ""))
        encoded = self.text_encoder.encode(text)
        augmentation_fallback = False
        if self.wave_augment is not None or self.ctc_alignment_padding_frames:
            ids = encoded.reshape(-1)
            required = len(ids) + int((ids[1:] == ids[:-1]).sum())
            if required > (spectrogram.shape[-1] + 3) // 4:
                original_spectrogram = self.get_spectrogram(original)
                if required <= (original_spectrogram.shape[-1] + 3) // 4:
                    audio, spectrogram = original, original_spectrogram
                    augmentation_fallback = True
                else:
                    available = (original_spectrogram.shape[-1] + 3) // 4
                    if 0 < required - available <= self.ctc_alignment_padding_frames:
                        # Training-only opt-in: add the smallest centered-STFT tail
                        # needed for ceil(feature_frames / 4) to fit CTC repeats.
                        hop = self.features.mel.hop_length
                        minimum_samples = (4 * required - 4) * hop
                        audio = torch.nn.functional.pad(
                            original, (0, max(0, minimum_samples - original.shape[-1]))
                        )
                        spectrogram = self.get_spectrogram(audio)
                        augmentation_fallback = self.wave_augment is not None
        return {
            "audio": audio,
            "original_audio": original,
            "spectrogram": spectrogram,
            "text": text,
            "text_encoded": encoded,
            "augmentation_fallback": augmentation_fallback,
            "audio_path": data["path"],
            "utterance_id": data["utterance_id"],
            "has_transcript": data.get("has_transcript", True),
        }

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        array, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(array.mean(axis=1)).unsqueeze(0)
        if not audio.numel():
            raise ValueError(f"empty audio: {path}")
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)
        return audio

    def get_spectrogram(self, audio):
        return self.features(audio)

    @staticmethod
    def _assert_index_is_valid(index):
        for item in index:
            for key in ("path", "audio_len"):
                if key not in item:
                    raise ValueError(f"missing index key {key}")

    @property
    def durations(self):
        return [item["audio_len"] for item in self._index]
