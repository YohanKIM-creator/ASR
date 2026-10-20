import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F


def collate_fn(dataset_items):
    if not dataset_items:
        raise ValueError("empty batch")
    specs = [
        item["spectrogram"].squeeze(0) if item["spectrogram"].ndim == 3 else item["spectrogram"]
        for item in dataset_items
    ]
    spec_lengths = torch.tensor([x.shape[-1] for x in specs], dtype=torch.long)
    max_time = int(spec_lengths.max())
    targets = [item["text_encoded"].reshape(-1).long() for item in dataset_items]
    result = {
        "spectrogram": torch.stack([F.pad(x, (0, max_time - x.shape[-1])) for x in specs]),
        "spectrogram_length": spec_lengths,
        "text_encoded": pad_sequence(targets, batch_first=True, padding_value=0),
        "text_encoded_length": torch.tensor([x.numel() for x in targets], dtype=torch.long),
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
        "utterance_id": [item.get("utterance_id", str(i)) for i, item in enumerate(dataset_items)],
        "has_transcript": [item.get("has_transcript", True) for item in dataset_items],
        "augmentation_fallback": [item.get("augmentation_fallback", False) for item in dataset_items],
        "audio": [item["audio"] for item in dataset_items],
        "original_audio": [item.get("original_audio", item["audio"]) for item in dataset_items],
    }
    return result
