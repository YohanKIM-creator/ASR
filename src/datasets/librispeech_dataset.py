import json
from pathlib import Path

import soundfile as sf

from src.datasets.base_dataset import BaseDataset

PARTS = (
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
)


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir="data/LibriSpeech", download=False, **kwargs):
        if part not in PARTS and part != "train_all":
            raise ValueError(f"unknown LibriSpeech split {part}")
        root = Path(data_dir).expanduser().resolve()
        selected = PARTS[:3] if part == "train_all" else [part]
        index = []
        for split in selected:
            if not (root / split).is_dir():
                if not download:
                    raise FileNotFoundError(
                        f"{root / split}: run scripts/prepare_librispeech.py --parts {split}"
                    )
                from scripts.prepare_librispeech import prepare

                prepare(root, [split])
            cache = root / (split + ".index.json")
            if cache.is_file():
                records = json.loads(cache.read_text())
            else:
                records = []
                for transcript in sorted((root / split).rglob("*.trans.txt")):
                    for line in transcript.read_text(encoding="utf-8").splitlines():
                        uid, text = line.split(" ", 1)
                        path = transcript.parent / (uid + ".flac")
                        info = sf.info(path)
                        records.append(
                            {
                                "path": str(path.relative_to(root)),
                                "text": text,
                                "utterance_id": uid,
                                "audio_len": info.frames / info.samplerate,
                                "has_transcript": True,
                            }
                        )
                if not records:
                    raise ValueError(f"no utterances found for {split}")
                cache.write_text(json.dumps(records), encoding="utf-8")
            index.extend([{**r, "path": str(root / r["path"])} for r in records])
        super().__init__(index=index, **kwargs)
