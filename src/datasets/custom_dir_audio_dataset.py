from pathlib import Path

import soundfile as sf

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir=None, audio_dir=None, transcription_dir=None, **kwargs):
        if data_dir is not None:
            root = Path(data_dir).expanduser()
            audio_dir = root / "audio"
            transcription_dir = root / "transcriptions"
        if audio_dir is None or not Path(audio_dir).is_dir():
            raise FileNotFoundError(f"audio directory missing: {audio_dir}")
        index, ids = [], set()
        for path in sorted(Path(audio_dir).iterdir()):
            if not path.is_file() or path.suffix.lower() not in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
                continue
            if path.stem in ids:
                raise ValueError(f"duplicate utterance ID: {path.stem}")
            ids.add(path.stem)
            transcript = Path(transcription_dir) / (path.stem + ".txt") if transcription_dir else None
            has_transcript = transcript is not None and transcript.is_file()
            info = sf.info(path)
            index.append(
                {
                    "path": str(path.resolve()),
                    "utterance_id": path.stem,
                    "text": transcript.read_text(encoding="utf-8").strip() if has_transcript else "",
                    "has_transcript": has_transcript,
                    "audio_len": info.frames / info.samplerate,
                }
            )
        super().__init__(index=index, **kwargs)


CustomDirAudioDataset = CustomDirDataset
