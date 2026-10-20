from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import pytest


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def create(directory):
    directory = Path(directory)
    (directory / "audio").mkdir(parents=True, exist_ok=True)
    (directory / "transcriptions").mkdir(exist_ok=True)
    for index, (label, frequency) in enumerate(
        [("a", 240), ("b", 510), ("a", 270), ("b", 540)]
    ):
        time = np.arange(12000 + index * 800, dtype=np.float32) / 16000
        waveform = 0.15 * np.sin(2 * np.pi * frequency * time)
        sf.write(directory / "audio" / f"sample-{index}.wav", waveform, 16000)
        (directory / "transcriptions" / f"sample-{index}.txt").write_text(label + "\n")
    print("Created four synthetic utterances in " + str(directory))
