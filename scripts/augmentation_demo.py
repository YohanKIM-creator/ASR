"""Log examples of the four waveform augmentations."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio

from src.logger import create_writer
from src.transforms.features import LogMelSpectrogram, SpecAugment
from src.transforms.wav_augs import AdditiveNoise, Gain, Reverberation, SpeedPerturb
from train import load_local_wandb_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="saved/augmentation-audit")
    parser.add_argument(
        "--mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--tracker", choices=["comet", "wandb"], default="comet")
    args = parser.parse_args()
    load_local_wandb_key()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    array, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    original = torch.from_numpy(array.mean(1))[None]
    if sr != 16000:
        original = torchaudio.functional.resample(original, sr, 16000)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    features = LogMelSpectrogram()
    writer = create_writer(
        args.tracker,
        directory,
        {"purpose": "individual augmentation audit", "source": Path(args.audio).name},
        run_name="augmentations-audit",
        mode=args.mode,
        log_checkpoints=False,
    )
    augmentations = {
        "original": lambda x: x,
        "gain": Gain(3, 3),
        "speed": SpeedPerturb(speeds=(0.9,)),
        "noise": AdditiveNoise(20, 20),
        "reverb": Reverberation(),
        "specaugment": lambda x: x,
    }
    try:
        for step, (name, augmentation) in enumerate(augmentations.items()):
            audio = augmentation(original)
            audio = audio / audio.abs().max().clamp_min(1)
            spectrogram = features(audio)
            if name == "specaugment":
                spectrogram = SpecAugment()(
                    spectrogram[None], torch.tensor([spectrogram.shape[-1]])
                )[0]
            wav_path, png_path = (
                directory / (name + ".wav"),
                directory / (name + ".png"),
            )
            sf.write(wav_path, audio.squeeze(0).numpy(), 16000, subtype="FLOAT")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(spectrogram.numpy(), origin="lower", aspect="auto")
            ax.set(title=name, xlabel="Frame (10 ms)", ylabel="Mel bin")
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            writer.log(
                {
                    "augmentation/name": name,
                    "augmentation/duration_ratio": audio.shape[-1] / original.shape[-1],
                    "augmentation/peak": float(audio.abs().max()),
                    "augmentation/rms": float(audio.square().mean().sqrt()),
                },
                step,
            )
            if getattr(writer, "experiment", None) is not None:
                writer.experiment.log_audio(
                    str(wav_path), file_name=wav_path.name, step=step, overwrite=True
                )
                writer.experiment.log_image(
                    str(png_path), name=png_path.name, step=step, overwrite=True
                )
            if writer.run is not None:
                writer.run.log(
                    {
                        "augmentation/name": name,
                        "augmentation/audio": writer.wandb.Audio(
                            str(wav_path), sample_rate=16000
                        ),
                        "augmentation/spectrogram": writer.wandb.Image(str(png_path)),
                    }
                )
    finally:
        writer.finish()
    print("Augmentation audit saved to " + str(directory))


if __name__ == "__main__":
    main()
