import json
import os
import random
from pathlib import Path

# A Colab kernel can pass an inline backend absent from the isolated worker.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

from src.metrics.utils import calc_cer, calc_wer


class WandBWriter:
    def __init__(
        self,
        directory,
        project_config,
        project_name="hse-asr-conformer",
        entity=None,
        run_id=None,
        run_name="conformer-char",
        mode="online",
        logger=None,
        log_checkpoints=True,
        **kwargs,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode, self.log_checkpoints = mode, log_checkpoints
        self.run = None
        self.run_id = run_id
        if mode != "disabled":
            import wandb

            self.wandb = wandb
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                id=run_id,
                name=run_name,
                config=project_config,
                mode=mode,
                resume="allow" if mode == "online" else None,
                dir=str(self.directory),
                settings=wandb.Settings(init_timeout=90),
            )
            self.run_id = self.run.id
        self.ledger = self.directory / "metrics.jsonl"
        self.examples_dir = self.directory / "examples"
        self.examples_dir.mkdir(exist_ok=True)

    def log(self, metrics, step):
        row = {"step": int(step), **metrics}
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.run is not None:
            # Explicit trainer step as a metric allows recovery from an older
            # checkpoint even if W&B received newer scalar logs before a crash.
            self.run.log({"trainer/global_step": int(step), **metrics})

    def log_examples(
        self,
        batch,
        predictions,
        text_encoder,
        step,
        partition,
        sample_rate=16000,
        count=3,
    ):
        selected = random.Random(int(step) + 17).sample(
            range(len(predictions)), min(count, len(predictions))
        )
        rows, media = [], {}
        for index in selected:
            reference = text_encoder.normalize_text(batch["text"][index])
            prediction = predictions[index]
            uid = batch["utterance_id"][index]
            row = {
                "id": uid,
                "target": reference,
                "prediction": prediction,
                "cer": calc_cer(reference, prediction),
                "wer": calc_wer(reference, prediction),
                "has_transcript": batch["has_transcript"][index],
            }
            rows.append(row)
            stem = f"{partition}-{step}-{index}"
            for kind, waveform in [
                ("original", batch["original_audio"][index]),
                ("augmented", batch["audio"][index]),
            ]:
                array = waveform.detach().cpu().squeeze(0).numpy()
                audio_path = self.examples_dir / (stem + "-" + kind + ".wav")
                sf.write(audio_path, array, sample_rate, subtype="FLOAT")
                if self.run is not None:
                    media[f"{partition}/{kind}_audio_{index}"] = self.wandb.Audio(
                        array, sample_rate=sample_rate, caption=reference
                    )
            time = int(batch["spectrogram_length"][index])
            spec = batch["spectrogram"][index, :, :time].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.imshow(spec, origin="lower", aspect="auto")
            ax.set(
                xlabel="Frame (10 ms)", ylabel="Mel bin", title=f"{partition}: {uid}"
            )
            image_path = self.examples_dir / (stem + "-spectrogram.png")
            fig.savefig(image_path, bbox_inches="tight")
            plt.close(fig)
            if self.run is not None:
                media[f"{partition}/spectrogram_{index}"] = self.wandb.Image(
                    str(image_path)
                )
        with (self.examples_dir / f"{partition}-{step}-predictions.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(rows, stream, ensure_ascii=False, indent=2)
        if self.run is not None:
            columns = list(rows[0]) if rows else []
            media[f"{partition}/predictions"] = self.wandb.Table(
                columns=columns, data=[[r[c] for c in columns] for r in rows]
            )
            self.run.log({"trainer/global_step": int(step), **media})

    def publish_best(self, path, metadata):
        if self.run is not None and self.log_checkpoints:
            artifact = self.wandb.Artifact(
                "ctc-conformer-" + self.run_id, type="model", metadata=metadata
            )
            artifact.add_file(str(path), name="best.pt")
            self.run.log_artifact(artifact, aliases=["best", "latest"])

    def finish(self):
        if self.run is not None:
            self.run.finish()
