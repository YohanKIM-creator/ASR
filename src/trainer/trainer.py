import subprocess
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.metrics.utils import CorpusMetrics
from src.trainer.base_trainer import BaseTrainer
from src.utils.precision import make_grad_scaler
from src.utils.checkpoint import CheckpointManager, capture_rng, restore_rng


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        text_encoder,
        config,
        device,
        dataloaders,
        writer,
        resume_checkpoint=None,
        **kwargs,
    ):
        self.model, self.criterion = model, criterion
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.text_encoder, self.config = text_encoder, config
        self.device, self.dataloaders, self.writer = torch.device(device), dataloaders, writer
        self.cfg = config.trainer
        self.project_config = OmegaConf.to_container(config, resolve=True)
        self.directory = Path(self.cfg.save_dir).expanduser().resolve()
        self.checkpoints = CheckpointManager(self.directory)
        self.spec_augment = instantiate(config.spec_augment) if config.get("spec_augment") else None
        self.amp = self.device.type == "cuda" and self.cfg.amp != "off"
        self.amp_dtype = torch.bfloat16 if self.cfg.amp == "bf16" else torch.float16
        self.scaler = make_grad_scaler(enabled=self.amp and self.amp_dtype == torch.float16)
        self.epoch = self.batch_in_epoch = self.global_step = 0
        self.stop_requested = False
        self.amp_overflows = 0
        try:
            self.git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            self.git_sha = "unknown"
        self.resume_contract = {
            key: self.project_config.get(key)
            for key in (
                "model",
                "optimizer",
                "lr_scheduler",
                "features",
                "augmentation",
                "spec_augment",
                "dataloader",
            )
        }
        # Paths may change between WSL and Colab/Drive. Compare dataset identity
        # and actual utterance metadata, not machine-local absolute paths.
        self.resume_contract["train_data"] = [
            (record["utterance_id"], record.get("text", ""), record["audio_len"])
            for record in self.dataloaders["train"].dataset._index
        ]
        self.resume_contract["training"] = {
            key: self.project_config["trainer"][key]
            for key in ("seed", "accumulation_steps", "max_grad_norm", "amp")
        }
        if resume_checkpoint is not None:
            self.resume(resume_checkpoint)

    def move_batch_to_device(self, batch):
        result = dict(batch)
        for name in ("spectrogram", "spectrogram_length", "text_encoded"):
            result[name] = batch[name].to(self.device, non_blocking=True)
        return result

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp)

    def predictions(self, outputs):
        return [
            self.text_encoder.ctc_decode(ids[: int(length)])
            for ids, length in zip(
                outputs["log_probs"].detach().cpu().argmax(-1), outputs["log_probs_length"].cpu()
            )
        ]

    def save(self, metric=None):
        improved = self.checkpoints.save(self.snapshot(), metric)
        if improved:
            self.writer.publish_best(
                self.directory / "best.pt",
                {"validation_wer": float(metric), "step": self.global_step, "git_sha": self.git_sha},
            )

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        monitor = None
        for partition, loader in self.dataloaders.items():
            if partition == "train":
                continue
            if partition.startswith("test"):
                raise ValueError("test datasets are forbidden in training/selection")
            counts, loss_sum, examples = CorpusMetrics(), 0.0, 0
            for batch_index, batch in enumerate(loader):
                batch = self.move_batch_to_device(batch)
                with self._autocast():
                    outputs = self.model(**batch)
                loss = self.criterion(**batch, **outputs)["loss"]
                predictions = self.predictions(outputs)
                if not all(batch["has_transcript"]):
                    raise ValueError("validation requires all transcripts")
                for reference, prediction in zip(batch["text"], predictions):
                    counts.update(reference, prediction)
                loss_sum += float(loss) * len(predictions)
                examples += len(predictions)
                if batch_index == 0:
                    self.writer.log_examples(
                        batch, predictions, self.text_encoder, self.global_step, partition
                    )
            result = counts.result()
            result["loss"] = loss_sum / max(1, examples)
            self.writer.log({f"{partition}/{key}": value for key, value in result.items()}, self.global_step)
            print(
                f"{partition} step={self.global_step} WER={result['wer']:.2%} CER={result['cer']:.2%} n={examples}",
                flush=True,
            )
            if partition == self.cfg.monitor_partition:
                monitor = result["wer"]
        if monitor is None:
            raise ValueError("monitor_partition missing from validation loaders")
        self.model.train()
        return monitor

    def train(self):
        if self.global_step >= self.cfg.max_steps or self.epoch >= self.cfg.n_epochs:
            return
        self.install_signal_handlers()
        session_start = self.global_step
        group_rng = None
        group_start = self.batch_in_epoch
        try:
            if self.global_step == 0:
                self.writer.log(
                    {
                        "train/loss": None,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/grad_norm": 0.0,
                        "model/parameters": sum(p.numel() for p in self.model.parameters()),
                    },
                    0,
                )
                self.save()
                self.save(self.evaluate())
            if self.stop_requested:
                self.save()
                return
            loader = self.dataloaders["train"]
            if any(not record.get("has_transcript", True) for record in loader.dataset._index):
                raise ValueError("training requires transcripts")
            if any(record.get("part", "").startswith(("dev", "test")) for record in loader.dataset._index):
                raise ValueError("held-out data in training")
            for epoch in range(self.epoch, self.cfg.n_epochs):
                self.epoch = epoch
                loader.batch_sampler.set_epoch(epoch)
                loader.dataset.set_epoch(epoch, self.cfg.seed)
                total_batches = len(loader)
                self.model.train()
                start_batch = self.batch_in_epoch
                group_items = group_loss = group_batches = 0
                group_audio_seconds = 0.0
                for batch_index, raw_batch in enumerate(loader):
                    if batch_index < start_batch:
                        continue
                    if group_batches == 0:
                        group_start = batch_index
                        group_rng = capture_rng()
                        self.optimizer.zero_grad(set_to_none=True)
                        group_size = min(self.cfg.accumulation_steps, total_batches - batch_index)
                        started = time.monotonic()
                    batch = self.move_batch_to_device(raw_batch)
                    if self.spec_augment is not None:
                        batch["spectrogram"] = self.spec_augment(
                            batch["spectrogram"], batch["spectrogram_length"]
                        )
                    with self._autocast():
                        outputs = self.model(**batch)
                        loss = self.criterion(**batch, **outputs)["loss"]
                    # Weight variable-duration microbatches by utterance count.
                    size = len(batch["text"])
                    self.scaler.scale(loss * size).backward()
                    group_items += size
                    group_loss += float(loss.detach()) * size
                    group_audio_seconds += sum(audio.shape[-1] / 16000 for audio in batch["audio"])
                    group_batches += 1
                    if group_batches < group_size:
                        continue
                    self.scaler.unscale_(self.optimizer)
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_items)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        if not self.scaler.is_enabled():
                            raise FloatingPointError("nonfinite gradient norm")
                        previous_scale = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)  # found_inf skips the optimizer update
                        self.scaler.update()
                        if self.scaler.get_scale() >= previous_scale:
                            raise FloatingPointError("nonfinite norm without AMP overflow detection")
                        self.amp_overflows += 1
                        self.batch_in_epoch = batch_index + 1
                        group_rng = None
                        group_items = group_loss = group_batches = 0
                        group_audio_seconds = 0.0
                        self.writer.log(
                            {"train/amp_overflow": 1, "train/loss_scale": self.scaler.get_scale()},
                            self.global_step,
                        )
                        self.save()
                        if self.stop_requested:
                            return
                        if self.amp_overflows >= 8:
                            raise FloatingPointError("eight consecutive AMP overflows")
                        continue
                    self.amp_overflows = 0
                    used_lr = self.optimizer.param_groups[0]["lr"]
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.global_step += 1
                    self.batch_in_epoch = batch_index + 1
                    group_rng = None
                    elapsed = time.monotonic() - started
                    self.writer.log(
                        {
                            "train/loss": group_loss / group_items,
                            "train/lr": used_lr,
                            "train/grad_norm": float(grad_norm),
                            "train/epoch": epoch,
                            "train/audio_seconds_per_second": group_audio_seconds / max(elapsed, 1e-6),
                            "train/update_seconds": elapsed,
                            "train/peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9
                            if self.device.type == "cuda"
                            else 0,
                        },
                        self.global_step,
                    )
                    if self.global_step == 1 or self.global_step % self.cfg.log_examples_every == 0:
                        self.writer.log_examples(
                            batch, self.predictions(outputs), self.text_encoder, self.global_step, "train"
                        )
                    if self.global_step % self.cfg.log_every == 0:
                        print(
                            f"step={self.global_step} epoch={epoch} loss={group_loss / group_items:.4f} lr={used_lr:.3g} grad={float(grad_norm):.3f}",
                            flush=True,
                        )
                    group_items = group_loss = group_batches = 0
                    group_audio_seconds = 0.0
                    if self.global_step % self.cfg.validate_every == 0:
                        self.save(self.evaluate())
                    elif self.global_step % self.cfg.save_every == 0:
                        self.save()
                    session_limit = self.cfg.get("session_max_steps")
                    if session_limit and self.global_step - session_start >= session_limit:
                        self.stop_requested = True
                    if self.stop_requested or self.global_step >= self.cfg.max_steps:
                        self.save(self.evaluate())
                        return
                self.epoch, self.batch_in_epoch = epoch + 1, 0
                self.save(self.evaluate())
            self.save()
        except BaseException:
            if group_rng is not None:
                # No partial update is saved. Replay this accumulation group.
                self.batch_in_epoch = group_start
                restore_rng(group_rng)
                self.optimizer.zero_grad(set_to_none=True)
            self.save()
            raise
        finally:
            self.restore_signal_handlers()
