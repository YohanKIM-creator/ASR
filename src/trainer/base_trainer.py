import signal

from src.utils.checkpoint import capture_rng, restore_rng


class BaseTrainer:
    def request_stop(self, signum=None, frame=None):
        self.stop_requested = True

    def install_signal_handlers(self):
        self._signal_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._signal_handlers[signum] = signal.signal(signum, self.request_stop)

    def restore_signal_handlers(self):
        for signum, handler in self._signal_handlers.items():
            signal.signal(signum, handler)

    def snapshot(self):
        return {
            "format_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": capture_rng(),
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "global_step": self.global_step,
            "config": self.project_config,
            "text_encoder": self.text_encoder.state_dict(),
            "run_id": self.writer.run_id,
            "git_sha": self.git_sha,
            "amp_overflows": self.amp_overflows,
            "resume_contract": self.resume_contract,
        }

    def resume(self, checkpoint):
        if checkpoint["resume_contract"] != self.resume_contract:
            raise ValueError(
                "resume contract changed (model/data/optimizer/schedule/augmentation). Use a new run for fine-tuning."
            )
        if checkpoint["text_encoder"] != self.text_encoder.state_dict():
            raise ValueError("tokenizer mismatch in resume")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.epoch, self.batch_in_epoch = (
            checkpoint["epoch"],
            checkpoint["batch_in_epoch"],
        )
        self.global_step = checkpoint["global_step"]
        self.amp_overflows = checkpoint.get("amp_overflows", 0)
        self.checkpoints.best_wer = float(checkpoint["best_wer"])
        restore_rng(checkpoint["rng"])
