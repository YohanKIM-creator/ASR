import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def atomic_torch_save(state, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # Drive FUSE does not support directory fsync everywhere.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    # Trusted, self-produced snapshots contain Python/NumPy RNG state.
    return torch.load(path, map_location="cpu", weights_only=False)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class CheckpointManager:
    def __init__(self, directory, best_wer=math.inf):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_wer = float(best_wer)

    def save(self, state, validation_wer=None):
        improved = (
            validation_wer is not None and math.isfinite(validation_wer) and validation_wer < self.best_wer
        )
        if improved:
            self.best_wer = float(validation_wer)
        state = {**state, "best_wer": self.best_wer}
        if improved:
            atomic_torch_save(state, self.directory / "best.pt")
        atomic_torch_save(state, self.directory / "last.pt")
        return improved
