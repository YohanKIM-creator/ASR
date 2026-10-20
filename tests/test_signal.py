import signal
import subprocess
import sys
import time
from pathlib import Path

from conftest import create
from src.utils.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_sigterm_saves_completed_step(tmp_path):
    data, saved = tmp_path / "data", tmp_path / "run"
    create(data)
    command = [
        sys.executable,
        "train.py",
        "--config-name",
        "smoke",
        f"datasets.train.data_dir={data}",
        f"datasets.dev_other.data_dir={data}",
        f"trainer.save_dir={saved}",
        "trainer.max_steps=10000",
        "trainer.n_epochs=10000",
        "trainer.validate_every=10000",
        "trainer.save_every=1",
        "trainer.log_examples_every=10000",
    ]
    with (tmp_path / "process.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError((tmp_path / "process.log").read_text())
                if (saved / "last.pt").is_file() and load_checkpoint(saved / "last.pt")[
                    "global_step"
                ] >= 1:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("training did not produce a completed step")
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
            assert process.returncode == 0, (tmp_path / "process.log").read_text()
            last = load_checkpoint(saved / "last.pt")
            best = load_checkpoint(saved / "best.pt")
            assert last["global_step"] >= 1
            assert best["best_wer"] < float("inf")
            assert last["batch_in_epoch"] >= 1
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
