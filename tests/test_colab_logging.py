import json
import os
import subprocess
import sys
from pathlib import Path


def test_logging_works_with_colab_inline_backend_environment():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.logger import WandBWriter; import matplotlib; print(matplotlib.get_backend())",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().lower() == "agg"


def test_augmentation_cli_works_with_colab_inline_backend_environment():
    result = subprocess.run(
        [sys.executable, "scripts/augmentation_demo.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_environment_check_only_reports_versions():
    result = subprocess.run(
        [sys.executable, "scripts/setup_colab_env.py", "--check"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    profile = json.loads(result.stdout)
    assert {"python", "torch", "torchaudio", "cuda", "native_bf16"} <= profile.keys()
    assert profile["torch"].split("+")[0] == profile["torchaudio"].split("+")[0]
