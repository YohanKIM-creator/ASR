"""Set up the Python environment for the Colab demo."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_environment():
    import numpy
    import torch
    import torchaudio

    from src.utils.precision import supports_native_bf16

    if torch.__version__.split("+")[0] != torchaudio.__version__.split("+")[0]:
        raise RuntimeError("torch/torchaudio versions differ")
    result = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torchaudio": str(torchaudio.__version__),
        "numpy": numpy.__version__,
        "cuda": torch.cuda.is_available(),
        "native_bf16": supports_native_bf16(),
    }
    if result["cuda"]:
        result["gpu"] = torch.cuda.get_device_name()
        result["capability"] = torch.cuda.get_device_capability()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv-colab")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the current environment without installing packages",
    )
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_environment(), indent=2))
        return
    if (
        Path("/content").is_dir()
        and os.geteuid() == 0
        and shutil.which("aria2c") is None
    ):
        subprocess.run(["apt-get", "-qq", "update"], check=True)
        subprocess.run(["apt-get", "-qq", "install", "-y", "aria2"], check=True)
    environment = args.venv.expanduser().resolve()
    python = environment / "bin/python"
    subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.13.3"], check=True)
    uv = [sys.executable, "-m", "uv"]
    if not python.is_file():
        subprocess.run([*uv, "venv", "--python", "3.12", str(environment)], check=True)
    actual = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()
    if actual != "3.12":
        raise RuntimeError(
            f"Expected Python 3.12 worker, got {actual}; select a new --venv path"
        )
    subprocess.run(
        [
            *uv,
            "pip",
            "install",
            "--python",
            str(python),
            "torch==2.5.1+cu121",
            "torchaudio==2.5.1+cu121",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    subprocess.run(
        [
            *uv,
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(ROOT / "requirements.txt"),
        ],
        check=True,
    )
    subprocess.run([*uv, "pip", "check", "--python", str(python)], check=True)
    frozen = subprocess.check_output(
        [*uv, "pip", "freeze", "--python", str(python)], text=True
    )
    output = ROOT / "outputs/colab-environment.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(frozen)
    subprocess.run(
        [str(python), str(ROOT / "scripts/setup_colab_env.py"), "--check"],
        cwd=ROOT,
        check=True,
    )
    print(json.dumps({"worker_python": str(python), "environment_log": str(output)}))


if __name__ == "__main__":
    main()
