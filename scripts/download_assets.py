"""Download model weights and the language model."""

import argparse
import gzip
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_librispeech import download

CHECKPOINTS = {
    76000: "https://www.comet.com/api/asset/download?experimentKey=00b40d3352ed4b02ab6e7c3854d83ca9&assetId=lCgwCus9wAaLLBPoVlQRolVBR",
    81500: "https://www.comet.com/api/asset/download?experimentKey=00b40d3352ed4b02ab6e7c3854d83ca9&assetId=ZQTRBkulO3vc5lZYynn83BEFM",
}
CHECKPOINT_URL = CHECKPOINTS[76000]
LM_URL = "https://www.openslr.org/resources/11/4-gram.arpa.gz"


def download_asset(url, target, gzip_source=False, force=False):
    """Download atomically; remember the URL to distinguish cached versions."""
    target = Path(target)
    source = target.with_name(target.name + ".url")
    if (
        not force
        and target.is_file()
        and target.stat().st_size
        and source.is_file()
        and source.read_text() == url
    ):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="." + target.name + ".", suffix=".download", dir=target.parent
    )
    os.close(descriptor)
    candidate = Path(name)
    unpacked = candidate.with_suffix(".unpacked")
    source_candidate = candidate.with_suffix(".url")
    try:
        if "drive.google.com" in url:
            import gdown

            if gdown.download(url, str(candidate), fuzzy=True) is None:
                raise RuntimeError(
                    "Google Drive download did not complete: " + str(target)
                )
        else:
            download(url, candidate)
        ready = candidate
        if gzip_source:
            with (
                gzip.open(candidate, "rb") as compressed,
                unpacked.open("wb") as output,
            ):
                for block in iter(lambda: compressed.read(1024 * 1024), b""):
                    output.write(block)
            ready = unpacked
        if not ready.stat().st_size:
            raise ValueError("Downloaded file is empty: " + str(target))
        with ready.open("rb") as stream:
            os.fsync(stream.fileno())
        # Invalidate the old URL before replacing its file, including on a crash.
        source.unlink(missing_ok=True)
        os.replace(ready, target)
        with source_candidate.open("w") as stream:
            stream.write(url)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(source_candidate, source)
        return target
    finally:
        for path in [
            candidate,
            unpacked,
            source_candidate,
            candidate.with_suffix(candidate.suffix + ".part"),
        ]:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-step", type=int, choices=sorted(CHECKPOINTS), default=76000
    )
    parser.add_argument("--checkpoint-url")
    parser.add_argument("--lm-url", default=LM_URL)
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Download files again even if cached"
    )
    parser.add_argument("--output", type=Path, default=Path("data/resources"))
    args = parser.parse_args()
    checkpoint_url = args.checkpoint_url or CHECKPOINTS[args.checkpoint_step]
    checkpoint = download_asset(
        checkpoint_url, args.output / "model.pt", force=args.force
    )
    print("checkpoint: " + str(checkpoint))
    if not args.skip_lm:
        language_model = download_asset(
            args.lm_url,
            args.output / "lm.arpa",
            gzip_source=urlsplit(args.lm_url).path.endswith(".gz"),
            force=args.force,
        )
        print("language_model: " + str(language_model))


if __name__ == "__main__":
    main()
