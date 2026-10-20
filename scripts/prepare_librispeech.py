"""Download LibriSpeech splits and verify their checksums."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

PARTS = (
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
)
BASE_URL = "https://www.openslr.org/resources/12/"


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError("unsafe archive member")
        tar.extractall(destination)


def download(url, target):
    target = Path(target)
    partial = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={offset}-"} if offset else {}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                expected = response.headers.get("Content-Length")
                received = 0
                with partial.open("ab" if append else "wb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        received += len(block)
                    if expected is not None and received != int(expected):
                        raise OSError("Incomplete download: " + str(target))
                    stream.flush()
                    os.fsync(stream.fileno())
            os.replace(partial, target)
            return target
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("download exhausted retries")


def download_parallel(url, target, connections, checksum):
    executable = shutil.which("aria2c")
    if executable is None:
        raise RuntimeError(
            "aria2c is required for --connections > 1; run Colab bootstrap first"
        )
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    subprocess.run(
        [
            executable,
            "--continue=true",
            "--always-resume=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--file-allocation=none",
            "--no-netrc=true",
            "--enable-color=false",
            "--summary-interval=15",
            "--max-tries=5",
            "--retry-wait=3",
            "--max-connection-per-server=" + str(connections),
            "--split=" + str(connections),
            "--min-split-size=16M",
            "--checksum=md5=" + checksum,
            "--check-integrity=true",
            "--dir=" + str(partial.parent),
            "--out=" + partial.name,
            url,
        ],
        check=True,
    )
    if not partial.is_file():
        raise RuntimeError("aria2 download did not produce the expected archive")
    os.replace(partial, target)
    return target


def prepare(root, parts, remove_archives=False, connections=1):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE_URL + "md5sum.txt", timeout=30) as response:
        checksums = {
            line.split()[1].lstrip("*"): line.split()[0]
            for line in response.read().decode().splitlines()
            if line.strip()
        }
    for part in parts:
        if part not in PARTS:
            raise ValueError("unsupported split " + part)
        marker = root / part / ".download-complete"
        if marker.is_file():
            print(part + " already verified", flush=True)
            continue
        archive = root.parent / (part + ".tar.gz")
        if not archive.is_file():
            print("Downloading " + part, flush=True)
            if connections > 1:
                expected = checksums.get(archive.name)
                if expected is None:
                    raise ValueError("No official checksum for " + archive.name)
                download_parallel(
                    BASE_URL + archive.name, archive, connections, expected
                )
            else:
                download(BASE_URL + archive.name, archive)
        digest = hashlib.md5()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        expected = checksums.get(archive.name)
        if expected is None or digest.hexdigest() != expected:
            raise ValueError(
                f"MD5 mismatch for {archive}; remove this invalid archive and retry"
            )
        safe_extract(archive, root.parent)
        if not (root / part).is_dir():
            raise ValueError(
                "archive extraction did not produce expected split directory"
            )
        marker.write_text(digest.hexdigest() + "\n")
        print("Verified and extracted " + part, flush=True)
        if remove_archives:
            archive.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/LibriSpeech")
    parser.add_argument("--parts", nargs="+", choices=PARTS, required=True)
    parser.add_argument("--remove-archives", action="store_true")
    parser.add_argument("--connections", type=int, choices=range(1, 17), default=1)
    args = parser.parse_args()
    prepare(args.root, args.parts, args.remove_archives, args.connections)
