"""Download and extract a dataset archive."""

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_librispeech import download, safe_extract


def fetch(url, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "dataset.download"
    if "drive.google.com" in url:
        import gdown

        if not gdown.download(url, str(archive), fuzzy=True, quiet=False):
            raise ValueError("Google Drive download failed")
    else:
        download(url, archive)
    extracted = directory / "extracted"
    extracted.mkdir(exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                target = (extracted / member.filename).resolve()
                if (
                    not target.is_relative_to(extracted)
                    or (member.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("unsafe archive member")
            stream.extractall(extracted)
    elif tarfile.is_tarfile(archive):
        safe_extract(archive, extracted)
    else:
        raise ValueError("dataset URL must point to a zip or tar archive")
    roots = [path.parent for path in extracted.rglob("audio") if path.is_dir()]
    if len(roots) != 1:
        raise ValueError("archive must contain exactly one dataset audio directory")
    return roots[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", default="data/custom-download")
    args = parser.parse_args()
    print(fetch(args.url, args.output))
