import zipfile

import pytest

from scripts.download_dataset import fetch


def test_dataset_archive_download_and_root_detection(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("my-dataset/audio/sample.wav", b"fixture")
        stream.writestr("my-dataset/transcriptions/sample.txt", "hello")
    root = fetch(archive.as_uri(), tmp_path / "download")
    assert root.name == "my-dataset"
    assert (root / "audio/sample.wav").read_bytes() == b"fixture"
    assert (root / "transcriptions/sample.txt").read_text() == "hello"


def test_user_archive_cannot_escape_output_directory(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "unexpected")
    with pytest.raises(ValueError, match="unsafe archive"):
        fetch(archive.as_uri(), tmp_path / "download")
    assert not (tmp_path / "download/escape.txt").exists()
