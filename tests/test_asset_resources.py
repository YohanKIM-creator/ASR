import gzip
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(source, output, lm=None):
    args = [
        sys.executable,
        str(ROOT / "scripts/download_assets.py"),
        "--checkpoint-url",
        source,
        "--output",
        str(output),
    ]
    return args + (["--lm-url", lm] if lm else ["--skip-lm"])


def test_gzip_language_model_becomes_a_ready_arpa_file(tmp_path):
    checkpoint = tmp_path / "source-model.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    expected = b"\\data\\\nngram 1=1\n\\1-grams:\n-1.0 HELLO 0.0\n\\end\\\n"
    archive = tmp_path / "source-lm.arpa.gz"
    archive.write_bytes(gzip.compress(expected))
    output = tmp_path / "assets"
    result = subprocess.run(
        command(checkpoint.as_uri(), output, archive.as_uri()),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "lm.arpa").read_bytes() == expected
    assert "language_model: " + str(output / "lm.arpa") in result.stdout


def test_invalid_gzip_language_model_does_not_publish_partial_arpa(tmp_path):
    checkpoint = tmp_path / "source-model.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    archive = tmp_path / "source-lm.arpa.gz"
    archive.write_bytes(b"invalid gzip archive")
    output = tmp_path / "assets"
    result = subprocess.run(
        command(checkpoint.as_uri(), output, archive.as_uri()),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "BadGzipFile" in result.stderr
    assert {p.name for p in output.iterdir()} == {"model.pt", "model.pt.url"}


def test_empty_checkpoint_download_can_be_retried_without_manual_cleanup(tmp_path):
    source = tmp_path / "source-model.pt"
    source.touch()
    output = tmp_path / "assets"
    args = command(source.as_uri(), output)
    failed = subprocess.run(args, capture_output=True, text=True)
    assert failed.returncode != 0 and "Downloaded file is empty" in failed.stderr
    assert not list(output.iterdir())
    expected = b"complete checkpoint"
    source.write_bytes(expected)
    retried = subprocess.run(args, capture_output=True, text=True)
    assert retried.returncode == 0, retried.stdout + retried.stderr
    assert (output / "model.pt").read_bytes() == expected


def test_changed_checkpoint_url_replaces_the_cached_model(tmp_path):
    output = tmp_path / "assets"
    for name, content in [("first.pt", b"first model"), ("second.pt", b"second model")]:
        source = tmp_path / name
        source.write_bytes(content)
        result = subprocess.run(
            command(source.as_uri(), output), capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (output / "model.pt").read_bytes() == content


def test_cached_model_is_reused_and_force_download_refreshes_it(tmp_path):
    source = tmp_path / "source-model.pt"
    source.write_bytes(b"first model")
    output = tmp_path / "assets"
    args = command(source.as_uri(), output)
    first = subprocess.run(args, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    source.unlink()
    cached = subprocess.run(args, capture_output=True, text=True)
    assert cached.returncode == 0, cached.stdout + cached.stderr
    assert (output / "model.pt").read_bytes() == b"first model"
    source.write_bytes(b"updated model")
    refreshed = subprocess.run(args + ["--force"], capture_output=True, text=True)
    assert refreshed.returncode == 0, refreshed.stdout + refreshed.stderr
    assert (output / "model.pt").read_bytes() == b"updated model"


def test_failed_google_drive_transfer_does_not_publish_partial_checkpoint(tmp_path):
    (tmp_path / "gdown.py").write_text(
        "from pathlib import Path\ndef download(url, output, **kwargs):\n"
        "    Path(output).write_bytes(b'partial provider response')\n    return None\n"
    )
    output = tmp_path / "assets"
    args = command("https://drive.google.com/file/d/test-file-id/view", output)
    result = subprocess.run(
        args,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode != 0
        and "Google Drive download did not complete" in result.stderr
    )
    assert not list(output.iterdir())


def test_interrupted_http_download_resumes_before_publishing_checkpoint(tmp_path):
    expected = b"complete model after an interrupted HTTP response"
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            request_range = self.headers.get("Range")
            ranges.append(request_range)
            offset = (
                int(request_range.removeprefix("bytes=").removesuffix("-"))
                if request_range
                else 0
            )
            payload = expected[offset:]
            self.send_response(206 if offset else 200)
            self.send_header("Content-Length", str(len(payload)))
            if offset:
                self.send_header(
                    "Content-Range",
                    f"bytes {offset}-{len(expected) - 1}/{len(expected)}",
                )
            self.end_headers()
            self.wfile.write(payload if offset else payload[:10])
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = f"http://127.0.0.1:{server.server_port}/model.pt"
        output = tmp_path / "assets"
        result = subprocess.run(command(source, output), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (output / "model.pt").read_bytes() == expected
        assert ranges == [None, "bytes=10-"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
