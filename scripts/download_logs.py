"""Download the wide288 training logs from Comet."""

import argparse
import json
import os
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

EXPERIMENT = "00b40d3352ed4b02ab6e7c3854d83ca9"
ASSET = "4ff9730f9eb348c2bc97ed67005b6961"
LOG_URL = f"https://www.comet.com/api/asset/download?experimentKey={EXPERIMENT}&assetId={ASSET}"
COMPRESSED_BYTES = 4414144
PAYLOAD_BYTES = 22475838
ROWS = 81754
LAST_STEP = 81541


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if (
            redirected is not None
            and urllib.parse.urlsplit(request.full_url).netloc
            != urllib.parse.urlsplit(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/training-logs")
    )
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"metrics-through-{LAST_STEP}.jsonl"
    if target.is_file():
        payload = target.read_bytes()
    else:
        headers = {"User-Agent": "ASR-training-log-downloader"}
        if os.environ.get("COMET_API_KEY"):
            headers["Authorization"] = os.environ["COMET_API_KEY"]
        request = urllib.request.Request(LOG_URL, headers=headers)
        opener = urllib.request.build_opener(SameOriginRedirect())
        with opener.open(request, timeout=60) as response:
            compressed = response.read()
        if len(compressed) != COMPRESSED_BYTES:
            raise ValueError("Compressed training log is incomplete")
        payload = zlib.decompress(compressed, wbits=32 + zlib.MAX_WBITS)
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError("Training log is incomplete")
    rows = [json.loads(line) for line in payload.splitlines()]
    steps = {
        row["step"] for row in rows if isinstance(row.get("train/loss"), (int, float))
    }
    if len(rows) != ROWS or steps != set(range(1, LAST_STEP + 1)):
        raise ValueError("Training log has missing rows or steps")
    if not target.exists():
        temporary = target.with_suffix(".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
    print(f"Training logs: {target}\nSteps: 1–{LAST_STEP}, rows: {len(rows)}")


if __name__ == "__main__":
    main()
