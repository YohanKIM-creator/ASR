"""Compute corpus WER and CER from transcript directories."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.utils import CorpusMetrics
from src.text_encoder import CTCTextEncoder


def calculate(reference_dir, prediction_dir):
    references = {path.stem: path for path in Path(reference_dir).glob("*.txt")}
    predictions = {path.stem: path for path in Path(prediction_dir).glob("*.txt")}
    if not references:
        raise ValueError("no reference transcripts")
    if set(references) != set(predictions):
        missing, extra = (
            set(references) - set(predictions),
            set(predictions) - set(references),
        )
        raise ValueError(
            f"transcript ID mismatch: missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}"
        )
    metrics = CorpusMetrics()
    for uid, reference_path in sorted(references.items()):
        normalize = CTCTextEncoder.normalize_text
        metrics.update(
            normalize(reference_path.read_text(encoding="utf-8")),
            normalize(predictions[uid].read_text(encoding="utf-8")),
        )
    result = metrics.result()
    result.update(wer_percent=100 * result["wer"], cer_percent=100 * result["cer"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = calculate(args.reference_dir, args.prediction_dir)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
