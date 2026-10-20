import ast
import json
from pathlib import Path

import pytest
import torch

from src.text_encoder import CTCTextEncoder

ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebooks_have_compilable_cells_and_no_embedded_keys():
    for name in ("inference_colab.ipynb",):
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert any(cell["cell_type"] == "markdown" for cell in notebook["cells"])
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            assert "ghp_" not in source and "wandb_v1_" not in source
            if cell["cell_type"] == "code":
                ast.parse(source)


def test_lm_decoder_uppercase_vocabulary_and_lowercase_output(tmp_path):
    pytest.importorskip("kenlm")
    pytest.importorskip("pyctcdecode")
    from src.decoding.language_model import LMDecoder

    model = tmp_path / "toy.arpa"
    model.write_text(
        "\\data\\\nngram 1=6\nngram 2=5\n\n\\1-grams:\n"
        "-3.0\t<unk>\t0\n-0.3\t<s>\t0\n-0.3\t</s>\t0\n"
        "-0.5\tTHE\t0\n-0.2\tDOG\t0\n-2.5\tBOG\t0\n"
        "\n\\2-grams:\n-0.1\t<s> THE\n-0.1\tTHE DOG\n"
        "-4.0\tTHE BOG\n-0.1\tDOG </s>\n-0.1\tBOG </s>\n\n\\end\\\n"
    )
    encoder = CTCTextEncoder()
    ids = encoder.encode("the bog").reshape(-1).tolist()
    logits = torch.full((len(ids) * 2, len(encoder)), -12.0)
    for index, token in enumerate(ids):
        logits[index * 2, token] = 0
        logits[index * 2 + 1, 0] = 0
        if token == encoder.char2ind["b"]:
            logits[index * 2, encoder.char2ind["d"]] = -0.2
    probs = logits.log_softmax(-1)
    assert encoder.ctc_decode(probs.argmax(-1)) == "the bog"
    decoder = LMDecoder(encoder, str(model), alpha=1.5, beta=0, uppercase=True)
    assert decoder.decode(probs, beam_size=32) == "the dog"
