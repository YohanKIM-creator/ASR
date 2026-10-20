import itertools
import subprocess
import sys
import unittest
from pathlib import Path
from string import ascii_lowercase

from src.text_encoder import CTCTextEncoder


class CharacterCodecTests(unittest.TestCase):
    def test_saved_character_checkpoint_keeps_token_ids(self):
        state = {"type": "char", "alphabet": list(ascii_lowercase + "' ")}
        encoder = CTCTextEncoder.from_state_dict(state)
        self.assertEqual(encoder.state_dict(), state)
        self.assertEqual(len(encoder), 29)
        self.assertEqual(encoder.blank_id, 0)
        self.assertEqual(encoder.char2ind["a"], 1)
        self.assertEqual(encoder.char2ind["'"], 27)
        self.assertEqual(encoder.char2ind[" "], 28)

    def test_text_and_repeated_ctc_labels_round_trip(self):
        encoder = CTCTextEncoder()
        ids = [8, 5, 12, 12, 15, 28, 23, 15, 18, 12, 4]
        self.assertEqual(encoder.decode(ids), "hello world")
        expanded = itertools.chain.from_iterable((value, value, 0) for value in ids)
        self.assertEqual(encoder.ctc_decode(expanded), "hello world")
        self.assertEqual(encoder.ctc_decode([1, 1, 0, 1]), "aa")
        self.assertEqual(encoder.normalize_text("  HELLO,  World! 123 "), "hello world")
        self.assertEqual(encoder.ctc_decode([0, 0]), "")

    def test_invalid_or_incompatible_vocabularies_fail(self):
        for alphabet in (["a", "a"], [""]):
            with self.assertRaises(ValueError):
                CTCTextEncoder(alphabet=alphabet)
        with self.assertRaises(ValueError):
            CTCTextEncoder.from_state_dict({"type": "unsupported", "alphabet": ["a"]})

    def test_text_operations_do_not_import_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys; from src.text_encoder import CTCTextEncoder; "
                "CTCTextEncoder().ctc_decode([1, 0, 2]); assert 'torch' not in sys.modules",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
