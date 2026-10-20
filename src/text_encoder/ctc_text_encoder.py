import re
from string import ascii_lowercase


class CTCTextEncoder:
    EMPTY_TOK = ""
    blank_id = 0

    def __init__(self, alphabet=None):
        self.alphabet = list(alphabet or (ascii_lowercase + "' "))
        if len(set(self.alphabet)) != len(self.alphabet) or "" in self.alphabet:
            raise ValueError("alphabet must contain unique nonblank characters")
        self.vocab = [self.EMPTY_TOK] + self.alphabet
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item):
        return self.ind2char[int(item)]

    @staticmethod
    def normalize_text(text):
        text = re.sub(r"[^a-z' ]", " ", str(text).lower())
        return " ".join(text.split())

    def encode(self, text):
        import torch

        text = self.normalize_text(text)
        ids = [self.char2ind[c] for c in text]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode(self, inds):
        ids = [int(i) for i in inds if int(i) != self.blank_id]
        return self.normalize_text("".join(self.ind2char[i] for i in ids))

    def ctc_decode(self, inds):
        tokens, previous = [], None
        for value in inds:
            value = int(value)
            if value != self.blank_id and value != previous:
                tokens.append(value)
            previous = value  # blank resets a repetition
        return self.decode(tokens)

    def ctc_beam_search(self, log_probs, beam_size=32, token_topk=None):
        from src.decoding.ctc_beam import prefix_beam_search

        best = prefix_beam_search(log_probs, beam_size, self.blank_id, token_topk)[0]
        return self.decode(best[0])

    def state_dict(self):
        return {"type": "char", "alphabet": self.alphabet}

    @classmethod
    def from_state_dict(cls, state):
        if state["type"] != "char":
            raise ValueError("only character CTC checkpoints are supported")
        return cls(alphabet=state["alphabet"])
