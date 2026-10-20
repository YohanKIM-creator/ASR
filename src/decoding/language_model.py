class LMDecoder:
    def __init__(self, encoder, path, alpha=0.5, beta=1.0, uppercase=True):
        if path is None:
            raise ValueError("set inferencer.lm_path to downloaded KenLM model")
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError as exc:
            raise ImportError("install requirements.txt for LM decoding") from exc
        labels = encoder.vocab.copy()
        if uppercase:
            labels = [label.upper() if not label.startswith("<") else label for label in labels]
        labels[0] = ""
        self.normalize = encoder.normalize_text
        self.decoder = build_ctcdecoder(labels, kenlm_model_path=path, alpha=alpha, beta=beta)

    def decode(self, log_probs, beam_size=64):
        return self.normalize(self.decoder.decode(log_probs.detach().cpu().numpy(), beam_width=beam_size))
