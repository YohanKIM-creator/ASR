import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.metrics.utils import CorpusMetrics
from src.text_encoder import CTCTextEncoder
from src.utils.checkpoint import load_checkpoint


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    torch.set_num_threads(config.inferencer.cpu_threads)
    checkpoint = load_checkpoint(config.inferencer.from_pretrained)
    encoder = CTCTextEncoder.from_state_dict(checkpoint["text_encoder"])
    device = config.inferencer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(checkpoint["config"]["model"], n_tokens=len(encoder)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    config.features = checkpoint["config"]["features"]
    loaders, _ = get_dataloaders(config, encoder, device)
    output_root = Path(config.inferencer.save_path).expanduser().resolve()
    lm_decoder = None
    if config.inferencer.decoder == "lm":
        from src.decoding.language_model import LMDecoder

        lm_decoder = LMDecoder(
            encoder,
            config.inferencer.lm_path,
            config.inferencer.lm_alpha,
            config.inferencer.lm_beta,
            config.inferencer.lm_uppercase,
        )
    for partition, loader in loaders.items():
        output_dir = output_root / partition
        output_dir.mkdir(parents=True, exist_ok=True)
        counts = CorpusMetrics()
        seen = set()
        with torch.inference_mode():
            for batch in loader:
                features = batch["spectrogram"].to(device)
                lengths = batch["spectrogram_length"].to(device)
                outputs = model(features, lengths)
                for index, (probs, length, uid) in enumerate(
                    zip(outputs["log_probs"], outputs["log_probs_length"], batch["utterance_id"])
                ):
                    if uid in seen:
                        raise ValueError("duplicate utterance id " + uid)
                    seen.add(uid)
                    probs = probs[: int(length)].cpu()
                    if config.inferencer.decoder == "greedy":
                        prediction = encoder.ctc_decode(probs.argmax(-1))
                    elif config.inferencer.decoder == "beam":
                        prediction = encoder.ctc_beam_search(
                            probs, config.inferencer.beam_size, config.inferencer.token_topk
                        )
                    elif lm_decoder is not None:
                        prediction = lm_decoder.decode(probs, config.inferencer.beam_size)
                    else:
                        raise ValueError("decoder must be greedy, beam or lm")
                    (output_dir / (uid + ".txt")).write_text(prediction + "\n", encoding="utf-8")
                    if batch["has_transcript"][index]:
                        counts.update(batch["text"][index], prediction)
        result = counts.result()
        result.update(
            {
                "predictions": len(seen),
                "decoder": config.inferencer.decoder,
                "checkpoint_step": checkpoint["global_step"],
            }
        )
        (output_root / (partition + "-metrics.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps({partition: result}, indent=2))


if __name__ == "__main__":
    main()
