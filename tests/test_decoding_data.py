import itertools
import math
from collections import defaultdict

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.calc_metrics import calculate
from src.datasets import CustomDirDataset
from src.datasets.batch_sampler import DurationBatchSampler
from src.datasets.collate import collate_fn
from src.decoding.ctc_beam import prefix_beam_search
from src.metrics.utils import CorpusMetrics, calc_wer
from src.text_encoder import CTCTextEncoder
from src.transforms.features import SpecAugment
from src.transforms.wav_augs import AdditiveNoise, Gain, Reverberation, SpeedPerturb


def collapse(path):
    return tuple(
        value for index, value in enumerate(path) if value != 0 and (index == 0 or value != path[index - 1])
    )


def test_beam_matches_exhaustive_ctc_path_mass():
    probabilities = np.array([[0.30, 0.40, 0.30], [0.45, 0.25, 0.30], [0.20, 0.60, 0.20], [0.35, 0.20, 0.45]])
    exact = defaultdict(float)
    for path in itertools.product(range(3), repeat=4):
        exact[collapse(path)] += math.prod(probabilities[t, token] for t, token in enumerate(path))
    beam = {
        prefix: math.exp(score)
        for prefix, score in prefix_beam_search(np.log(probabilities), beam_size=1000)
        if math.isfinite(score)
    }
    assert set(beam) == set(exact)
    for prefix, mass in exact.items():
        assert beam[prefix] == pytest.approx(mass, abs=1e-12)
    assert sum(beam.values()) == pytest.approx(1.0)


def test_greedy_and_beam_known_case():
    encoder = CTCTextEncoder(alphabet=["a"])
    probabilities = torch.tensor([[0.6, 0.4], [0.6, 0.4]])
    greedy = encoder.ctc_decode(probabilities.argmax(-1))
    beam = encoder.ctc_beam_search(probabilities.log(), beam_size=8)
    assert greedy == "" and beam == "a"
    assert calc_wer("a", beam) < calc_wer("a", greedy)
    assert encoder.ctc_decode([1, 1, 0, 1, 1]) == "aa"


def test_corpus_metrics_counts_not_utterance_mean():
    metrics = CorpusMetrics()
    metrics.update("one", "")
    metrics.update("two three four five", "two three four five")
    assert metrics.result()["wer"] == 0.2
    assert calc_wer("", "") == 0 and calc_wer("", "inserted") == 1


def test_custom_dataset_resampling_and_missing_transcripts(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "transcriptions").mkdir()
    waveform = (
        np.stack([np.sin(np.arange(8000) / 20), np.sin(np.arange(8000) / 20)], -1).astype(np.float32) * 0.1
    )
    sf.write(tmp_path / "audio/stereo.wav", waveform, 8000)
    sf.write(tmp_path / "audio/unlabeled.flac", waveform[:, 0], 8000)
    (tmp_path / "transcriptions/stereo.txt").write_text("Hello, WORLD!")
    dataset = CustomDirDataset(data_dir=tmp_path, text_encoder=CTCTextEncoder())
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["audio"][0].shape == (1, 16000)
    assert batch["spectrogram"].shape[1] == 80
    assert batch["has_transcript"] == [True, False]
    assert batch["text"][0] == "hello world"
    assert batch["text_encoded"].dtype == torch.long
    assert batch["text_encoded_length"][1] == 0


def test_duration_sampler_complete_deterministic_and_budget():
    durations = [1, 3, 2, 4, 1.5, 2.5, 3.5]
    sampler = DurationBatchSampler(durations, max_seconds=8, max_items=3)
    first = list(sampler)
    assert sorted(itertools.chain.from_iterable(first)) == list(range(7))
    assert first == list(sampler)
    for batch in first:
        assert max(durations[i] for i in batch) * len(batch) <= 8
    sampler.set_epoch(3)
    assert sorted(itertools.chain.from_iterable(sampler)) == list(range(7))


def test_four_waveform_augmentations_and_spec_masks():
    torch.manual_seed(7)
    audio = 0.2 * torch.sin(torch.arange(16000).float() / 20)[None]
    augmentations = [Gain(3, 3), SpeedPerturb(speeds=(0.9,)), AdditiveNoise(20, 20), Reverberation()]
    for augmentation in augmentations:
        output = augmentation(audio)
        assert torch.isfinite(output).all()
        assert output.ndim == 2 and output.shape[0] == 1
        assert output.shape != audio.shape or not torch.equal(output, audio)
    noisy = augmentations[2](audio)
    snr = 10 * torch.log10(audio.square().mean() / (noisy - audio).square().mean())
    assert float(snr) == pytest.approx(20, abs=0.01)
    spec = torch.ones(2, 80, 100)
    masked = SpecAugment()(spec, torch.tensor([50, 100]))
    assert not torch.equal(masked, spec)
    assert torch.equal(masked[0, :, 50:], spec[0, :, 50:])


def test_metrics_require_exact_utterance_ids(tmp_path):
    reference, prediction = tmp_path / "ref", tmp_path / "pred"
    reference.mkdir()
    prediction.mkdir()
    (reference / "a.txt").write_text("one two")
    (prediction / "a.txt").write_text("one too")
    assert calculate(reference, prediction)["wer_percent"] == 50
    (prediction / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="ID mismatch"):
        calculate(reference, prediction)


def test_speed_augmentation_cannot_make_ctc_alignment_impossible(tmp_path):
    from src.transforms.wav_augs import WaveAugment

    (tmp_path / "audio").mkdir()
    (tmp_path / "transcriptions").mkdir()
    sf.write(tmp_path / "audio/fast.wav", np.sin(np.arange(16000) / 20).astype(np.float32) * 0.1, 16000)
    (tmp_path / "transcriptions/fast.txt").write_text("a" * 13)
    augmentation = WaveAugment(speed_p=1, gain_p=0, noise_p=0, reverb_p=0)
    augmentation.transforms[0].speeds = (1.1,)
    dataset = CustomDirDataset(data_dir=tmp_path, text_encoder=CTCTextEncoder(), wave_augment=augmentation)
    sample = dataset[0]
    required = 25  # 13 labels plus 12 required blank separators
    assert (sample["spectrogram"].shape[-1] + 3) // 4 >= required
    assert sample["audio"].shape[-1] == sample["original_audio"].shape[-1]


@pytest.mark.parametrize("decoder", ["greedy", "beam"])
def test_decoded_transcript_metrics_match_independent_cli(tmp_path, decoder):
    encoder = CTCTextEncoder()
    path = [encoder.char2ind["a"], encoder.char2ind[" "], 0, encoder.char2ind[" "], encoder.char2ind["b"]]
    if decoder == "greedy":
        prediction = encoder.ctc_decode(path)
    else:
        log_probs = torch.full((len(path), len(encoder)), -1000.0)
        log_probs[torch.arange(len(path)), path] = 0
        prediction = encoder.ctc_beam_search(log_probs, beam_size=16, token_topk=2)
    reference_dir, prediction_dir = tmp_path / "reference", tmp_path / "prediction"
    reference_dir.mkdir()
    prediction_dir.mkdir()
    (reference_dir / "sample.txt").write_text("a b\n", encoding="utf-8")
    (prediction_dir / "sample.txt").write_text(prediction + "\n", encoding="utf-8")
    online = CorpusMetrics()
    online.update("a b", prediction)
    independent = calculate(reference_dir, prediction_dir)
    assert online.result() == {key: independent[key] for key in online.result()}
    assert prediction == "a b"
