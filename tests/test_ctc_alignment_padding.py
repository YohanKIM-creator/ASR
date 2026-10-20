import numpy as np
import soundfile as sf
import torch

from src.datasets import CustomDirDataset
from src.datasets.collate import collate_fn
from src.loss import CTCLossWrapper
from src.model import ConformerCTC
from src.text_encoder import CTCTextEncoder


def test_train_only_minimal_padding_keeps_signal_and_allows_ctc(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "transcriptions").mkdir()
    waveform = np.sin(np.arange(58800) / 20).astype(np.float32) * 0.1
    sf.write(tmp_path / "audio/short.wav", waveform, 16000, subtype="FLOAT")
    (tmp_path / "transcriptions/short.txt").write_text("abc" * 31)
    encoder = CTCTextEncoder()
    dataset = CustomDirDataset(
        data_dir=tmp_path,
        text_encoder=encoder,
        ctc_alignment_padding_frames=1,
    )
    sample = dataset[0]
    assert sample["original_audio"].shape[-1] == 58800
    assert sample["audio"].shape[-1] == 58880
    torch.testing.assert_close(sample["audio"][..., :58800], sample["original_audio"], rtol=0, atol=0)
    batch = collate_fn([sample])
    model = ConformerCTC(
        n_feats=80,
        n_tokens=len(encoder),
        dim=16,
        num_layers=1,
        heads=2,
        kernel_size=7,
        dropout=0,
        subsampling_channels=4,
    ).eval()
    output = model(**batch)
    assert output["log_probs_length"].tolist() == [93]
    assert torch.isfinite(CTCLossWrapper()(**batch, **output)["loss"])


def test_validation_default_preserves_original_input_even_with_long_target(tmp_path):
    import pytest

    (tmp_path / "audio").mkdir()
    (tmp_path / "transcriptions").mkdir()
    waveform = np.sin(np.arange(58800) / 20).astype(np.float32) * 0.1
    sf.write(tmp_path / "audio/short.wav", waveform, 16000, subtype="FLOAT")
    (tmp_path / "transcriptions/short.txt").write_text("abc" * 31)
    encoder = CTCTextEncoder()
    sample = CustomDirDataset(data_dir=tmp_path, text_encoder=encoder)[0]
    assert sample["audio"].shape[-1] == 58800
    torch.testing.assert_close(sample["audio"], sample["original_audio"], rtol=0, atol=0)
    batch = collate_fn([sample])
    model = ConformerCTC(
        n_feats=80,
        n_tokens=len(encoder),
        dim=16,
        num_layers=1,
        heads=2,
        kernel_size=7,
        dropout=0,
        subsampling_channels=4,
    ).eval()
    output = model(**batch)
    assert output["log_probs_length"].tolist() == [92]
    with pytest.raises(ValueError, match="CTC alignment impossible"):
        CTCLossWrapper()(**batch, **output)
