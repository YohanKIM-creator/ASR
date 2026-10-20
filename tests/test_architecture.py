import pytest
import torch
from torch.nn import functional as F

from src.loss import CTCLossWrapper
from src.model import ConformerCTC
from src.text_encoder import CTCTextEncoder


def tiny_model(n_feats=8, n_tokens=4, dropout=0):
    return ConformerCTC(
        n_feats=n_feats,
        n_tokens=n_tokens,
        dim=16,
        num_layers=2,
        heads=2,
        kernel_size=7,
        dropout=dropout,
        subsampling_channels=4,
    )


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 15, 16, 17, 65])
def test_lengths_and_padding_independence(length):
    torch.manual_seed(11)
    model = tiny_model().eval()
    single = torch.randn(1, 8, length)
    with torch.no_grad():
        output = model(single, torch.tensor([length]))
        padded = torch.cat([F.pad(single, (0, 80 - length)), torch.randn(1, 8, 80)])
        batched = model(padded, torch.tensor([length, 80]))
    expected = (length + 3) // 4
    assert output["log_probs"].shape == (1, expected, 4)
    assert int(batched["log_probs_length"][0]) == expected
    torch.testing.assert_close(
        output["log_probs"][0], batched["log_probs"][0, :expected], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(output["log_probs"].exp().sum(-1), torch.ones(1, expected))


def test_ctc_repeats_guard_and_gradients():
    model = tiny_model(n_tokens=3)
    batch = {
        "spectrogram": torch.randn(2, 8, 24),
        "spectrogram_length": torch.tensor([24, 20]),
        "text_encoded": torch.tensor([[1, 1], [1, 2]]),
        "text_encoded_length": torch.tensor([2, 2]),
    }
    outputs = model(**batch)
    loss = CTCLossWrapper()(**batch, **outputs)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.output.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="alignment impossible"):
        CTCLossWrapper()(
            log_probs=torch.log_softmax(torch.randn(1, 2, 3), -1),
            log_probs_length=torch.tensor([2]),
            text_encoded=torch.tensor([[1, 1]]),
            text_encoded_length=torch.tensor([2]),
        )


def test_one_batch_conformer_overfit():
    torch.manual_seed(4)
    model = tiny_model(n_tokens=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    batch = {
        "spectrogram": torch.randn(2, 8, 24),
        "spectrogram_length": torch.tensor([24, 24]),
        "text_encoded": torch.tensor([[1], [2]]),
        "text_encoded_length": torch.tensor([1, 1]),
    }
    criterion = CTCLossWrapper()
    losses = []
    for _ in range(140):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(**batch, **outputs)["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    encoder = CTCTextEncoder(alphabet=["a", "b"])
    model.eval()
    predictions = [encoder.ctc_decode(ids) for ids in model(**batch)["log_probs"].argmax(-1)]
    assert predictions == ["a", "b"]
    assert losses[-1] < 0.03 and losses[-1] < losses[0] * 0.02


def test_gradient_checkpointing_backward():
    model = tiny_model(dropout=0.1)
    model.gradient_checkpointing = True
    output = model(torch.randn(2, 8, 40), torch.tensor([40, 33]))
    output["log_probs"].square().mean().backward()
    assert model.blocks[0].attention.qkv.weight.grad is not None
