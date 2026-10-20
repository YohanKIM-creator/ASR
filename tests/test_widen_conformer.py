import torch

from src.model import ConformerCTC
from src.loss import CTCLossWrapper
from src.text_encoder import CTCTextEncoder


def test_widened_conformer_keeps_function_for_padding_and_rope():
    from scripts.widen_conformer import widen_model

    torch.manual_seed(41)
    old = ConformerCTC(
        n_feats=40,
        dim=16,
        heads=2,
        num_layers=2,
        expansion=4,
        kernel_size=5,
        dropout=0.1,
        subsampling_channels=4,
    ).eval()
    new = ConformerCTC(
        n_feats=40,
        dim=32,
        heads=4,
        num_layers=2,
        expansion=4,
        kernel_size=5,
        dropout=0.1,
        subsampling_channels=4,
    ).eval()
    widen_model(old, new)
    features = torch.randn(3, 40, 97)
    lengths = torch.tensor([97, 65, 33])
    with torch.inference_mode():
        expected = old(features, lengths)
        actual = new(features, lengths)
    assert torch.equal(actual["log_probs_length"], expected["log_probs_length"])
    assert torch.allclose(actual["log_probs"], expected["log_probs"], atol=2e-5, rtol=2e-5)
    assert new.num_parameters > old.num_parameters * 3
    assert old.blocks[0].attention.head_dim == new.blocks[0].attention.head_dim


def test_widened_checkpointing_model_can_break_duplicate_symmetry():
    from scripts.widen_conformer import widen_model

    torch.manual_seed(42)
    old = ConformerCTC(dim=16, heads=2, num_layers=2, kernel_size=5, dropout=0.1, subsampling_channels=4)
    new = ConformerCTC(
        dim=32,
        heads=4,
        num_layers=2,
        kernel_size=5,
        dropout=0.1,
        subsampling_channels=4,
        gradient_checkpointing=True,
    )
    widen_model(old, new)
    encoder = CTCTextEncoder()
    target = encoder.encode("hello world").repeat(3, 1)
    outputs = new(torch.randn(3, 80, 97), torch.tensor([97, 85, 65]))
    loss = CTCLossWrapper()(text_encoded=target, text_encoded_length=torch.tensor([11, 11, 11]), **outputs)[
        "loss"
    ]
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW(new.parameters(), lr=0.001)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
    optimizer.step()
    weight = new.blocks[0].attention.qkv.weight
    assert not torch.equal(weight[:16, :16], weight[16:32, :16])
    state = new.state_dict()
    filters = state["blocks.0.convolution.depthwise.weight"]
    assert not torch.allclose(filters[:16], filters[16:32], atol=1e-8, rtol=1e-7)
