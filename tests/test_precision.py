import pytest
import torch

from src.utils.init_utils import supports_native_bf16


@pytest.mark.parametrize(
    "cuda, capability, emulated_support, expected",
    [
        (True, (7, 5), True, False),  # T4 reports emulation support on torch 2.11.
        (True, (8, 0), True, True),  # A100 has native bf16.
        (False, None, False, False),
    ],
)
def test_amp_uses_native_gpu_precision(
    monkeypatch, cuda, capability, emulated_support, expected
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: emulated_support)
    assert supports_native_bf16() is expected
