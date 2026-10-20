import os
import random

import numpy as np
import torch


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def supports_native_bf16():
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return (
            torch.cuda.get_device_capability()[0] >= 8
            and torch.cuda.is_bf16_supported()
        )


def make_grad_scaler(enabled):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
