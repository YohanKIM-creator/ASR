import torch
from torch import nn


class CTCLossWrapper(nn.Module):
    def __init__(self, blank=0, reduction="mean", zero_infinity=False):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.blank = blank

    def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch):
        lengths = text_encoded_length.detach().cpu().long()
        input_lengths = log_probs_length.detach().cpu().long()
        targets = text_encoded.long()
        if targets.ndim != 2:
            raise ValueError("padded targets must be B,S")
        for row, target_len, input_len in zip(targets, lengths, input_lengths):
            actual = row[: int(target_len)]
            if (actual == self.blank).any():
                raise ValueError("blank inside active target")
            repeated = int((actual[1:] == actual[:-1]).sum().item())
            if int(target_len) + repeated > int(input_len):
                raise ValueError("CTC alignment impossible: target plus repeats exceeds input")
        loss = self.ctc(log_probs.float().transpose(0, 1), targets, input_lengths, lengths)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("nonfinite CTC loss")
        return {"loss": loss}
