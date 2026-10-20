import math

from torch.optim.lr_scheduler import LambdaLR


class WarmupCosine(LambdaLR):
    def __init__(self, optimizer, warmup_steps=10000, total_steps=200000, min_lr_ratio=0.05):
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        self.warmup_steps, self.total_steps, self.min_lr_ratio = warmup_steps, total_steps, min_lr_ratio
        super().__init__(optimizer, self.multiplier)

    def multiplier(self, step):
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        progress = min(1, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
