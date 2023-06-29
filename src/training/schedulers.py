import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CustomCosineAnnealingWarmupScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        self.cycle_length = int(total_steps * cycles)
        super(CustomCosineAnnealingWarmupScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            # Warmup phase: linearly increase the learning rate
            return float(step) / float(max(1, self.warmup_steps))
        else:
            # Cosine annealing phase
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * self.cycles * 2 * progress))
