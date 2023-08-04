import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.optim.lr_scheduler import _LRScheduler

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

class InvSqrtScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, scale=0.01, print_lr=False, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.scale = scale
        self.print_lr = print_lr
        super(InvSqrtScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            lr = 0.05 if self.num_warmup_steps == 0 else 0
        elif self.last_epoch < self.num_warmup_steps:
            lr = float(self.last_epoch) / float(self.num_warmup_steps)
        else:
            lr = (1. / math.sqrt(self.last_epoch)) * self.scale

        if self.print_lr: 
            print(lr)

        return [lr for _ in self.optimizer.param_groups]

    def get_last_lr(self):
        """ Return last computed learning rate by scheduler. """
        return self.get_lr()


@torch.inference_mode()
def update_grad(model, lr, momentum):
    grad = sum(param.grad.norm().item() for param in model.parameters()) / sum(p.numel() for p in model.parameters())
    return adaptive_momentum_scheduler(grad,lr,momentum)

def adaptive_momentum_scheduler(grad, lr, momentum, momentum_decay=0.9, lr_min=1e-5, lr_max=0.1):
    # Update the momentum
    momentum = momentum_decay * momentum + (1 - momentum_decay) * grad

    # Adjust the learning rate based on the momentum
    if momentum > 0:
        # If the momentum is high, decrease the learning rate
        lr /= (1 + momentum)
    else:
        # If the momentum is low, increase the learning rate
        lr *= (1 - momentum)

    # Clip the learning rate to be within [lr_min, lr_max]
    lr = max(lr_min, min(lr, lr_max))

    return lr, momentum
