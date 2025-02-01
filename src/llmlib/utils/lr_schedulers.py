import torch
import math


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        init_lr=0.0,
        eta_min=0.0,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.init_lr = init_lr
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase from init_lr to base_lr
            return [
                self.init_lr
                + (base_lr - self.init_lr) * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.eta_min
                + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
