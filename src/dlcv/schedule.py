import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupThenScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, post_warmup_scheduler: _LRScheduler, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.post_warmup_scheduler = post_warmup_scheduler
        self.finished_warmup = False
        super(WarmupThenScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            if not self.finished_warmup:
                self.post_warmup_scheduler.base_lrs = self.base_lrs
                self.finished_warmup = True
            self.post_warmup_scheduler.last_epoch = self.last_epoch - self.warmup_steps
            return self.post_warmup_scheduler.get_lr()

    def step(self, epoch=None):
        if self.finished_warmup:
            self.post_warmup_scheduler.step(epoch)
        else:
            super(WarmupThenScheduler, self).step(epoch)