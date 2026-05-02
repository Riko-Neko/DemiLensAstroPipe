from torch.optim import lr_scheduler

import math


class WarmupSchedulerWrapper:
    def __init__(self, optimizer, scheduler_class, scheduler_kwargs,
                 warmup_steps = 0, base_lr = 1e-3, warmup_enabled = False):
        """
        :param optimizer: Optimizer object
        :param scheduler_class: Scheduler class from torch.optim.lr_scheduler
        :param scheduler_kwargs: Parameters dictionary for the scheduler, e.g., {"T_max": 30, "eta_min": 1e-5}
        :param warmup_steps: Number of warmup steps
        :param base_lr: Base learning rate
        :param warmup_enabled: Whether to enable warmup
        """
        self.optimizer = optimizer
        self.scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.warmup_enabled = warmup_enabled
        self.current_step = 0

    def step(self):
        """Update learning rate at each step"""
        self.current_step += 1
        if self.warmup_enabled and self.current_step <= self.warmup_steps:
            # Warmup phase: linearly increase learning rate
            lr = self.base_lr * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Normal scheduler phase
            self.scheduler.step()

    def get_lr(self):
        """Get current learning rate"""
        if self.warmup_enabled and self.current_step <= self.warmup_steps:
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            return self.scheduler.get_last_lr()[0]
