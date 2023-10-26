import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
from cycling_utils import InterruptableDistributedSampler, atomic_torch_save, MetricsTracker

class BaseCycler():
    def __init__(self, model, dataloader, optimizer, save_path, batch_size, scheduler=None, metrics_tracker=None, save_interval=100):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.metrics_tracker = metrics_tracker

        self.dataloader.sampler = InterruptableDistributedSampler(self.dataloader.dataset)

        # Register a pre forward hook
        self.model.register_module_forward_pre_hook(self.save_state)

        if os.path.exists(self.save_path):
            self.load()

    def load(self):

        print(f"Loading checkpoint from {checkpoint_latest}")
        checkpoint = torch.load(self.save_path)
        print(f"{checkpoint.keys()=}")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.dataloader.sampler.load_state_dict(checkpoint["sampler_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.metrics_tracker.load_state_dict(checkpoint["metrics"])
        print(f"Loaded checkpoint from {checkpoint_latest}")
       

    def save_state(self, module, grad_input, grad_output):
        
        # ignore first iteration - model params unchanged

        if self.iteration % self.save_interval == 1:
            state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'sampler_state': self.dataloader.sampler.state_dict(),
                'epoch': self.epoch,
                'metrics': self.metrics_tracker,
                'iteration': self.iteration
            }
            atomic_torch_save(state, self.save_path)
        
        self.dataloader.sampler.advance(self.batch_size)

    def on_epoch_end(self):
        self.dataloader.sampler._reset_progress()
        self.epoch += 1
        self.dataloader.sampler.set_epoch(self.epoch)
        self.metrics_tracker["train"].end_epoch()
