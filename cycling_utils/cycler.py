import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
from cycling_utils import InterruptableDistributedSampler, atomic_torch_save, MetricsTracker

class BaseCycler():
    def __init__(self, model, dataloader, optimizer, save_path, scheduler=None, metrics_tracker=None, save_interval=100):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.save_interval = save_interval
        self.metrics_tracker = metrics_tracker
        self.iteration = 0
        self.epoch = 0
        # Register a pre forward hook
        self.model.module.register_forward_pre_hook(self.save_state)
        if os.path.exists(self.save_path):
            self.load()

    def load(self):
        
        print(f"Loading checkpoint from {self.save_path}")
        checkpoint = torch.load(self.save_path)
        print(f"{checkpoint.keys()=}")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.dataloader.sampler.load_state_dict(checkpoint["sampler_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.metrics_tracker = checkpoint["metrics"]
        self.iteration = checkpoint["iteration"]
        self.epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from {self.save_path}")
       

    def save_state(self, *args):
         
        _, batch = args
        # ignore first iteration - model params unchanged
        
        if self.iteration % self.save_interval == 0 and self.model.training and self.iteration != 0:
            state = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'sampler_state': self.dataloader.sampler.state_dict(),
                'epoch': self.epoch,
                'metrics': self.metrics_tracker,
                'iteration': self.iteration
            }
            atomic_torch_save(state, self.save_path)
        
            self.dataloader.sampler.advance(len(batch[0]))

        self.iteration += 1

    def on_epoch_end(self):
        self.dataloader.sampler._reset_progress()
        self.epoch += 1
        self.iteration = 0
        self.dataloader.sampler.set_epoch(self.epoch)
        self.metrics_tracker["train"].end_epoch()
