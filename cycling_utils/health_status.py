import os
import torch
import socket
from collections import defaultdict
import torch.distributed as dist

class HealthChecker:
    def __init__(self):
        self.host_faults = dict()

    def check_grads(self, model, weight_check_fun=None, grad_check_fun=None, path=[], name="root"):
        '''
        Recursively checks the health of weights and grads of a model.
        check_fun should be a function which takes a single value and returns a boolean where True should
        indicate that the value is acceptable to the user.
        '''
        local_results = []  # List to store results for each parameter

        if len(list(model.named_children())) == 0:
            for pname, ptensor in list(filter(lambda ptensor: ptensor[1].grad is not None, model.named_parameters())):

                weight_nans = torch.isnan(ptensor).sum()
                weight_infs = (~torch.isfinite(ptensor)).sum()
                weight_check = 0
                if weight_check_fun is not None:
                    weight_check = (~ptensor.apply_(weight_check_fun)).sum()

                grad_nans = torch.isnan(ptensor.grad.data).sum()
                grad_infs = (~torch.isfinite(ptensor.grad.data)).sum()
                grad_check = 0
                if grad_check_fun is not None:
                    grad_check = (~ptensor.grad.data.apply_(grad_check_fun)).sum()

                # Store results for this parameter
                result = {
                    'weight_nans': weight_nans.item(),
                    'weight_infs': weight_infs.item(),
                    'weight_custom': weight_check.item(),
                    'grad_nans': grad_nans.item(),
                    'grad_infs': grad_infs.item(),
                    'grad_custom': grad_check.item(),
                }

                local_results.append(result)

        else:
            for n, c in model.named_children():
                # Recursively check children
                local_results.extend(self.check_grads(c, weight_check_fun, grad_check_fun, path + [name], n))

        # Summarise the faults that have occurred by type on this device
        local_summary = defaultdict(int)
        fault_types = ['weight_nans', 'weight_infs', 'weight_custom', 'grad_nans', 'grad_infs', 'grad_custom']
        for result in local_results:
            for typ in fault_types:
                local_summary[typ] += result[typ]
        
        # Prepare for reduction of faults from all ranks
        local_faults = torch.tensor([local_summary[k] for k in fault_types], device='cuda', requires_grad=False)
        global_faults = torch.zeros((int(os.environ["WORLD_SIZE"]), len(fault_types)), device='cuda', requires_grad=False)
        global_faults[int(os.environ["RANK"])] = local_faults

        # Reduce the global fault tensor onto rank 0
        dist.reduce(global_faults, dst=0)

        # Encode host / device names for reduction from all ranks
        host_device = f"HOST {socket.gethostname()}, GPU {os.environ['LOCAL_RANK']}" # host / device name
        host_device = 'X'+host_device if len(host_device) < 16 else host_device # Padded with an X to standardise length
        host_device = torch.tensor(list(host_device.encode('utf-8')), device='cuda', requires_grad=False) # Encode as tensor of ints
        global_hosts = torch.zeros((int(os.environ["WORLD_SIZE"]), 16), device='cuda', requires_grad=False)
        global_hosts[int(os.environ["RANK"])] = host_device

        # Reduce the global hosts tensor onto rank 0
        dist.reduce(global_hosts, dst=0)

        # Decode host names back to english
        global_hosts = [[chr(c) for c in host] for host in global_hosts] # Decode back to characters
        global_hosts = [host[1:] if host[0]=='X' else host for host in global_hosts] # Remove padding if any
        global_hosts = [''.join(host) for host in global_hosts]

        # Update count of faults on each device
        for host, faults in zip(global_hosts, global_faults):
            if host in self.host_faults:
                self.host_faults[host] += faults
            else:
                self.host_faults[host] = faults