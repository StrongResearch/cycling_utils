import os
import torch
import torch.distributed as dist


def torch_distributed_readiness():
    dist.init_process_group("nccl")
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    world_size_check = torch.tensor([global_rank + 1], device=local_rank)
    dist.all_reduce(world_size_check)
    all_reduce_result = world_size_check.item()
    expected_result = world_size * (world_size + 1) / 2
    assert (
        all_reduce_result == expected_result
    ), f"All reduce check failed, expected_result: {expected_result}, all_reduce_result: {all_reduce_result}"
    dist.destroy_process_group()
