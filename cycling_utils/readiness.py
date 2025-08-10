import os
import torch
import torch.distributed as dist


def torch_distributed_readiness():
    assert torch.cuda.is_available(), "CUDA not available."
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


""" Useage:
Alongside user project files, create two additional files (examples below):

1. readiness_check.py
2. readiness_check.sh

Then in the experiment launch file, after sourcing venv and before launching torchrun, insert:

"bash /path/to/readiness_check.sh &&"

The idea here is that the shell script essentially launches a mini project with torchrun before the 
main project starts. The mini project initializes a process group and completes an all-reduce.
Upon failure of the mini-project, it will re-try up to 5 times before failing completely.

The hypothesis here is that something in torch may not yet be ready when torchrun starts, so we
run the mini project to catch this failure if it's going to happen.

# - readiness_check.py - #

from cycling_utils import torch_distributed_readiness
torch_distributed_readiness()

# - readiness_check.sh - #

#!/bin/bash

MAX_RETRIES=5
RETRY_COUNT=0

# Function to run the Python script
run_python_script() {
    torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$RANK /path/to/readiness_check.py
    return $?
}

# Attempt to run the Python script with retries
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES"
    if run_python_script; then
        echo "Python script executed successfully"
        exit 0
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "Python script failed. Retrying in 5 seconds..."
            sleep 5
        fi
    fi
done

# If we reached here, all attempts failed
echo "Error: Python script failed after $MAX_RETRIES attempts" >&2
exit 1
"""
