# cycling utils

Utilities for cycling jobs on ISC infra or making checkpointing more robust.

## dependencies

The `cycling_utils` package has the following dependencies.
- `torch`
- `torchvision`

These are not installed with `cycling_utils` to allow the user freedom to install any version of torch and torchvision they require for their project.

## features 

Regardless of whether it's a hardware failure or you are cycling jobs on ISC infra, the ability to resume a machine learning job from a checkpoint in a robust way should be important for anyone. These utilities have several useful helpers for  safely resuming from a checkpoint, importantly:
- Interruptable sampling
- Dataset Sharding
- Atomic saving

You will also find here some helpful utilities for tracking your metrics and logging progress. 

### interruptable sampling

It is important when suspending and resuming training to pick up right where you left off, including which epoch and how far through that epoch you had progressed. We provide a sampler for torch dataloaders which has a state dict so it can be saved in a checkpoint. Roughly how it works:

```
from cycling_utils import InterruptableDistributedSampler

dataset = ... # your dataset
sampler  = InterruptableDistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)

# if a checkpoint exists you can load from a checkpoint
if resume_from_checkpoint:
  sampler.load_state_dict(checkpoint["sampler"])

# use in your train script
for batch in loader:

  # advance by your batch size each epoch
  # batch_size = ...
  sampler.advance(batch_size)

  # save sampler.state_dict() in your checkpoint

  atomic_torch_save({
    "sampler":sampler.state_dict(),
    # anything else
  })
```

### dataset sharding

It is common for training code to move the dataset to the GPU before starting training. In some cases the available VRAM on a single GPU will be insufficient to store the entire dataset. In that case it may be possible to shard the dataset across all GPUs in the cluster so that each GPU hosts a potentially disjoint subset of the whole dataset. We provide a Dataset for this purpose, which is based on the DatasetFolder (or ImageFolder) dataset from Torchvision. To complement this we also provide a simplified sampler which performs a similar function to the InterruptableDistributedSampler above, but does not same selectively to distribute the data across the cluster as this is taken care of by the dataset.

```bash
from cycling_utils import DistributedShardedDataset, InterruptibleSampler

dataset = DistributedShardedDataset(
  root=root,
  rank=rank,
  world_size=world_size,
  samples_pct_per_replica=samples_pct_per_replica
)
sampler  = InterruptibleSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)

# proceed as above
```

### atomic saving

If a process writing to a checkpoint (eg `latest.pt`) is interrupted, it can corrupt the file. If you try and resume from a corrupted checkpoint, this is going to fail your job, and you are going to have to go back to an earlier checkpoint, wasting time. To remedy this, you need to save in an atomic way - making sure the checkpoint saves completely or not at all. For this, we provide a drop in replacement for `torch.save`:
```
from cycling_utils import atomic_torch_save

# usual ml code
model = ...
optimizer = ...  

# save state dicts and anything else like usual!
atomic_torch_save({
  "model":model.state_dict(),
  "optimizer":optimizer.state_dict()
}, checkpoint_output_path)
```

The Strong Compute ISC uses an Artifacts system for saving experiment outputs. Saving Checkpoint type Artifacts requires using the AtomicDirectory saver. The User is responsible for implementing AtomicDirectory saver and saving checkpoints at their desired frequency.

The AtomicDirectory saver is designed for use in a distributed process group (e.g. via torchrun). Each process must initialize the saver. AtomicDirectory accepts the following arguments at initialization:

- output_directory: root directory for all ouputs from the experiment, should always be set to the $CHECKPOINT_ARTIFACT_PATH environment variable when training on the Strong Compute ISC.
- is_master: a boolean to indicate whether the process running the AtomicDirectory saver is the master rank in the process group.
- name: a name for the AtomicDirectory saver. If the user is running multiple savers in parallel, each must be given a unique name.
- keep_last: the number of previous checkpoints to retain locally, should always be -1 when saving Checkpoint Artifacts to the $CHECKPOINT_ARTIFACT_PATH on the Strong Compute.

The AtomicDirectory saver works by saving each checkpoint to a new directory, and then saving a symlink to that directory which should be read upon resume to obtain the path to the latest checkpoint directory.

Checkpoint Artifacts saved to $CHECKPOINT_ARTIFACT_PATH are synchronized every 10 minutes and/or at the end of each cycle on Strong Compute. Upon synchronization, the latest symlinked checkpoint/s saved by AtomicDirectory saver/s in the $CHECKPOINT_ARTIFACT_PATH directory will be shipped to Checkpoint Artifacts for the experiment. Any non-latest checkpoints saved since the previous Checkpoint Artifact sychronization will be deleted and not shipped.

The user can force non-latest checkpoints to also ship to Checkpoint Artifacts by calling `prepare_checkpoint_directory`
with `force_save = True`. This can be used, for example:
- to ensure milestone checkpoints are archived for later analysis, or
- to ensure that checkpoints are saved each time model performance improves.

The optional `strategy` argument to `prepare_checkpoint_directory` determines what happens if processes differ on the `force_save` argument.

- `strategy = "any"` (default) will `force_save` the checkpoint if any process passes `force_save = True`
- `strategy = "all"` will `force_save` the checkpoint if and only if ALL processes pass `force_save = True`

Example usage of AtomicDirectory on the Strong Compute ISC launching with torchrun as follows.

```
>>> import os
>>> import torch
>>> import torch.distributed as dist
>>> from cycling_utils import AtomicDirectory, atomic_torch_save

>>> dist.init_process_group("nccl")
>>> rank = int(os.environ["RANK"])
>>> output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]

>>> # Initialize the AtomicDirectory - called by ALL ranks
>>> saver = AtomicDirectory(output_directory, is_master=rank==0)

>>> # Resume from checkpoint
>>> latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
>>> if os.path.exists(latest_symlink_file_path):
>>>    latest_checkpoint_path = os.readlink(latest_symlink_file_path)

>>>     # Load files from latest_checkpoint_path
>>>     checkpoint_path = os.path.join(latest_checkpoint_path, "checkpoint.pt")
>>>     checkpoint = torch.load(checkpoint_path)
>>>     ...

>>> for epoch in epochs:
>>>     for step, batch in enumerate(batches):

>>>         ...training...

>>>         if is_save_step:
>>>             # prepare the checkpoint directory - called by ALL ranks
>>>             checkpoint_directory = saver.prepare_checkpoint_directory()

>>>             # saving files to the checkpoint_directory
>>>             if is_master_rank:
>>>                 checkpoint = {...}
>>>                 checkpoint_path = os.path.join(checkpoint_directory, "checkpoint.pt")
>>>                 atomic_torch_save(checkpoint, checkpoint_path)

>>>             # finalizing checkpoint with symlink - called by ALL ranks
>>>             saver.symlink_latest(checkpoint_directory)
```

