# cycling utils

Utilities for cycling jobs on ISC infra or making checkpointing more robust.

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

If a process writing to a checkpoint (eg `latest.pt`) is interrupted, it can corrupt the file. If you try and resume from a corrupted checkpoint, this is going to fail your job, and you are going to have to go back to an earlier checkpoint, wasting time. To remedy this, you need to save in an atomic way - only making making sure the file is written if it suceeds. For this, we provide a drop in replacement for `torch.save`:
```
from cycling_utils import atomic_torch_save

# usual ml code
model = ...
optimizer = ...  

# save state dicts and anything else like usual!
atomic_torch_save({
  "model":model.state_dict(),
  "optimizer":optimizer.state_dict()
})
```
Alternatively it is often necessary to save multiple files which together make up your checkpoint, such that the checkpoint is not easily packaged as a dictionary object above. For this we have developed the AtomicDirectory saver which can be used to save checkpoints to successive directories, potentially removing redundant checkpoint directories as you go, in a way which is atomic.
```
# Import
from cycling_utils import AtomicDirectory
saver = AtomicDirectory(output_directory)

# Resume
latest_sym = os.path.join(output_directory, saver.symlink_name)
if os.path.exists(latest_sym):
  latest_path = os.readlink(latest_sym)
  # Load files from latest_path
  ...

# Train
for epoch in epochs:
  for batch in batches:

    # Prepare the checkpoint directory before starting iteration
    checkpoint_directory = saver.prepare_checkpoint_directory()

    # Saving files to the checkpoint_directory
    torch.save(obj, checkpoint_directory)
    ...

    # Finalzing checkpoint directory
    saver.atomic_symlink(checkpoint_directory)
```


