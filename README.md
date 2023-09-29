# cycling utils

Utilities for cycling jobs on ISC infra or making checkpointing more robust.

## features 

Regardless of whether it's a hardware failure or you are cycling jobs on ISC infra, the ability to resume a machine learning job from a checkpoint in a robust way should be important for anyone. These utilities have several useful helpers for making resuming from a checkpoint more safe:

- Atomic torch.save
- Interruptable distributed sampler for torch data loaders 
- Torch lightning integration

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

### interruptable sampling

Another useful utility is to keep track of where you are in a dataset. This is so you can resume from the last sample rather than oversampling from the beginning. We provide a sampler for torch dataloaders which has a state dict so it can be saved in a checkpoint. Roughly how it works:

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
```

