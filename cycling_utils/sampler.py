import math
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain, repeat
import torch
from torch.utils.data import Dataset, DistributedSampler, Sampler


class HasNotResetProgressError(Exception):
    pass


class AdvancedTooFarError(Exception):
    pass


class InterruptableDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        This is a DistributedSampler that can be suspended and resumed.

        This works by keeping track of the epoch and progress within the epoch.
        The progress is the number of samples that have been returned by the
        sampler. The epoch is the number of times the sampler has been iterated
        over.

        The epoch is incremented at the start of each epoch. The epoch is set
        to 0 at initialization.

        The progress is incremented by the number of samples returned by the
        sampler. The progress is reset to 0 at the end of each epoch.

        Suspending and resuming the sampler is done by saving and loading the
        state dict. The state dict contains the epoch and progress. This works
        because the permutation of the dataset is deterministic given the seed
        and epoch.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.progress = 0
        self._has_reset_progress = True

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            self.indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(self.indices)
            if padding_size <= len(self.indices):
                self.indices += self.indices[:padding_size]
            else:
                self.indices += (
                    self.indices * math.ceil(padding_size / len(self.indices))
                )[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            self.indices = self.indices[: self.total_size]
        assert len(self.indices) == self.total_size

        # subsample
        self.indices = self.indices[self.rank : self.total_size : self.num_replicas]
        assert len(self.indices) == self.num_samples

    def reset_progress(self):
        self.progress = 0
        self._has_reset_progress = True

    def set_epoch(self, epoch):
        if not self._has_reset_progress:
            raise HasNotResetProgressError(
                "You must reset progress before setting epoch e.g. `sampler.reset_progress()`\nor use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`"
            )
        self.epoch = epoch

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        if not self.progress <= self.num_samples:
            raise AdvancedTooFarError(
                f"progress should be less than or equal to the number of samples. progress: {self.progress}, num_samples: {self.num_samples}"
            )
        self.epoch = state_dict["epoch"]

    def advance(self, n: int):
        """
        Record that n samples have been consumed.
        """
        self.progress += n
        self._has_reset_progress = False
        if self.progress > self.num_samples:
            raise AdvancedTooFarError(
                "You have advanced too far. You can only advance up to the total size of the dataset."
            )

    def __iter__(self):
        for idx in self.indices[self.progress :]:
            yield idx

    @contextmanager
    def in_epoch(self, epoch):
        """
        This context manager is used to set the epoch. It is used like this:

        ```
        for epoch in range(0, 10):
            with sampler.in_epoch(epoch):
                for step, (x, ) in enumerate(dataloader):
                    # work would be done here...
        ```
        """
        self.set_epoch(epoch)
        yield
        self.reset_progress()


class InterruptableDistributedGroupedBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        group_ids: list[int],
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        This is a DistributedSampler that can be suspended and resumed.
        This works by keeping track of the sample batches that have already been
        dispatched.

        This InterruptableDistributedGroupedBatchSampler also enables the sampling
        strategy exhibited in the torch vision detection reference wherein batches
        are created from images from within the same 'group', defined in the
        torchvision example by similarity of image aspect ratio.

        https://github.com/pytorch/vision/tree/main/references/detection

        Any grouping can be similarly applied by passing suitable group_ids.

        For this reason, InterruptableDistributedGroupedBatchSampler progress is
        tracked in units of batches, not samples. This is an important
        distinction from the InterruptableDistributedSampler which tracks progress
        in units of samples. The progress is reset to 0 at the end of each epoch.

        The epoch is set to 0 at initialization and incremented at the start
        of each epoch.

        Suspending and resuming the sampler is done by saving and loading the
        state dict. The state dict contains the epoch and progress.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        # OVERALL STATUS INDICATOR
        self.progress = 0
        self._has_reset_progress = True
        self.batch_size = batch_size
        self.group_ids = group_ids

    def _create_batches(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make dataset evenly divisible accross ranks
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make dataset evenly divisible accross ranks
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample indices to use on this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # PRE-COMPUTE GROUPED BATCHES
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)
        self.num_batches = math.ceil(len(indices) / self.batch_size)

        batches = []  # pre-computed so progress refers to batches, not samples.
        for idx in indices:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                batches.append(buffer_per_group[group_id])
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        num_remaining = self.num_batches - len(batches)
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with the largest number
            # of elements
            for group_id, _ in sorted(
                buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True
            ):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = self._repeat_to_at_least(
                    samples_per_group[group_id], remaining
                )
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                batches.append(buffer_per_group[group_id])
                num_remaining -= 1
                if num_remaining == 0:
                    break

        # Check that the batches are all good to go
        assert len(batches) == self.num_batches
        return batches

    def _repeat_to_at_least(self, iterable, n):
        repeat_times = math.ceil(n / len(iterable))
        repeated = chain.from_iterable(repeat(iterable, repeat_times))
        return list(repeated)

    def _reset_progress(self):
        self.progress = 0
        self._has_reset_progress = True

    def set_epoch(self, epoch: int) -> None:
        raise NotImplementedError(
            "Use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`"
        )

    def _set_epoch(self, epoch):
        if not self._has_reset_progress:
            raise HasNotResetProgressError(
                "You must reset progress before setting epoch e.g. `sampler.reset_progress()`\nor use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`"
            )
        self.epoch = epoch
        self.batches = self._create_batches()

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        if not self.progress <= self.num_batches:
            raise AdvancedTooFarError(
                f"progress should be less than or equal to the number of batches. progress: {self.progress}, num_batches: {self.num_batches}"
            )
        self.epoch = state_dict["epoch"]

    def advance(self):
        """
        Record that one batch has been consumed.
        """
        self.progress += 1
        if self.progress > self.num_batches:
            raise AdvancedTooFarError(
                f"You have advanced too far. You can only advance up to the total number of batches: {self.num_batches}."
            )

    def __iter__(self):
        # slice from progress to pick up where we left off
        for batch in self.batches[self.progress :]:
            yield batch

    def __len__(self):
        return self.num_batches

    @contextmanager
    def in_epoch(self, epoch):
        """
        This context manager is used to set the epoch. It is used like this:
        ```
        for epoch in range(0, 10):
            with sampler.in_epoch(epoch):
                for step, (x, ) in enumerate(dataloader):
                    # work would be done here...
        ```
        """
        self._set_epoch(epoch)
        yield
        self._reset_progress()


class InterruptableSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        This 'InterruptibleSampler' class is similar to the 'InterruptibleDistributedSampler' class in that it keeps
        track of progress through the dataset, but does not include functionality to distribute the dataset accross
        GPUs in the cluster.

        This class is intended for use with the 'DistributedShardedDataset' dataset class which achieves distribution of
        an ImageFolder-like dataset in shards to each GPU in the cluster.
        """
        super().__init__()
        self.progress = 0
        self._has_reset_progress = True
        self.epoch = 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

    def _reset_progress(self):
        self.progress = 0
        self._has_reset_progress = True

    def set_epoch(self, epoch):
        if not self._has_reset_progress:
            raise HasNotResetProgressError(
                "You must reset progress before setting epoch e.g. `sampler.reset_progress()`\nor use `with sampler.in_epoch(epoch)` instead of `sampler.set_epoch(epoch)`"
            )
        self.epoch = epoch

    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        if self.progress > len(self.dataset):
            raise AdvancedTooFarError(
                f"Progress [{self.progress}] should be less than or equal to the size of the dataset [{len(self.dataset)}]"
            )
        self.epoch = state_dict["epoch"]

    def advance(self, n: int):
        """
        Record that n samples have been consumed.
        """
        self.progress += n
        if self.progress > len(self.dataset):
            raise AdvancedTooFarError(
                "You have advanced too far. You can only advance up to the total size of the dataset."
            )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # slice from progress to pick up where we left off
        for idx in indices[self.progress :]:
            yield idx

    @contextmanager
    def in_epoch(self, epoch):
        """
        This context manager is used to set the epoch. It is used like this:

        ```
        for epoch in range(0, 10):
            with sampler.in_epoch(epoch):
                for step, (x, ) in enumerate(dataloader):
                    # work would be done here...
        ```
        """
        self.set_epoch(epoch)
        yield
        self._reset_progress()
