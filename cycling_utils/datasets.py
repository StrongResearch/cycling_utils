import math
import random
import torch
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets.folder import make_dataset, find_classes, pil_loader
    
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class DistributedShardedDataset:
    """
    This 'DistributedShardedDataset' dataset class is based on the ImageFolder dataset class by pytorch, and is designed 
    to enable instantiation of a unique dataset for each GPU with a (potentially overlapping) shard of the underlying 
    dataset.

    The data is first loaded into RAM as lists of samples and corresponding targets optionally transformed according to 
    the 'sample_load_transform' and 'target_load_transform' callables.

    If the 'device_id' is not None, then the samples and targets are separately concatenated into contiguous tensors
    and moved onto the device. Note that in order to achieve this, the samples must first be transformed to be all the 
    same shape tensors.

    This dataset class is intended for use with the 'InterruptibleSampler' class which does not distribute the data
    to each GPU (as this is taken care of by this dataset class) but keeps track of progress through the dataset.
    """
    def __init__(
        self,
        root: str,
        rank: int,
        world_size: int,
        samples_pct_per_replica: float,
        device_id: str = None,
        seed: int = 0,
        shuffle: bool = True,
        loader: Callable[[str], Any] = pil_loader,
        extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
        sample_load_transform: Optional[Callable] = lambda sample: sample,
        target_load_transform: Optional[Callable] = lambda target: target,
        sample_getitem_transform: Optional[Callable] = None,
        target_getitem_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,        
    ) -> None:
        
        self.root = root
        self.seed = seed
        self.shuffle = shuffle
        self.loader = loader
        self.sample_load_transform = sample_load_transform
        self.sample_getitem_transform = sample_getitem_transform
        self.target_load_transform = target_load_transform
        self.target_getitem_transform = target_getitem_transform
        
        # Get classes, class index map, and sample paths
        self.classes, self.class_to_idx = self.find_classes(self.root)
        global_samples = self.make_dataset(self.root, self.class_to_idx, extensions, is_valid_file)

        # Shuffle if shuffling
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(global_samples)

        # Deal ourselves in
        num_local = math.ceil(samples_pct_per_replica * len(global_samples))
        local_samples = []
        for idx, sample in enumerate(cycle(global_samples)):
            if (idx - rank) % world_size == 0:
                local_samples.append(sample)
            if len(local_samples) >= num_local:
                break
        assert len(local_samples) == num_local, "MORE LOCAL SAMPLES THAN EXPECTED"

        # Load into RAM
        self.samples = [self.sample_load_transform(self.loader(path)) for path, _ in local_samples]
        self.targets = [self.target_load_transform(target) for _, target in local_samples]

        if device_id is not None:
            # Load into VRAM
            self.samples = torch.stack(self.samples).to(device_id)
            self.targets = torch.stack(self.targets).to(device_id)

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Returns a list of samples of the form [(path_to_sample, class), ...]."""
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
 
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a standard dataset structured folder."""
        return find_classes(directory)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.samples[index], self.targets[index]
        if self.sample_getitem_transform is not None:
            sample = self.sample_getitem_transform(sample)
        if self.target_getitem_transform is not None:
            target = self.target_getitem_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
