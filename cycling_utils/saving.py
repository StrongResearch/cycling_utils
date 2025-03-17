import os, re
from pathlib import Path
from shutil import rmtree
import torch
from torch.distributed import barrier

def atomic_torch_save(obj, f: str | Path, **kwargs):
    f = str(f)
    temp_f = f + ".temp"
    torch.save(obj, temp_f, **kwargs)
    os.replace(temp_f, f)

class AtomicDirectory:
    """
    The AtomicDirectory saver works by saving each checkpoint to a new directory, and then saving a symlink to that directory
    which should be read upon resume to obtain the path to the latest checkpoint directory.

    The AtomicDirectory accepts the following arguments at initialization:

    - output_directory: root directory for all ouputs from the experiment, should always be set to the $CHECKPOINT_ARTIFACT_PATH environment variable when training on the Strong Compute ISC.
    - is_master: a boolean to indicate whether the process running the AtomicDirectory saver is the master rank in the process group.
    - name: a name for the AtomicDirectory saver. If the user is running multiple savers in parallel, each must be given a unique name.
    - keep_last: the number of previous checkpoints to retain on disk, should always be -1 when saving Checkpoint Artifacts on Strong Compute.

    Example usage of AtomicDirectory on Strong Compute launching with torchrun as follows.

    >>> import os 
    >>> import torch
    >>> import torch.distributed as dist
    >>> from cycling_utils import AtomicDirectory, atomic_torch_save
    
    >>> dist.init_process_group("nccl")
    >>> rank = int(os.environ["RANK"]) 
    >>> output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]

    >>> # Initialize the AtomicDirectory
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
    >>>     for batch in batches:

    >>>         ...training...

    >>>         if is_save_step:
    >>>             checkpoint_directory = saver.prepare_checkpoint_directory()

    >>>             # saving files to the checkpoint_directory
    >>>             checkpoint = {...}
    >>>             checkpoint_path = os.path.join(checkpoint_directory, "checkpoint.pt")
    >>>             atomic_torch_save(checkpoint, checkpoint_path)

    >>>             # finalizing checkpoint with symlink
    >>>             saver.symlink_latest(checkpoint_directory)

    The AtomicDirectory saver is designed for use with Checkpoint Artifacts on Strong Compute. The User is responsible for 
    implementing AtomicDirectory saver and saving checkpoints at their desired frequency.

    Checkpoint Artifacts are synchronized every 10 minutes and/or at the end of each cycle on Strong Compute. Upon synchronization, 
    the latest symlinked checkpoint/s saved by AtomicDirectory saver/s in the $CHECKPOINT_ARTIFACT_PATH directory will be shipped 
    to Checkpoint Artifacts for the experiment. Any non-latest checkpoints saved since the previous Checkpoint Artifact sychronization 
    will be deleted and not shipped.

    The user can force non-latest checkpoints to also ship to Checkpoint Artifacts by calling `prepare_checkpoint_directory`
    with `force_save = True`. This can be used, for example:
    - to ensure every Nth saved checkpoint is archived for later analysis, or 
    - to ensure that checkpoints are saved each time model performance improves. 
    """

    def __init__(
        self,
        output_directory,
        is_master = False,
        name = "AtomicDirectory",
        keep_last = -1,
    ):
        self.output_directory = output_directory
        self.is_master = is_master
        self.name = name
        self.keep_last = keep_last
        self.symlink_name = f"{self.name}.latest_checkpoint"

        try:
            os.makedirs(output_directory, exist_ok=True)
        except:
            raise Exception("Unable to find or create output directory.")
        
    def is_checkpoint_directory(self, path_str):
        pattern = r'checkpoint_(\d+)(_force)?$'
        path = Path(path_str)
        match = re.search(pattern, path.name)
        if path.exists() and path.is_dir() and path.name.startswith(f"{self.name}_checkpoint") and match:
            # Combine the matched groups to form the suffix
            suffix = match.group(1)  # This captures the integer part
            if match.group(2):  # This captures "_force" if it exists
                suffix += match.group(2)
            return suffix
        else:
            return None

    def prepare_checkpoint_directory(self, force_save=False):
        barrier()
        output_directory_contents = os.listdir(self.output_directory)
        maybe_checkpoint_suffixes = [self.is_checkpoint_directory(os.path.join(self.output_directory, path_str)) for path_str in output_directory_contents]
        checkpoint_paths = {path: suffix for path, suffix in zip(output_directory_contents, maybe_checkpoint_suffixes) if suffix}
        symlink_found = self.symlink_name in output_directory_contents

        if checkpoint_paths and not symlink_found:
            print("Found one or more checkpoint dirs but no symlink to latest. Will assume all should be deleted.")

        if symlink_found and not checkpoint_paths:
            raise Exception(f"Found symlink but no finalized checkpoint dirs: {checkpoint_paths}, {output_directory_contents}")

        latest_sequential_index = -1

        if symlink_found:

            symlink_path = os.readlink(os.path.join(self.output_directory, self.symlink_name))
            latest_sequential_index = int(checkpoint_paths[Path(symlink_path).name].split("_")[0])

            # determine directories that can be deleted
            if self.keep_last > 0:
                deletable = [
                    os.path.join(self.output_directory, path) for path,suffix in checkpoint_paths.items()
                    if not suffix.endswith("_force")
                    and (
                        int(suffix.split("_")[0]) < latest_sequential_index - self.keep_last + 2 
                        or int(suffix.split("_")[0]) > latest_sequential_index
                    )
                ]
            else:
                deletable = [
                    os.path.join(self.output_directory, path) for path, suffix in checkpoint_paths.items()
                    if not suffix.endswith("_force")
                    and int(suffix.split("_")[0]) > latest_sequential_index
                ]

            # Delete deletable
            barrier()
            if self.is_master:
                for path in deletable:
                    rmtree(path)
                for path in deletable:
                    assert not Path(path).exists()

        # create the next checkpoint directory
        next_checkpoint_name = f"{self.name}_checkpoint_{latest_sequential_index + 1}"
        if force_save:
            next_checkpoint_name += "_force"

        # Create new checkpoint_directory to save to
        next_checkpoint_directory = os.path.join(self.output_directory, next_checkpoint_name)
        if self.is_master:
            os.makedirs(next_checkpoint_directory, exist_ok=True)
            assert Path(next_checkpoint_directory).exists(), "ERROR: Just made directory but does not exist."
            assert Path(next_checkpoint_directory).is_dir(), "ERROR: Path just created is not a directory."
            assert len(os.listdir(next_checkpoint_directory)) == 0, "ERROR: Next checkpoint directory already populated."
            if force_save:
                assert Path(next_checkpoint_directory).name.endswith("_force"), "ERROR: Force path missing force tag."

        # Return path to save to
        barrier()
        return next_checkpoint_directory

    def symlink_latest(self, checkpoint_directory):
        barrier()
        if self.is_master:
            # Create a new symlink with name suffixed with temp
            parent_dir = Path(checkpoint_directory).parent.absolute()
            os.symlink(
                checkpoint_directory,
                os.path.join(parent_dir, self.symlink_name + "_temp"),
            )
            # Replace any existing current symlink with the new temp symlink
            os.replace(
                os.path.join(parent_dir, self.symlink_name + "_temp"),
                os.path.join(parent_dir, self.symlink_name),
            )
