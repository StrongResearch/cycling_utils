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
    This is an extension on the concept of atomic saving to support saving to directories, which is useful where it is
    necessary to use a specific saving technology that is not easily adapted to the dictionary-based torch save pattern.

    This works by preparing and saving to a new directory each checkpoint, and then saving a symlink to that directory
    which should be read upon resume to obtain the path to the latest checkpoint directory.

    The AtomicDirectory accepts the following arguments:
        output_directory: root directory for all ouputs from the experiment, (e.g. where the rank logs are saved).
        symlink_name: filename given to the symlink used to designate the latest checkpoint directory.
        chk_dir_prefix: prefix given to the checkpoint directories, used to identify them as checkpoint directories.
        cleanup: whether to delete old checkpoint directories and their contents.

    Example usage (saving every training batch) is as follows. Note that saver.prepare_checkpoint_directory() is run
    at the start of the iteration in which a checkpoint is saved. This is to ensure that the clean-up step is
    prioritised and that subsequent work is not lost.

    >>> # Initialize the AtomicDirectory
    >>> saver = AtomicDirectory(output_directory)

    >>> # Resume from checkpoint
    >>> latest_sym = os.path.join(output_directory, saver.symlink_name)
    >>> if os.path.exists(latest_sym):
    >>>    latest_path = os.readlink(latest_sym)

    >>>     # Load files from latest_path
    >>>     ...

    >>> for epoch in epochs:
    >>>     for batch in batches:

    >>>         # Prepare the checkpoint directory before starting computation for the iteration
    >>>         checkpoint_directory = saver.prepare_checkpoint_directory()

    >>>         # Saving files to the checkpoint_directory
    >>>         ...

    >>>         # Updating symlink to direct to the latest checkpoint
    >>>         saver.atomic_symlink(checkpoint_directory)

    
    For best-endeavours checkpoint saving:
    >>> saver = AtomicDirectory(output_directory, is_master=rank==0)
    >>> ...loop
    >>>     checkpoint_directory = saver.prepare_checkpoint_directory()
    >>>     ...saving...
    >>>     saver.atomic_symlink(checkpoint_directory)

    To force-save a checkpoint for retrieval later:
    >>> saver = AtomicDirectory(output_directory, is_master=rank==0)
    >>> ...loop
    >>>     checkpoint_directory = saver.prepare_checkpoint_directory(force_save=True)
    >>>     ...saving...
    >>>     saver.atomic_symlink(checkpoint_directory)
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
        
    def is_checkpoint_directory(path_str):
        pattern = r"_checkpoint_(\d+(_s)?)$"
        path = Path(path_str)
        match = re.match(pattern, path.name)
        if path.exists() and path.is_dir() and match:
            return match.group(1)
        else:
            return None

    def prepare_checkpoint_directory(self, force_save=False):

        output_directory_contents = os.listdir(self.output_directory)
        maybe_checkpoint_suffixes = [self.is_checkpoint_directory(path_str) for path_str in output_directory_contents]
        checkpoint_paths = {path: suffix for path, suffix in zip(output_directory_contents, maybe_checkpoint_suffixes) if suffix}
        symlink_found = self.symlink_name in output_directory_contents

        if checkpoint_paths and not symlink_found:
            print("Found one or more checkpoint dirs but no symlink to latest. Will assume all to be deleted.")

        if symlink_found and not checkpoint_paths:
            raise Exception("Found symlink but no finalized checkpoint dirs.")

        latest_sequential_index = -1

        if symlink_found:

            symlink_path = os.readlink(os.path.join(self.output_directory, self.symlink_name))

            latest_sequential_index = int(checkpoint_paths[symlink_path].split("_")[0])

            # determine directories that can be deleted
            if self.keep_last > 0:
                deletable = [
                    path for path,suffix in checkpoint_paths.items()
                    if not suffix.endswith("_force")
                    and (
                        int(suffix.split("_")[0]) < latest_sequential_index - self.keep_last + 2 
                        or int(suffix.split("_")[0]) > latest_sequential_index
                    )
                ]
            else:
                deletable = [
                    path for path, suffix in checkpoint_paths.items()
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
            assert (
                Path(next_checkpoint_directory).exists() and Path(next_checkpoint_directory).is_dir()
                and len(os.listdir(next_checkpoint_directory)) == 0
            ), "ERROR: fault creating new save dir."

        # Return path to save to
        barrier()
        return next_checkpoint_directory

    def symlink_latest(self, checkpoint_directory):
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
