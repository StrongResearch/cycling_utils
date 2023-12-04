import os
from pathlib import Path
from shutil import rmtree

import torch

def atomic_torch_save(obj, f: str | Path, **kwargs):
    f = str(f)
    temp_f = f + ".temp"
    torch.save(obj, temp_f, **kwargs)
    os.replace(temp_f, f)

class AtomicDirectory:
    '''
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
    '''
    def __init__(self, output_directory, symlink_name="latest_pt", chk_dir_prefix="CHK", cleanup=True):
        self.output_directory = output_directory
        self.symlink_name = symlink_name
        self.chk_dir_prefix = chk_dir_prefix
        self.cleanup = cleanup

    def prepare_checkpoint_directory(self):
        if os.environ["RANK"] == 0:
            # Catalogue any checkpoint directories already in the output_directory
            checkpoints = [
                d for d in os.listdir(self.output_directory) 
                if os.path.isdir(os.path.join(self.output_directory,d)) and d.startswith(self.chk_dir_prefix)
            ]
            if checkpoints:
                # Full paths to checkpoint directories
                checkpoint_paths = [os.path.join(self.output_directory,d) for d in checkpoints]
                # Obtain the checkpoint indices from the directory names
                checkpoint_indices = [int(d[3:]) for d in checkpoints]
                # Default latest path and index
                latest_path, latest_index = None, -1
                # If there is also a valid symlink in the output_directory
                if os.path.exists(os.path.join(self.output_directory, self.symlink_name)):
                    # The full path to the current checkpoint directory is the one pointed to by the symlink
                    latest_path = os.readlink(os.path.join(self.output_directory, self.symlink_name))
                    # The index of the latest checkpoint
                    latest_index = int(latest_path.split(self.chk_dir_prefix)[1])
                # Obsolete checkpoint directories
                if self.cleanup:
                    # All but the latest path
                    obsolete = [path for path in checkpoint_paths if path!=latest_path]
                else:
                    # Any with index greater than the latest path index
                    obsolete = [d for d,i in zip(checkpoint_paths,checkpoint_indices) if i>latest_index]
                # Delete obsolete
                for d in obsolete:
                    rmtree(d)
                # New checkpoint_directory to save to
                next_checkpoint_directory = os.path.join(self.output_directory, f"{self.chk_dir_prefix}{latest_index + 1}")
            else:
                next_checkpoint_directory = os.path.join(self.output_directory, f"{self.chk_dir_prefix}0")
            # Create new checkpoint_directory to save to
            os.mkdir(next_checkpoint_directory)
            assert os.path.isdir(next_checkpoint_directory) and len(os.listdir(next_checkpoint_directory)) == 0, 
                "ERROR: fault creating new save dir."
            # Return path to save to
            return next_checkpoint_directory
        else:
            return None
    
    def atomic_symlink(self, checkpoint_directory):
        if os.environ["RANK"] == 0:
            # Create a new symlink with name suffixed with temp
            os.symlink(checkpoint_directory, self.symlink_name+"_temp")
            # Replace any existing current symlink with the new temp symlink
            os.replace(self.symlink_name+"_temp", self.symlink_name)
