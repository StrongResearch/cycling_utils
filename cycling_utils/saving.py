import os
import re
from pathlib import Path
from shutil import rmtree
import torch
from torch.distributed import barrier, all_reduce, all_gather


def atomic_torch_save(obj, f: str | Path, **kwargs):
    f = str(f)
    temp_f = f + ".temp"
    torch.save(obj, temp_f, **kwargs)
    os.replace(temp_f, f)


class AtomicDirectory:
    """
    The Strong Compute ISC uses an Artifacts system for saving experiment outputs. Saving Checkpoint type Artifacts requires using
    the AtomicDirectory saver. The User is responsible for implementing AtomicDirectory saver and saving checkpoints at their
    desired frequency.

    The AtomicDirectory saver works by saving each checkpoint to a new directory, and then saving a symlink to that directory
    which should be read upon resume to obtain the path to the latest checkpoint directory.

    The AtomicDirectory saver is designed for use in a distributed process group (e.g. via torchrun) and can be run in either
    synchronous or non-synchronous mode.

    In synchronous mode, each process in the distributed process group accesses the same checkpoint directory created by the
    AtomicDirectory saver each checkpoint save. This is suitable for synchronous distributed training procedures like
    DistributedDataParallel.

    In asynchronous mode, each process creates its own dedicated checkpoint directory and can save checkpoints independently of
    other processes.

    Each checkpointing process must initialize the saver. AtomicDirectory accepts the following arguments at initialization:

    - output_directory: root directory for all ouputs from the experiment, should always be set to the $CHECKPOINT_ARTIFACT_PATH
    environment variable when training on the Strong Compute ISC.
    - is_master: a boolean to indicate whether the process running the AtomicDirectory saver is the master rank in the process
    group. For asynchronous mode, all ranks must pass is_master = True.
    - name: a name for the AtomicDirectory saver. If the user is running multiple savers in parallel, each must be given a unique
    name.
    - keep_last: the number of previous checkpoints to retain locally, should always be -1 when saving Checkpoint Artifacts to the
    $CHECKPOINT_ARTIFACT_PATH on the Strong Compute.
    - strategy: must be one of "sync_any" (default), "sync_all", or "async", determining whether the saver runs in synchronous or
    asynchronous mode, and force saving (see below). All AtomicDirectory savers MUST be initialized withh the same strategy.

    Checkpoint Artifacts saved to $CHECKPOINT_ARTIFACT_PATH are synchronized every 10 minutes and/or at the end of each cycle on
    Strong Compute. Upon synchronization, the latest symlinked checkpoint/s saved by AtomicDirectory saver/s in the
    $CHECKPOINT_ARTIFACT_PATH directory will be shipped to Checkpoint Artifacts for the experiment. Any non-latest checkpoints
    saved since the previous Checkpoint Artifact sychronization will be deleted and not shipped.

    The user can force non-latest checkpoints to also ship to Checkpoint Artifacts by calling `prepare_checkpoint_directory`
    with `force_save = True`. This can be used, for example:
    - to ensure milestone checkpoints are archived for later analysis, or
    - to ensure that checkpoints are saved each time model performance improves.

    The `strategy` argument to the AtomicDirectory saver at initialization determines what happens if processes disagree on the
    `force_save` argument.

    - `strategy = "sync_any"` (default) will `force_save` the checkpoint if ANY process passes `force_save = True`
    - `strategy = "sync_all"` will `force_save` the checkpoint if and only if ALL processes pass `force_save = True`
    - `strategy = "async"` will `force_save` the checkpoint if the saving process passes `force_save = True`
    - `strategy = "offline"` will `force_save` the checkpoint if the saving process passes `force_save = True`

    Each of the strategies "sync_any", "sync_all", and "async"

    Further, the `strategy = "offline"` argument should be passed if the AtomicDirectory saver is intended for use outside of a
    torchrun distributed process group. In such cases, the user must ensure that all instances of the AtomicDirectory saver are
    initialized with a unique 'name'.

    Example usage of AtomicDirectory in synchronous mode on the Strong Compute ISC launching with torchrun as follows.

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
    """

    def __init__(
        self,
        output_directory,
        is_master=False,
        name="AtomicDirectory",
        keep_last=-1,
        strategy="sync_any",
    ):
        self.output_directory = output_directory
        self.is_master = is_master
        self.keep_last = keep_last
        self.strategy = strategy
        self.rank = os.getenv("RANK", "NONE")
        self.world_size = os.getenv("WORLD_SIZE", "NONE")

        # make sure all processes have been initialized with the same
        strategy_map = {"sync_any": 0, "sync_all": 1, "async": 2, "offline": 3}
        if strategy in strategy_map:
            strategy_int = strategy_map[strategy]
        else:
            raise f"ERROR: AtomicDirectory saver must be initialized with strategy = 'sync_any', 'sync_all', 'async', or 'offline' but rank \
                {self.rank} was passed '{strategy}'."

        if strategy != "offline":

            assert (
                self.rank != "NONE"
            ), "ERROR: AtomicDirectory requires RANK environment variable set if strategy is not 'offline'."
            assert (
                self.world_size != "NONE"
            ), "ERROR: AtomicDirectory requires WORLD_SIZE environment variable set if strategy is not 'offline'."

            local_strategy_tensor = torch.tensor(
                strategy_int, dtype=torch.int64, requires_grad=False, device="cuda"
            )
            global_strategy_list = [
                torch.zeros(1, dtype=torch.int64, requires_grad=False, device="cuda")
                for _ in range(int(self.world_size))
            ]
            all_gather(global_strategy_list, local_strategy_tensor)
            unique_global_strategy_ints = set([t.item() for t in global_strategy_list])
            assert (
                len(unique_global_strategy_ints) == 1
            ), "ERROR: AtomicDirectory savers initialized with different strategies."

        if strategy == "async":
            self.name = name + f"_rank_{self.rank}"
        else:
            self.name = name

        self.symlink_name = f"{self.name}.latest_checkpoint"

        try:
            os.makedirs(output_directory, exist_ok=True)
        except Exception as e:
            print(
                f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
            )
            raise e

    def is_checkpoint_directory(self, path_str):
        pattern = r"_checkpoint_(\d+)(_force)?$"
        path = Path(path_str)
        match = re.search(pattern, path.name)
        if (
            path.exists()
            and path.is_dir()
            and path.name.startswith(f"{self.name}_checkpoint")
            and match
        ):
            # Combine the matched groups to form the suffix
            suffix = match.group(1)  # This captures the integer part
            if match.group(2):  # This captures "_force" if it exists
                suffix += match.group(2)
            return suffix
        else:
            return None

    def prepare_checkpoint_directory(self, force_save=False):

        if self.strategy in ["sync_any", "sync_all"]:
            barrier()

        output_directory_contents = os.listdir(self.output_directory)
        maybe_checkpoint_suffixes = [
            self.is_checkpoint_directory(os.path.join(self.output_directory, path_str))
            for path_str in output_directory_contents
        ]
        checkpoint_paths = {
            path: suffix
            for path, suffix in zip(
                output_directory_contents, maybe_checkpoint_suffixes
            )
            if suffix
        }
        symlink_found = self.symlink_name in output_directory_contents

        if checkpoint_paths and not symlink_found:
            print(
                "Found one or more checkpoint dirs but no symlink to latest. Will assume all should be deleted."
            )

        if symlink_found and not checkpoint_paths:
            raise Exception(
                f"Found symlink but no finalized checkpoint dirs: {checkpoint_paths}, {output_directory_contents}"
            )

        latest_sequential_index = -1
        deletable = []

        if symlink_found:

            symlink_path = os.readlink(
                os.path.join(self.output_directory, self.symlink_name)
            )
            latest_sequential_index = int(
                checkpoint_paths[Path(symlink_path).name].split("_")[0]
            )

            # determine directories that can be deleted
            incomplete_deletable = [
                os.path.join(self.output_directory, path)
                for path, suffix in checkpoint_paths.items()
                if int(suffix.split("_")[0]) > latest_sequential_index
            ]

            obsolete_deletable = []
            if self.keep_last > 0:
                obsolete_deletable = [
                    os.path.join(self.output_directory, path)
                    for path, suffix in checkpoint_paths.items()
                    if not suffix.endswith("_force")
                    and int(suffix.split("_")[0])
                    < latest_sequential_index - self.keep_last + 2
                ]

            deletable = incomplete_deletable + obsolete_deletable

        # Delete deletable
        if self.strategy in ["sync_any", "sync_all"]:
            barrier()

        if self.is_master:
            for path in deletable:
                rmtree(path)
            for path in deletable:
                assert not Path(path).exists()

        if self.strategy in ["sync_any", "sync_all"]:
            barrier()

        # name the next checkpoint directory
        next_checkpoint_name = f"{self.name}_checkpoint_{latest_sequential_index + 1}"

        # determine force saving based on strategy
        if self.strategy in ["sync_any", "sync_all"]:
            global_force = torch.tensor(
                1 if force_save else 0,
                dtype=torch.int64,
                requires_grad=False,
                device="cuda",
            )
            all_reduce(global_force)

            effective_force_save = False
            if (global_force.item() > 0 and self.strategy == "sync_any") or (
                global_force.item() == int(self.world_size)
                and self.strategy == "sync_all"
            ):
                effective_force_save = True

        elif self.strategy in ["async", "offline"]:
            effective_force_save = force_save

        if effective_force_save:
            next_checkpoint_name += "_force"
        next_checkpoint_directory = os.path.join(
            self.output_directory, next_checkpoint_name
        )

        # create the next checkpoint directory
        if self.is_master:
            os.makedirs(next_checkpoint_directory, exist_ok=True)

        while True:
            if Path(next_checkpoint_directory).exists():
                break

        if self.strategy in ["sync_any", "sync_all"]:
            barrier()

        assert Path(
            next_checkpoint_directory
        ).exists(), "ERROR: Just made directory but does not exist."
        assert Path(
            next_checkpoint_directory
        ).is_dir(), "ERROR: Path just created is not a directory."
        assert (
            len(os.listdir(next_checkpoint_directory)) == 0
        ), "ERROR: Next checkpoint directory already populated."
        if effective_force_save:
            assert Path(next_checkpoint_directory).name.endswith(
                "_force"
            ), "ERROR: Force path missing force tag."

        if self.strategy in ["sync_any", "sync_all"]:
            barrier()

        return next_checkpoint_directory

    def symlink_latest(self, checkpoint_directory):

        if self.strategy in ["sync_any", "sync_all"]:
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

        if self.strategy in ["sync_any", "sync_all"]:
            barrier()
