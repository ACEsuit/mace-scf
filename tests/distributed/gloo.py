"""Reusable in-process gloo (CPU) multi-process test utility.

Faster than launching a real `torchrun`/subprocess per test (no interpreter
startup or import overhead, and no need to shell out to a launcher binary
or fake env-var rendezvous): `torch.multiprocessing.spawn` starts `world_size`
worker processes directly, each running a plain module-level function with
a real `gloo` process group already initialized.

Worker return values can't cross the process boundary directly (`mp.spawn`
discards them), so each worker `torch.save`s its result to a shared temp
directory and the parent reads them back in rank order once every worker
has joined. Worker exceptions propagate through `mp.spawn`'s own
`ProcessRaisedException`/`ProcessExitedException` (which already embed the
failing rank's full traceback), so a bug in a worker shows up as a normal,
readable pytest failure -- not a hang or a bare exit code.
"""

import os
import shutil
import tempfile
import time
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(local_rank, world_size, tmp_dir, fn, args, kwargs):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_dir}/rdvz",
        rank=local_rank,
        world_size=world_size,
    )
    try:
        result = fn(local_rank, world_size, *args, **kwargs)
        torch.save(result, os.path.join(tmp_dir, f"result_{local_rank}.pt"))
    finally:
        dist.destroy_process_group()


def run_gloo(
    fn: Callable[..., Any],
    world_size: int = 2,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout_s: float = 60.0,
) -> list:
    """Run fn(rank, world_size, *args, **kwargs) in world_size gloo processes.

    fn (and args/kwargs) must be module-level/picklable -- no closures or
    lambdas, since mp.spawn's "spawn" start method pickles the target and
    its arguments to hand off to fresh child interpreters.

    Returns a list of length world_size with fn's return value from each
    rank, ordered by rank.
    """
    kwargs = kwargs or {}
    tmp_dir = tempfile.mkdtemp(prefix="mace_scf_gloo_test_")
    try:
        context = mp.spawn(
            _worker,
            args=(world_size, tmp_dir, fn, args, kwargs),
            nprocs=world_size,
            join=False,
        )
        # ProcessContext.join() already returns as soon as a process
        # finishes (or raises on failure) rather than blocking for the
        # full timeout, so no artificial poll interval is needed here.
        deadline = time.monotonic() + timeout_s
        while not context.join(timeout=max(deadline - time.monotonic(), 0)):
            if time.monotonic() >= deadline:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                raise TimeoutError(
                    f"gloo worker(s) did not complete within {timeout_s}s -- "
                    "possible deadlock (e.g. unbalanced collective calls "
                    "across ranks)"
                )

        return [
            torch.load(
                os.path.join(tmp_dir, f"result_{rank}.pt"), weights_only=False
            )
            for rank in range(world_size)
        ]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
