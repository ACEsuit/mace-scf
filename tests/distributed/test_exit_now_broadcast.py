"""Regression test for the exit_now early-stop broadcast in
mace_scf/utils/train.py::train(): rank 0 alone decides whether training
should stop, fills a shared tensor, and every rank broadcasts/reads it to
agree on `should_stop` together -- instead of only rank 0 leaving the epoch
loop, which used to deadlock the other rank(s) waiting on a peer that never
shows up to the next collective call.

Calls the real train() (not a reimplementation of the broadcast pattern), 2
ranks via gloo, with lr=0 (keeps validation loss constant) and patience=1
(so the second evaluation triggers the stop) -- mirroring the scenario
tests/distributed/test_run_train_distributed.py's
test_distributed_early_stop_terminates covers end-to-end on real GPUs/SLURM,
but fast and GPU-free here. If the broadcast regressed to rank-0-only, this
test would hang and gloo.py's run_gloo would raise TimeoutError rather than
silently pass.
"""

import json
from pathlib import Path

import pytest

from mace_scf.utils.train import train

from . import fixtures
from .gloo import run_gloo


def _run_until_early_stop(rank, world_size, tmp_dir):
    kwargs = fixtures.build_train_kwargs(
        tmp_dir, rank, world_size, patience=1, lr=0.0, end_epoch=5
    )
    train(**kwargs)
    return "completed"


@pytest.mark.distributed
def test_early_stop_does_not_deadlock(tmp_path):
    # above gloo.py's 60s default: each worker imports the full mace_scf/mace
    # stack and builds+forwards a real FixedPointCore model, which alone can
    # take well over a minute on a loaded shared node. If the exit_now
    # broadcast were broken, this would hang until the timeout instead of
    # returning "completed" -- that's the regression this test guards.
    results = run_gloo(
        _run_until_early_stop, world_size=2, args=(str(tmp_path),), timeout_s=300.0
    )
    assert results == ["completed", "completed"]

    records = []
    for path in Path(tmp_path).glob("*.txt"):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    eval_epochs = {r["epoch"] for r in records if r.get("mode") == "eval"}

    assert len(eval_epochs) < 6, (
        f"expected patience=1 to stop well before all 6 epochs "
        f"(0..end_epoch=5), got eval records for epochs {eval_epochs}"
    )
