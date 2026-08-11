"""Two-rank GPU integration smoke tests of scripts/run_train.py --distributed.

Unlike the rest of tests/distributed/ (fast, CPU-only, gloo-via-mp.spawn),
this uses two GPUs.

Launched with --launcher slurm (via `srun`), not torchrun:
mace.tools.slurm_distributed.DistributedEnvironment always prefers SLURM's
own env vars when present, so a torchrun rendezvous nested inside a SLURM
allocation gets its rank/world_size silently overridden.

Skips cleanly outside a suitable environment. This file's own
_srun_command_prefix() runs a *nested* `srun --ntasks=2 ...` step inside
whatever allocation pytest is already running under -- that nested step
needs the outer allocation to reserve 2 tasks up front
(--ntasks-per-node=2), not a single-task allocation with a bigger
--cpus-per-task for pytest itself to subdivide; the latter reliably fails
with "More processors requested than permitted" or hangs retrying "node is
busy", no matter how the CPU count is tuned. Use sbatch, not a bare
interactive srun
    #SBATCH --partition=gpu
    #SBATCH --constraint="gpu"
    #SBATCH --gres=gpu:a100:2
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=<relevant fraction of total cores>
    module load cuda/<version>
    python -m pytest tests/distributed/test_run_train_distributed.py -v
"""

import os
import sys
from collections import Counter

import pytest
import torch
from ase.io import read, write

from . import fixtures
from ..run_train_pipeline.harness import (
    DEFAULT_REFERENCE_DATA,
    RunTrainCase,
    deep_update,
    make_run_train_config,
    model_config_overrides,
    run_train_case,
)

requires_slurm_gpus = pytest.mark.skipif(
    not (torch.cuda.device_count() >= 2 and "SLURM_JOB_ID" in os.environ),
    reason=(
        "requires 2 CUDA GPUs inside an active SLURM allocation "
        "(scripts/run_train.py --distributed uses nccl+cuda unconditionally, "
        "and --launcher slurm needs real SLURM env vars)"
    ),
)

# The full water fixture has 200 configs; a 16-config subset keeps the
# linearize_solve stage and final error tables affordable.
NUM_SMOKE_CONFIGS = 16


def _trimmed_reference_data(tmp_path):
    frames = read(DEFAULT_REFERENCE_DATA, index=":")[:NUM_SMOKE_CONFIGS]
    path = tmp_path / "water_clusters_small.xyz"
    write(path, frames)
    return path


NPROC = 2
DISTRIBUTED_ARGS = ["--distributed", "--launcher", "slurm"]

TWO_STAGE_SCHEDULE_OVERRIDES = {
    "train_schedule": {
        1: {
            "name": "stage2",
            "start": 2,
            "end": 3,
            "lr": 0.001,
            "loss": {
                "energy_per_atom": 1.0,
                "forces": 10.0,
                "atomic_multipoles": 1.0,
                "dipole_per_atom": 1.0,
            },
            "fixed_point_training_options": {
                "mode": "linearize_solve",
                "scf": fixtures.SCF_OPTIONS,
            },
        },
    },
}


def _srun_command_prefix():
    return ["srun", f"--ntasks={NPROC}", sys.executable]


@pytest.mark.distributed
@requires_slurm_gpus
def test_distributed_fixed_point_two_stage(tmp_path):
    name = "distributed_fixedpoint_two_stage"
    config = make_run_train_config(
        model="FixedPoint",
        name=name,
        overrides=deep_update(
            model_config_overrides("FixedPoint"),
            TWO_STAGE_SCHEDULE_OVERRIDES,
        ),
    )
    config["device"] = "cuda"
    case = RunTrainCase(
        name=name,
        config=config,
        extra_args=DISTRIBUTED_ARGS,
        timeout_s=900,
        command_prefix=_srun_command_prefix(),
    )
    result = run_train_case(
        tmp_path / name,
        case,
        reference_data=_trimmed_reference_data(tmp_path),
        overwrite=True,
    )
    result.assert_success()

    metrics = result.metrics()
    opt_records = [m for m in metrics if m.get("mode") == "opt"]
    eval_records = [m for m in metrics if m.get("mode") == "eval"]
    assert opt_records and eval_records

    # 12 train configs sharded over 2 ranks give 6 per rank and 3 optimizer
    # steps per epoch with batch_size=2; a single-process run would take 6.
    # Rank duplication would double these counts.
    opt_epochs = Counter(record["epoch"] for record in opt_records)
    assert set(opt_epochs.values()) == {3}, opt_epochs

    # logger.log(eval_metrics) is rank-0-gated (see mace_scf/utils/train.py)
    # and valid_err_log intentionally re-logs eval metrics from rank 0, so
    # each evaluated epoch appears exactly twice; if the rank gate ever
    # regresses, unconditional per-rank logging would give three or four.
    eval_epochs = Counter(record["epoch"] for record in eval_records)
    assert set(eval_epochs.values()) == {2}, eval_epochs

    assert len(result.checkpoints()) >= 2
    assert len(result.models()) == 2  # one .model per stage

    summary = result.summary()
    assert summary["final_error_table"] is not None
    log_text = result.log_text()
    assert "world_size=2" in log_text or "Processes: 2" in log_text


@pytest.mark.distributed
@requires_slurm_gpus
def test_distributed_early_stop_terminates(tmp_path):
    name = "distributed_mace_early_stop"
    config = make_run_train_config(
        model="MACE",
        name=name,
        overrides={
            "patience": 1,
            "train_schedule": {
                0: {
                    "name": "stage1",
                    "start": 0,
                    "end": 6,
                    "lr": 0.0,
                    "loss": {
                        "energy_per_atom": 1.0,
                        "forces": 10.0,
                    },
                },
            },
        },
    )
    config["device"] = "cuda"
    case = RunTrainCase(
        name=name,
        config=config,
        extra_args=DISTRIBUTED_ARGS,
        timeout_s=600,
        command_prefix=_srun_command_prefix(),
    )
    # lr=0 keeps the validation loss constant, so the second evaluation
    # triggers the patience stop; before the exit_now broadcast this
    # deadlocked because only rank 0 left the training loop.
    result = run_train_case(
        tmp_path / name,
        case,
        reference_data=_trimmed_reference_data(tmp_path),
        overwrite=True,
    )
    result.assert_success()
    assert "Stopping optimization" in result.log_text()
    eval_epochs = {
        record["epoch"] for record in result.metrics() if record.get("mode") == "eval"
    }
    assert len(eval_epochs) < 6
