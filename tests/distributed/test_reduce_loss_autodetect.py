"""mace.modules.loss.reduce_loss's `ddp=None` default auto-detects via
is_ddp_enabled() (dist.is_initialized() and world_size > 1). None of
mace_scf/electrostatics/loss.py's per-term loss functions pass an explicit
`ddp=` value, so this auto-detect path is what production actually runs.

reduce_loss's DDP-branch return value uses only each rank's own local sum,
so it isn't the true global mean by itself -- that only holds after DDP's
backward-pass gradient averaging. So this test doesn't compare against a
full-batch-mean reference; it checks that ddp=None picks the same branch
an explicit ddp=True/False would.
"""

import pytest
import torch

from mace.modules.loss import reduce_loss

from .gloo import run_gloo


def _autodetect_matches_explicit_ddp_true(rank, world_size):
    torch.manual_seed(rank)
    raw_loss = torch.rand(4 + rank)  # different size/values per rank
    auto = reduce_loss(raw_loss.clone(), ddp=None)
    explicit = reduce_loss(raw_loss.clone(), ddp=True)
    return torch.equal(auto, explicit)


@pytest.mark.distributed
def test_autodetect_enables_ddp_branch_under_real_process_group():
    results = run_gloo(_autodetect_matches_explicit_ddp_true, world_size=2)
    assert all(results), (
        "ddp=None should auto-enable the DDP-aware branch (matching "
        "explicit ddp=True) once a real >=2-rank process group is "
        "initialized -- this is the path production code actually uses"
    )


def test_autodetect_matches_plain_mean_without_a_process_group():
    raw_loss = torch.rand(5)
    auto = reduce_loss(raw_loss.clone(), ddp=None)
    off = reduce_loss(raw_loss.clone(), ddp=False)
    assert torch.equal(auto, off)
