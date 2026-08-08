"""take_step() clips gradients via `clip_grad_norm_(model.parameters(), ...)`
using the plain, never-DDP-wrapped `model` reference rather than the
DDP-wrapped `model_eval_wrapper` -- see mace_scf/utils/train.py. This only
clips the tensors DDP actually synced if `model.parameters()` and the DDP
wrapper's `.parameters()` are the *same* tensor objects (every wrapper
class stores `self.model = model` by direct reference and registers no
parameters of its own, so they are).

Two checks: (1) DDP-wrapping the wrapper still exposes model's parameters
by identity (no accidental copy), and (2) running take_step() on a
different data shard per rank still produces bit-identical post-clip
gradients across ranks -- if clipping were operating on a stale,
pre-all-reduce copy, each rank's differently-shaped local shard would
produce differently-clipped gradients and this would fail.
"""

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from mace_scf.electrostatics.fixed_point_state import FixedPointTrainingOptions
from mace_scf.utils.model_training_wrappers import FixedPointWrapper
from mace_scf.utils.train import take_step

from . import fixtures
from .gloo import run_gloo

# Small enough that take_step's clip_grad_norm_ actually rescales gradients
# (rather than being a no-op because the unclipped norm is already smaller).
MAX_GRAD_NORM = 1e-3


def _clip_and_return_grads(rank, world_size, max_grad_norm):
    model = fixtures.build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    wrapper = FixedPointWrapper(
        model=model,
        optimizer=optimizer,
        output_args=fixtures.OUTPUT_ARGS,
        training_options=FixedPointTrainingOptions(
            mode="direct", scf=None, linear_solve="inverse"
        ),
    )
    ddp_wrapper = DDP(wrapper)

    model_param_ids = {id(p) for p in model.parameters()}
    wrapper_param_ids = {id(p) for p in ddp_wrapper.parameters()}
    identity_ok = model_param_ids <= wrapper_param_ids

    batches = fixtures.shard_batches(model, n_shards=world_size, shard_size=2)
    batch = batches[rank]
    loss_fn = fixtures.make_loss_fn()

    _, loss_dict = take_step(
        model=model,
        model_eval_wrapper=ddp_wrapper,
        loss_fn=loss_fn,
        batch=batch,
        optimizer=optimizer,
        ema=None,
        max_grad_norm=max_grad_norm,
        device=torch.device("cpu"),
    )
    assert loss_dict["grad_clip_applied"] == True, "Gradient not clipped -- test setup is wrong"

    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return identity_ok, grads


@pytest.mark.distributed
def test_ddp_sees_model_parameters_by_identity_and_clips_synced_grads():
    # above gloo.py's 60s default: each worker imports the full mace_scf/mace
    # stack and builds+forwards a real FixedPointCore model, which alone can
    # take well over a minute on a loaded shared node.
    results = run_gloo(
        _clip_and_return_grads,
        world_size=2,
        args=(MAX_GRAD_NORM,),
        timeout_s=180.0,
    )

    for rank, (identity_ok, _) in enumerate(results):
        assert identity_ok, (
            f"rank {rank}: model.parameters() are not a subset of the DDP "
            "wrapper's parameters -- clip_grad_norm_(model.parameters(), ...) "
            "in take_step() would be clipping a different set of tensors "
            "than the ones DDP actually syncs"
        )

    grads_by_rank = [grads for _, grads in results]
    names = set(grads_by_rank[0])
    assert names, "no parameter received a gradient -- check the fixture/loss setup"
    for grads in grads_by_rank[1:]:
        assert set(grads) == names

    for name in names:
        assert torch.equal(grads_by_rank[0][name], grads_by_rank[1][name]), (
            f"post-clip gradient for {name!r} differs across ranks -- "
            "clip_grad_norm_ is not operating on DDP-synced gradients"
        )
