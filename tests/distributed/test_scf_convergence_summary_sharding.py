"""world_size=1 vs world_size=N behavior for the SCF convergence summary's
DDP sharding + result gathering
(mace_scf/utils/scf_convergence_summary.py: _rebuild_loader/
_evaluate_one_setting).

Before this, `create_scf_convergence_summary` was only ever called on rank
0, and `_rebuild_loader` rebuilt its DataLoader straight from
`loader.dataset` (ignoring any DistributedSampler the caller's loader had),
so rank 0 always processed the *entire* diagnostic dataset itself -- the
other ranks sat idle. `_rebuild_loader` now stride-shards the dataset
across ranks when a process group is initialized, and
`_evaluate_one_setting` all_gather_objects each rank's per-graph
statuses/steps/final_changes back together. This changed *who* does the
work, not what gets computed, so the merged multiset of per-graph results
across ranks should be identical to a single, unsharded process.

Two tiers of coverage, in one file since they're two views of the same
mechanism:
  - `_rebuild_loader` alone, with a fake in-memory dataset (a plain list)
    rather than real ExtAtomicData/model batches -- fast and model-free,
    since _rebuild_loader only touches `loader.dataset` and returns a new
    DataLoader wrapping a Subset of it; nothing here needs to actually
    collate a batch.
  - `_evaluate_one_setting` end-to-end with a real (small) FixedPointCore
    model and a tiny num_scf_steps, exercising the all_gather_object step
    too -- this guards against misdirected work (a graph silently dropped
    or double-counted by the stride split) rather than any aggregation
    math, so SCF convergence quality isn't what's under test.
"""

import pytest

from ase.io import read
from mace.tools import torch_geometric

from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions
from mace_scf.utils.scf_convergence_summary import (
    _evaluate_one_setting,
    _rebuild_loader,
)

from ..utils import dataset_from_atoms
from . import fixtures
from .gloo import run_gloo


# --- _rebuild_loader alone: fast, model-free -------------------------------


class _FakeLoader:
    def __init__(self, dataset):
        self.dataset = dataset


def _loader_items(loader):
    """Read back every item _rebuild_loader's DataLoader would iterate,
    without relying on Subset supporting __iter__ (it only guarantees
    __len__/__getitem__)."""
    dataset = loader.dataset
    return [dataset[i] for i in range(len(dataset))]


def test_no_sharding_outside_a_process_group():
    # Not run under run_gloo -- no process group initialized, so
    # _rebuild_loader must return the full, unsharded dataset unchanged.
    dataset = list(range(7))
    loader = _rebuild_loader(_FakeLoader(dataset), batch_size=1)
    assert _loader_items(loader) == dataset


def _shard_items(rank, world_size, dataset_size):
    loader = _rebuild_loader(
        _FakeLoader(list(range(dataset_size))), batch_size=1
    )
    return _loader_items(loader)


@pytest.mark.distributed
@pytest.mark.parametrize("dataset_size", [4, 5, 17])
def test_shards_cover_every_item_exactly_once(dataset_size):
    world_size = 3
    results = run_gloo(
        _shard_items, world_size=world_size, args=(dataset_size,), timeout_s=60.0
    )

    merged = sorted(item for shard in results for item in shard)
    assert merged == list(range(dataset_size)), (
        "expected every item to be covered exactly once across all ranks' "
        f"shards, got {merged}"
    )

    sizes = [len(shard) for shard in results]
    assert sum(sizes) == dataset_size
    # a plain stride split should never differ by more than one item
    # between the fullest and emptiest rank.
    assert max(sizes) - min(sizes) <= 1, sizes


# --- _evaluate_one_setting end-to-end: real model, real gather -------------

NUM_CONFIGS = 5  # odd, so a 2-rank stride split is uneven

# Cheap settings, chosen (by trying a few candidates against the fixture
# configs) to make every graph's status/step-count/final_change distinct
# rather than uniform
SCF_OPTIONS = FixedPointSCFOptions(
    num_scf_steps=8,
    scf_tolerance=1e-3,
    mixing_parameter=0.2,
    constant_charge=False,
    use_autograd_forces=False,
    initial_density="local_guess",
    initial_fermi_level="from_data",
)


def _build_loader():
    atoms_list = read(str(fixtures.CONFIGS_PATH), index=":")[:NUM_CONFIGS]
    dataset = dataset_from_atoms(
        atoms_list,
        cutoff=4.5,
        fermi_level_key="the_VBM",
        atomic_multipoles_max_l=fixtures.ATOMIC_MULTIPOLES_MAX_L,
    )
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, drop_last=False
    )


def _run_one_setting(rank, world_size):
    model = fixtures.build_model()
    loader = _build_loader()
    result = _evaluate_one_setting(
        model,
        {"valid": loader},
        fixtures.OUTPUT_ARGS,
        "cpu",
        SCF_OPTIONS,
        batch_size=1,
    )
    return result["valid"]


@pytest.mark.distributed
def test_scf_summary_shard_gather_matches_single_process_reference():
    # Reference: no process group initialized, so _rebuild_loader takes the
    # un-sharded path -- one process covers all NUM_CONFIGS graphs itself.
    ref = _run_one_setting(0, 1)
    assert len(ref["statuses"]) == NUM_CONFIGS

    # If every config produced the same status/step-count, the sorted-list
    # comparisons below would pass even if the sharding silently dropped one
    # graph and duplicated another -- guard the fixture itself against
    # regressing to that uniform, less discriminating case.
    assert len(set(ref["statuses"])) > 1, (
        f"test fixture produced the same status for every config "
        f"({ref['statuses']}) -- sorted-list comparisons can't catch a "
        "shard/gather bug that shuffles graphs when every value looks the "
        "same; tune SCF_OPTIONS/NUM_CONFIGS to restore diversity"
    )
    assert len(set(ref["steps"])) > 1, (
        f"test fixture produced the same step-count for every config "
        f"({ref['steps']}) -- see above"
    )

    # Distributed: 2 ranks, each shards half (stride split) of the same
    # dataset and all_gather_objects the merged result back onto every rank.
    results = run_gloo(_run_one_setting, world_size=2, timeout_s=180.0)

    for rank, result in enumerate(results):
        assert len(result["statuses"]) == NUM_CONFIGS, (
            f"rank {rank}: expected every one of {NUM_CONFIGS} graphs to be "
            f"covered exactly once after gathering, got {len(result['statuses'])}"
        )
        assert sorted(result["statuses"]) == sorted(ref["statuses"]), rank
        assert sorted(result["steps"]) == sorted(ref["steps"]), rank
        assert sorted(result["final_changes"]) == pytest.approx(
            sorted(ref["final_changes"])
        ), rank
