"""world_size=1 vs world_size=N equivalence for MaceSCFLoss (DDP eval-metric
aggregation).

Uses synthetic batch/output objects rather than a real model forward pass,
so every conditional branch in MaceSCFLoss.update() can be exercised
directly -- this is a test of the aggregation, not of what a given model
happens to output.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mace_scf.utils.eval_metrics import MaceSCFLoss

from .gloo import run_gloo

ATOMS_PER_STRUCT = 3
NUM_STRUCTS = 4  # split 2/2 across 2 ranks


def _build_structure(struct_id: int, seed: int):
    """One synthetic structure's worth of every quantity MaceSCFLoss.update()
    looks for, plus a matching model "output" dict. Deterministic in
    struct_id+seed so the same structure is produced whether it ends up in
    the single-process reference run or on whichever rank owns it."""
    g = torch.Generator().manual_seed(seed * 1000 + struct_id)
    n = ATOMS_PER_STRUCT

    def r(*shape):
        return torch.randn(*shape, generator=g)

    batch = SimpleNamespace(
        energy=r(1),
        forces=r(n, 3),
        stress=r(1, 3, 3),
        virials=r(1, 3, 3),
        total_charge=r(1),
        fermi_level=r(1),
        dipole=r(1, 3),
        dipole_weight=torch.ones(1, 3),
        density_coefficients=r(n, 4),
        electrostatic_potentials=r(n, 1),
        polarizability=r(1, 3, 3),
        polarizability_weight=torch.ones(1),
        ptr=torch.tensor([0, n]),
        batch=torch.zeros(n, dtype=torch.long),
    )
    output = {
        "energy": r(1),
        "forces": r(n, 3),
        "stress": r(1, 3, 3),
        "virials": r(1, 3, 3),
        "fermi_level": r(1),
        "dipole": r(1, 3),
        "density_coefficients": r(n, 4),
        "electrostatic_potentials": r(n, 1),
        "polarizability": r(1, 3, 3),
    }
    return batch, output


def _merge(structs):
    """Concatenate per-structure (batch, output) pairs into one Batch-like
    object + output dict spanning all of them, recomputing ptr/batch."""
    batches, outputs = zip(*structs)
    n = ATOMS_PER_STRUCT
    num_graphs = len(structs)
    # concatenate every field _build_structure put on the batch except the
    # two (ptr, batch) that need recomputing for the merged structure count,
    # rather than a hand-maintained field list that could drift out of sync
    # with _build_structure.
    concat_fields = set(vars(batches[0])) - {"ptr", "batch"}
    merged_batch = SimpleNamespace(
        **{
            field: torch.cat([getattr(b, field) for b in batches])
            for field in concat_fields
        },
        ptr=torch.arange(0, n * num_graphs + 1, n),
        batch=torch.repeat_interleave(torch.arange(num_graphs), n),
    )
    merged_output = {
        key: torch.cat([o[key] for o in outputs]) for key in outputs[0]
    }
    return merged_batch, merged_output


def _loss_fn(pred, ref):
    return torch.mean((pred["energy"] - ref.energy) ** 2)


def _compute_shard_metrics(rank, world_size, seed):
    structs_per_rank = NUM_STRUCTS // world_size
    # struct_ids == (0, 1) for rank 0, (2, 3) for rank 1, etc.
    # so each rank gets half of the structures
    struct_ids = range(rank * structs_per_rank, (rank + 1) * structs_per_rank)
    batch, output = _merge([_build_structure(i, seed) for i in struct_ids])

    metric = MaceSCFLoss(loss_fn=_loss_fn)
    metric.update(batch, output)
    total_loss, aux = metric.compute()
    return total_loss, aux


@pytest.mark.distributed
def test_maceloss_matches_single_process_reference():
    seed = 7

    # Reference: everything in one process, one MaceSCFLoss, no sharding.
    ref_batch, ref_output = _merge(
        [_build_structure(i, seed) for i in range(NUM_STRUCTS)]
    )
    ref_metric = MaceSCFLoss(loss_fn=_loss_fn)
    ref_metric.update(ref_batch, ref_output)
    ref_total_loss, ref_aux = ref_metric.compute()

    # Distributed: 2 ranks, each owns half of the structures,
    # MaceSCFLoss.compute() triggers torchmetrics' all-reduce/all-gather.
    # timeout_s raised above gloo.py's default: each worker imports the
    # full mace_scf/mace stack, which alone can take over a minute.
    results = run_gloo(
        _compute_shard_metrics, world_size=2, args=(seed,), timeout_s=300.0
    )

    for rank, (total_loss, aux) in enumerate(results):
        assert set(aux) == set(ref_aux), (
            f"rank {rank} aux keys {set(aux)} != reference keys {set(ref_aux)}"
        )
        for key in ref_aux:
            np.testing.assert_allclose(
                aux[key],
                ref_aux[key],
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"rank {rank}, key={key!r}",
            )
        np.testing.assert_allclose(
            total_loss, ref_total_loss, rtol=1e-6, atol=1e-10,
            err_msg=f"rank {rank}, total_loss (the value train.py branches on)",
        )
