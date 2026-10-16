"""Parameter gradients of energy, force and stress losses: linearize_solve vs unroll_scf.
i.e. checking second order gradients

To print the size of linearize_solve error add -s to pytest call.

Kept separate from test_fixed_point_wrapper_training_modes.py: uses
a random in-code model so it runs without the git-ignored .model fixtures.
"""

import os
import resource
import time

import numpy as np
import pytest
import torch

from tests.utils import (
    CUBIC_CELL,
    TRICLINIC_CELL,
    copy_batch,
    dataset_from_atoms,
    make_fixed_point_model_random,
    make_fixed_point_wrapper,
    periodic_water_dimer,
    wrap_loader,
)

UPDATE_SCALE = 0.2  # largest tried scale at which both models converge in both charge modes
FINITE_DIFFERENCE_STEP = 1e-5
NUM_DIRECTIONS = 2
GRADIENT_RTOL = 1e-6  # linearize_solve vs unroll_scf, relative L2; linear model observed <1e-8
FINITE_DIFFERENCE_RTOL = 1e-5  # unroll_scf vs finite difference; observed <~2e-7
FINITE_DIFFERENCE_ATOL = 1e-8

MODELS = {
    "linear": "NoNonLinearity",
    "nonlinear": "MLPNonLinearity",
}
OUTPUT_KEYS = {"energy": "energy", "force": "forces", "stress": "stress"}
OBJECTIVES = tuple(OUTPUT_KEYS)


def _build_batch(batch_size):
    atoms_list = [periodic_water_dimer(TRICLINIC_CELL), periodic_water_dimer(CUBIC_CELL)]
    dataset = dataset_from_atoms(
        atoms_list[:batch_size], cutoff=3.0, atomic_multipoles_max_l=1
    )
    return next(wrap_loader(dataset, batch_size=batch_size))


def _loss(output, objective, weights):
    value = output[OUTPUT_KEYS[objective]]
    return value.sum() if weights is None else (value * weights).sum()


def _objective_weights(objective, batch, seed):
    """Fixed random weights that turn forces or stress into a scalar loss."""
    if objective == "energy":
        return None
    num_graphs = batch["ptr"].numel() - 1
    shape = batch["positions"].shape if objective == "force" else (num_graphs, 3, 3)
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _parameter_gradient(wrapper, model, batch, objective, weights):
    output = wrapper(copy_batch(batch), training=True)
    params = list(model.parameters())
    grads = torch.autograd.grad(
        _loss(output, objective, weights), params, allow_unused=True
    )
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, params)
        ]
    ).detach()


def _shift_parameters(model, direction, step):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            n = param.numel()
            param.add_(step * direction[offset : offset + n].view_as(param))
            offset += n


def _directional_finite_difference(wrapper, model, batch, objective, weights, direction):
    h = FINITE_DIFFERENCE_STEP
    values = []
    for step in (h, -2 * h):
        _shift_parameters(model, direction, step)
        output = wrapper(copy_batch(batch), training=False)
        values.append(_loss(output, objective, weights).item())
    _shift_parameters(model, direction, h)
    return (values[0] - values[1]) / (2 * h)


def _scf_status(wrapper, model, batch):
    batch = copy_batch(batch)
    runner = wrapper._runner
    local_state = model.local_part(batch)
    result = runner.converge(
        model,
        batch,
        local_state,
        runner.get_initial_density(local_state, batch),
        runner.get_initial_fermi(model, local_state, batch),
    )
    return result.status


_RESULTS = {}


def _results(model_kind, constant_charge, batch_size):
    """Gradients and finite differences for all objectives, computed once per case."""
    key = (model_kind, constant_charge, batch_size)
    if key in _RESULTS:
        return _RESULTS[key]

    torch.set_default_dtype(torch.float64)
    model = make_fixed_point_model_random(
        seed=0, nonlinearity_cls=MODELS[model_kind], update_scale=UPDATE_SCALE
    )
    batch = _build_batch(batch_size)
    linearize = make_fixed_point_wrapper(model, "linearize_solve", constant_charge, stress=True)
    unroll = make_fixed_point_wrapper(model, "unroll_scf", constant_charge, stress=True)
    status = _scf_status(unroll, model, batch)
    assert status == "converged", f"SCF did not converge for {key}: {status}"

    generator = torch.Generator().manual_seed(1)
    results = {}
    for seed, objective in enumerate(OBJECTIVES, start=2):
        weights = _objective_weights(objective, batch, seed)
        start = time.perf_counter()
        grad_linearize = _parameter_gradient(linearize, model, batch, objective, weights)
        time_linearize = time.perf_counter() - start
        start = time.perf_counter()
        grad_unroll = _parameter_gradient(unroll, model, batch, objective, weights)
        time_unroll = time.perf_counter() - start
        directions = []
        for _ in range(NUM_DIRECTIONS):
            direction = torch.randn(grad_unroll.shape, generator=generator)
            direction /= direction.norm()
            directions.append(
                (
                    torch.dot(grad_linearize, direction).item(),
                    torch.dot(grad_unroll, direction).item(),
                    _directional_finite_difference(
                        unroll, model, batch, objective, weights, direction
                    ),
                )
            )
        results[objective] = {
            "relative_error": (
                (grad_linearize - grad_unroll).norm() / grad_unroll.norm()
            ).item(),
            "cosine": (
                torch.dot(grad_linearize, grad_unroll)
                / grad_linearize.norm()
                / grad_unroll.norm()
            ).item(),
            "directions": directions,
            "time_linearize": time_linearize,
            "time_unroll": time_unroll,
        }
    _RESULTS[key] = results
    return results


CASES = [
    pytest.param(
        model_kind,
        objective,
        constant_charge,
        batch_size,
        id=f"{model_kind}-{objective}-cc_{constant_charge}-bs_{batch_size}",
    )
    for model_kind in MODELS
    for objective in OBJECTIVES
    for constant_charge in (False, True)
    for batch_size in (1, 2)
]


@pytest.mark.parametrize("model_kind, objective, constant_charge, batch_size", CASES)
def test_unroll_scf_gradient_matches_finite_difference(
    model_kind, objective, constant_charge, batch_size
):
    results = _results(model_kind, constant_charge, batch_size)[objective]
    for _, unroll, finite_difference in results["directions"]:
        assert np.isclose(
            unroll,
            finite_difference,
            rtol=FINITE_DIFFERENCE_RTOL,
            atol=FINITE_DIFFERENCE_ATOL,
        ), f"unroll_scf={unroll}, finite_difference={finite_difference}"


@pytest.mark.parametrize("model_kind, objective, constant_charge, batch_size", CASES)
def test_linearize_solve_gradient_matches_unroll_scf(
    model_kind, objective, constant_charge, batch_size
):
    results = _results(model_kind, constant_charge, batch_size)[objective]
    directional = ", ".join(
        f"lin={lin:.6e} unroll={unroll:.6e} fd={fd:.6e}"
        for lin, unroll, fd in results["directions"]
    )
    print(
        f"\n{model_kind} {objective} cc={constant_charge} bs={batch_size}: "
        f"relative_error={results['relative_error']:.3e} "
        f"cosine={results['cosine']:.8f} "
        f"time linearize_solve={results['time_linearize']:.2f}s "
        f"unroll_scf={results['time_unroll']:.2f}s [{directional}]"
    )
    assert results["relative_error"] < GRADIENT_RTOL, (
        f"linearize_solve vs unroll_scf relative gradient error "
        f"{results['relative_error']:.3e}, cosine {results['cosine']:.8f}"
    )


LARGE_SYSTEM_REPEATS = (3, 3, 4)  # 36 water dimers, 216 atoms
INCLUDE_LARGE_SYSTEM_BENCHMARK = os.environ.get("MACE_TEST_LARGE_SYSTEM", "0") == "1"


@pytest.mark.skipif(
    not INCLUDE_LARGE_SYSTEM_BENCHMARK,
    reason="set MACE_TEST_LARGE_SYSTEM=1 to run the 216-atom cost benchmark",
)
def test_linearize_solve_cost_large_system():
    """Time and peak memory of one force-loss gradient on 216 atoms.

    Small systems cannot show the cost of the Jacobian build. Add -rA to see the
    printed numbers. On a CPU, peak memory is the growth of the process peak RSS,
    which cannot be reset, so it is only valid for linearize_solve, which runs first.
    """
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_fixed_point_model_random(
        seed=0, nonlinearity_cls=MODELS["nonlinear"], update_scale=UPDATE_SCALE
    ).to(device)
    atoms = periodic_water_dimer(CUBIC_CELL).repeat(LARGE_SYSTEM_REPEATS)
    dataset = dataset_from_atoms([atoms], cutoff=3.0, atomic_multipoles_max_l=1)
    batch = next(wrap_loader(dataset, device=device, batch_size=1))
    force_weights = torch.randn(
        batch["positions"].shape, generator=torch.Generator().manual_seed(1)
    ).to(device)

    gradients = {}
    for mode in ("linearize_solve", "unroll_scf"):
        wrapper = make_fixed_point_wrapper(model, mode, constant_charge=True)
        peak_rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _parameter_gradient(wrapper, model, batch, "force", force_weights)  # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        gradients[mode] = _parameter_gradient(wrapper, model, batch, "force", force_weights)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**30
        else:  # ru_maxrss is in KiB on Linux
            peak = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak_rss_before) / 2**20
        print(f"\n{mode} {len(atoms)} atoms on {device}: time={elapsed:.2f}s peak_memory={peak:.2f}GiB")

    relative_error = (
        (gradients["linearize_solve"] - gradients["unroll_scf"]).norm()
        / gradients["unroll_scf"].norm()
    ).item()
    print(f"relative_error={relative_error:.3e}")
    assert relative_error < GRADIENT_RTOL
