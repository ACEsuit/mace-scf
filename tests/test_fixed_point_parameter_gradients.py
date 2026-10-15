"""Parameter gradients of energy and force losses: linearize_solve vs unroll_scf.
i.e. checking second order gradients

To print the size of linearize_solve error add -s to pytest call.

TODO: Merge with tests/test_fixed_point_wrapper_training_modes.py.
I did not do this because it loads .model files from fixtures/models
and I am not sure how to obtain these models.
"""

import time

import numpy as np
import pytest
import torch

from mace_scf.electrostatics.fixed_point_state import (
    FixedPointSCFOptions,
    FixedPointTrainingOptions,
)
from mace_scf.utils.model_training_wrappers import FixedPointWrapper
from tests.utils import (
    CUBIC_CELL,
    TRICLINIC_CELL,
    dataset_from_atoms,
    make_fixed_point_model_random,
    periodic_water_dimer,
    wrap_loader,
)

UPDATE_SCALE = 0.2  # largest tried scale at which both models converge in both charge modes
SCF_TOLERANCE = 1e-12
NUM_SCF_STEPS = 400
FINITE_DIFFERENCE_STEP = 1e-5
NUM_DIRECTIONS = 2
GRADIENT_RTOL = 1e-6  # linearize_solve vs unroll_scf, relative L2; linear model observed <1e-8
FINITE_DIFFERENCE_RTOL = 1e-5  # unroll_scf vs finite difference; observed <~2e-7
FINITE_DIFFERENCE_ATOL = 1e-8

MODELS = {
    "linear": "NoNonLinearity",
    "nonlinear": "MLPNonLinearity",
}


def _build_wrapper(model, mode, constant_charge):
    scf_options = FixedPointSCFOptions(
        num_scf_steps=NUM_SCF_STEPS,
        scf_tolerance=SCF_TOLERANCE,
        constant_charge=constant_charge,
        mixing_parameter=0.5,
        initial_density="local_guess",
        initial_fermi_level="from_data",
        use_autograd_forces=True,
    )
    wrapper = FixedPointWrapper(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        output_args={"forces": True, "virials": False, "stress": False},
        training_options=FixedPointTrainingOptions(mode=mode, scf=scf_options),
    )
    if mode == "linearize_solve":
        # A silent fallback to unroll_scf would make the comparison vacuous.
        def _no_fallback(*args, **kwargs):
            raise AssertionError("linearize_solve fell back to unroll_scf")

        wrapper._forward_unroll_scf = _no_fallback
        wrapper._linearize_fallback_unroll = _no_fallback
    return wrapper


def _build_batch(batch_size):
    atoms_list = [periodic_water_dimer(TRICLINIC_CELL), periodic_water_dimer(CUBIC_CELL)]
    dataset = dataset_from_atoms(
        atoms_list[:batch_size], cutoff=3.0, atomic_multipoles_max_l=1
    )
    return next(wrap_loader(dataset, batch_size=batch_size))


def _copy_batch(batch):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def _loss(output, objective, force_weights):
    if objective == "energy":
        return output["energy"].sum()
    return (output["forces"] * force_weights).sum()


def _parameter_gradient(wrapper, model, batch, objective, force_weights):
    output = wrapper(_copy_batch(batch), training=True)
    params = list(model.parameters())
    grads = torch.autograd.grad(
        _loss(output, objective, force_weights), params, allow_unused=True
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


def _directional_finite_difference(wrapper, model, batch, objective, force_weights, direction):
    h = FINITE_DIFFERENCE_STEP
    values = []
    for step in (h, -2 * h):
        _shift_parameters(model, direction, step)
        output = wrapper(_copy_batch(batch), training=True)
        values.append(_loss(output, objective, force_weights).item())
    _shift_parameters(model, direction, h)
    return (values[0] - values[1]) / (2 * h)


def _scf_status(wrapper, model, batch):
    batch = _copy_batch(batch)
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
    """Gradients and finite differences for both objectives, computed once per case."""
    key = (model_kind, constant_charge, batch_size)
    if key in _RESULTS:
        return _RESULTS[key]

    torch.set_default_dtype(torch.float64)
    model = make_fixed_point_model_random(
        seed=0, nonlinearity_cls=MODELS[model_kind], update_scale=UPDATE_SCALE
    )
    batch = _build_batch(batch_size)
    linearize = _build_wrapper(model, "linearize_solve", constant_charge)
    unroll = _build_wrapper(model, "unroll_scf", constant_charge)
    status = _scf_status(unroll, model, batch)
    assert status == "converged", f"SCF did not converge for {key}: {status}"

    generator = torch.Generator().manual_seed(1)
    force_weights = torch.randn(batch["positions"].shape, generator=generator)
    results = {}
    for objective in ("energy", "force"):
        start = time.perf_counter()
        grad_linearize = _parameter_gradient(linearize, model, batch, objective, force_weights)
        time_linearize = time.perf_counter() - start
        start = time.perf_counter()
        grad_unroll = _parameter_gradient(unroll, model, batch, objective, force_weights)
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
                        unroll, model, batch, objective, force_weights, direction
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
    for objective in ("energy", "force")
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
