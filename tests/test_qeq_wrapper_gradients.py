"""Focused gradient tests for the MACEQEq training wrapper."""

import os

import numpy as np
import pytest
import torch
from ase.io import read

from mace_scf.utils.model_training_wrappers import QEqModelWrapper

from .paths import reference_model, reference_output, require_file
from .utils import dataset_from_atoms, split_to_graphs, wrap_loader


MODEL_PATH = reference_model("QEqfit_stage2")
CONFIGS_PATH = reference_output("qeq_stage2")
DEVICE = os.environ.get("MACE_DEVICE", "cpu")
OUTPUT_ARGS = {
    "forces": True,
    "virials": False,
    "stress": False,
}


def _load_reference_model(device=DEVICE):
    require_file(MODEL_PATH, "Reference QEq model")
    return torch.load(MODEL_PATH, map_location=device, weights_only=False).to(device)


def _load_reference_atoms(num_configs=1):
    require_file(CONFIGS_PATH, "Reference QEq configs")
    return read(CONFIGS_PATH, index=":")[:num_configs]


def _build_dataset(model, atoms_list):
    return dataset_from_atoms(
        atoms_list,
        cutoff=float(model.r_max),
        charges_key="expected_charges",
        atomic_multipoles_max_l=0,
    )


def _build_wrapper(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    return QEqModelWrapper(model=model, optimizer=optimizer, output_args=OUTPUT_ARGS)


def _cleanup_model(model):
    if hasattr(model, "batch_positions"):
        del model.batch_positions


def _first_parameter_for_gradient(model):
    for name, param in model.named_parameters():
        if "enegs_readouts" in name and param.requires_grad:
            return name, param
    raise AssertionError("No suitable electronegativity readout parameter found")


def _scalar_objective(output, objective):
    if objective == "energy":
        return output["energy"].sum()
    if objective == "forces":
        return output["forces"][0, 0]
    if objective == "charges":
        return output["density_coefficients"].reshape(-1)[0]
    raise ValueError(f"Unknown objective: {objective}")


def _wrapper_parameter_gradient(wrapper, model, batch_dict, objective):
    model.zero_grad(set_to_none=True)
    output = wrapper(batch_dict, training=True)
    loss = _scalar_objective(output, objective)
    _, param = _first_parameter_for_gradient(model)
    grad = torch.autograd.grad(loss, param)[0].reshape(-1)[0].detach().cpu().item()
    _cleanup_model(model)
    return grad


def _finite_difference_parameter_gradient(
    wrapper, model, batch_dict, objective, delta=1e-5
):
    _, param = _first_parameter_for_gradient(model)
    initial_value = param.detach().clone()
    flat = initial_value.reshape(-1)

    with torch.no_grad():
        plus_value = initial_value.clone()
        plus_value.reshape(-1)[0] = flat[0] + delta
        param.copy_(plus_value)
    plus_output = wrapper(batch_dict, training=True)
    plus_value = _scalar_objective(plus_output, objective).detach().cpu().item()
    _cleanup_model(model)

    with torch.no_grad():
        minus_value = initial_value.clone()
        minus_value.reshape(-1)[0] = flat[0] - delta
        param.copy_(minus_value)
    minus_output = wrapper(batch_dict, training=True)
    minus_value = _scalar_objective(minus_output, objective).detach().cpu().item()
    _cleanup_model(model)

    with torch.no_grad():
        param.copy_(initial_value)

    return (plus_value - minus_value) / (2.0 * delta)


def _central_difference_dataset(model, base_atoms, delta=1e-5):
    atoms_minus = base_atoms.copy()
    atoms_center = base_atoms.copy()
    atoms_plus = base_atoms.copy()
    atoms_minus.positions[0, 0] -= delta
    atoms_plus.positions[0, 0] += delta
    return _build_dataset(model, [atoms_minus, atoms_center, atoms_plus])


def test_qeq_wrapper_output_contract():
    torch.set_default_dtype(torch.float64)
    model = _load_reference_model()
    atoms_list = _load_reference_atoms(num_configs=1)
    dataset = _build_dataset(model, atoms_list)
    batch_dict = next(wrap_loader(dataset, batch_size=1, device=DEVICE))
    wrapper = _build_wrapper(model)

    output = wrapper(batch_dict, training=True)

    assert "energy" in output
    assert "qeq_energy" in output
    assert "forces" in output
    assert "density_coefficients" in output
    assert "dipole" in output
    assert "enegs" in output
    assert "hardness" in output
    assert output["energy"].shape == (1,)
    assert output["qeq_energy"].shape == (1,)
    assert output["forces"].shape == batch_dict["positions"].shape
    assert output["density_coefficients"].shape[0] == batch_dict["positions"].shape[0]

    _cleanup_model(model)


def test_qeq_wrapper_force_matches_energy_gradient():
    torch.set_default_dtype(torch.float64)
    model = _load_reference_model()
    wrapper = _build_wrapper(model)
    base_atoms = _load_reference_atoms(num_configs=1)[0]
    delta = 1e-3
    dataset = _central_difference_dataset(model, base_atoms, delta=delta)

    all_energies = []
    all_forces = []
    for batch_dict in wrap_loader(dataset, batch_size=1, device=DEVICE):
        output = wrapper(batch_dict, training=True)
        all_energies += list(output["energy"].detach().cpu().numpy())
        all_forces += split_to_graphs(
            output["forces"].detach().cpu().numpy(),
            batch_dict["ptr"].detach().cpu().numpy(),
        )
        _cleanup_model(model)

    numerical_force = -(all_energies[2] - all_energies[0]) / (2.0 * delta)
    analytic_force = all_forces[1][0, 0]
    error = abs(numerical_force - analytic_force)
    print(
        f"qeq_parameter_gradient_error={error} "
        f"analytic={numerical_force} finite_difference={analytic_force}"
    )

    assert np.allclose(
        analytic_force,
        numerical_force,
        rtol=1e-8,
        atol=1e-8,
    ), (
        f"Force / energy-gradient mismatch: analytic={analytic_force}, "
        f"finite_difference={numerical_force}"
    )


@pytest.mark.parametrize("objective", ["energy", "forces", "charges"])
def test_qeq_wrapper_parameter_gradient_matches_finite_difference(objective):
    torch.set_default_dtype(torch.float64)
    model = _load_reference_model()
    atoms_list = _load_reference_atoms(num_configs=1)
    dataset = _build_dataset(model, atoms_list)
    batch_dict = next(wrap_loader(dataset, batch_size=1, device=DEVICE))
    wrapper = _build_wrapper(model)

    analytic = _wrapper_parameter_gradient(wrapper, model, batch_dict, objective)
    finite_difference = _finite_difference_parameter_gradient(
        wrapper, model, batch_dict, objective
    )
    error = abs(analytic - finite_difference)
    print(
        f"objective={objective} qeq_parameter_gradient_error={error} "
        f"analytic={analytic} finite_difference={finite_difference}"
    )

    assert np.allclose(
        analytic,
        finite_difference,
        rtol=1e-5,
        atol=1e-5,
    ), (
        f"{objective} parameter-gradient mismatch: analytic={analytic}, "
        f"finite_difference={finite_difference}"
    )
