import os

import numpy as np
import pytest
import torch
from ase.io import read

from mace_scf.utils.model_training_wrappers import FixedPointWrapper
from mace_scf.electrostatics.fixed_point_state import (
    FixedPointSCFOptions,
    FixedPointTrainingOptions,
)

from .paths import reference_config, reference_model, require_file
from .utils import dataset_from_atoms, split_to_graphs, wrap_loader


MODEL_PATH = reference_model("fixedpoint_onebodylinear")
CONFIGS_PATH = reference_config("mixed_test_configs.xyz")
DEVICE = os.environ.get("MACE_DEVICE", "cpu")
INCLUDE_FERMI_LEVEL_OBJECTIVE = os.environ.get("MACE_TEST_FERMI_LEVEL", "1") == "1"
INCLUDE_ESP_OBJECTIVE = os.environ.get("MACE_TEST_ESPS", "1") == "1"
INCLUDE_FORCE_OBJECTIVE = os.environ.get("MACE_TEST_FORCES", "0") == "1"
INCLUDE_IMPLICIT_NORMAL_CG = (
    os.environ.get("MACE_TEST_IMPLICIT_NORMAL_CG", "0") == "1"
)
IMPLICIT_LINEAR_SOLVES = ["inverse"]
if INCLUDE_IMPLICIT_NORMAL_CG:
    IMPLICIT_LINEAR_SOLVES.append("normal_cg")
PERIODIC = bool(os.environ.get(
    "USE_PBC",
    False,
))
OUTPUT_ARGS = {
    "forces": True,
    "virials": False,
    "stress": False,
}


def _scf_objectives():
    objectives = ["energy"]
    if INCLUDE_FERMI_LEVEL_OBJECTIVE:
        objectives.append("fermi_level")
    if INCLUDE_ESP_OBJECTIVE:
        objectives.append("esp_scalar")
    if INCLUDE_FORCE_OBJECTIVE:
        objectives.append("force_scalar")
    return objectives


def _direct_objectives():
    objectives = ["energy"]
    if INCLUDE_ESP_OBJECTIVE:
        objectives.append("esp_scalar")
    if INCLUDE_FORCE_OBJECTIVE:
        objectives.append("force_scalar")
    return objectives


def _mode_charge_batch_cases(modes):
    return [
        pytest.param(
            mode,
            constant_charge,
            batch_size,
            id=f"{mode}-cc_{constant_charge}-bs_{batch_size}",
        )
        for mode in modes
        for constant_charge in [False, True]
        for batch_size in [1, 2]
    ]


def _objective_batch_cases(objectives):
    return [
        pytest.param(
            objective,
            batch_size,
            id=f"{objective}-bs_{batch_size}",
        )
        for objective in objectives
        for batch_size in [1, 2]
    ]


def _scf_objective_mode_charge_batch_linear_solve_cases(modes):
    return [
        pytest.param(
            mode,
            objective,
            constant_charge,
            batch_size,
            linear_solve,
            id=(
                f"{mode}-{linear_solve}-{objective}-"
                f"cc_{constant_charge}-bs_{batch_size}"
            ),
        )
        for mode in modes
        for objective in _scf_objectives()
        for constant_charge in [False, True]
        for batch_size in [1, 2]
        for linear_solve in (
            IMPLICIT_LINEAR_SOLVES if mode == "implicit" else ["inverse"]
        )
        if not (objective == "fermi_level" and not constant_charge)
    ]


def _scf_objective_charge_batch_cases():
    return [
        pytest.param(
            objective,
            constant_charge,
            batch_size,
            id=f"{objective}-cc_{constant_charge}-bs_{batch_size}",
        )
        for objective in _scf_objectives()
        for constant_charge in [False, True]
        for batch_size in [1, 2]
        if not (objective == "fermi_level" and not constant_charge)
    ]


def _load_reference_model(device=DEVICE):
    require_file(MODEL_PATH, "Reference fixed-point model")
    model = torch.load(MODEL_PATH, map_location=device).to(device)
    model.coulomb_energy.set_pbc_handling("mixed_periodic")
    model.electric_potential_descriptor.set_pbc_handling("mixed_periodic")
    model.return_electrostatic_potentials = INCLUDE_ESP_OBJECTIVE
    return model


def _load_reference_atoms(num_configs=2):
    require_file(CONFIGS_PATH, "Reference configs")
    atoms_list = read(CONFIGS_PATH, index=":")
    if PERIODIC:
        for atoms in atoms_list:
            atoms.set_pbc([True, True, True])
    return atoms_list[:num_configs]


def _build_dataset(model, atoms_list):
    return dataset_from_atoms(
        atoms_list,
        cutoff=float(model.r_max),
        atomic_multipoles_key="some_multipoles",
        fermi_level_key="the_VBM",
        atomic_multipoles_max_l=int(model.coulomb_energy.density_max_l),
    )


def _build_wrapper(model, mode, constant_charge, linear_solve="inverse"):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scf_options = None
    if mode != "direct":
        scf_options = FixedPointSCFOptions(
            num_scf_steps=100,
            scf_tolerance=1e-9,
            constant_charge=constant_charge,
            mixing_parameter=0.5,
            initial_density="from_data",
            initial_fermi_level="from_data",
            use_autograd_forces=True,
        )
    return FixedPointWrapper(
        model=model,
        optimizer=optimizer,
        output_args=OUTPUT_ARGS,
        training_options=FixedPointTrainingOptions(
            mode=mode,
            scf=scf_options,
            linear_solve=linear_solve,
        ),
    )


def _cleanup_model(model):
    if hasattr(model, "batch_positions"):
        del model.batch_positions


def _first_parameter_for_gradient(model):
    for name, param in model.named_parameters():
        if "field_dependent_charges_map" in name and param.requires_grad:
            return name, param
    raise AssertionError("No suitable field-dependent charge-map parameter found")


def _scalar_objective(output, objective):
    if objective == "energy":
        return output["energy"].sum()
    if objective == "fermi_level":
        return output["fermi_level"].sum()
    if objective == "esp_scalar":
        if "esps" not in output or output["esps"] is None:
            pytest.skip("Model output does not include esps")
        return output["esps"].sum()
    if objective == "force_scalar":
        if "forces" not in output or output["forces"] is None:
            pytest.skip("Model output does not include forces")
        return output["forces"][0,:].sum()
    raise ValueError(f"Unknown objective: {objective}")


def _wrapper_objective_gradient(wrapper, model, batch_dict, objective):
    model.zero_grad(set_to_none=True)
    output = wrapper(batch_dict, training=True)
    loss = _scalar_objective(output, objective)
    _, param = _first_parameter_for_gradient(model)
    grad = torch.autograd.grad(loss, param)[0].reshape(-1)[0].detach().cpu().item()
    _cleanup_model(model)
    return grad


def _finite_difference_objective_gradient(wrapper, model, batch_dict, objective, delta=1e-5):
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
    atoms_centre_a = base_atoms.copy()
    atoms_centre_b = base_atoms.copy()
    atoms_plus = base_atoms.copy()
    atoms_minus.positions[0, 0] -= delta
    atoms_plus.positions[0, 0] += delta
    return _build_dataset(model, [atoms_minus, atoms_centre_a, atoms_centre_b, atoms_plus])


def _run_force_gradient_check(wrapper, model, batch_size, constant_charge, delta=1e-5):
    base_atoms = _load_reference_atoms(num_configs=1)[0]
    dataset = _central_difference_dataset(model, base_atoms, delta=delta)

    all_forces = []
    all_energies = []

    for batch_dict in wrap_loader(dataset, batch_size=batch_size, device=DEVICE):
        output = wrapper(batch_dict, training=True)
        all_energies += list(output["energy"].detach().cpu().numpy())
        all_forces += split_to_graphs(
            output["forces"].detach().cpu().numpy(),
            batch_dict["ptr"].detach().cpu().numpy(),
        )
        _cleanup_model(model)

    all_energies = np.asarray(all_energies)
    scanned_force_component = np.asarray([forces[0, 0] for forces in all_forces])
    numerical_force_component = -(
        all_energies[3] - all_energies[0]
    ) / (2.0 * delta)
    analytic_force_component = scanned_force_component[1:3]
    errors = analytic_force_component - numerical_force_component

    print(
        f"max_abs_force_gradient_error={np.max(np.abs(errors))} "
        f"numerical_force_component={numerical_force_component}"
    )
    #print(f"analytic_force_component={analytic_force_component}")
    #print(f"numerical_force_component={numerical_force_component}")

    assert np.allclose(
        analytic_force_component,
        numerical_force_component,
        rtol=1e-6,
        atol=2e-7,
    ), (
        f"Force / energy-gradient mismatch for mode={wrapper.mode}, "
        f"constant_charge={constant_charge}, batch_size={batch_size}"
    )


@pytest.mark.parametrize(
    "mode, constant_charge, batch_size",
    _mode_charge_batch_cases(["direct", "unroll_scf", "implicit"])
    + _mode_charge_batch_cases(["linearize_solve"]),
)
def test_wrapper_output_contract(mode, constant_charge, batch_size):
    torch.set_default_dtype(torch.float64)
    model = _load_reference_model()
    atoms_list = _load_reference_atoms(num_configs=2)
    dataset = _build_dataset(model, atoms_list)
    wrapper = _build_wrapper(model, mode=mode, constant_charge=constant_charge)

    batch_dict = next(wrap_loader(dataset, batch_size=batch_size, device=DEVICE))
    output = wrapper(batch_dict, training=True)

    assert "energy" in output
    assert "density_coefficients" in output
    if mode != "implicit":
        assert "charges_history" in output
    assert "fermi_level" in output
    assert "forces" in output

    assert output["energy"].shape[0] == batch_size
    assert output["density_coefficients"].shape[0] == batch_dict["positions"].shape[0]
    assert output["forces"].shape == batch_dict["positions"].shape
    if mode != "implicit":
        assert output["charges_history"].shape[0] == batch_dict["positions"].shape[0]

    if mode == "direct":
        assert output["charges_history"].shape[-1] == 1
    elif mode == 'unroll_scf':
        assert output["charges_history"].shape[-1] >= 1

    _cleanup_model(model)


@pytest.mark.parametrize(
    "mode, constant_charge, batch_size",
    _mode_charge_batch_cases(["unroll_scf", "implicit"])
    + _mode_charge_batch_cases(["linearize_solve"]),
)
def test_wrapper_force_energy_consistency(mode, constant_charge, batch_size):
    torch.set_default_dtype(torch.float64)
    model = _load_reference_model()
    wrapper = _build_wrapper(model, mode=mode, constant_charge=constant_charge)
    _run_force_gradient_check(
        wrapper,
        model,
        batch_size=batch_size,
        constant_charge=constant_charge,
    )


@pytest.mark.parametrize(
    "objective, batch_size",
    _objective_batch_cases(_direct_objectives()),
)
def test_direct_wrapper_parameter_gradient_matches_finite_difference(objective, batch_size):
    torch.set_default_dtype(torch.float64)
    model = _load_reference_model()
    atoms_list = _load_reference_atoms(num_configs=2)
    dataset = _build_dataset(model, atoms_list)
    batch_dict = next(wrap_loader(dataset, batch_size=batch_size, device=DEVICE))
    wrapper = _build_wrapper(model, mode="direct", constant_charge=True)

    analytic = _wrapper_objective_gradient(wrapper, model, batch_dict, objective)
    finite_difference = _finite_difference_objective_gradient(
        wrapper,
        model,
        batch_dict,
        objective,
    )
    print(
        f"\tmax_abs_gradient_error={np.max(np.abs(analytic-finite_difference))} "
        f"numerical_gradient={finite_difference}"
    )

    assert np.allclose(
        analytic,
        finite_difference,
        rtol=1e-6,
        atol=1e-6,
    ), (
        f"\tmax_abs_gradient_error={np.max(np.abs(analytic-finite_difference))} "
        f"numerical_gradient={finite_difference}"
    )


@pytest.mark.parametrize(
    "mode, objective, constant_charge, batch_size, linear_solve",
    _scf_objective_mode_charge_batch_linear_solve_cases(["unroll_scf", "implicit"])
    + _scf_objective_mode_charge_batch_linear_solve_cases(["linearize_solve"]),
)
def test_scf_wrapper_parameter_gradient_matches_finite_difference(
    mode,
    objective,
    constant_charge,
    batch_size,
    linear_solve,
):
    torch.set_default_dtype(torch.float64)
    model = _load_reference_model()
    atoms_list = _load_reference_atoms(num_configs=2)
    dataset = _build_dataset(model, atoms_list)
    batch_dict = next(wrap_loader(dataset, batch_size=batch_size, device=DEVICE))
    wrapper = _build_wrapper(
        model,
        mode=mode,
        constant_charge=constant_charge,
        linear_solve=linear_solve,
    )

    analytic = _wrapper_objective_gradient(wrapper, model, batch_dict, objective)
    finite_difference = _finite_difference_objective_gradient(
        wrapper,
        model,
        batch_dict,
        objective,
    )
    print(
        f"mode={mode}, linear_solve={linear_solve}, objective={objective}, "
        f"constant_charge={constant_charge}, batch_size={batch_size}\n"
        f"\tmax_abs_gradient_error={np.max(np.abs(analytic-finite_difference))} "
        f"numerical_gradient={finite_difference}"
    )

    assert np.allclose(
        analytic,
        finite_difference,
        rtol=1e-6,
        atol=1e-6,
    ), (
        f"mode={mode}, objective={objective}, constant_charge={constant_charge}, "
        f"linear_solve={linear_solve}, analytic={analytic}, "
        f"finite_difference={finite_difference}"
    )


@pytest.mark.parametrize(
    "objective, constant_charge, batch_size",
    _scf_objective_charge_batch_cases(),
)
def test_implicit_and_unroll_gradients_agree(objective, constant_charge, batch_size):
    torch.set_default_dtype(torch.float64)
    atoms_list = _load_reference_atoms(num_configs=2)

    unroll_model = _load_reference_model()
    implicit_model = _load_reference_model()

    dataset_unroll = _build_dataset(unroll_model, atoms_list)
    dataset_implicit = _build_dataset(implicit_model, atoms_list)

    batch_unroll = next(wrap_loader(dataset_unroll, batch_size=batch_size, device=DEVICE))
    batch_implicit = next(wrap_loader(dataset_implicit, batch_size=batch_size, device=DEVICE))

    unroll_wrapper = _build_wrapper(
        unroll_model,
        mode="unroll_scf",
        constant_charge=constant_charge,
    )
    implicit_wrapper = _build_wrapper(
        implicit_model,
        mode="implicit",
        constant_charge=constant_charge,
    )

    unroll_grad = _wrapper_objective_gradient(
        unroll_wrapper,
        unroll_model,
        batch_unroll,
        objective,
    )
    implicit_grad = _wrapper_objective_gradient(
        implicit_wrapper,
        implicit_model,
        batch_implicit,
        objective,
    )
    print(
        f"\tmax_abs_gradient_error={np.max(np.abs(implicit_grad-unroll_grad))} "
        f"reference_gradient={unroll_grad}"
    )

    assert np.allclose(
        implicit_grad,
        unroll_grad,
        rtol=1e-6,
        atol=1e-6,
    ), (
        f"\tmax_abs_gradient_error={np.max(np.abs(implicit_grad-unroll_grad))} "
        f"reference_gradient={unroll_grad}"
    )


@pytest.mark.parametrize(
    "objective, constant_charge, batch_size",
    _scf_objective_charge_batch_cases(),
)
def test_linearize_solve_and_implicit_gradients_agree(
    objective, constant_charge, batch_size
):
    torch.set_default_dtype(torch.float64)
    atoms_list = _load_reference_atoms(num_configs=2)

    implicit_model = _load_reference_model()
    linearize_model = _load_reference_model()

    dataset_implicit = _build_dataset(implicit_model, atoms_list)
    dataset_linearize = _build_dataset(linearize_model, atoms_list)

    batch_implicit = next(wrap_loader(dataset_implicit, batch_size=batch_size, device=DEVICE))
    batch_linearize = next(wrap_loader(dataset_linearize, batch_size=batch_size, device=DEVICE))

    implicit_wrapper = _build_wrapper(
        implicit_model,
        mode="implicit",
        constant_charge=constant_charge,
    )
    linearize_wrapper = _build_wrapper(
        linearize_model,
        mode="linearize_solve",
        constant_charge=constant_charge,
    )

    implicit_grad = _wrapper_objective_gradient(
        implicit_wrapper,
        implicit_model,
        batch_implicit,
        objective,
    )
    linearize_grad = _wrapper_objective_gradient(
        linearize_wrapper,
        linearize_model,
        batch_linearize,
        objective,
    )
    print(
        f"\tmax_abs_gradient_error={np.max(np.abs(linearize_grad-implicit_grad))} "
        f"reference_gradient={implicit_grad}"
    )

    assert np.allclose(
        linearize_grad,
        implicit_grad,
        rtol=1e-6,
        atol=1e-6,
    ), (
        f"\tmax_abs_gradient_error={np.max(np.abs(linearize_grad-implicit_grad))} "
        f"reference_gradient={implicit_grad}"
    )
