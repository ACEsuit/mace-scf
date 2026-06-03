import os

import mace.data
import mace.tools
import numpy as np
import pytest
import torch
from ase.io import read

import mace_scf.data
from mace_scf.utils.model_training_wrappers import LocalSourcesModelWrapper

from .paths import reference_config, reference_model, require_file
from .utils import split_to_graphs, wrap_loader


LSC_MODEL_PATH = reference_model("splitcharge_l1_stage1")
CONFIGS_PATH = reference_config("mixed_test_configs.xyz")
DEVICE = os.environ.get("MACE_DEVICE", "cpu")

OUTPUT_ARGS = {
    "forces": True,
    "virials": False,
    "stress": False,
}


def _load_reference_model(device=DEVICE):
    require_file(LSC_MODEL_PATH, "Reference LSC model")
    return torch.load(LSC_MODEL_PATH, map_location=device).to(device)


def _load_reference_atoms():
    require_file(CONFIGS_PATH, "Reference configs")
    return read(CONFIGS_PATH, index="0")


def _build_dataset(model, atoms_list, cutoff=None):
    keyspec = mace.data.KeySpecification()
    keyspec = mace_scf.data.update_keyspec_from_kwargs(
        keyspec,
        {"charges_key": "formal_oxidation_states"},
    )
    configs = mace.data.config_from_atoms_list(atoms_list, key_specification=keyspec)
    z_table = mace.tools.get_atomic_number_table_from_zs(
        list(set(atoms_list[0].get_atomic_numbers()))
    )
    return [
        mace_scf.data.ExtAtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=float(model.r_max if cutoff is None else cutoff),
            heads=model.heads,
        )
        for config in configs
    ]


def _central_difference_atoms(atoms_obj, delta=1e-4):
    atoms_minus = atoms_obj.copy()
    atoms_centre_a = atoms_obj.copy()
    atoms_centre_b = atoms_obj.copy()
    atoms_plus = atoms_obj.copy()
    atoms_minus.positions[0, 0] -= delta
    atoms_plus.positions[0, 0] += delta
    return [atoms_minus, atoms_centre_a, atoms_centre_b, atoms_plus]


def _build_wrapper(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    return LocalSourcesModelWrapper(
        optimizer=optimizer,
        output_args=OUTPUT_ARGS,
    )


def _force_gradient_values(wrapper, model, batch_size, delta=1e-5):
    base_atoms = _load_reference_atoms()
    dataset = _build_dataset(model, _central_difference_atoms(base_atoms, delta=delta))

    all_forces = []
    all_energies = []

    for batch_dict in wrap_loader(dataset, batch_size=batch_size, device=DEVICE):
        output = wrapper(model, batch_dict, training=True)
        all_energies += list(output["energy"].detach().cpu().numpy())
        all_forces += split_to_graphs(
            output["forces"].detach().cpu().numpy(),
            batch_dict["ptr"].detach().cpu().numpy(),
        )
    all_energies = np.asarray(all_energies)
    scanned_force_component = np.asarray([forces[0, 0] for forces in all_forces])
    numerical_force_component = -(
        all_energies[3] - all_energies[0]
    ) / (2.0 * delta)
    analytic_force_component = scanned_force_component[1:3]
    errors = analytic_force_component - numerical_force_component

    print(
        f"batch_size={batch_size} "
        f"max_abs_force_gradient_error={np.max(np.abs(errors))} "
        f"mean_abs_force_gradient_error={np.mean(np.abs(errors))}"
    )
    print(f"analytic_force_component={analytic_force_component}")
    print(f"numerical_force_component={numerical_force_component}")

    return analytic_force_component, numerical_force_component


@pytest.mark.parametrize("batch_size", [1, 2])
def test_lsc_force_scan_calibration(batch_size):
    torch.set_default_dtype(torch.float64)
    model = _load_reference_model()
    wrapper = _build_wrapper(model)

    analytic_force_component, numerical_force_component = _force_gradient_values(
        wrapper,
        model,
        batch_size=batch_size,
    )

    assert np.allclose(
        analytic_force_component,
        numerical_force_component,
        rtol=1e-7,
        atol=1e-9,
    ), (
        f"LSC force-gradient mismatch for batch_size={batch_size}: "
        f"analytic={analytic_force_component}, numerical={numerical_force_component}"
    )
