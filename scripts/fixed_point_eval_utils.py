import ast

import ase.io
import mace.data
import mace_scf.data
import numpy as np
import torch
from mace.data import Configurations
from mace.tools import torch_geometric, torch_tools, utils

from mace_scf.electrostatics.fixed_point import FixedPoint
from mace_scf.electrostatics.fixed_point_core import FixedPointCore
from mace_scf.electrostatics.fixed_point_options import (
    validate_fixed_point_scf_options,
)
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions


EVAL_SCF_DEFAULTS = {
    "num_scf_steps": 100,
    "scf_tolerance": 1e-6,
    "mixing_parameter": 0.25,
    "constant_charge": True,
    "use_autograd_forces": True,
    "initial_density": "local_guess",
    "initial_fermi_level": "zero",
}

COMMON_MANDATORY_OUTPUTS = (
    "energy",
    "density_coefficients",
    "charges_history",
    "electrostatic_features",
    "dipole",
    "electrostatic_energy",
    "electron_energy",
    "fermi_level",
)


def log_dataset_contents(dataset: Configurations, dataset_name: str) -> None:
    log_string = f"{dataset_name} ["
    for prop_name in dataset[0].properties.keys():
        if prop_name == "dipole":
            log_string += (
                f"{prop_name} components: "
                f"{int(np.sum([np.sum(config.property_weights[prop_name]) for config in dataset]))}, "
            )
        else:
            log_string += (
                f"{prop_name}: "
                f"{int(np.sum([config.property_weights[prop_name] for config in dataset]))}, "
            )
    log_string = log_string[:-2] + "]"
    print("")
    print(log_string)
    print("")


def normalize_scf_options(scf_options_text):
    raw_options = (
        {} if scf_options_text is None else ast.literal_eval(scf_options_text)
    )
    if not isinstance(raw_options, dict):
        raise TypeError("scf_options must parse to a dict")
    return validate_fixed_point_scf_options(raw_options, EVAL_SCF_DEFAULTS)


def load_fixed_point_model(model_path: str, device: str):
    model = torch.load(f=model_path, map_location=device).to(device)
    if isinstance(model, FixedPoint) and not isinstance(model, FixedPointCore):
        from scripts.convert_models import convert_fixedpoint_to_core

        model = convert_fixedpoint_to_core(model)
    for param in model.parameters():
        param.requires_grad = False
    validate_fixed_point_model(model)
    return model


def validate_fixed_point_model(model) -> None:
    if not hasattr(model, "heads"):
        raise ValueError("Fixed-point evaluation requires a model with `heads`.")
    if not hasattr(model, "coulomb_energy") or not hasattr(
        model.coulomb_energy, "density_max_l"
    ):
        raise ValueError(
            "Fixed-point evaluation requires a model with "
            "`coulomb_energy.density_max_l`."
        )


def build_fixed_point_keyspec(
    external_field_key: str,
    fermi_level_key: str,
    atomic_multipoles_key: str,
    total_charge_key: str,
):
    return mace.data.KeySpecification(
        info_keys={
            "external_field": external_field_key,
            "fermi_level": fermi_level_key,
            "total_charge": total_charge_key,
        },
        arrays_keys={
            "atomic_multipoles": atomic_multipoles_key,
        },
    )


def load_atoms_and_configs(configs_path: str, keyspec):
    atoms_list = ase.io.read(configs_path, index=":")
    configs = [
        mace.data.config_from_atoms(
            atoms,
            key_specification=keyspec,
        )
        for atoms in atoms_list
    ]
    return atoms_list, configs


def check_restart_info(dataset, restart_fermi_level, restart_multipoles):
    num_fermi_levels = sum(
        int(config.property_weights["fermi_level"]) for config in dataset
    )
    num_density_coefficients = sum(
        int(config.property_weights["atomic_multipoles"]) for config in dataset
    )
    num_configs = len(dataset)

    if restart_fermi_level and (num_fermi_levels < num_configs):
        raise ValueError(
            "Not all configurations have Fermi-level information, cannot use "
            "--initial_fermi_level from_data. "
            f"num_fermi_levels={num_fermi_levels}, num_configs={num_configs}"
        )
    if restart_multipoles and (num_density_coefficients < num_configs):
        raise ValueError(
            "Not all configurations have atomic multipoles information, cannot use "
            "--initial_density from_data. "
            f"num_atomic_multipoles={num_density_coefficients}, num_configs={num_configs}"
        )


def check_direct_mode_info(dataset):
    num_fermi_levels = sum(
        int(config.property_weights["fermi_level"]) for config in dataset
    )
    num_density_coefficients = sum(
        int(config.property_weights["atomic_multipoles"]) for config in dataset
    )
    num_configs = len(dataset)

    if num_fermi_levels < num_configs:
        raise ValueError(
            "Direct mode requires Fermi-level information from "
            "--fermi_level_key. "
            f"num_fermi_levels={num_fermi_levels}, num_configs={num_configs}"
        )
    if num_density_coefficients < num_configs:
        raise ValueError(
            "Direct mode requires atomic multipoles from "
            "--atomic_multipoles_key. "
            f"num_atomic_multipoles={num_density_coefficients}, "
            f"num_configs={num_configs}"
        )


def build_fixed_point_dataset(configs, model):
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atomic_multipoles_max_l = int(model.coulomb_energy.density_max_l)
    return [
        mace_scf.data.ExtAtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=float(model.r_max),
            atomic_multipoles_max_l=atomic_multipoles_max_l,
            heads=model.heads,
        )
        for config in configs
    ]


def build_fixed_point_dataloader(dataset, batch_size: int):
    return torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def get_requested_output_keys(
    compute_force: bool,
    return_contributions: bool,
    model,
):
    requested_output_keys = set()
    if compute_force:
        requested_output_keys.add("forces")
    if return_contributions:
        requested_output_keys.add("contributions")
    if getattr(model, "return_electrostatic_potentials", False):
        requested_output_keys.add("esps")
    return requested_output_keys


def run_scf_batch(
    model,
    batch,
    scf_options: FixedPointSCFOptions,
    compute_force: bool,
    restart_multipoles: bool,
    restart_fermi_level: bool,
):
    runner = FixedPointSCFRunner(
        FixedPointSCFOptions(
            num_scf_steps=scf_options.num_scf_steps,
            scf_tolerance=scf_options.scf_tolerance,
            mixing_parameter=scf_options.mixing_parameter,
            constant_charge=scf_options.constant_charge,
            use_autograd_forces=scf_options.use_autograd_forces,
            initial_density="from_data" if restart_multipoles else "local_guess",
            initial_fermi_level="from_data" if restart_fermi_level else "zero",
        )
    )
    return runner.eval(
        model=model,
        data=batch.to_dict(),
        training=False,
        compute_force=compute_force,
    )


def run_direct_fixed_point_batch(
    model,
    batch,
    compute_force: bool,
):
    batch_dict = batch.to_dict()
    for param in model.parameters():
        param.requires_grad = False

    local_state = model.local_part(
        batch_dict,
        compute_force=compute_force,
    )

    fermi_level_features = model.features_from_fermi_level(
        batch_dict["batch"],
        local_state.positions,
        batch_dict["fermi_level"],
    )

    field_dep, field_feats = model.scf_step(
        batch_dict,
        local_state,
        charge_density_in=batch_dict["density_coefficients"],
        total_charges=batch_dict["density_coefficients"],
        fermi_level_features=fermi_level_features,
    )
    density = local_state.field_independent_charge_density + field_dep

    output = model.build_observables(
        data=batch_dict,
        local_state=local_state,
        density=density,
        fermi_level=batch_dict["fermi_level"],
        field_feats=field_feats,
        training=False,
        compute_force=compute_force,
        compute_virials=False,
        compute_stress=False,
    )
    output["charges_history"] = torch.stack([density.detach()], dim=-1)
    return output


def require_tensor_output(output, key: str) -> None:
    if key not in output:
        raise KeyError(f"Missing mandatory model output: {key}")
    if output[key] is None:
        raise ValueError(f"Mandatory model output is None: {key}")


def validate_fixed_point_output(output, mandatory_keys, requested_output_keys) -> None:
    for key in mandatory_keys:
        require_tensor_output(output, key)
    for key in requested_output_keys:
        require_tensor_output(output, key)


def split_output_tensor(batch, output, key):
    return split_numpy_by_batch_ptr(batch, torch_tools.to_numpy(output[key]))


def split_numpy_by_batch_ptr(batch, array):
    split_indices = torch_tools.to_numpy(batch.ptr[1:])
    return np.split(array, indices_or_sections=split_indices, axis=0)[:-1]
