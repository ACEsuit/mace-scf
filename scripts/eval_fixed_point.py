import argparse
import logging
import warnings

import ase.io
import numpy as np
from mace.tools import torch_tools

from fixed_point_eval_utils import (
    COMMON_MANDATORY_OUTPUTS,
    build_fixed_point_dataloader,
    build_fixed_point_dataset,
    build_fixed_point_keyspec,
    check_direct_mode_info,
    check_restart_info,
    get_requested_output_keys,
    load_atoms_and_configs,
    load_fixed_point_model,
    log_dataset_contents,
    normalize_scf_options,
    run_direct_fixed_point_batch,
    run_scf_batch,
    split_output_tensor,
    validate_fixed_point_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["scf", "direct"],
        default="scf",
        help="evaluation mode: SCF solve or direct fixed-point update",
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--pbc_handling",
        help="electrostatic boundary-condition handling",
        type=str,
        choices=[
            "realspace",
            "pbc",
            "slab",
            "molecule_in_box",
            "mixed_periodic",
            "auto",
        ],
        default="mixed_periodic",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dont_compute_force",
        action="store_true",
    )
    parser.add_argument(
        "--return_contributions",
        help="write model energy contributions to the output xyz",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for stored output keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--scf_history",
        help="how much of the scf history to return",
        type=str,
        default="absolute_change",
        choices=[
            "none",
            "absolute_change",
            "full_history",
        ],
    )
    parser.add_argument(
        "--external_field_key",
        help="key for external field",
        type=str,
        default="external_field",
    )
    parser.add_argument(
        "--fermi_level_key",
        help="key for external field",
        type=str,
        default="fermi_level",
    )
    parser.add_argument(
        "--atomic_multipoles_key",
        type=str,
        default="AIMS_atom_multipoles",
    )
    parser.add_argument(
        "--total_charge_key",
        type=str,
        default="total_charge",
    )
    parser.add_argument(
        "--scf_options",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--initial_density",
        choices=["local_guess", "from_data"],
        default=None,
        help="initial SCF density source",
    )
    parser.add_argument(
        "--initial_fermi_level",
        choices=["zero", "from_data"],
        default=None,
        help="initial SCF Fermi-level source",
    )
    parser.add_argument(
        "--scf_restart_multipoles",
        action="store_true",
        help="deprecated alias for --initial_density from_data",
    )
    parser.add_argument(
        "--scf_restart_fermi_level",
        action="store_true",
        help="deprecated alias for --initial_fermi_level from_data",
    )
    return parser.parse_args()


def _resolve_initial_state_options(args):
    initial_density = args.initial_density or "local_guess"
    initial_fermi_level = args.initial_fermi_level or "zero"

    if args.scf_restart_multipoles:
        if args.initial_density is not None and args.initial_density != "from_data":
            raise ValueError(
                "--scf_restart_multipoles conflicts with "
                "--initial_density local_guess."
            )
        initial_density = "from_data"

    if args.scf_restart_fermi_level:
        if (
            args.initial_fermi_level is not None
            and args.initial_fermi_level != "from_data"
        ):
            raise ValueError(
                "--scf_restart_fermi_level conflicts with "
                "--initial_fermi_level zero."
            )
        initial_fermi_level = "from_data"

    return initial_density, initial_fermi_level


def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    if args.compute_stress:
        raise ValueError("Stress is not supported for the FixedPoint evaluation script.")

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.getLogger("mace_scf").setLevel(logging.DEBUG)

    initial_density, initial_fermi_level = _resolve_initial_state_options(args)
    if args.mode != "scf" and args.scf_options is not None:
        warnings.warn(
            "SCF options are ignored unless --mode scf is selected.",
            DeprecationWarning,
            stacklevel=2,
        )
    scf_options = (
        normalize_scf_options(args.scf_options) if args.mode == "scf" else None
    )
    model = load_fixed_point_model(args.model, args.device)
    model.coulomb_energy.set_pbc_handling(args.pbc_handling)
    model.electric_potential_descriptor.set_pbc_handling(args.pbc_handling)
    requested_output_keys = get_requested_output_keys(
        compute_force=not args.dont_compute_force,
        return_contributions=args.return_contributions,
        model=model,
    )

    keyspec = build_fixed_point_keyspec(
        external_field_key=args.external_field_key,
        fermi_level_key=args.fermi_level_key,
        atomic_multipoles_key=args.atomic_multipoles_key,
        total_charge_key=args.total_charge_key,
    )
    atoms_list, configs = load_atoms_and_configs(args.configs, keyspec)
    log_dataset_contents(dataset=configs, dataset_name="Evaluation dataset")
    if args.mode == "scf":
        check_restart_info(
            configs,
            restart_fermi_level=initial_fermi_level == "from_data",
            restart_multipoles=initial_density == "from_data",
        )
    else:
        check_direct_mode_info(configs)

    dataset = build_fixed_point_dataset(configs, model)
    data_loader = build_fixed_point_dataloader(dataset, args.batch_size)

    energies_list = []
    contributions_list = []
    forces_collection = []
    density_coefficients_collection = []
    electrostatic_features_collection = []
    dipoles_list = []
    full_charge_histories_list = []
    electrostatic_energies_list = []
    electron_energies_list = []
    fermi_levels_list = []
    esps_collection = []
    num_scf_steps_list = []

    mandatory_keys = COMMON_MANDATORY_OUTPUTS
    write_scf_history = args.scf_history if args.mode == "scf" else "none"

    for batch in data_loader:
        batch = batch.to(device)
        if args.mode == "scf":
            output = run_scf_batch(
                model=model,
                batch=batch,
                scf_options=scf_options,
                compute_force=not args.dont_compute_force,
                restart_multipoles=initial_density == "from_data",
                restart_fermi_level=initial_fermi_level == "from_data",
            )
        elif args.mode == "direct":
            output = run_direct_fixed_point_batch(
                model=model,
                batch=batch,
                compute_force=not args.dont_compute_force,
            )
        else:
            raise ValueError(f"Unknown eval mode: {args.mode}")
        validate_fixed_point_output(output, mandatory_keys, requested_output_keys)

        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))
        if "forces" in requested_output_keys:
            forces_collection.append(split_output_tensor(batch, output, "forces"))
        full_charge_histories_list += split_output_tensor(batch, output, "charges_history")
        density_coefficients_collection.append(
            split_output_tensor(batch, output, "density_coefficients")
        )
        electrostatic_features_collection.append(
            split_output_tensor(batch, output, "electrostatic_features")
        )
        dipoles_list.append(torch_tools.to_numpy(output["dipole"]))
        electrostatic_energies_list.append(
            torch_tools.to_numpy(output["electrostatic_energy"])
        )
        electron_energies_list.append(torch_tools.to_numpy(output["electron_energy"]))
        fermi_levels_list.append(torch_tools.to_numpy(output["fermi_level"]))
        if "esps" in requested_output_keys:
            esps_collection.append(split_output_tensor(batch, output, "esps"))
        num_scf_steps_list.extend(
            charge_history.shape[-1]
            for charge_history in full_charge_histories_list[-batch.num_graphs :]
        )

    energies = np.concatenate(energies_list, axis=0)
    dipoles = np.concatenate(dipoles_list, axis=0)
    electrostatic_energies = np.concatenate(electrostatic_energies_list, axis=0)
    electron_energies = np.concatenate(electron_energies_list, axis=0)
    fermi_levels = np.concatenate(fermi_levels_list, axis=0)

    if "forces" in requested_output_keys:
        forces_list = [forces for forces_list in forces_collection for forces in forces_list]
    density_coefficients_list = [
        density_coefficients
        for density_coefficients_list in density_coefficients_collection
        for density_coefficients in density_coefficients_list
    ]
    electrostatic_features_list = [
        electrostatic_features
        for electrostatic_features_list in electrostatic_features_collection
        for electrostatic_features in electrostatic_features_list
    ]
    if "esps" in requested_output_keys:
        esps_list = [esps for esps_list in esps_collection for esps in esps_list]

    assert len(atoms_list) == len(energies) == len(density_coefficients_list) == dipoles.shape[0]
    assert len(atoms_list) == electrostatic_energies.shape[0] == electron_energies.shape[0] == fermi_levels.shape[0]
    assert len(atoms_list) == len(num_scf_steps_list)

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    if write_scf_history != "none":
        assert len(atoms_list) == len(full_charge_histories_list)
        scf_convergence_relative = [
            [
                np.average(np.abs(single_config_data[..., i] - single_config_data[..., -1]))
                for i in range(single_config_data.shape[-1])
            ]
            for single_config_data in full_charge_histories_list
        ]

        if write_scf_history in {"absolute_change", "full_history"}:
            scf_convergence_approach = []
            for single_ats_charge_history in full_charge_histories_list:
                abs_average_charge_change = [
                    np.average(np.abs(single_ats_charge_history[..., 0]))
                ]
                for i in range(single_ats_charge_history.shape[-1] - 1):
                    abs_average_charge_change.append(
                        np.average(
                            np.abs(
                                single_ats_charge_history[..., i]
                                - single_ats_charge_history[..., i + 1]
                            )
                        )
                    )
                scf_convergence_approach.append(abs_average_charge_change)
        if write_scf_history == "full_history":
            reshaped_charges = [
                np.asarray(
                    [
                        np.reshape(single_atoms_charges, (-1), order="F")
                        for single_atoms_charges in config_charges
                    ]
                )
                for config_charges in full_charge_histories_list
            ]

    for i, (atoms, energy, density_coefficients, dipole) in enumerate(
        zip(atoms_list, energies, density_coefficients_list, dipoles)
    ):
        atoms.calc = None
        atoms.info[args.info_prefix + "energy"] = energy
        if "forces" in requested_output_keys:
            atoms.arrays[args.info_prefix + "forces"] = forces_list[i]
        atoms.arrays[args.info_prefix + "density_coefficients"] = density_coefficients
        atoms.info[args.info_prefix + "dipole"] = dipole

        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

        if write_scf_history == "absolute_change":
            atoms.info["scf_convergence"] = np.asarray(scf_convergence_approach[i])
            atoms.info["scf_convergence_relative"] = np.asarray(scf_convergence_relative[i])
        elif write_scf_history == "full_history":
            atoms.arrays["scf_charge_history"] = reshaped_charges[i]
            atoms.info["scf_convergence"] = np.asarray(scf_convergence_approach[i])
            atoms.info["scf_convergence_relative"] = np.asarray(scf_convergence_relative[i])

        atoms.info[args.info_prefix + "electrostatic_energy"] = electrostatic_energies[i]
        atoms.info[args.info_prefix + "electron_energy"] = electron_energies[i]
        atoms.info[args.info_prefix + "fermi_level"] = fermi_levels[i]
        atoms.info[args.info_prefix + "num_scf_steps"] = num_scf_steps_list[i]

        if "esps" in requested_output_keys:
            atoms.arrays[args.info_prefix + "esps"] = np.asarray(esps_list[i])

        atoms.arrays[args.info_prefix + "electrostatic_features"] = np.asarray(
            electrostatic_features_list[i]
        )

    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
