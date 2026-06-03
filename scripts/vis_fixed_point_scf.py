import argparse

import ase.io
import numpy as np
from mace.tools import torch_tools

from fixed_point_eval_utils import (
    COMMON_MANDATORY_OUTPUTS,
    build_fixed_point_dataloader,
    build_fixed_point_dataset,
    build_fixed_point_keyspec,
    check_restart_info,
    load_atoms_and_configs,
    load_fixed_point_model,
    log_dataset_contents,
    normalize_scf_options,
    run_scf_batch,
    validate_fixed_point_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        ],
        default="mixed_periodic",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--info_prefix",
        type=str,
        default="MACE_",
        help="prefix for stored output keys",
    )
    parser.add_argument(
        "--external_field_key",
        type=str,
        default="external_field",
    )
    parser.add_argument(
        "--fermi_level_key",
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
        "--scf_restart_multipoles",
        action="store_true",
        help="use density coefficients from the input file as the initial SCF guess",
    )
    parser.add_argument(
        "--scf_restart_fermi_level",
        action="store_true",
        help="use fermi levels from the input file as the initial SCF guess",
    )
    parser.add_argument(
        "--include_final_frame",
        action="store_true",
        default=False,
        help="also write a final frame using the converged density_coefficients output",
    )
    return parser.parse_args()


def _get_partial_dipoles(density_coefficients: np.ndarray) -> np.ndarray:
    if density_coefficients.shape[1] > 1:
        return density_coefficients[:, [3, 1, 2]]
    return np.zeros((density_coefficients.shape[0], 3), dtype=density_coefficients.dtype)


def _step_abs_change(charge_history: np.ndarray, step_index: int) -> float:
    if step_index == 0:
        return float(np.average(np.abs(charge_history[..., 0])))
    return float(
        np.average(
            np.abs(charge_history[..., step_index] - charge_history[..., step_index - 1])
        )
    )


def _step_abs_change_to_final(charge_history: np.ndarray, step_index: int) -> float:
    return float(
        np.average(np.abs(charge_history[..., step_index] - charge_history[..., -1]))
    )


def _make_scf_frames(
    atoms,
    charge_history: np.ndarray,
    final_density_coefficients: np.ndarray,
    dipole: np.ndarray,
    fermi_level: float,
    electrostatic_energy: float,
    electron_energy: float,
    energy: float,
    info_prefix: str,
    include_final_frame: bool,
    config_index: int,
):
    frames = []
    num_scf_steps = charge_history.shape[-1]
    for step_index in range(num_scf_steps):
        frame = atoms.copy()
        step_density = charge_history[..., step_index]
        frame.calc = None
        frame.info[info_prefix + "config_index"] = config_index
        frame.info[info_prefix + "scf_step"] = step_index
        frame.info[info_prefix + "num_scf_steps"] = num_scf_steps
        frame.info[info_prefix + "energy"] = energy
        frame.info[info_prefix + "fermi_level"] = fermi_level
        frame.info[info_prefix + "electrostatic_energy"] = electrostatic_energy
        frame.info[info_prefix + "electron_energy"] = electron_energy
        frame.info[info_prefix + "dipole"] = dipole
        frame.info[info_prefix + "scf_abs_change"] = _step_abs_change(
            charge_history, step_index
        )
        frame.info[info_prefix + "scf_abs_change_to_final"] = _step_abs_change_to_final(
            charge_history, step_index
        )
        frame.arrays[info_prefix + "density_coefficients"] = step_density
        frame.arrays[info_prefix + "partial_charges"] = step_density[:, 0]
        frame.arrays[info_prefix + "partial_dipoles"] = _get_partial_dipoles(step_density)
        frames.append(frame)

    if include_final_frame:
        frame = atoms.copy()
        frame.calc = None
        frame.info[info_prefix + "config_index"] = config_index
        frame.info[info_prefix + "scf_step"] = num_scf_steps
        frame.info[info_prefix + "num_scf_steps"] = num_scf_steps
        frame.info[info_prefix + "energy"] = energy
        frame.info[info_prefix + "fermi_level"] = fermi_level
        frame.info[info_prefix + "electrostatic_energy"] = electrostatic_energy
        frame.info[info_prefix + "electron_energy"] = electron_energy
        frame.info[info_prefix + "dipole"] = dipole
        frame.info[info_prefix + "scf_abs_change"] = 0.0
        frame.info[info_prefix + "scf_abs_change_to_final"] = 0.0
        frame.info[info_prefix + "is_final_output"] = 1
        frame.arrays[info_prefix + "density_coefficients"] = final_density_coefficients
        frame.arrays[info_prefix + "partial_charges"] = final_density_coefficients[:, 0]
        frame.arrays[info_prefix + "partial_dipoles"] = _get_partial_dipoles(
            final_density_coefficients
        )
        frames.append(frame)

    return frames


def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)
    print(device)

    scf_options = normalize_scf_options(args.scf_options)
    model = load_fixed_point_model(args.model, args.device)
    model.coulomb_energy.set_pbc_handling(args.pbc_handling)
    model.electric_potential_descriptor.set_pbc_handling(args.pbc_handling)

    keyspec = build_fixed_point_keyspec(
        external_field_key=args.external_field_key,
        fermi_level_key=args.fermi_level_key,
        atomic_multipoles_key=args.atomic_multipoles_key,
        total_charge_key=args.total_charge_key,
    )
    atoms_list, configs = load_atoms_and_configs(args.configs, keyspec)
    log_dataset_contents(dataset=configs, dataset_name="Visualization dataset")
    check_restart_info(
        configs, args.scf_restart_fermi_level, args.scf_restart_multipoles
    )

    dataset = build_fixed_point_dataset(configs, model)
    data_loader = build_fixed_point_dataloader(dataset, args.batch_size)

    visualisation_frames = []
    config_offset = 0
    mandatory_keys = COMMON_MANDATORY_OUTPUTS

    for batch in data_loader:
        batch = batch.to(device)
        output = run_scf_batch(
            model=model,
            batch=batch,
            scf_options=scf_options,
            compute_force=False,
            restart_multipoles=args.scf_restart_multipoles,
            restart_fermi_level=args.scf_restart_fermi_level,
        )
        validate_fixed_point_output(output, mandatory_keys, set())

        split_indices = torch_tools.to_numpy(batch.ptr[1:])
        energies = torch_tools.to_numpy(output["energy"])
        density_coefficients = np.split(
            torch_tools.to_numpy(output["density_coefficients"]),
            indices_or_sections=split_indices,
            axis=0,
        )[:-1]
        charge_histories = np.split(
            torch_tools.to_numpy(output["charges_history"]),
            indices_or_sections=split_indices,
            axis=0,
        )[:-1]
        dipoles = torch_tools.to_numpy(output["dipole"])
        fermi_levels = torch_tools.to_numpy(output["fermi_level"])
        electrostatic_energies = torch_tools.to_numpy(output["electrostatic_energy"])
        electron_energies = torch_tools.to_numpy(output["electron_energy"])

        batch_atoms = atoms_list[config_offset : config_offset + len(density_coefficients)]
        for local_index, atoms in enumerate(batch_atoms):
            visualisation_frames.extend(
                _make_scf_frames(
                    atoms=atoms,
                    charge_history=charge_histories[local_index],
                    final_density_coefficients=density_coefficients[local_index],
                    dipole=dipoles[local_index],
                    fermi_level=float(fermi_levels[local_index]),
                    electrostatic_energy=float(electrostatic_energies[local_index]),
                    electron_energy=float(electron_energies[local_index]),
                    energy=float(energies[local_index]),
                    info_prefix=args.info_prefix,
                    include_final_frame=args.include_final_frame,
                    config_index=config_offset + local_index,
                )
            )
        config_offset += len(density_coefficients)

    ase.io.write(args.output, images=visualisation_frames, format="extxyz")


if __name__ == "__main__":
    main()
