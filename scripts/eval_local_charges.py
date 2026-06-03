import argparse

import ase.io
import numpy as np
import torch

import mace_scf.data
import mace.data
from mace.tools import torch_geometric, torch_tools, utils


MANDATORY_ALWAYS = (
    "energy",
    "contributions",
    "forces",
    "density_coefficients",
    "dipole",
)
OPTIONAL_MODEL_OUTPUT_KEYS = (
    "fermi_level",
    "polarizability",
)
PBC_HANDLING_CHOICES = (
    "realspace",
    "pbc",
    "slab",
    "molecule_in_box",
    "mixed_periodic",
    "auto",
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
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
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
        "--pbc_handling",
        type=str,
        choices=PBC_HANDLING_CHOICES,
        default="mixed_periodic",
    )
    parser.add_argument(
        "--external_field_key",
        help="key for external field",
        type=str,
        default="external_field",
    )
    parser.add_argument(
        "--fermi_level_key",
        help="key for fermi level",
        type=str,
        default="fermi_level",
    )
    parser.add_argument(
        "--formal_charges_key",
        help="array key for per-atom formal charges, used by local symmetric charge models",
        type=str,
        default="formal_oxidation_states",
    )
    return parser.parse_args()


def _require_tensor_output(output, key: str) -> None:
    if key not in output:
        raise KeyError(f"Missing mandatory model output: {key}")
    if output[key] is None:
        raise ValueError(f"Mandatory model output is None: {key}")


def _get_optional_model_outputs(output) -> set:
    return {
        key
        for key in OPTIONAL_MODEL_OUTPUT_KEYS
        if key in output and output[key] is not None
    }


def _model_uses_formal_charges(model) -> bool:
    return hasattr(model, "formal_charges")


def main():
    args = parse_args()
    torch_tools.set_default_dtype("float64")
    device = torch_tools.init_device(args.device)

    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(args.device)
    model.coulomb_energy.set_pbc_handling(args.pbc_handling)

    for param in model.parameters():
        param.requires_grad = False

    if not hasattr(model, "heads"):
        raise ValueError(
            "scripts/eval_configs.py requires a model with `heads`."
        )
    if not hasattr(model, "coulomb_energy") or not hasattr(
        model.coulomb_energy, "density_max_l"
    ):
        raise ValueError(
            "scripts/eval_configs.py requires a local electrostatics model "
            "with `coulomb_energy.density_max_l`."
        )

    arrays_keys = {}
    if _model_uses_formal_charges(model):
        arrays_keys["charges"] = args.formal_charges_key

    keyspec = mace.data.KeySpecification(
        info_keys={
            "external_field": args.external_field_key,
            "fermi_level": args.fermi_level_key,
            "total_charge": "total_charge",
        },
        arrays_keys=arrays_keys,
    )

    atoms_list = ase.io.read(args.configs, index=":")
    configs = [
        mace.data.config_from_atoms(atoms, key_specification=keyspec)
        for atoms in atoms_list
    ]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atomic_multipoles_max_l = int(model.coulomb_energy.density_max_l)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            mace_scf.data.ExtAtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=float(model.r_max),
                atomic_multipoles_max_l=atomic_multipoles_max_l,
                heads=model.heads,
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    energies_list = []
    contributions_list = []
    stresses_list = []
    forces_collection = []
    density_coefficients_collection = []
    dipoles_list = []
    fermi_levels_list = []
    polarizabilities_list = []
    optional_model_outputs = None

    for batch in data_loader:
        batch = batch.to(device)
        output = model(
            batch.to_dict(),
            compute_stress=args.compute_stress,
            compute_force=True,
        )

        for key in MANDATORY_ALWAYS:
            _require_tensor_output(output, key)
        if args.compute_stress:
            _require_tensor_output(output, "stress")

        current_optional_model_outputs = _get_optional_model_outputs(output)
        if (
            optional_model_outputs is not None
            and current_optional_model_outputs != optional_model_outputs
        ):
            raise ValueError(
                "Inconsistent optional model outputs across batches: "
                f"expected {sorted(optional_model_outputs)}, "
                f"got {sorted(current_optional_model_outputs)}"
            )
        optional_model_outputs = current_optional_model_outputs

        split_indices = torch_tools.to_numpy(batch.ptr[1:])
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if args.compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))
        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=split_indices,
            axis=0,
        )
        forces_collection.append(forces[:-1])

        density_coefficients = np.split(
            torch_tools.to_numpy(output["density_coefficients"]),
            indices_or_sections=split_indices,
            axis=0,
        )
        density_coefficients_collection.append(density_coefficients[:-1])
        dipoles_list.append(torch_tools.to_numpy(output["dipole"]))

        if "fermi_level" in current_optional_model_outputs:
            fermi_levels_list.append(torch_tools.to_numpy(output["fermi_level"]))
        if "polarizability" in current_optional_model_outputs:
            polarizabilities_list.append(torch_tools.to_numpy(output["polarizability"]))

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    density_coefficients_list = [
        density_coefficients
        for density_coefficients_list in density_coefficients_collection
        for density_coefficients in density_coefficients_list
    ]
    dipoles = np.concatenate(dipoles_list, axis=0)

    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    if "fermi_level" in optional_model_outputs:
        fermi_levels = np.concatenate(fermi_levels_list, axis=0)
    if "polarizability" in optional_model_outputs:
        polarizabilities = np.concatenate(polarizabilities_list, axis=0)

    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None
        atoms.info[args.info_prefix + "energy"] = energy
        atoms.arrays[args.info_prefix + "forces"] = forces
        atoms.arrays[args.info_prefix + "density_coefficients"] = density_coefficients_list[i]
        atoms.info[args.info_prefix + "dipole"] = dipoles[i]

        if args.compute_stress:
            atoms.info[args.info_prefix + "stress"] = stresses[i]

        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

        if "fermi_level" in optional_model_outputs:
            atoms.info[args.info_prefix + "fermi_level"] = fermi_levels[i]
        if "polarizability" in optional_model_outputs:
            atoms.info[args.info_prefix + "polarizability"] = polarizabilities[i]

    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
