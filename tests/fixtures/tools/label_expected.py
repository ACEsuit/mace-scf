import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read, write

from mace_scf.calculators.localsources import (
    MACEFixedChargeBaseline,
    MACELocalSplitCharges,
    MACELocalCharges,
)
from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF

import torch
torch.set_default_dtype(torch.float64)


def _clear_expected_keys(atoms):
    for key in list(atoms.info.keys()):
        if key.startswith("expected_"):
            del atoms.info[key]
    for key in list(atoms.arrays.keys()):
        if key.startswith("expected_"):
            del atoms.arrays[key]


def _clear_prefix_keys(atoms, prefix):
    for key in list(atoms.info.keys()):
        if key.startswith(prefix):
            del atoms.info[key]
    for key in list(atoms.arrays.keys()):
        if key.startswith(prefix):
            del atoms.arrays[key]


def _store_expected(atoms, key, value):
    if value is None:
        return
    expected_key = f"expected_{key}"
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[0] == len(atoms):
            atoms.arrays[expected_key] = value
            return
        atoms.info[expected_key] = value.tolist() if value.ndim > 0 else float(value)
        return
    if np.isscalar(value):
        atoms.info[expected_key] = float(value)
        return
    atoms.info[expected_key] = value


def _build_fixed_point_calculator(args, scf_restart: bool):
    return MACEFixedPointSCF(
        model_path=args.model_path,
        device=args.device,
        scf_options={
            "mixing_parameter": args.mixing_parameter,
            "constant_charge": args.constant_charge,
            "num_scf_steps": args.max_num_scf_steps,
            "scf_tolerance": args.scf_tolerance,
        },
        use_pbc_evaluator=args.use_pbc_evaluator,
        atomic_multipoles_key=args.atomic_multipoles_key,
        fermi_level_key=args.fermi_level_key,
        external_field_key=args.external_field_key,
        total_charge_key=args.total_charge_key,
        scf_restart=scf_restart,
    )


def build_calculator(args):
    if args.calculator == "local_symmetric":
        return MACELocalSplitCharges(
            model_path=args.model_path,
            device=args.device,
            formal_charges_key=args.formal_charges_key,
            external_field_key=args.external_field_key,
            fermi_level_key=args.fermi_level_key,
            pbc_handling=args.pbc_handling,
        )
    if args.calculator in {"nonpolarizable", "local_charges"}:
        return MACELocalCharges(
            model_path=args.model_path,
            device=args.device,
            external_field_key=args.external_field_key,
            fermi_level_key=args.fermi_level_key,
            pbc_handling=args.pbc_handling,
        )
    if args.calculator == "fixed_charge_baseline":
        return MACEFixedChargeBaseline(
            model_path=args.model_path,
            device=args.device,
            formal_charges_key=args.formal_charges_key,
            external_field_key=args.external_field_key,
            fermi_level_key=args.fermi_level_key,
            pbc_handling=args.pbc_handling,
        )
    if args.calculator == "fixed_point":
        return _build_fixed_point_calculator(args, scf_restart=False)
    if args.calculator == "fixed_point_restart":
        return _build_fixed_point_calculator(args, scf_restart=True)
    raise ValueError(f"Unknown calculator: {args.calculator}")


def _label_atoms(atoms, calc):
    _clear_expected_keys(atoms)
    atoms.calc = calc
    atoms.calc.calculate(atoms)
    for key, value in atoms.calc.results.items():
        _store_expected(atoms, key, value)
    atoms.calc = None
    return atoms


def _store_eval_expected(atoms, prefix="MACE_"):
    labeled = atoms.copy()
    _clear_expected_keys(labeled)
    _clear_prefix_keys(labeled, prefix)

    source_keys = {}
    for key, value in atoms.info.items():
        if key.startswith(prefix):
            source_keys[key[len(prefix):]] = value
    for key, value in atoms.arrays.items():
        if key.startswith(prefix):
            source_keys[key[len(prefix):]] = value

    for key, value in source_keys.items():
        _store_expected(labeled, key, value)
    return labeled


def _label_eval_local_charges(args, configs):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "configs.xyz"
        output_path = tmpdir / "eval_output.xyz"
        write(input_path, configs)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "eval_local_charges.py"),
            "--configs",
            str(input_path),
            "--model",
            args.model_path,
            "--output",
            str(output_path),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--pbc_handling",
            args.pbc_handling,
            "--external_field_key",
            args.external_field_key,
            "--fermi_level_key",
            args.fermi_level_key,
            "--formal_charges_key",
            args.formal_charges_key,
            "--compute_stress",
        ]
        if args.return_contributions:
            cmd.append("--return_contributions")
        subprocess.run(cmd, check=True)
        evaluated = read(output_path, index=":")
        return [_store_eval_expected(atoms) for atoms in evaluated]


def _perturb_atoms(atoms, displacement_magnitude: float):
    perturbed = atoms.copy()
    indices = np.arange(len(perturbed), dtype=float)
    directions = np.stack(
        (
            np.sin(indices + 1.0),
            np.cos(indices + 2.0),
            np.sin(0.5 * indices + 3.0),
        ),
        axis=1,
    )
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms
    perturbed.positions = perturbed.positions + displacement_magnitude * directions
    return perturbed


def _label_fixed_point_restart(args, configs):
    if len(configs) != 1:
        raise ValueError(
            "fixed_point_restart labeling expects exactly one input configuration."
        )

    calc = build_calculator(args)
    first_atoms = _label_atoms(configs[0], calc)
    second_atoms = _perturb_atoms(first_atoms, args.restart_perturbation)
    second_atoms = _label_atoms(second_atoms, calc)
    return [first_atoms, second_atoms]


def main():
    parser = argparse.ArgumentParser(
        description="Label configs with expected_<property> values by running calculators."
    )
    parser.add_argument(
        "--calculator",
        required=True,
        choices=[
            "local_symmetric",
            "nonpolarizable",
            "fixed_charge_baseline",
            "local_charges",
            "eval_local_charges",
            "fixed_point",
            "fixed_point_restart",
        ],
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-pbc-evaluator", action="store_true")
    parser.add_argument(
        "--pbc-handling",
        default="mixed_periodic",
        choices=[
            "realspace",
            "pbc",
            "slab",
            "molecule_in_box",
            "mixed_periodic",
            "auto",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--return-contributions", action="store_true")

    parser.add_argument("--formal-charges-key", default="formal_oxidation_states")
    parser.add_argument("--atomic-multipoles-key", default="DMA_coeficients")
    parser.add_argument("--fermi-level-key", default="fermi_level")
    parser.add_argument("--external-field-key", default="external_field")
    parser.add_argument("--total-charge-key", default="total_charge")

    parser.add_argument("--constant-charge", action="store_true")
    parser.add_argument("--mixing-parameter", type=float, default=0.2)
    parser.add_argument("--max-num-scf-steps", type=int, default=100)
    parser.add_argument("--scf-tolerance", type=float, default=1e-5)
    parser.add_argument("--restart-perturbation", type=float, default=0.05)

    args = parser.parse_args()

    configs = read(args.configs, index=":")
    if args.calculator == "fixed_point_restart":
        labeled_configs = _label_fixed_point_restart(args, configs)
    elif args.calculator == "eval_local_charges":
        labeled_configs = _label_eval_local_charges(args, configs)
    else:
        calc = build_calculator(args)
        labeled_configs = [_label_atoms(atoms, calc) for atoms in configs]

    write(args.output, labeled_configs)


if __name__ == "__main__":
    main()
