import argparse
from pathlib import Path

import numpy as np
from ase.io import write

try:
    from .case_utils import (
        build_fixed_point_calculator,
        case_usage,
        discover_case_dirs,
        expected_output_keys,
        load_case_config,
        load_input_atoms,
        materialized_cases_dir,
        resolve_model_path,
    )
except ImportError:
    from tests.fixed_point_cases.case_utils import (
        build_fixed_point_calculator,
        case_usage,
        discover_case_dirs,
        expected_output_keys,
        load_case_config,
        load_input_atoms,
        materialized_cases_dir,
        resolve_model_path,
    )

RESTART_PERTURBATION = 0.05


def _clear_expected_keys(atoms):
    for key in list(atoms.info.keys()):
        if key.startswith("expected_"):
            del atoms.info[key]
    for key in list(atoms.arrays.keys()):
        if key.startswith("expected_"):
            del atoms.arrays[key]


def _store_expected(atoms, key, value):
    if value is None:
        raise ValueError(f"Requested expected output {key} is None")

    expected_key = f"expected_{key}"
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[0] == len(atoms):
            if value.ndim <= 2:
                atoms.arrays[expected_key] = value
                return
            atoms.arrays[expected_key] = value.reshape(len(atoms), -1)
            atoms.info[f"{expected_key}_shape"] = list(value.shape[1:])
            return
        atoms.info[expected_key] = value.tolist() if value.ndim > 0 else float(value)
        return
    if np.isscalar(value):
        atoms.info[expected_key] = float(value)
        return
    atoms.info[expected_key] = value


def _label_atoms(atoms, calc, output_keys):
    labeled = atoms.copy()
    _clear_expected_keys(labeled)
    labeled.calc = calc
    labeled.calc.calculate(labeled)

    for key in output_keys:
        if key not in labeled.calc.results:
            raise KeyError(f"Missing calculator result key: {key}")
        _store_expected(labeled, key, labeled.calc.results[key])
    labeled.calc = None
    return labeled


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


def _label_expected_case(case_dir: Path, case_config, model_dir: Path, device: str):
    calc = build_fixed_point_calculator(case_config, model_dir=model_dir, device=device)
    inputs = load_input_atoms(case_dir)
    output_keys = expected_output_keys(case_config)
    return [_label_atoms(atoms, calc, output_keys) for atoms in inputs]


def _label_restart_case(case_dir: Path, case_config, model_dir: Path, device: str):
    calc = build_fixed_point_calculator(case_config, model_dir=model_dir, device=device)
    inputs = load_input_atoms(case_dir)
    output_keys = expected_output_keys(case_config)
    if len(inputs) != 1:
        raise ValueError(
            f"Restart case {case_dir.name} must materialize exactly one input frame."
        )

    first_atoms = _label_atoms(inputs[0], calc, output_keys)
    second_atoms = _perturb_atoms(first_atoms, RESTART_PERTURBATION)
    second_atoms = _label_atoms(second_atoms, calc, output_keys)
    return [first_atoms, second_atoms]


def _label_case(case_dir: Path, model_dir: Path, device: str, overwrite: bool):
    case_config = load_case_config(case_dir)
    model_path = resolve_model_path(case_config, model_dir=model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for case {case_dir.name}: {model_path}")

    output_path = case_dir / "expected.xyz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Expected output already exists for case {case_dir.name}: {output_path}"
        )

    usage = case_usage(case_config)
    if usage == "single_point":
        labeled_configs = _label_expected_case(case_dir, case_config, model_dir, device)
    elif usage == "restart":
        labeled_configs = _label_restart_case(case_dir, case_config, model_dir, device)
    else:
        raise ValueError(f"Unknown usage for case {case_dir.name}: {usage}")

    write(output_path, labeled_configs)


def main():
    parser = argparse.ArgumentParser(
        description="Label materialized fixed-point calculator cases with expected outputs."
    )
    parser.add_argument(
        "--cases-dir",
        default=str(materialized_cases_dir()),
        help="Directory containing materialized fixed-point case folders.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional list of case names to label. Defaults to all discovered cases.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing .model files. Defaults to tests/fixtures/models.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing expected.xyz files.",
    )
    args = parser.parse_args()

    case_dirs = discover_case_dirs(cases_dir=Path(args.cases_dir), case_names=args.cases)
    model_dir = None if args.model_dir is None else Path(args.model_dir)

    for case_dir in case_dirs:
        _label_case(
            case_dir=case_dir,
            model_dir=model_dir,
            device=args.device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
