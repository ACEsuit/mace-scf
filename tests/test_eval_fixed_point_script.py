import os
import subprocess
import sys

import numpy as np
import pytest
import yaml
from ase.io import read, write

from tests.paths import REPO_ROOT, reference_model, require_file, script_env


SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_fixed_point.py"
DEVICE = os.environ.get("MACE_DEVICE", "cpu")
DEFAULT_RTOL = float(os.environ.get("EXPECTED_RTOL", "1e-7"))
DEFAULT_ATOL = float(os.environ.get("EXPECTED_ATOL", "1e-6"))

SCF_CASES = [
    "clusters_force_pbc",
    "constant_charge_baseline",
    "constant_fermi_baseline",
    "constant_q_bulk",
    "constant_q_clusters",
    "constant_q_slab",
    "custom_keys",
    "full_scf_history",
    "ignore_nonconverged",
    "scf_max_steps",
    "scf_tolerance",
]

OUTPUT_KEYS = (
    "energy",
    "forces",
    "density_coefficients",
    "dipole",
    "fermi_level",
    "electrostatic_energy",
    "electron_energy",
    "num_scf_steps",
)


def _case_dir(case_name):
    return REPO_ROOT / "tests" / "fixed_point_cases" / "materialized" / case_name


def _load_case(case_name):
    case_dir = _case_dir(case_name)
    with (case_dir / "case.yaml").open("r", encoding="utf-8") as handle:
        return case_dir, yaml.safe_load(handle)


def _model_path(case_config):
    model_name = case_config["model"]["name"]
    return reference_model(model_name)


def _run_scf_eval(case_name, input_path, pbc_handling, tmp_path, batch_size=1, suffix=""):
    case_dir, case_config = _load_case(case_name)
    model_path = _model_path(case_config)
    require_file(model_path, "Model")
    require_file(input_path, "Input configs")

    keys = case_config.get("keys", {})
    scf_options = dict(case_config.get("scf_options", {}))
    output_path = tmp_path / f"{case_name}{suffix}.xyz"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--mode",
        "scf",
        "--configs",
        str(input_path),
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--device",
        DEVICE,
        "--batch_size",
        str(batch_size),
        "--pbc_handling",
        pbc_handling,
        "--atomic_multipoles_key",
        keys.get("atomic_multipoles_key", "initial_density_coefficients"),
        "--fermi_level_key",
        keys.get("fermi_level_key", "initial_fermi_level"),
        "--external_field_key",
        keys.get("external_field_key", "external_field"),
        "--total_charge_key",
        keys.get("total_charge_key", "total_charge"),
        "--scf_options",
        repr(scf_options),
        "--scf_history",
        "none",
    ]
    if scf_options.get("initial_fermi_level") == "from_data":
        cmd.extend(["--initial_fermi_level", "from_data"])
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=script_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"eval_fixed_point.py failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return read(output_path, index=":")


def _legacy_fixed_point_pbc_handling(atoms, case_config):
    calculator_options = case_config.get("calculator_options", {})
    explicit_method = calculator_options.get("pbc_handling")
    if explicit_method is not None:
        return explicit_method

    force_periodic = calculator_options.get("use_pbc_evaluator", False)
    pbc = tuple(bool(x) for x in atoms.pbc)

    if force_periodic:
        if pbc == (False, False, False):
            return "molecule_in_box"
        if pbc == (True, True, True):
            return "pbc"
        if pbc == (True, True, False):
            return "slab"
        raise ValueError(f"Unsupported forced-periodic pbc pattern: {pbc}")

    if pbc == (False, False, False):
        return "realspace"
    if pbc == (True, True, True):
        return "pbc"
    if pbc == (True, True, False):
        return "slab"
    raise ValueError(f"Unsupported legacy pbc pattern: {pbc}")


def _group_indices_by_legacy_pbc(expected_atoms, case_config):
    groups = {}
    for index, atoms in enumerate(expected_atoms):
        pbc_handling = _legacy_fixed_point_pbc_handling(atoms, case_config)
        groups.setdefault(pbc_handling, []).append(index)
    return groups


def _expected_value(atoms, key):
    info_key = f"expected_{key}"
    if info_key in atoms.info:
        shape_key = f"{info_key}_shape"
        value = atoms.info[info_key]
        if shape_key in atoms.info:
            trailing_shape = tuple(int(x) for x in atoms.info[shape_key])
            return np.asarray(value).reshape((len(atoms), *trailing_shape))
        return value
    return atoms.arrays[info_key]


def _actual_value(atoms, key):
    output_key = f"MACE_{key}"
    if output_key in atoms.info:
        return atoms.info[output_key]
    return atoms.arrays[output_key]


def _assert_allclose(name, expected, actual):
    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)
    if (
        name == "density_coefficients"
        and expected_arr.ndim == 1
        and actual_arr.ndim == 2
        and actual_arr.shape[1] == 1
        and expected_arr.shape[0] == actual_arr.shape[0]
    ):
        expected_arr = expected_arr.reshape(-1, 1)
    assert expected_arr.shape == actual_arr.shape, (
        f"{name} shape mismatch: expected {expected_arr.shape}, got {actual_arr.shape}"
    )
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        err_msg=f"{name} mismatch",
    )


@pytest.mark.parametrize("case_name", SCF_CASES)
def test_eval_fixed_point_scf_matches_materialized_references(
    case_name,
    tmp_path,
):
    case_dir, case_config = _load_case(case_name)
    input_path = case_dir / "input.xyz"
    if not input_path.exists():
        pytest.skip(f"Input configs not found: {input_path}")
    input_atoms = read(input_path, index=":")
    expected_atoms = read(case_dir / "expected.xyz", index=":")
    assert len(input_atoms) == len(expected_atoms)

    for pbc_handling, indices in _group_indices_by_legacy_pbc(
        expected_atoms,
        case_config,
    ).items():
        grouped_input_path = tmp_path / f"{case_name}_{pbc_handling}_input.xyz"
        grouped_input_atoms = [input_atoms[index] for index in indices]
        grouped_expected_atoms = [expected_atoms[index] for index in indices]
        write(grouped_input_path, images=grouped_input_atoms, format="extxyz")
        output_atoms = _run_scf_eval(
            case_name,
            grouped_input_path,
            pbc_handling,
            tmp_path,
            suffix=f"_{pbc_handling}",
        )

        assert len(grouped_expected_atoms) == len(output_atoms)
        for expected, output in zip(grouped_expected_atoms, output_atoms):
            for key in OUTPUT_KEYS:
                _assert_allclose(
                    key,
                    _expected_value(expected, key),
                    _actual_value(output, key),
                )
