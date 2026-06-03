import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read, write
from ase.stress import full_3x3_to_voigt_6_stress

from tests.paths import REPO_ROOT, reference_model, reference_output, require_file, script_env

SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_local_charges.py"
DEVICE = os.environ.get("MACE_DEVICE", "cpu")
DEFAULT_RTOL = float(os.environ.get("EXPECTED_RTOL", "1e-6"))
DEFAULT_ATOL = float(os.environ.get("EXPECTED_ATOL", "1e-6"))

FIXED_PBC_MODES = (
    "realspace",
    "pbc",
    "slab",
    "molecule_in_box",
    "mixed_periodic",
)

EVAL_EXPECTED_CASES = [
    *[
        {
            "model": model_name,
            "expected": f"{model_name}_{pbc_handling}",
            "pbc_handling": pbc_handling,
        }
        for model_name in ("splitcharge_l0_stage1", "splitcharge_l1_stage1")
        for pbc_handling in FIXED_PBC_MODES
    ],
    *[
        {
            "model": model_name,
            "expected": f"{model_name}_{pbc_handling}",
            "pbc_handling": pbc_handling,
        }
        for model_name in ("nonpol_l0_stage1", "nonpol_l1_stage1")
        for pbc_handling in ("realspace", "pbc")
    ],
]

EVAL_AUTO_CASE = {
    "model": "splitcharge_l1_stage1",
    "expected": "splitcharge_l1_stage1_eval_auto",
    "pbc_handling": "auto",
}

BATCH_INVARIANCE_CASE = {
    "model": "splitcharge_l1_stage1",
    "expected": "splitcharge_l1_stage1_mixed_periodic",
    "pbc_handling": "mixed_periodic",
}


def _expected_path(case):
    return reference_output(case["expected"])


def _model_path(case):
    return reference_model(case["model"])


def _require_case_files(case):
    expected_path = _expected_path(case)
    model_path = _model_path(case)
    require_file(expected_path, "Expected configs")
    require_file(model_path, "Model")
    return expected_path


def _run_eval(case, tmp_path, batch_size, configs_path=None, output_stem=None):
    if configs_path is None:
        configs_path = _require_case_files(case)
    else:
        configs_path = Path(configs_path)
        model_path = _model_path(case)
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

    output_path = tmp_path / f"{output_stem or case['expected']}_batch{batch_size}.xyz"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--configs",
        str(configs_path),
        "--model",
        str(_model_path(case)),
        "--output",
        str(output_path),
        "--device",
        DEVICE,
        "--batch_size",
        str(batch_size),
        "--pbc_handling",
        case["pbc_handling"],
        "--external_field_key",
        "external_field",
        "--fermi_level_key",
        "the_VBM",
        "--formal_charges_key",
        "formal_oxidation_states",
        "--compute_stress",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=script_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"eval_local_charges.py failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return read(output_path, index=":")


def _as_voigt_stress(value):
    arr = np.asarray(value)
    if arr.shape == (3, 3):
        return full_3x3_to_voigt_6_stress(arr)
    return arr


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


def _assert_matches_expected(expected_atoms, output_atoms):
    assert len(expected_atoms) == len(output_atoms)
    for expected, output in zip(expected_atoms, output_atoms):
        _assert_allclose(
            "energy",
            expected.info["expected_energy"],
            output.info["MACE_energy"],
        )
        assert "MACE_dipole" in output.info
        if "expected_dipole" in expected.info:
            _assert_allclose(
                "dipole",
                expected.info["expected_dipole"],
                output.info["MACE_dipole"],
            )
        _assert_allclose(
            "forces",
            expected.arrays["expected_forces"],
            output.arrays["MACE_forces"],
        )
        _assert_allclose(
            "density_coefficients",
            expected.arrays["expected_density_coefficients"],
            output.arrays["MACE_density_coefficients"],
        )
        assert "MACE_stress" in output.info
        if "expected_stress" in expected.info:
            _assert_allclose(
                "stress",
                _as_voigt_stress(expected.info["expected_stress"]),
                _as_voigt_stress(output.info["MACE_stress"]),
            )
        if "expected_polarizability" in expected.info:
            _assert_allclose(
                "polarizability",
                expected.info["expected_polarizability"],
                output.info.get("MACE_polarizability", np.zeros((3, 3))),
            )


def _assert_outputs_match(atoms_a, atoms_b):
    assert len(atoms_a) == len(atoms_b)
    for first, second in zip(atoms_a, atoms_b):
        _assert_allclose("energy", first.info["MACE_energy"], second.info["MACE_energy"])
        _assert_allclose("dipole", first.info["MACE_dipole"], second.info["MACE_dipole"])
        _assert_allclose(
            "forces",
            first.arrays["MACE_forces"],
            second.arrays["MACE_forces"],
        )
        _assert_allclose(
            "stress",
            _as_voigt_stress(first.info["MACE_stress"]),
            _as_voigt_stress(second.info["MACE_stress"]),
        )
        _assert_allclose(
            "density_coefficients",
            first.arrays["MACE_density_coefficients"],
            second.arrays["MACE_density_coefficients"],
        )


@pytest.mark.parametrize("case", EVAL_EXPECTED_CASES, ids=lambda case: case["expected"])
def test_eval_local_charges_expected_outputs(case, tmp_path):
    expected_atoms = read(_require_case_files(case), index=":")
    output_atoms = _run_eval(case, tmp_path, batch_size=8)
    _assert_matches_expected(expected_atoms, output_atoms)


def test_eval_local_charges_auto_expected_outputs(tmp_path):
    expected_atoms = read(_require_case_files(EVAL_AUTO_CASE), index=":")
    output_atoms = _run_eval(EVAL_AUTO_CASE, tmp_path, batch_size=8)
    _assert_matches_expected(expected_atoms, output_atoms)


def test_eval_local_charges_auto_dispatch_matches_explicit_modes(tmp_path):
    source = read(reference_output("splitcharge_l1_stage1_mixed_periodic"), index=":")
    fff_path = tmp_path / "fff.xyz"
    periodic_path = tmp_path / "periodic.xyz"
    write(fff_path, [next(atoms for atoms in source if not any(atoms.pbc))])
    write(periodic_path, [next(atoms for atoms in source if any(atoms.pbc))])

    auto_case = {
        "model": "splitcharge_l1_stage1",
        "expected": "auto_tmp",
        "pbc_handling": "auto",
    }
    realspace_case = dict(auto_case, pbc_handling="realspace")
    mixed_case = dict(auto_case, pbc_handling="mixed_periodic")

    _assert_outputs_match(
        _run_eval(auto_case, tmp_path, batch_size=8, configs_path=fff_path, output_stem="auto_fff"),
        _run_eval(realspace_case, tmp_path, batch_size=8, configs_path=fff_path, output_stem="realspace_fff"),
    )
    _assert_outputs_match(
        _run_eval(auto_case, tmp_path, batch_size=8, configs_path=periodic_path, output_stem="auto_periodic"),
        _run_eval(mixed_case, tmp_path, batch_size=8, configs_path=periodic_path, output_stem="mixed_periodic"),
    )


def test_eval_local_charges_batch_size_invariance(tmp_path):
    output_batch_1 = _run_eval(BATCH_INVARIANCE_CASE, tmp_path, batch_size=1)
    output_batch_2 = _run_eval(BATCH_INVARIANCE_CASE, tmp_path, batch_size=2)
    _assert_outputs_match(output_batch_1, output_batch_2)
