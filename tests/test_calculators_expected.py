"""Regression tests for local-source ASE calculator outputs."""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from mace_scf.calculators.localsources import (
    MACELocalSplitCharges,
    MACELocalCharges,
)
from tests.paths import reference_model, reference_output, require_file


DEVICE = os.environ.get("MACE_DEVICE", "cpu")
LOCAL_FORMAL_CHARGES_KEY = os.environ.get(
    "LOCAL_FORMAL_CHARGES_KEY",
    "formal_oxidation_states",
)
LOCAL_EXTERNAL_FIELD_KEY = os.environ.get(
    "LOCAL_EXTERNAL_FIELD_KEY",
    "external_field",
)
LOCAL_FERMI_LEVEL_KEY = os.environ.get(
    "LOCAL_FERMI_LEVEL_KEY",
    "fermi_level",
)
DEFAULT_RTOL = float(os.environ.get("EXPECTED_RTOL", "1e-6"))
DEFAULT_ATOL = float(os.environ.get("EXPECTED_ATOL", "1e-6"))
MIN_CELL_NONPERIODIC_WARNING = "Warning: min cell dimension for pbc=False is 30A"

FIXED_PBC_MODES = (
    "realspace",
    "pbc",
    "slab",
    "molecule_in_box",
    "mixed_periodic",
)


CALCULATOR_EXPECTED_CASES = [
    *[
        {
            "name": model_name,
            "expected_name": f"{model_name}_{pbc_handling}",
            "calculator": "local_symmetric",
            "pbc_handling": pbc_handling,
        }
        for model_name in ("splitcharge_l0_stage1", "splitcharge_l1_stage1")
        for pbc_handling in FIXED_PBC_MODES
    ],
    *[
        {
            "name": model_name,
            "expected_name": f"{model_name}_{pbc_handling}",
            "calculator": "local_charges",
            "pbc_handling": pbc_handling,
        }
        for model_name in ("nonpol_l0_stage1", "nonpol_l1_stage1")
        for pbc_handling in ("realspace", "pbc")
    ],
]

CALCULATOR_AUTO_CASE = {
    "name": "splitcharge_l1_stage1",
    "expected_name": "splitcharge_l1_stage1_auto",
    "calculator": "local_symmetric",
    "pbc_handling": "auto",
}

LEGACY_CASES = [
    {
        "name": "splitcharge_l0_stage1",
        "expected_name": "splitcharge_l0_stage1",
        "calculator": "local_symmetric",
    },
    {
        "name": "splitcharge_l1_stage1",
        "expected_name": "splitcharge_l1_stage1",
        "calculator": "local_symmetric",
    },
    {
        "name": "nonpol_l0_stage1",
        "expected_name": "nonpol_l0_stage1",
        "calculator": "local_charges",
    },
    {
        "name": "nonpol_l1_stage1",
        "expected_name": "nonpol_l1_stage1",
        "calculator": "local_charges",
    },
]


def _expected_path(expected_name: str) -> str:
    return str(reference_output(expected_name))


def _model_path(model_name: str) -> str:
    return str(reference_model(model_name))


def _load_expected(atoms):
    expected = {}
    for key, value in atoms.info.items():
        if key.startswith("expected_"):
            expected[key[len("expected_") :]] = value
    for key, value in atoms.arrays.items():
        if key.startswith("expected_"):
            expected[key[len("expected_") :]] = value
    return expected


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


def _assert_results_close(expected_results, actual_results):
    for key, expected_value in expected_results.items():
        assert key in actual_results, f"Missing result key: {key}"
        _assert_allclose(key, expected_value, actual_results[key])


def _build_calculator(case_or_name, pbc_handling=None):
    if isinstance(case_or_name, dict):
        model_name = case_or_name["name"]
        calculator = case_or_name["calculator"]
        pbc_handling = case_or_name.get("pbc_handling", pbc_handling)
    else:
        model_name = case_or_name
        calculator = (
            "local_symmetric"
            if model_name.startswith("splitcharge")
            else "local_charges"
        )

    model_path = _model_path(model_name)
    if calculator == "local_symmetric":
        return MACELocalSplitCharges(
            model_path=model_path,
            device=DEVICE,
            formal_charges_key=LOCAL_FORMAL_CHARGES_KEY,
            external_field_key=LOCAL_EXTERNAL_FIELD_KEY,
            fermi_level_key=LOCAL_FERMI_LEVEL_KEY,
            pbc_handling=pbc_handling,
        )
    if calculator == "local_charges":
        return MACELocalCharges(
            model_path=model_path,
            device=DEVICE,
            external_field_key=LOCAL_EXTERNAL_FIELD_KEY,
            fermi_level_key=LOCAL_FERMI_LEVEL_KEY,
            pbc_handling=pbc_handling,
        )
    raise ValueError(f"Unknown calculator type: {calculator}")


def _calculate(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.calc.calculate(atoms)
    return deepcopy(atoms.calc.results)


def _calculate_expected_warning(atoms, calc):
    if any(atoms.pbc):
        return _calculate(atoms, calc)
    with pytest.warns(UserWarning, match=MIN_CELL_NONPERIODIC_WARNING):
        return _calculate(atoms, calc)


def _require_case_files(case):
    expected_path = _expected_path(case["expected_name"])
    model_path = _model_path(case["name"])
    require_file(Path(expected_path), "Expected configs")
    require_file(Path(model_path), "Model")
    return expected_path


def _legacy_pbc_handling(atoms) -> str:
    pbc = tuple(bool(x) for x in atoms.pbc)
    if pbc == (False, False, False):
        return "realspace"
    if pbc == (True, True, True):
        return "pbc"
    if pbc == (True, True, False):
        return "slab"
    raise ValueError(f"Unsupported pbc pattern in legacy expected test: {pbc}")


@pytest.mark.parametrize(
    "case",
    CALCULATOR_EXPECTED_CASES,
    ids=lambda case: case["expected_name"],
)
def test_calculator_expected_outputs(case):
    expected_path = _require_case_files(case)
    calc = _build_calculator(case)

    for atoms in read(expected_path, index=":"):
        _assert_results_close(
            _load_expected(atoms), _calculate_expected_warning(atoms, calc)
        )


def test_calculator_auto_expected_outputs():
    expected_path = _require_case_files(CALCULATOR_AUTO_CASE)
    calc = _build_calculator(CALCULATOR_AUTO_CASE)

    for atoms in read(expected_path, index=":"):
        _assert_results_close(
            _load_expected(atoms), _calculate_expected_warning(atoms, calc)
        )


def test_calculator_auto_dispatch_matches_explicit_modes():
    case = CALCULATOR_AUTO_CASE
    expected_path = _require_case_files(case)
    configs = read(expected_path, index=":")

    fff_atoms = next(atoms for atoms in configs if not any(atoms.pbc))
    periodic_atoms = next(atoms for atoms in configs if any(atoms.pbc))

    auto_calc = _build_calculator(case["name"], pbc_handling="auto")
    realspace_calc = _build_calculator(case["name"], pbc_handling="realspace")
    mixed_calc = _build_calculator(case["name"], pbc_handling="mixed_periodic")

    _assert_results_close(
        _calculate_expected_warning(fff_atoms, realspace_calc),
        _calculate_expected_warning(fff_atoms, auto_calc),
    )
    _assert_results_close(
        _calculate(periodic_atoms, mixed_calc),
        _calculate(periodic_atoms, auto_calc),
    )


@pytest.mark.parametrize("case", LEGACY_CASES, ids=lambda case: case["expected_name"])
def test_calculator_against_legacy_expected_outputs(case):
    expected_path = _require_case_files(case)

    for atoms in read(expected_path, index=":"):
        calc = _build_calculator(
            case,
            pbc_handling=_legacy_pbc_handling(atoms),
        )
        _assert_results_close(
            _load_expected(atoms), _calculate_expected_warning(atoms, calc)
        )
