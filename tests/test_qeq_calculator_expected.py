"""Regression test for MACEQEq ASE calculator outputs."""

import os
from copy import deepcopy

import numpy as np
from ase.io import read

from mace_scf.calculators.qeq import MACEQEqCalculator
from tests.paths import reference_model, reference_output, require_file


CONFIGS_PATH = reference_output("qeq_stage2")
MODEL_PATH = reference_model("QEqfit_stage2")
DEVICE = os.environ.get("MACE_DEVICE", "cpu")
DEFAULT_RTOL = float(os.environ.get("EXPECTED_RTOL", "1e-6"))
DEFAULT_ATOL = float(os.environ.get("EXPECTED_ATOL", "1e-6"))


def _require_qeq_files():
    require_file(CONFIGS_PATH, "QEq expected configs")
    require_file(MODEL_PATH, "QEq model")


def _assert_allclose(name, expected, actual):
    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)
    if (
        name == "charges"
        and expected_arr.ndim == 1
        and actual_arr.ndim == 2
        and actual_arr.shape[1] == 1
        and expected_arr.shape[0] == actual_arr.shape[0]
    ):
        expected_arr = expected_arr.reshape(-1, 1)
    assert (
        expected_arr.shape == actual_arr.shape
    ), f"{name} shape mismatch: expected {expected_arr.shape}, got {actual_arr.shape}"
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        err_msg=f"{name} mismatch",
    )


def _calculate(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.calc.calculate(atoms)
    return deepcopy(atoms.calc.results)


def test_qeq_calculator_expected_outputs():
    _require_qeq_files()
    calc = MACEQEqCalculator(model_path=str(MODEL_PATH), device=DEVICE)

    for atoms in read(CONFIGS_PATH, index=":"):
        results = _calculate(atoms, calc)
        _assert_allclose("energy", atoms.info["expected_energy"], results["energy"])
        _assert_allclose("forces", atoms.arrays["expected_forces"], results["forces"])
        _assert_allclose(
            "charges", atoms.arrays["expected_charges"], results["charges"]
        )
