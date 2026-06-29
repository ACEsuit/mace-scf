"""Parity tests for compile-oriented local-source execution."""

import os
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from mace_scf.calculators.localsources import (
    MACELocalSplitCharges,
    MACELocalCharges,
)
from mace_scf.electrostatics.compiled_localsources import (
    build_compiled_local_source_evaluator,
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
JELLIUM_SLAB_BOUNDS = (4.0, 10.0)


LOCAL_SYMMETRIC_CASES = [
    ("splitcharge_l0_stage1", "pbc"),
    ("splitcharge_l0_stage1", "slab"),
    ("splitcharge_l1_stage1", "pbc"),
    ("splitcharge_l1_stage1", "slab"),
]

NONPOLARIZABLE_CASES = [
    ("nonpol_l0_stage1", "pbc"),
    ("nonpol_l1_stage1", "pbc"),
]


def _model_path(model_name: str) -> str:
    return str(reference_model(model_name))


def _expected_path(model_name: str, pbc_handling: str) -> str:
    return str(reference_output(f"{model_name}_{pbc_handling}"))


def _require_case_files(model_name: str, pbc_handling: str):
    model_path = _model_path(model_name)
    expected_path = _expected_path(model_name, pbc_handling)
    require_file(Path(model_path), "Model")
    require_file(Path(expected_path), "Expected configs")
    return model_path, expected_path


def _build_local_symmetric_calculator(model_path: str, pbc_handling: str, **kwargs):
    return MACELocalSplitCharges(
        model_path=model_path,
        device=DEVICE,
        formal_charges_key=LOCAL_FORMAL_CHARGES_KEY,
        external_field_key=LOCAL_EXTERNAL_FIELD_KEY,
        fermi_level_key=LOCAL_FERMI_LEVEL_KEY,
        pbc_handling=pbc_handling,
        **kwargs,
    )


@pytest.mark.parametrize("model_name,pbc_handling", LOCAL_SYMMETRIC_CASES)
def test_local_symmetric_compiled_core_matches_model(model_name, pbc_handling):
    model_path, expected_path = _require_case_files(model_name, pbc_handling)
    atoms = read(expected_path, index=0)
    calc = _build_local_symmetric_calculator(
        model_path=model_path,
        pbc_handling=pbc_handling,
    )
    batch = next(iter(calc._build_data_loader(atoms))).to(calc.device)
    data = batch.to_dict()

    reference = calc.model(
        data,
        compute_force=True,
        compute_stress=False,
    )
    evaluator = build_compiled_local_source_evaluator(
        calc.model,
        pbc_handling=pbc_handling,
        enabled=False,
    )
    actual = evaluator.evaluate(data)

    np.testing.assert_allclose(
        actual["energy"].detach().cpu().numpy(),
        reference["energy"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["forces"].detach().cpu().numpy(),
        reference["forces"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["density_coefficients"].detach().cpu().numpy(),
        reference["density_coefficients"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["dipole"].detach().cpu().numpy(),
        reference["dipole"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


def test_kspace_plan_reused_for_fixed_cell():
    model_path, expected_path = _require_case_files("splitcharge_l0_stage1", "pbc")
    atoms = read(expected_path, index=0)
    calc = _build_local_symmetric_calculator(model_path=model_path, pbc_handling="pbc")
    batch = next(iter(calc._build_data_loader(atoms))).to(calc.device)
    data = batch.to_dict()
    evaluator = build_compiled_local_source_evaluator(
        calc.model,
        pbc_handling="pbc",
        enabled=False,
    )

    plan_1 = evaluator.kspace_planner.get_plan(
        data["cell"],
        data["rcell"],
        "pbc",
    )
    plan_2 = evaluator.kspace_planner.get_plan(
        data["cell"],
        data["rcell"],
        "pbc",
    )

    assert plan_1 is plan_2


def test_local_symmetric_compiled_core_matches_model_with_jellium_slab():
    model_path, expected_path = _require_case_files("splitcharge_l0_stage1", "slab")
    atoms = read(expected_path, index=0)
    calc = _build_local_symmetric_calculator(
        model_path=model_path,
        pbc_handling="slab",
        compensating_jellium=True,
        jellium_slab_bounds=JELLIUM_SLAB_BOUNDS,
    )
    batch = next(iter(calc._build_data_loader(atoms))).to(calc.device)
    data = batch.to_dict()

    reference = calc.model(
        data,
        compute_force=True,
        compute_stress=False,
    )
    evaluator = build_compiled_local_source_evaluator(
        calc.model,
        pbc_handling="slab",
        enabled=False,
    )
    actual = evaluator.evaluate(data)

    np.testing.assert_allclose(
        actual["energy"].detach().cpu().numpy(),
        reference["energy"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["forces"].detach().cpu().numpy(),
        reference["forces"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["density_coefficients"].detach().cpu().numpy(),
        reference["density_coefficients"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["dipole"].detach().cpu().numpy(),
        reference["dipole"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


def test_local_symmetric_jellium_rejects_pbc_mode():
    model_path, _ = _require_case_files("splitcharge_l0_stage1", "pbc")
    with pytest.raises(
        ValueError, match="compensating_jellium=True requires pbc_handling='slab'"
    ):
        _build_local_symmetric_calculator(
            model_path=model_path,
            pbc_handling="pbc",
            compensating_jellium=True,
            jellium_slab_bounds=JELLIUM_SLAB_BOUNDS,
            use_compile=True,
        )


def _build_nonpolarizable_calculator(model_path: str, pbc_handling: str):
    return MACELocalCharges(
        model_path=model_path,
        device=DEVICE,
        external_field_key=LOCAL_EXTERNAL_FIELD_KEY,
        fermi_level_key=LOCAL_FERMI_LEVEL_KEY,
        pbc_handling=pbc_handling,
    )


@pytest.mark.parametrize("model_name,pbc_handling", NONPOLARIZABLE_CASES)
def test_nonpolarizable_compiled_core_matches_model(model_name, pbc_handling):
    model_path, expected_path = _require_case_files(model_name, pbc_handling)
    atoms = read(expected_path, index=0)
    calc = _build_nonpolarizable_calculator(
        model_path=model_path,
        pbc_handling=pbc_handling,
    )
    batch = next(iter(calc._build_data_loader(atoms))).to(calc.device)
    data = batch.to_dict()

    reference = calc.model(
        data,
        compute_force=True,
        compute_stress=False,
    )
    evaluator = build_compiled_local_source_evaluator(
        calc.model,
        pbc_handling=pbc_handling,
        enabled=False,
    )
    actual = evaluator.evaluate(data)

    np.testing.assert_allclose(
        actual["energy"].detach().cpu().numpy(),
        reference["energy"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["forces"].detach().cpu().numpy(),
        reference["forces"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["density_coefficients"].detach().cpu().numpy(),
        reference["density_coefficients"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        actual["dipole"].detach().cpu().numpy(),
        reference["dipole"].detach().cpu().numpy(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
