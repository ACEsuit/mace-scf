"""Parity tests for compile-oriented fixed-point SCF execution."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read

from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
from mace_scf.electrostatics.compiled_fixed_point import (
    build_compiled_fixed_point_evaluator,
)
from tests.paths import REPO_ROOT, reference_model, require_file


DEVICE = os.environ.get("MACE_DEVICE", "cpu")
DEFAULT_RTOL = float(os.environ.get("EXPECTED_RTOL", "1e-6"))
DEFAULT_ATOL = float(os.environ.get("EXPECTED_ATOL", "1e-6"))


COMPILED_FIXED_POINT_CASES = [
    ("pbc", 2),
    ("slab", 4),
]
JELLIUM_SLAB_BOUNDS = (4.0, 10.0)
HAS_TORCH_COMPILER_NAMESPACE = hasattr(torch, "compiler")


def _model_path() -> str:
    return str(reference_model("fixedpoint_l1_stage2"))


def _input_path() -> str:
    return str(
        REPO_ROOT
        / "tests"
        / "fixed_point_cases"
        / "materialized"
        / "constant_fermi_baseline"
        / "input.xyz"
    )


def _require_case_files():
    model_path = _model_path()
    input_path = _input_path()
    require_file(Path(model_path), "Model")
    require_file(Path(input_path), "Input configs")
    return model_path, input_path


def _build_calculator(
    model_path: str,
    pbc_handling: str,
    *,
    compensating_jellium: bool = False,
    use_compile: bool = False,
):
    return MACEFixedPointSCF(
        model_path=model_path,
        device=DEVICE,
        pbc_handling=pbc_handling,
        compensating_jellium=compensating_jellium,
        jellium_slab_bounds=JELLIUM_SLAB_BOUNDS if compensating_jellium else None,
        use_compile=use_compile,
        atomic_multipoles_key="DMA_coeficients",
        fermi_level_key="fermi_level",
        external_field_key="external_field",
        total_charge_key="total_charge",
        scf_options={
            "constant_charge": False,
            "num_scf_steps": 3,
            "mixing_parameter": 0.2,
            "scf_tolerance": 1e-300,
            "initial_density": "local_guess",
            "initial_fermi_level": "from_data",
        },
        compile_scope="scf_chunk",
        compile_chunk_size=1,
        compile_enabled=False,
    )


@pytest.mark.parametrize("pbc_handling,atoms_index", COMPILED_FIXED_POINT_CASES)
def test_compiled_fixed_point_core_matches_runner(pbc_handling, atoms_index):
    model_path, input_path = _require_case_files()
    atoms = read(input_path, index=atoms_index)
    calc = _build_calculator(model_path, pbc_handling)
    batch = calc._build_batch(atoms).to(calc.device)
    batch_dict = batch.to_dict()
    restart_state = {
        "initial_charge_density": None,
        "fermi_level": None,
    }

    reference = calc._run_model(batch, restart_state)
    evaluator = build_compiled_fixed_point_evaluator(
        calc.model,
        pbc_handling=pbc_handling,
        num_scf_steps=calc.scf_options.num_scf_steps,
        mixing_parameter=calc.scf_options.mixing_parameter,
        enabled=False,
    )
    actual = evaluator.evaluate(batch_dict)

    for key in (
        "energy",
        "forces",
        "density_coefficients",
        "fermi_level",
        "dipole",
        "electrostatic_energy",
        "electron_energy",
    ):
        np.testing.assert_allclose(
            actual[key].detach().cpu().numpy(),
            reference[key].detach().cpu().numpy(),
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
            err_msg=f"{key} mismatch",
        )


def test_compiled_fixed_point_calculator_rejects_one_scf_step():
    model_path, _ = _require_case_files()
    with pytest.raises(ValueError, match="num_scf_steps > 1"):
        MACEFixedPointSCF(
            model_path=model_path,
            device=DEVICE,
            pbc_handling="pbc",
            use_compile=True,
            scf_options={
                "constant_charge": False,
                "num_scf_steps": 1,
                "mixing_parameter": 0.2,
            },
        )


def test_compensating_jellium_requires_slab_mode():
    model_path, _ = _require_case_files()
    with pytest.raises(
        ValueError, match="compensating_jellium=True requires pbc_handling='slab'"
    ):
        _build_calculator(
            model_path,
            "pbc",
            compensating_jellium=True,
        )


def test_compensating_jellium_swaps_blocks():
    model_path, _ = _require_case_files()
    calc = _build_calculator(
        model_path,
        "slab",
        compensating_jellium=True,
    )

    from graph_longrange.jellium_energy import JelliumSlabSolvatedEnergy
    from graph_longrange.jellium_features import JelliumSlabSolvatedFeatures

    assert isinstance(calc.model.coulomb_energy, JelliumSlabSolvatedEnergy)
    assert isinstance(
        calc.model.electric_potential_descriptor, JelliumSlabSolvatedFeatures
    )


def test_compiled_fixed_point_jellium_core_matches_runner():
    model_path, input_path = _require_case_files()
    atoms = read(input_path, index=4)
    calc = _build_calculator(
        model_path,
        "slab",
        compensating_jellium=True,
    )
    batch = calc._build_batch(atoms).to(calc.device)
    batch_dict = batch.to_dict()
    restart_state = {
        "initial_charge_density": None,
        "fermi_level": None,
    }

    reference = calc._run_model(batch, restart_state)
    evaluator = build_compiled_fixed_point_evaluator(
        calc.model,
        pbc_handling="slab",
        num_scf_steps=calc.scf_options.num_scf_steps,
        mixing_parameter=calc.scf_options.mixing_parameter,
        enabled=False,
    )
    actual = evaluator.evaluate(batch_dict)

    for key in (
        "energy",
        "forces",
        "density_coefficients",
        "fermi_level",
        "dipole",
        "electrostatic_energy",
        "electron_energy",
    ):
        np.testing.assert_allclose(
            actual[key].detach().cpu().numpy(),
            reference[key].detach().cpu().numpy(),
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
            err_msg=f"{key} mismatch",
        )


def test_compiled_fixed_point_jellium_calculator_matches_eager():
    if not HAS_TORCH_COMPILER_NAMESPACE:
        pytest.skip(
            "This test exercises active torch.compile and requires a newer "
            "PyTorch with the torch.compiler namespace."
        )
    model_path, input_path = _require_case_files()
    atoms = read(input_path, index=4)
    eager = _build_calculator(
        model_path,
        "slab",
        compensating_jellium=True,
        use_compile=False,
    )
    compiled = _build_calculator(
        model_path,
        "slab",
        compensating_jellium=True,
        use_compile=True,
    )

    eager.calculate(atoms)
    compiled.calculate(atoms)

    for key in (
        "energy",
        "free_energy",
        "forces",
        "density_coefficients",
        "dipole",
        "fermi_level",
        "electrostatic_energy",
        "electron_energy",
        "electrostatic_features",
    ):
        np.testing.assert_allclose(
            np.asarray(compiled.results[key]),
            np.asarray(eager.results[key]),
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
            err_msg=f"{key} mismatch",
        )
