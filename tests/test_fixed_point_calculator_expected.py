import os

import numpy as np
import pytest
from ase.io import read

from tests.fixed_point_cases.case_utils import (
    build_fixed_point_calculator,
    case_usage,
    discover_case_dirs,
    expected_output_keys,
    load_case_config,
    resolve_model_path,
)


DEFAULT_RTOL = float(os.environ.get("EXPECTED_RTOL", "1e-7"))
DEFAULT_ATOL = float(os.environ.get("EXPECTED_ATOL", "1e-6"))
CASE_DIRS = discover_case_dirs()


def _load_expected(atoms):
    expected = {}
    for key, value in atoms.info.items():
        if key.startswith("expected_"):
            if key.endswith("_shape"):
                continue
            expected_name = key[len("expected_") :]
            shape_key = f"expected_{expected_name}_shape"
            if shape_key in atoms.info:
                trailing_shape = tuple(int(x) for x in atoms.info[shape_key])
                array_value = np.asarray(value).reshape((len(atoms), *trailing_shape))
                expected[expected_name] = array_value
            else:
                expected[expected_name] = value
    for key, value in atoms.arrays.items():
        if key.startswith("expected_"):
            expected_name = key[len("expected_") :]
            shape_key = f"expected_{expected_name}_shape"
            if shape_key in atoms.info:
                trailing_shape = tuple(int(x) for x in atoms.info[shape_key])
                expected[expected_name] = value.reshape((len(atoms), *trailing_shape))
            else:
                expected[expected_name] = value
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


def _case_dirs_for_usage(usage):
    selected = []
    for case_dir in CASE_DIRS:
        case_config = load_case_config(case_dir)
        if case_usage(case_config) == usage:
            selected.append(case_dir)
    return selected


def _legacy_fixed_point_pbc_method(atoms, case_config):
    calculator_options = case_config.get("calculator_options", {})
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


def _case_pbc_method(atoms, case_config):
    calculator_options = case_config.get("calculator_options", {})
    explicit_method = calculator_options.get("pbc_handling")
    if explicit_method is None:
        explicit_method = _legacy_fixed_point_pbc_method(atoms, case_config)
    return explicit_method


def _pbc_pattern(atoms):
    return tuple(bool(x) for x in atoms.pbc)


def _assert_outputs_for_atoms(atoms, output_keys, expected_path):
    expected = _load_expected(atoms)
    for key in output_keys:
        assert key in expected, f"Missing expected_{key} in {expected_path}"
        assert key in atoms.calc.results, f"Missing result key: {key}"
        _assert_allclose(key, expected[key], atoms.calc.results[key])


def _assert_case_outputs_single_point(case_dir):
    case_config = load_case_config(case_dir)
    expected_path = case_dir / "expected.xyz"
    model_path = resolve_model_path(case_config)
    if not expected_path.exists():
        pytest.skip(f"Expected configs not found: {expected_path}")
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    configs = read(expected_path, index=":")
    output_keys = expected_output_keys(case_config)

    for atoms in configs:
        calc = build_fixed_point_calculator(
            case_config,
            pbc_handling=_case_pbc_method(atoms, case_config),
        )
        atoms.calc = calc
        atoms.calc.calculate(atoms)
        _assert_outputs_for_atoms(atoms, output_keys, expected_path)


def _assert_case_outputs_restart(case_dir):
    case_config = load_case_config(case_dir)
    expected_path = case_dir / "expected.xyz"
    model_path = resolve_model_path(case_config)
    if not expected_path.exists():
        pytest.skip(f"Expected configs not found: {expected_path}")
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    configs = read(expected_path, index=":")
    output_keys = expected_output_keys(case_config)
    first_atoms = configs[0]
    expected_pbc = _pbc_pattern(first_atoms)
    expected_method = _case_pbc_method(first_atoms, case_config)

    for atoms in configs[1:]:
        actual_pbc = _pbc_pattern(atoms)
        actual_method = _case_pbc_method(atoms, case_config)
        if actual_pbc != expected_pbc:
            raise ValueError(
                f"Restart case {case_dir.name} mixes pbc patterns: "
                f"{expected_pbc} then {actual_pbc}"
            )
        if actual_method != expected_method:
            raise ValueError(
                f"Restart case {case_dir.name} mixes pbc_handling modes: "
                f"{expected_method} then {actual_method}"
            )

    calc = build_fixed_point_calculator(
        case_config,
        pbc_handling=expected_method,
    )
    for atoms in configs:
        atoms.calc = calc
        atoms.calc.calculate(atoms)
        _assert_outputs_for_atoms(atoms, output_keys, expected_path)


@pytest.mark.parametrize(
    "case_dir",
    _case_dirs_for_usage("single_point"),
    ids=lambda case_dir: case_dir.name,
)
def test_fixed_point_calculator_expected_cases(case_dir):
    _assert_case_outputs_single_point(case_dir)


@pytest.mark.parametrize(
    "case_dir",
    _case_dirs_for_usage("restart"),
    ids=lambda case_dir: case_dir.name,
)
def test_fixed_point_calculator_restart_cases(case_dir):
    _assert_case_outputs_restart(case_dir)
