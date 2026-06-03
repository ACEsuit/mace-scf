import os
from pathlib import Path

import yaml

from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
from tests.paths import REFERENCE_MODELS_DIR


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_DIR = Path(__file__).resolve().parent / "materialized"
DEFAULT_MODEL_DIR = REFERENCE_MODELS_DIR
DEFAULT_DEVICE = os.environ.get("MACE_DEVICE", "cpu")


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def materialized_cases_dir() -> Path:
    return Path(os.environ.get("FIXED_POINT_CASES_DIR", str(DEFAULT_CASES_DIR)))


def load_case_config(case_dir: Path):
    case_yaml = case_dir / "case.yaml"
    with case_yaml.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def discover_case_dirs(cases_dir: Path = None, case_names=None):
    if cases_dir is None:
        cases_dir = materialized_cases_dir()
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        return []

    discovered = sorted(
        case_dir
        for case_dir in cases_dir.iterdir()
        if case_dir.is_dir() and (case_dir / "case.yaml").exists()
    )
    if case_names is None:
        return discovered

    requested = set(case_names)
    return [case_dir for case_dir in discovered if case_dir.name in requested]


def case_usage(case_config) -> str:
    return case_config.get("usage", "single_point")


def expected_output_keys(case_config):
    return list(case_config.get("checks", {}).get("expected_outputs", []))


def load_input_atoms(case_dir: Path):
    from ase.io import read

    case_config = load_case_config(case_dir)
    inputs = case_config["inputs"]
    input_path = case_dir / inputs.get("xyz", "input.xyz")
    return read(input_path, index=inputs.get("index", ":"))


def resolve_model_path(case_config, model_dir: Path = None) -> Path:
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    model_dir = Path(model_dir)

    model_config = case_config.get("model", {})
    model_path = model_config.get("path")
    if model_path is not None:
        return _resolve_repo_path(model_path)

    model_name = model_config["name"]
    if model_name.endswith(".model"):
        return model_dir / model_name
    return model_dir / f"{model_name}.model"


def build_fixed_point_calculator(
    case_config,
    model_dir: Path = None,
    device: str = None,
    pbc_handling: str = None,
):
    if device is None:
        device = DEFAULT_DEVICE

    keys = case_config.get("keys", {})
    calculator_options = case_config.get("calculator_options", {})
    scf_options = dict(case_config.get("scf_options", {}))

    restart_density = calculator_options.get("restart_density", False)
    restart_fermi_level = calculator_options.get("restart_fermi_level", False)
    if pbc_handling is None:
        pbc_handling = calculator_options.get("pbc_handling")

    return MACEFixedPointSCF(
        model_path=resolve_model_path(case_config, model_dir=model_dir),
        device=device,
        scf_options=scf_options,
        pbc_handling=pbc_handling,
        atomic_multipoles_key=keys.get(
            "atomic_multipoles_key", "initial_density_coefficients"
        ),
        fermi_level_key=keys.get("fermi_level_key", "initial_fermi_level"),
        external_field_key=keys.get("external_field_key", "external_field"),
        total_charge_key=keys.get("total_charge_key", "total_charge"),
        ignore_nonconverged=calculator_options.get("ignore_nonconverged", False),
        save_full_scf_history=calculator_options.get("save_full_scf_history", False),
        scf_restart=restart_density or restart_fermi_level,
        restart_density=restart_density,
        restart_fermi_level=restart_fermi_level,
    )
