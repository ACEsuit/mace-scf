import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
REFERENCE_CONFIGS_DIR = Path(
    os.environ.get("MACE_SCF_TEST_CONFIGS_DIR", str(FIXTURES_DIR / "configs"))
)
REFERENCE_EXPECTED_DIR = Path(
    os.environ.get("MACE_SCF_TEST_EXPECTED_DIR", str(FIXTURES_DIR / "expected"))
)
REFERENCE_MODELS_DIR = Path(
    os.environ.get("MACE_SCF_TEST_MODELS_DIR", str(FIXTURES_DIR / "models"))
)


def reference_config(name: str) -> Path:
    return REFERENCE_CONFIGS_DIR / name


def reference_output(name: str) -> Path:
    return REFERENCE_EXPECTED_DIR / f"{name}.expected.xyz"


def reference_model(name: str) -> Path:
    if name.endswith(".model"):
        return REFERENCE_MODELS_DIR / name
    return REFERENCE_MODELS_DIR / f"{name}.model"


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        pytest.skip(f"{description} not found: {path}")
    return path


def script_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
