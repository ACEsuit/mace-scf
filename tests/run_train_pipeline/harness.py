import ast
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TRAIN_SCRIPT = REPO_ROOT / "scripts" / "run_train.py"
DEFAULT_REFERENCE_DATA = (
    REPO_ROOT / "tests" / "run_train_pipeline" / "reference" / "water_clusters.xyz"
)
EXPECTED_DIR = REPO_ROOT / "tests" / "run_train_pipeline" / "expected"
ACTUAL_DIR = REPO_ROOT / "tests" / "run_train_pipeline" / "actual"
INITIAL_MODEL_CASES = (
    "MACE",
    "LocalSplitCharges",
    "LocalCharges",
    "FixedPoint",
    "FixedChargeBaselinedMACE",
)
SKIPPED_INITIAL_MODEL_CASES = {}
IGNORED_SUMMARY_PATHS = ()


def deep_update(base: Mapping[str, Any], updates: Optional[Mapping[str, Any]]):
    result = deepcopy(dict(base))
    if updates is None:
        return result
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _without_none(value):
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value


def make_run_train_config(
    *,
    model: str = "MACE",
    name: str = "run_train_default_metrics",
    train_file: str = "water_clusters.xyz",
    overrides: Optional[Mapping[str, Any]] = None,
):
    config = {
        "name": name,
        "seed": 5,
        "device": "cpu",
        "work_dir": ".",
        "train_file": train_file,
        "valid_fraction": 0.25,
        "model": model,
        "hidden_irreps": "4x0e+4x1o",
        "num_interactions": 1,
        "num_channels": None,
        "max_L": None,
        "r_max": 4.5,
        "E0s": "{1: -12.6746244439181, 8: -2041.03979050724}",
        "batch_size": 2,
        "valid_batch_size": 2,
        "eval_interval": 1,
        "patience": 1000,
        "max_num_epochs": 1,
        "optimizer": "schedulefree",
        "static_bond_transfer_block": "NoFieldSymmetricPredictionSourceBlock",
        "error_table": "PerAtomRMSE",
        "compute_forces": True,
        "compute_stress": False,
        "ema": True,
        "amsgrad": True,
        "restart_latest": True,
        "save_all_checkpoints": False,
        "atomic_multipoles_max_l": 1,
        "atomic_multipoles_smearing_width": 1.5,
        "kspace_cutoff_factor": 1.0,
        "atomic_formal_charges": "{1: 1.0, 8: -2.0}",
        "heads": {
            "DFT": {
                "info_keys": {
                    "energy": "AIMS_energy",
                    "dipole": "AIMS_dipole",
                    "stress": "none",
                    "total_charge": "total_charge",
                },
                "arrays_keys": {
                    "forces": "AIMS_forces",
                    "atomic_multipoles": "AIMS_atom_multipoles",
                },
            }
        },
        "train_schedule": {
            0: {
                "name": "stage1",
                "start": 0,
                "end": 1,
                "loss": {
                    "energy_per_atom": 1.0,
                    "forces": 10.0,
                },
                "lr": 0.01,
            }
        },
    }
    return _without_none(deep_update(config, overrides))


ELECTROSTATIC_DEFAULT_METRICS_OVERRIDES = {
    "error_table": "EnergyDensityDipoleRMSE",
    "train_schedule": {
        0: {
            "loss": {
                "energy_per_atom": 1.0,
                "forces": 10.0,
                "atomic_multipoles": 1.0,
                "dipole_per_atom": 1.0,
            }
        }
    },
}


def model_config_overrides(model: str):
    if model == "MACE":
        return {}
    if model == "LocalSplitCharges":
        return ELECTROSTATIC_DEFAULT_METRICS_OVERRIDES
    if model == "LocalCharges":
        return ELECTROSTATIC_DEFAULT_METRICS_OVERRIDES
    if model == "FixedPoint":
        return deep_update(
            ELECTROSTATIC_DEFAULT_METRICS_OVERRIDES,
            {
                "field_feature_widths": "[1.5]",
                "fermi_level_offset": 0.0,
                "fixedpoint_update_config": {
                    "type": "OneBodyVariableUpdate",
                    "potential_embedding_cls": "BiasedLinearPotentialEmbedding",
                    "nonlinearity_cls": "NoNonLinearity",
                },
                "train_schedule": {
                    0: {
                        "fixed_point_training_options": {
                            "mode": "direct",
                        },
                    }
                },    
                "heads": {
                    "DFT" : {
                        "info_keys" : {
                            "fermi_level": "the_fermi_level",
                            "external_field": "the_external_field",
                        }
                    }
                },
            },
        )
    if model == "FixedChargeBaselinedMACE":
        return {
            "error_table": "EnergyDensityDipoleRMSE",
            "train_schedule": {
                0: {
                    "loss": {
                        "energy_per_atom": 1.0,
                        "forces": 10.0,
                        "dipole_per_atom": 1.0,
                    }
                }
            },
        }
    raise ValueError(f"No default run_train test overrides for model={model!r}")


@dataclass
class RunTrainCase:
    name: str
    config: Mapping[str, Any]
    extra_args: list[str] = field(default_factory=list)
    timeout_s: int = 180


@dataclass
class RunTrainResult:
    case: RunTrainCase
    run_dir: Path
    config_path: Path
    train_file: Path
    process: subprocess.CompletedProcess

    @property
    def stdout(self):
        return self.process.stdout

    @property
    def stderr(self):
        return self.process.stderr

    @property
    def returncode(self):
        return self.process.returncode

    def assert_success(self):
        assert self.returncode == 0, (
            f"run_train.py failed for {self.case.name}\n"
            f"Command: {' '.join(self.process.args)}\n"
            f"STDOUT:\n{self.stdout}\n"
            f"STDERR:\n{self.stderr}"
        )

    def log_files(self):
        return sorted((self.run_dir / "logs").glob("*.log"))

    def log_text(self):
        return "\n".join(path.read_text() for path in self.log_files())

    def metrics_files(self):
        return sorted((self.run_dir / "results").glob("*.txt"))

    def metrics(self):
        records = []
        for path in self.metrics_files():
            for line in path.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
        return records

    def checkpoints(self):
        return sorted((self.run_dir / "checkpoints").glob("*.pt"))

    def models(self):
        return sorted(self.run_dir.glob("*.model"))

    def summary(self):
        return summarize_run_train_result(self)


def script_env():
    env = os.environ.copy()
    paths = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def prepare_run_dir(
    run_dir: Path,
    case: RunTrainCase,
    *,
    reference_data: Path = DEFAULT_REFERENCE_DATA,
    overwrite: bool = False,
):
    run_dir = Path(run_dir)
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("logs", "results", "checkpoints", "downloads"):
        (run_dir / dirname).mkdir(exist_ok=True)

    train_file = run_dir / "water_clusters.xyz"
    shutil.copyfile(reference_data, train_file)

    config = deep_update(
        case.config,
        {
            "work_dir": str(run_dir),
            "log_dir": str(run_dir / "logs"),
            "results_dir": str(run_dir / "results"),
            "checkpoints_dir": str(run_dir / "checkpoints"),
            "model_dir": str(run_dir),
            "train_file": str(train_file),
        },
    )
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, train_file


def run_train_case(
    run_dir: Path,
    case: RunTrainCase,
    *,
    reference_data: Path = DEFAULT_REFERENCE_DATA,
    overwrite: bool = False,
    check: bool = False,
):
    config_path, train_file = prepare_run_dir(
        run_dir,
        case,
        reference_data=reference_data,
        overwrite=overwrite,
    )
    cmd = [
        sys.executable,
        str(RUN_TRAIN_SCRIPT),
        "--config",
        str(config_path),
        *case.extra_args,
    ]
    process = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=script_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=case.timeout_s,
    )
    result = RunTrainResult(
        case=case,
        run_dir=Path(run_dir),
        config_path=config_path,
        train_file=train_file,
        process=process,
    )
    if check:
        result.assert_success()
    return result


def make_default_metrics_case(model: str):
    name = f"default_{model.lower()}_metrics"
    config = make_run_train_config(
        model=model,
        name=name,
        overrides=model_config_overrides(model),
    )
    return RunTrainCase(name=name, config=config)


def make_model_smoke_case(model: str):
    return make_default_metrics_case(model)


def expected_summary_path(model: str):
    return summary_path(EXPECTED_DIR, model)


def expected_log_path(model: str):
    return log_path(EXPECTED_DIR, model)


def actual_summary_path(model: str):
    return summary_path(ACTUAL_DIR, model)


def actual_log_path(model: str):
    return log_path(ACTUAL_DIR, model)


def summary_path(directory: Path, model: str):
    return directory / f"{model}.json"


def log_path(directory: Path, model: str):
    return directory / f"{model}.log"


def write_run_outputs(result: RunTrainResult, model: str, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    summary_path(directory, model).write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True) + "\n"
    )
    logs = result.log_files()
    if logs:
        shutil.copyfile(logs[0], log_path(directory, model))


def write_reference_outputs(result: RunTrainResult, model: str):
    write_run_outputs(result, model, EXPECTED_DIR)


def write_actual_outputs(result: RunTrainResult, model: str):
    write_run_outputs(result, model, ACTUAL_DIR)


def load_expected_summary(model: str):
    return json.loads(expected_summary_path(model).read_text())


def load_actual_summary(model: str):
    return json.loads(actual_summary_path(model).read_text())


def _search_one(pattern: str, text: str, default=None):
    match = re.search(pattern, text)
    if match is None:
        return default
    return match.group(1)


def _search_int(pattern: str, text: str, default=None):
    value = _search_one(pattern, text, default=None)
    return default if value is None else int(value)


def _search_float(pattern: str, text: str, default=None):
    value = _search_one(pattern, text, default=None)
    return default if value is None else float(value)


def _search_literal(pattern: str, text: str, default=None):
    value = _search_one(pattern, text, default=None)
    if value is None:
        return default
    return ast.literal_eval(value)


def _jsonable(value):
    return json.loads(json.dumps(value))


def _clean_metrics(records):
    volatile_keys = {
        "grad_clip_applied",
        "grad_norm_before_clip",
        "opt_step",
        "time",
    }
    return [
        {
            key: value
            for key, value in record.items()
            if key not in volatile_keys
            if isinstance(value, (int, float, str, bool)) or value is None
        }
        for record in records
    ]


def _parse_table_cell(value: str):
    stripped = value.strip()
    if stripped == "":
        return ""
    try:
        as_float = float(stripped)
    except ValueError:
        return stripped
    if as_float.is_integer() and re.fullmatch(r"[-+]?\d+", stripped):
        return int(as_float)
    return as_float


def _parse_pretty_tables(log_text: str):
    tables = []
    lines = log_text.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("+"):
            index += 1
            continue

        table_lines = []
        while index < len(lines) and (
            lines[index].startswith("+") or lines[index].startswith("|")
        ):
            table_lines.append(lines[index])
            index += 1

        if len(table_lines) < 4 or not table_lines[1].startswith("|"):
            continue

        header = [cell.strip() for cell in table_lines[1].strip("|").split("|")]
        rows = []
        for line in table_lines[3:-1]:
            if not line.startswith("|"):
                continue
            cells = [_parse_table_cell(cell) for cell in line.strip("|").split("|")]
            if len(cells) == len(header):
                rows.append(dict(zip(header, cells)))
        if rows:
            tables.append({"columns": header, "rows": rows})
    return tables


def summarize_run_train_result(result: RunTrainResult):
    log_text = result.log_text()
    config = result.case.config
    metrics = _clean_metrics(result.metrics())
    train_count = _search_int(
        r"Total number of configurations: train=(\d+), valid=\d+", log_text
    )
    valid_count = _search_int(
        r"Total number of configurations: train=\d+, valid=(\d+)", log_text
    )
    z_table_text = _search_one(r"AtomicNumberTable: \((.*?)\)", log_text, "")
    z_table = [int(item.strip()) for item in z_table_text.split(",") if item.strip()]

    return {
        "case": {
            "name": result.case.name,
            "model": config["model"],
        },
        "config": {
            "seed": config["seed"],
            "r_max": config["r_max"],
            "E0s": config["E0s"],
            "hidden_irreps": config["hidden_irreps"],
            "num_interactions": config["num_interactions"],
            "batch_size": config["batch_size"],
            "valid_batch_size": config["valid_batch_size"],
            "optimizer": config["optimizer"],
            "error_table": config["error_table"],
            "atomic_multipoles_max_l": config["atomic_multipoles_max_l"],
            "kspace_cutoff_factor": config["kspace_cutoff_factor"],
            "train_schedule": _jsonable(config["train_schedule"]),
        },
        "data": {
            "train_count": train_count,
            "valid_count": valid_count,
            "z_table": z_table,
            "atomic_energies": _search_literal(
                r"Atomic energies: (\[[^\n]+\])", log_text, default=[]
            ),
            "avg_num_neighbors": _search_float(
                r"Average number of neighbors: ([0-9.eE+-]+)", log_text
            ),
        },
        "model": {
            "num_parameters": _search_int(r"Number of parameters: (\d+)", log_text),
        },
        "runtime": {
            "scheduler_is_noop": (
                "ScheduleFree optimizer selected; LR scheduler options are ignored."
                in log_text
            ),
            "optimizer_log_contains_schedulefree": "AdamWScheduleFree" in log_text,
            "loss_line": _search_one(r"INFO: (WeightedLoss\([^\n]+\))", log_text),
        },
        "outputs": {
            "num_checkpoints": len(result.checkpoints()),
            "num_models": len(result.models()),
        },
        "metrics": metrics,
        "final_error_table": (_parse_pretty_tables(log_text) or [None])[-1],
    }


def _path_matches(pattern: str, path: str):
    pattern_re = re.escape(pattern)
    pattern_re = pattern_re.replace(r"\[\*\]", r"\[\d+\]")
    pattern_re = pattern_re.replace(r"\*", r"[^.]+")
    return re.fullmatch(pattern_re, path) is not None


def _path_is_ignored(path: str, ignored_paths=IGNORED_SUMMARY_PATHS):
    return any(_path_matches(pattern, path) for pattern in ignored_paths)


def summary_differences(
    actual,
    expected,
    *,
    atol=1e-10,
    rtol=1e-8,
    path="summary",
    ignored_paths=IGNORED_SUMMARY_PATHS,
):
    if _path_is_ignored(path, ignored_paths):
        return []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [(path, actual, expected, f"expected dict, got {type(actual)}")]
        diffs = []
        actual_keys = set(actual)
        expected_keys = set(expected)
        for key in sorted(actual_keys - expected_keys):
            child_path = f"{path}.{key}"
            if not _path_is_ignored(child_path, ignored_paths):
                diffs.append((child_path, actual[key], None, "unexpected key"))
        for key in sorted(expected_keys - actual_keys):
            child_path = f"{path}.{key}"
            if not _path_is_ignored(child_path, ignored_paths):
                diffs.append((child_path, None, expected[key], "missing key"))
        for key in sorted(actual_keys & expected_keys):
            diffs.extend(
                summary_differences(
                    actual[key],
                    expected[key],
                    atol=atol,
                    rtol=rtol,
                    path=f"{path}.{key}",
                    ignored_paths=ignored_paths,
                )
            )
        return diffs
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [(path, actual, expected, f"expected list, got {type(actual)}")]
        diffs = []
        if len(actual) != len(expected):
            diffs.append(
                (
                    path,
                    f"len={len(actual)}",
                    f"len={len(expected)}",
                    "length mismatch",
                )
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            diffs.extend(
                summary_differences(
                    actual_item,
                    expected_item,
                    atol=atol,
                    rtol=rtol,
                    path=f"{path}[{index}]",
                    ignored_paths=ignored_paths,
                )
            )
        return diffs
    if isinstance(expected, float):
        if not isinstance(actual, (float, int)):
            return [(path, actual, expected, "expected numeric")]
        diff = abs(float(actual) - expected)
        limit = atol + rtol * abs(expected)
        if diff > limit:
            return [(path, actual, expected, f"diff={diff}, limit={limit}")]
        return []
    if actual != expected:
        return [(path, actual, expected, "value mismatch")]
    return []


def assert_summary_close(actual, expected, *, atol=1e-10, rtol=1e-8, path="summary"):
    diffs = summary_differences(actual, expected, atol=atol, rtol=rtol, path=path)
    if not diffs:
        return
    first_path, actual_value, expected_value, reason = diffs[0]
    extra = "" if len(diffs) == 1 else f" (+ {len(diffs) - 1} more differences)"
    raise AssertionError(
        f"{first_path}: actual={actual_value!r}, expected={expected_value!r}; "
        f"{reason}{extra}"
    )
