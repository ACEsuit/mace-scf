import argparse
import copy
import shutil
from pathlib import Path

import yaml
from ase.io import read, write


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_FILE = Path(__file__).resolve().parent / "base.yaml"
DEFAULT_CASES_DIR = Path(__file__).resolve().parent / "cases"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "materialized"


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dump_yaml(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _materialize_input(case_config, target_dir: Path):
    input_config = case_config["inputs"]
    source_xyz = _resolve_repo_path(input_config["xyz"])
    source_index = input_config.get("index", ":")

    selected_configs = read(source_xyz, index=source_index)
    target_input = target_dir / "input.xyz"
    write(target_input, selected_configs)

    case_config["inputs"] = {
        "xyz": "input.xyz",
        "index": ":",
    }


def _materialize_case(case_file: Path, base_config, output_dir: Path, overwrite: bool):
    case_override = _load_yaml(case_file)
    case_config = _deep_merge(base_config, case_override)
    case_name = case_config.get("name", case_file.stem)
    case_config["name"] = case_name
    case_config["materialized_from"] = str(case_file.relative_to(REPO_ROOT))

    target_dir = output_dir / case_name
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Case directory already exists: {target_dir}. "
                "Use --overwrite to replace existing materialized cases."
            )
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    _materialize_input(case_config, target_dir)
    _dump_yaml(target_dir / "case.yaml", case_config)


def main():
    parser = argparse.ArgumentParser(
        description="Materialize fixed-point calculator test case directories from YAML templates."
    )
    parser.add_argument(
        "--base-file",
        default=str(DEFAULT_BASE_FILE),
        help=(
            "Base YAML template shared by fixed-point test cases. "
            "The default template uses tests/fixtures/configs/mixed_test_configs.xyz."
        ),
    )
    parser.add_argument(
        "--cases-dir",
        default=str(DEFAULT_CASES_DIR),
        help=(
            "Directory containing per-case override YAML files. "
            "Dedicated bulk/slab/cluster cases should usually override inputs.xyz there."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where materialized case folders will be written.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional list of case names to materialize. Defaults to all cases in the directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing materialized case directories.",
    )
    args = parser.parse_args()

    base_file = Path(args.base_file)
    cases_dir = Path(args.cases_dir)
    output_dir = Path(args.output_dir)

    base_config = _load_yaml(base_file)
    case_files = sorted(cases_dir.glob("*.yaml"))
    if args.cases is not None:
        requested_names = set(args.cases)
        case_files = [
            case_file for case_file in case_files if case_file.stem in requested_names
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for case_file in case_files:
        _materialize_case(
            case_file=case_file,
            base_config=base_config,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
