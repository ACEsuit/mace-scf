import os

from ase.io import read, write
import pytest

from .harness import (
    DEFAULT_REFERENCE_DATA,
    INITIAL_MODEL_CASES,
    RunTrainCase,
    SKIPPED_INITIAL_MODEL_CASES,
    assert_summary_close,
    deep_update,
    expected_summary_path,
    load_expected_summary,
    make_default_metrics_case,
    run_train_case,
    write_actual_outputs,
    write_reference_outputs,
)


FERMI_INFO_KEY = "the_fermi_level"


def _models_from_env():
    models = list(INITIAL_MODEL_CASES)
    only = os.environ.get("RUN_TRAIN_MODELS")
    if only:
        requested = [
            item
            for item in only.replace(",", " ").split()
            if item
        ]
        unknown = set(requested) - set(INITIAL_MODEL_CASES)
        if unknown:
            raise ValueError(f"Unknown RUN_TRAIN_MODELS entries: {sorted(unknown)}")
        models = requested

    skip = os.environ.get("RUN_TRAIN_SKIP_MODELS")
    if skip:
        skipped = {
            item
            for item in skip.replace(",", " ").split()
            if item
        }
        unknown = skipped - set(INITIAL_MODEL_CASES)
        if unknown:
            raise ValueError(f"Unknown RUN_TRAIN_SKIP_MODELS entries: {sorted(unknown)}")
        models = [model for model in models if model not in skipped]
    return models


def _model_params_from_env():
    params = []
    for model in _models_from_env():
        skip_reason = SKIPPED_INITIAL_MODEL_CASES.get(model)
        if skip_reason is None:
            params.append(model)
        else:
            params.append(pytest.param(model, marks=pytest.mark.skip(reason=skip_reason)))
    return params


def _reference_data_or_skip():
    reference_data = os.environ.get("RUN_TRAIN_REFERENCE_XYZ", DEFAULT_REFERENCE_DATA)
    if not os.path.exists(reference_data):
        pytest.skip(f"run_train reference data not found: {reference_data}")
    return reference_data


def _write_shifted_fermi_reference(source, target, shift):
    atoms_list = read(source, index=":")
    for atoms in atoms_list:
        if FERMI_INFO_KEY not in atoms.info:
            raise AssertionError(f"{source} is missing {FERMI_INFO_KEY!r}")
        atoms.info[FERMI_INFO_KEY] = float(atoms.info[FERMI_INFO_KEY]) + shift
    write(target, atoms_list, format="extxyz")
    return target


def _skip_unless_fixedpoint_enabled():
    if "FixedPoint" not in _models_from_env():
        pytest.skip("FixedPoint is disabled by RUN_TRAIN_MODELS/RUN_TRAIN_SKIP_MODELS")
    if "FixedPoint" in SKIPPED_INITIAL_MODEL_CASES:
        pytest.skip(SKIPPED_INITIAL_MODEL_CASES["FixedPoint"])


@pytest.mark.parametrize("model", _model_params_from_env())
def test_run_train_default_metrics(tmp_path, model):
    reference_data = _reference_data_or_skip()
    case = make_default_metrics_case(model)
    result = run_train_case(
        tmp_path / case.name,
        case,
        reference_data=reference_data,
        check=True,
    )

    log_text = result.log_text()
    assert "Atomic energies:" in log_text
    assert "Average number of neighbors:" in log_text
    assert "Number of parameters:" in log_text
    assert "Optimizer:" in log_text

    metrics = result.metrics()
    assert any(record.get("mode") == "eval" for record in metrics)
    assert result.checkpoints()
    assert result.models()

    write_actual_outputs(result, model)

    if os.environ.get("RUN_TRAIN_UPDATE_EXPECTED") == "1":
        write_reference_outputs(result, model)
        return

    expected_path = expected_summary_path(model)
    if not expected_path.exists():
        pytest.skip(f"run_train expected summary not found: {expected_path}")
    assert_summary_close(result.summary(), load_expected_summary(model))


def test_run_train_fixedpoint_shifted_fermi_level_matches_default_metrics(tmp_path):
    _skip_unless_fixedpoint_enabled()
    reference_data = _reference_data_or_skip()
    fermi_shift = 3.0
    shifted_reference_data = _write_shifted_fermi_reference(
        reference_data,
        tmp_path / "water_clusters_shifted_fermi.xyz",
        fermi_shift,
    )

    base_case = make_default_metrics_case("FixedPoint")
    case = RunTrainCase(
        name="default_fixedpoint_shifted_fermi_metrics",
        config=deep_update(
            base_case.config,
            {"name": "default_fixedpoint_shifted_fermi_metrics"},
        ),
        extra_args=["--fermi_level_offset", str(fermi_shift)],
        timeout_s=base_case.timeout_s,
    )
    result = run_train_case(
        tmp_path / case.name,
        case,
        reference_data=shifted_reference_data,
        check=True,
    )

    log_text = result.log_text()
    assert f"using manually specified Fermi level offset {fermi_shift}" in log_text
    assert f"using Fermi level offset {fermi_shift}" in log_text

    expected = load_expected_summary("FixedPoint")
    actual = result.summary()
    actual["case"]["name"] = expected["case"]["name"]
    assert_summary_close(actual, expected)


def test_run_train_harness_writes_debug_artifacts(tmp_path):
    reference_data = _reference_data_or_skip()
    output_dir = tmp_path / "debug_run"
    result = run_train_case(
        output_dir,
        make_default_metrics_case("MACE"),
        reference_data=reference_data,
        overwrite=True,
        check=True,
    )

    assert result.config_path.exists()
    assert result.train_file.exists()
    assert result.log_files()
    assert result.metrics_files()
