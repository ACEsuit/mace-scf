"""Regression tests for the fixed-point mode="implicit" guards in
mace_scf/utils/check_args.py::check_config_conflicts: implicit mode with
--distributed and with --compute_stress.
"""

import sys
import types
from pathlib import Path

import pytest
import yaml

import mace_scf.utils
from mace_scf.utils.check_args import check_config_conflicts

from . import fixtures
from ..run_train_pipeline.harness import (
    DEFAULT_REFERENCE_DATA,
    deep_update,
    make_run_train_config,
    model_config_overrides,
)


def _parse_fixedpoint_args(
    tmp_path, *, distributed: bool, mode: str, compute_stress: bool = False
):
    training_options = {"mode": mode}
    if mode != "direct":
        training_options["scf"] = fixtures.SCF_OPTIONS
    config = make_run_train_config(
        model="FixedPoint",
        name="implicit_distributed_guard",
        overrides=deep_update(
            model_config_overrides("FixedPoint"),
            {
                "distributed": distributed,
                "compute_stress": compute_stress,
                "train_schedule": {
                    0: {"fixed_point_training_options": training_options}
                },
            },
        ),
    )
    config["train_file"] = str(DEFAULT_REFERENCE_DATA)
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return mace_scf.utils.extended_arg_parser().parse_args(
        ["--config", str(config_path)]
    )


def test_distributed_plus_implicit_mode_raises(tmp_path):
    pytest.importorskip("torchopt")
    args = _parse_fixedpoint_args(tmp_path, distributed=True, mode="implicit")
    with pytest.raises(NotImplementedError, match="implicit"):
        check_config_conflicts(args)


def test_distributed_plus_direct_mode_is_allowed(tmp_path):
    args = _parse_fixedpoint_args(tmp_path, distributed=True, mode="direct")
    check_config_conflicts(args)  # must not raise


def test_stress_plus_implicit_mode_raises(tmp_path, monkeypatch):
    # Implicit-mode option parsing only checks that torchopt imports.
    monkeypatch.setitem(sys.modules, "torchopt", types.ModuleType("torchopt"))
    args = _parse_fixedpoint_args(
        tmp_path, distributed=False, mode="implicit", compute_stress=True
    )
    with pytest.raises(NotImplementedError, match="stress"):
        check_config_conflicts(args)


def test_stress_plus_linearize_solve_mode_is_allowed(tmp_path):
    args = _parse_fixedpoint_args(
        tmp_path, distributed=False, mode="linearize_solve", compute_stress=True
    )
    check_config_conflicts(args)  # must not raise
