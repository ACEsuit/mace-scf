"""Regression test for the --distributed + fixed-point mode="implicit" guard
in mace_scf/utils/check_args.py::check_config_conflicts. 
"""

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


def _parse_fixedpoint_args(tmp_path, *, distributed: bool, mode: str):
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
