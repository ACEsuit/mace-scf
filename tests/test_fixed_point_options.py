import pytest

from mace_scf.electrostatics.fixed_point_options import (
    fixed_point_training_options_from_stage,
    validate_fixed_point_scf_options,
    validate_fixed_point_training_options,
)
from mace_scf.electrostatics.fixed_point_state import (
    FixedPointSCFOptions,
    FixedPointTrainingOptions,
)


def test_direct_training_options_do_not_create_scf_options():
    options = validate_fixed_point_training_options({"mode": "direct"})

    assert options == FixedPointTrainingOptions(mode="direct", scf=None)


def test_unroll_training_options_parse_nested_scf_options():
    options = validate_fixed_point_training_options(
        {
            "mode": "unroll_scf",
            "scf": {
                "num_scf_steps": 10,
                "mixing_parameter": 0.3,
                "initial_density": "from_data",
                "initial_fermi_level": "from_data",
            },
        }
    )

    assert options.mode == "unroll_scf"
    assert options.scf == FixedPointSCFOptions(
        num_scf_steps=10,
        scf_tolerance=1e-6,
        mixing_parameter=0.3,
        constant_charge=True,
        use_autograd_forces=True,
        initial_density="from_data",
        initial_fermi_level="from_data",
    )


def test_linearize_solve_training_options_parse_constant_fermi_scf_options():
    options = validate_fixed_point_training_options(
        {
            "mode": "linearize_solve",
            "scf": {
                "num_scf_steps": 10,
                "mixing_parameter": 0.3,
                "constant_charge": False,
            },
        }
    )

    assert options.mode == "linearize_solve"
    assert options.scf.constant_charge is False
    assert options.scf.num_scf_steps == 10
    assert options.scf.mixing_parameter == 0.3


def test_linearize_solve_training_options_parse_constant_charge_scf_options():
    options = validate_fixed_point_training_options(
        {
            "mode": "linearize_solve",
            "scf": {
                "num_scf_steps": 10,
                "mixing_parameter": 0.3,
                "constant_charge": True,
            },
        }
    )

    assert options.mode == "linearize_solve"
    assert options.scf.constant_charge is True
    assert options.scf.num_scf_steps == 10
    assert options.scf.mixing_parameter == 0.3


def test_old_fixed_point_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        validate_fixed_point_training_options({"mode": "fixed_point"})


def test_old_scf_training_options_stage_key_is_converted():
    stage = {
        "loss": {"atomic_multipoles": {"weight": 1.0}},
        "scf_training_options": {
            "mode": "unroll_scf",
            "num_scf_steps": 8,
            "mixing_parameter": 0.4,
        },
    }

    with pytest.deprecated_call(match="scf_training_options is deprecated"):
        options = fixed_point_training_options_from_stage(stage)

    assert options.mode == "unroll_scf"
    assert options.scf.num_scf_steps == 8
    assert options.scf.mixing_parameter == 0.4


def test_new_training_options_reject_flat_scf_fields():
    with pytest.raises(ValueError, match="must be nested under scf"):
        validate_fixed_point_training_options(
            {"mode": "unroll_scf", "num_scf_steps": 8, "mixing_parameter": 0.4}
        )


def test_scf_training_modes_require_explicit_step_count_and_mixing():
    with pytest.raises(ValueError, match="must explicitly set"):
        validate_fixed_point_training_options(
            {"mode": "unroll_scf", "scf": {"num_scf_steps": 8}}
        )


def test_direct_mode_warns_and_ignores_old_flat_scf_fields():
    stage = {
        "loss": {"atomic_multipoles": {"weight": 1.0}},
        "scf_training_options": {
            "mode": "direct",
            "num_scf_steps": 0,
            "mixing_parameter": 1.0,
        },
    }

    with pytest.deprecated_call():
        options = fixed_point_training_options_from_stage(stage)

    assert options.mode == "direct"
    assert options.scf is None


def test_scf_options_reject_nonpositive_step_count():
    with pytest.raises(ValueError, match="num_scf_steps must be positive"):
        validate_fixed_point_scf_options(
            {"num_scf_steps": 0},
            {},
        )
