import warnings
from dataclasses import fields

from .fixed_point_state import FixedPointSCFOptions, FixedPointTrainingOptions


FIXED_POINT_MODES = ("direct", "unroll_scf", "implicit", "linearize_solve")
SCF_OPTION_KEYS = {field.name for field in fields(FixedPointSCFOptions)}
TRAINING_OPTION_KEYS = {
    "mode",
    "scf",
    "linear_solve",
    "fixedpoint_scf_stability",
}
REQUIRED_TRAINING_SCF_KEYS = {"num_scf_steps", "mixing_parameter"}

FIXED_POINT_SCF_DEFAULTS = {
    "num_scf_steps": 100,
    "scf_tolerance": 1e-6,
    "mixing_parameter": 0.25,
    "constant_charge": True,
    "use_autograd_forces": True,
    "initial_density": "local_guess",
    "initial_fermi_level": "from_data",
}


def _warn_deprecated(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def validate_fixed_point_scf_options(
    raw: dict,
    defaults: dict,
) -> FixedPointSCFOptions:
    unknown_keys = set(raw) - SCF_OPTION_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown SCF option(s): {sorted(unknown_keys)}")

    options = dict(defaults)
    options.update(raw)

    num_scf_steps = options["num_scf_steps"]
    if not isinstance(num_scf_steps, int) or isinstance(num_scf_steps, bool):
        raise TypeError("num_scf_steps must be an integer")
    if num_scf_steps <= 0:
        raise ValueError("num_scf_steps must be positive for SCF modes")

    scf_tolerance = float(options["scf_tolerance"])
    if scf_tolerance <= 0:
        raise ValueError("scf_tolerance must be positive")

    mixing_parameter = float(options["mixing_parameter"])
    if not (0 < mixing_parameter <= 1):
        raise ValueError("mixing_parameter must satisfy 0 < mixing_parameter <= 1")

    constant_charge = options["constant_charge"]
    if not isinstance(constant_charge, bool):
        raise TypeError("constant_charge must be bool")

    use_autograd_forces = options["use_autograd_forces"]
    if not isinstance(use_autograd_forces, bool):
        raise TypeError("use_autograd_forces must be bool")

    initial_density = options["initial_density"]
    if initial_density not in ("local_guess", "from_data"):
        raise ValueError("initial_density must be one of {'local_guess', 'from_data'}")

    initial_fermi_level = options["initial_fermi_level"]
    if initial_fermi_level not in ("zero", "from_data"):
        raise ValueError("initial_fermi_level must be one of {'zero', 'from_data'}")

    return FixedPointSCFOptions(
        num_scf_steps=num_scf_steps,
        scf_tolerance=scf_tolerance,
        mixing_parameter=mixing_parameter,
        constant_charge=constant_charge,
        use_autograd_forces=use_autograd_forces,
        initial_density=initial_density,
        initial_fermi_level=initial_fermi_level,
    )


def validate_fixed_point_training_options(
    raw: dict,
    loss=None,
) -> FixedPointTrainingOptions:
    flat_scf_keys = set(raw) & SCF_OPTION_KEYS
    if flat_scf_keys:
        raise ValueError("SCF options in fixed_point_training_options must be nested under scf")

    unknown_keys = set(raw) - TRAINING_OPTION_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown fixed_point_training_options key(s): {sorted(unknown_keys)}")

    if "mode" not in raw:
        raise ValueError("fixed_point_training_options must include mode")
    mode = raw["mode"]
    if mode not in FIXED_POINT_MODES:
        raise ValueError(f"mode must be one of {FIXED_POINT_MODES}, got {mode!r}")

    fixedpoint_scf_stability = (
        "fixedpoint_scf_stability" in loss if loss is not None else False
    )
    if "fixedpoint_scf_stability" in raw:
        fixedpoint_scf_stability = bool(raw["fixedpoint_scf_stability"])

    if mode == "direct":
        if raw.get("scf") is not None:
            raise ValueError("mode='direct' must not include an scf block")
        return FixedPointTrainingOptions(
            mode=mode,
            scf=None,
            linear_solve=raw.get("linear_solve", "inverse"),
            fixedpoint_scf_stability=fixedpoint_scf_stability,
        )

    if mode == "implicit":
        try:
            import torchopt  # noqa: F401
        except ImportError as exc:
            raise ImportError("torchopt 0.7.3 is required for implicit mode.") from exc

    if "scf" not in raw:
        raise ValueError(f"fixed_point_training_options.scf must be set for mode={mode!r}")
    if not isinstance(raw["scf"], dict):
        raise TypeError("fixed_point_training_options.scf must be a dict")

    missing_required = REQUIRED_TRAINING_SCF_KEYS - set(raw["scf"])
    if missing_required:
        raise ValueError(
            "fixed_point_training_options.scf must explicitly set "
            f"{sorted(missing_required)}"
        )

    scf_options = validate_fixed_point_scf_options(raw["scf"], FIXED_POINT_SCF_DEFAULTS)

    return FixedPointTrainingOptions(
        mode=mode,
        scf=scf_options,
        linear_solve=raw.get("linear_solve", "inverse"),
        fixedpoint_scf_stability=fixedpoint_scf_stability,
    )


def convert_legacy_scf_training_options(raw: dict) -> dict:
    flat_scf = {key: raw[key] for key in SCF_OPTION_KEYS if key in raw}
    nested_scf = raw.get("scf")
    if nested_scf is not None and flat_scf:
        raise ValueError(
            "Use either nested scf_training_options.scf or old flat SCF keys, not both."
        )

    canonical = {
        key: value
        for key, value in raw.items()
        if key not in SCF_OPTION_KEYS and key != "scf"
    }

    if canonical.get("mode") == "direct":
        return canonical

    if nested_scf is not None:
        canonical["scf"] = nested_scf
    elif flat_scf:
        canonical["scf"] = flat_scf
    return canonical


def fixed_point_training_options_from_stage(stage: dict) -> FixedPointTrainingOptions:
    has_new = "fixed_point_training_options" in stage
    has_old = "scf_training_options" in stage
    if has_new and has_old:
        raise ValueError(
            "Use only one of fixed_point_training_options or deprecated "
            "scf_training_options."
        )
    if has_new:
        return validate_fixed_point_training_options(
            stage["fixed_point_training_options"],
            loss=stage.get("loss"),
        )
    if has_old:
        _warn_deprecated(
            "scf_training_options is deprecated for FixedPoint; use "
            "fixed_point_training_options."
        )
        return validate_fixed_point_training_options(
            convert_legacy_scf_training_options(stage["scf_training_options"]),
            loss=stage.get("loss"),
        )
    raise ValueError("fixed_point_training_options must be set for model=FixedPoint")
