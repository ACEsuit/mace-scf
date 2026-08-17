import argparse
from typing import Optional
from mace.tools import build_default_arg_parser, build_preprocess_arg_parser


def strict_str2bool(value):
    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower == "true":
        return True
    if value_lower == "false":
        return False
    raise argparse.ArgumentTypeError("Expected 'True' or 'False'")


# some options need to have choices cleared
def remove_options(parser, options):
    for option in options:
        for action in parser._actions:
            if vars(action)['option_strings'][0] == option:
                parser._handle_conflict_resolve(None,[(option,action)])
                break


def preprocess_extended_arg_parser() -> argparse.ArgumentParser:
    parser = build_preprocess_arg_parser()
    # overwrite all key defaults to None - mace_scf only uses the config file for keys
    for item in parser._actions:
        if item.option_strings[0][-4:] == "_key":
            item.default = None
    return parser


def extended_arg_parser() -> argparse.ArgumentParser:
    parser = build_default_arg_parser()

    # new arguments
    parser.add_argument(
        "--atomic_multipoles_key",
        help="Key of density coefficients_key in training xyz",
        type=str,
        default="density_coefficients",
    )
    # new arguments
    parser.add_argument(
        "--electrostatic_potentials_key",
        help="Key of local electrostatic potentials in training xyz",
        type=str,
        default="electrostatic_potentials",
    )
    parser.add_argument(
        "--external_field_key",
        type=str,
        default="external_field",
    )
    parser.add_argument(
        "--fermi_level_key",
        type=str,
        default="fermi_level",
    )
    parser.add_argument(
        "--include_local_electron_energy",
        type=strict_str2bool,
        default=True
    )
    parser.add_argument("--atomic_multipoles_max_l", type=int, default=1)
    parser.add_argument("--atomic_multipoles_smearing_width", type=float, default=1.5)
    parser.add_argument("--kspace_cutoff_factor", type=float, default=1.0)
    parser.add_argument("--atomic_formal_charges", type=str, default="{}")
    parser.add_argument(
        "--formal_charges_from_data", action="store_true", default=False
    )
    parser.add_argument("--field_feature_max_l", type=int, default=1)
    parser.add_argument("--field_feature_widths", type=str, default="[1.0, 2.0]")
    parser.add_argument(
        "--dont_include_pbc_corrections", action="store_true"
    )
    parser.add_argument(
        "--include_electrostatic_self_interaction", type=strict_str2bool, default=True
    )
    parser.add_argument(
        "--valid_set_seed", type=int, default=None
    )
    parser.add_argument(
        "--quadrupole_feature_corrections", action="store_true", default=False
    )
    parser.add_argument(
        "--electrostatic_pbc_method",
        type=str,
        default="mixed_periodic",
        choices=[
            "realspace",
            "pbc",
            "slab",
            "molecule_in_box",
            "mixed_periodic",
            "auto",
        ],
    )
    parser.add_argument(
        "--static_bond_transfer_block", 
        help="Block to use for static bond transfer",
        type=str, 
        default="OxidationDependentSymmetricPredictionSourceBlock"
    )
    parser.add_argument(
        "--oxidation_state_mixer", 
        help="Block to use for oxidation state embedding",
        type=str, 
        default="SumOxidationStateMixer"
    )
    parser.add_argument(
        "--return_electrostatic_potentials", action="store_true", default=False
    )
    parser.add_argument(
        "--train_schedule", required=True
    )
    parser.add_argument(
        "--fixedpoint_update_config", type=str, default=None,
    )
    parser.add_argument(
        "--field_readout_config", type=str, default=None,
    )
    parser.add_argument(
        "--field_block_weight_decay", type=float, default=5e-7
    )
    parser.add_argument(
        "--local_charges_weight_decay", type=float, default=5e-7
    )
    parser.add_argument(
        "--field_feature_norms",
        type=str,
        default="average"
    )
    parser.add_argument(
        "--fermi_level_offset",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--field_norm_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--oxidation_state_range",
        type=str,
        default="(-6.0, 6.0)",
    )
    parser.add_argument(
        "--polarizable_charges",
        action="store_true",
        default=False
    )
    
    parser.add_argument(
        "--beta_two",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--log_on_test_sets",
        action='store_true'
    )
    parser.add_argument(
        "--scf-convergence-summary",
        action="store_true",
        default=False,
        help="perform a post-training fixed-point SCF convergence summary.",
    )
    parser.add_argument(
        "--use_linear_final_readout",
        action='store_true'
    )
    parser.add_argument(
        "--include_field_si",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--use_linear_local_charges",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--atom_density_scaling",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--allow_low_density_pbc",
        action="store_true",
        default=False,
        help="Allow fully periodic low-density configurations that look like clusters in large cells.",
    )
    parser.add_argument(
        "--override_pbc_checks",
        action="store_true",
        default=False,
        help="Skip the check that each config's pbc is compatible with --electrostatic_pbc_method.",
    )
    parser.add_argument(
        "--low_density_pbc_max_volume_per_atom",
        type=float,
        default=100.0,
        help="Maximum volume per atom for pbc=TTT configs before treating them as suspicious.",
    )
    parser.add_argument(
        "--wandb-watch",
        choices=["off", "gradients", "parameters", "all"],
        default="off",
        help="Control wandb.watch parameter/gradient histogram logging.",
    )
    parser.add_argument(
        "--wandb-watch-log-freq",
        type=int,
        default=50,
        help=(
            "Optimization-step frequency for wandb.watch and gradient summary "
            "logging. If unset, neither is logged."
        ),
    )
    parser.add_argument(
        "--debug-log-grad-summary",
        action="store_true",
        default=False,
        help="Log scalar parameter and gradient summaries during optimization.",
    )
    parser.add_argument(
        "--log-all-ranks",
        action="store_true",
        default=False,
        help=(
            "In distributed training, write each rank's log records to its own "
            "'{tag}_rank{rank}.log'/'{tag}_rank{rank}_debug.log' file under "
            "--log_dir, instead of only logging on rank 0."
        ),
    )
    parser.add_argument(
        "--fixedpoint-initial-charge-head-scale",
        type=float,
        default=0.01,
        help=(
            "Multiply initial FixedPoint charge-head weights by this factor after "
            "model construction. Set to 1.0 to disable the rescaling."
        ),
    )

    remove_options(
        parser,
        [
            "--loss",
            "--error_table",
            "--model",
            "--heads",
            "--optimizer",
            "--compute_polarizability",
            "--compute_atomic_dipole",
            "--restart_latest",
        ],
    )
    parser.add_argument("--heads", required=True)
    parser.add_argument("--error_table", required=True)
    parser.add_argument(
        "--compute_polarizability",
        help="Select True to compute polarizability",
        type=strict_str2bool,
        default=True,
    )
    parser.add_argument(
        "--restart_latest",
        help="restart optimizer from latest checkpoint",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--model",
        choices=[
            "ScaleShiftMACE",
            "MACE",
            "LocalSplitCharges",
            "FixedPoint",
            "LocalCharges",
            "FixedChargeBaselinedMACE",
            "MACEQEq" #added model for qeq
        ],
        type=str,
        default="MACE",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "schedulefree"],
        type=str,
        default="schedulefree",
    )

    # QEq arguments
    parser.add_argument(
        "--qeq_charges",
        help="How self-consistent charges are computed for MaceQEq",
        type=str,
        choices=["autograd"],
        default="autograd",
    )
    parser.add_argument(
        "--train_hardness",
        action="store_true",
        help="If hardness should be trainable or not",
        default=False,
    )
    parser.add_argument(
        "--read_enegs",
        action="store_true",
        help=(
            "If electronegativities should be read from data and used as starting point"
        ),
        default=False,
    )
    parser.add_argument(
        "--read_hardness",
        action="store_true",
        help=(
            "If hardness parameters should be read from data and used as starting "
            "point or nontrainable hyperparameters"
        ),
        default=False,
    )
    parser.add_argument(
        "--enegs_key",
        help="Key of enegs in training xyz",
        type=str,
        default="enegs",
    )
    parser.add_argument(
        "--default_hardness",
        help="Default hardness value used when hardness is not read or trained",
        type=float,
        default=2.0,
    )

    # overwrite all key defaults to None
    for item in parser._actions:
        if item.option_strings[0][-4:] == "_key":
            item.default = None

    return parser
