import argparse
import ast
import logging
import os
from e3nn import o3
from mace_scf.electrostatics import field_blocks
from mace_scf.electrostatics.fixed_point_options import (
    fixed_point_training_options_from_stage,
)

try:
    import configargparse
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("configargparse is required for using mace via mace_scf.")


def check_config_conflicts(args: argparse.Namespace):
    fill_default_dirs(args)
    check_and_fix_heads(args)
    check_and_fix_train_schedule(args)
    compute_and_fill_irreps(args)
    check_train_test_files(args)
    check_unsupported_training_options(args)
    fill_fixedpoint_update_config(args)
    fill_field_readout_config(args)

    if args.model == "FixedPoint":
        for train_stage in args.train_schedule:
            train_stage["fixed_point_training_options"] = (
                fixed_point_training_options_from_stage(train_stage)
            )
            train_stage.pop("scf_training_options", None)
            if args.distributed and train_stage["fixed_point_training_options"].mode == "implicit":
                raise NotImplementedError(
                    "fixed-point mode 'implicit' is not tested with "
                    "--distributed."
                )
    else:
        for train_stage in args.train_schedule:
            assert "scf_training_options" not in train_stage, f"scf_training_options should not be set for model={args.model}"
            assert "fixed_point_training_options" not in train_stage, f"fixed_point_training_options should not be set for model={args.model}"

    # small things
    if args.field_feature_max_l is None:
        args.field_feature_max_l = args.atomic_multipoles_max_l
    if args.valid_set_seed is None:
        args.valid_set_seed = args.seed
    if args.wandb_watch_log_freq is not None and args.wandb_watch_log_freq < 1:
        raise ValueError("wandb_watch_log_freq must be a positive integer")
    
    args.config_type_weights = set_configfigtype_weights(args.config_type_weights)


def check_and_fix_heads(args: argparse.Namespace):
    args.heads = ast.literal_eval(args.heads)
    assert type(args.heads) == dict and len(args.heads) == 1, print(args.heads)
    
    # assert people are only using the info_keys and arrays_keys syntax
    thedict = [value for value in args.heads.values()][0]
    assert len(thedict) == 2
    assert 'info_keys' in thedict
    assert 'arrays_keys' in thedict

    if not "total_charge" in thedict["info_keys"]:
        raise ValueError("info keys (config_yaml/info_keys) must include total_charge key. If all configs have zero charge, set total_charge: total_charge.")

    for key, value in vars(args).items():
        if key[-4:] == "_key":
            assert value is None, "keys can only be specified in the configfile dictionaries"

    # patch for annoying stuff:
    if "stress" not in thedict["info_keys"]:
        thedict["info_keys"]["stress"] = "none"
    if "energy" not in thedict["info_keys"]:
        thedict["info_keys"]["energy"] = "none"
    if "forces" not in thedict["arrays_keys"]:
        thedict["arrays_keys"]["forces"] = "none"
    # must be distinct
    if "head" not in thedict["info_keys"]:
        thedict["info_keys"]["head"] = "head"


def check_and_fix_train_schedule(args: argparse.Namespace):
    args.train_schedule = ast.literal_eval(args.train_schedule)
    assert type(args.train_schedule) == dict

    keys = list(args.train_schedule.keys())
    assert keys == list(range(len(keys)))
    train_schedule_list = [args.train_schedule[index] for index in keys]

    current_epoch = 0
    for train_stage in train_schedule_list:
        fill_default_train_settings(train_stage)
        assert train_stage["start"] == current_epoch, "start of one train stage must be greater than end of previous stage"
        current_epoch = train_stage["end"] + 1

    args.train_schedule = train_schedule_list


def fill_default_train_settings(train_stage_dict):
    required_settings = ["start", "end", "loss", "name"]
    default_optional_settings = {
        "lr": 0.01,
    }
    for key in required_settings:
        assert key in train_stage_dict, f"Missing key {key} in train_stage_dict"
    for key, value in default_optional_settings.items():
        if key not in train_stage_dict:
            train_stage_dict[key] = value

    loss_dict = {}
    for key, val in train_stage_dict["loss"].items():
        if not (type(val) in [float, dict]):
            raise TypeError(f"loss {key} must be followed by a weight (float) or a dictionary of options")
        if type(val) == dict:
            assert "weight" in val, "loss dictionary must contain a weight"
        else:
            val = {"weight": val}
        loss_dict[key] = val
    train_stage_dict["loss"] = loss_dict



def fill_default_dirs(args: argparse.Namespace):
    # set default dirs
    if args.log_dir is None:
        args.log_dir = os.path.join(args.work_dir, "logs")
    if args.model_dir is None:
        args.model_dir = args.work_dir
    if args.checkpoints_dir is None:
        args.checkpoints_dir = os.path.join(args.work_dir, "checkpoints")
    if args.results_dir is None:
        args.results_dir = os.path.join(args.work_dir, "results")
    if args.downloads_dir is None:
        args.downloads_dir = os.path.join(args.work_dir, "downloads")


def compute_and_fill_irreps(args):
    if args.num_channels is not None and args.max_L is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )
    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"


def set_configfigtype_weights(config_type_weights_str):
    try:
        config_type_weights = ast.literal_eval(config_type_weights_str)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}
    return config_type_weights


def check_train_test_files(args):
    if args.train_file is None or not args.train_file.endswith(".xyz"):
        raise ValueError("Only .xyz train_file inputs are supported in this repo right now")
    if args.valid_file is not None and not args.valid_file.endswith(".xyz"):
        raise ValueError("Only .xyz valid_file inputs are supported in this repo right now")
    if args.test_file is not None and not args.test_file.endswith(".xyz"):
        raise ValueError("Only .xyz test_file inputs are supported in this repo right now")
    if args.test_dir is not None:
        raise ValueError("test_dir HDF5/sharded test inputs are not supported in this repo right now")


def check_unsupported_training_options(args):
    if args.statistics_file is not None:
        raise ValueError(
            "statistics_file is not supported in this repo right now. "
            "Pass r_max, E0s, and avg_num_neighbors directly, or use "
            "--compute_avg_num_neighbors."
        )


def fill_fixedpoint_update_config(args):
    if args.fixedpoint_update_config is None:
        args.fixedpoint_update_config = {
            "type": "OneBodyVariableUpdate",
            "potential_embedding_cls": "BiasedLinearPotentialEmbedding",
            "nonlinearity_cls": "NoNonLinearity",
        }
    else:
        args.fixedpoint_update_config = ast.literal_eval(args.fixedpoint_update_config)
        assert "type" in args.fixedpoint_update_config

    cls_variables = [
        "type",
        "potential_embedding_cls",
        "central_atom_mixer_cls",
        "central_atom_feats_mixer_cls",
        "interaction_cls",
        "nonlinearity_cls",
    ]
    for key, value in args.fixedpoint_update_config.items():
        if key in cls_variables:
            args.fixedpoint_update_config[key] = getattr(field_blocks, value)


def fill_field_readout_config(args):
    if args.field_readout_config is None:
        args.field_readout_config = {"type": "StrictQuadraticFieldEnergyReadout"}
    else:
        args.field_readout_config = ast.literal_eval(args.field_readout_config)
        assert "type" in args.field_readout_config

    cls_variables = ["type"] + [item for item in args.field_readout_config.keys() if item.endswith("_cls")]
    for key, value in args.field_readout_config.items():
        if key in cls_variables:
            args.field_readout_config[key] = getattr(field_blocks, value)
