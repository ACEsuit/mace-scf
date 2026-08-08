import ast
import json
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import asdict, is_dataclass
import types

import numpy as np
from mace_scf.utils.load_data import load_train_valid_sets_from_xyz
import torch
import torch.nn.functional
from torch_ema import ExponentialMovingAverage
from torch.nn.parallel import DistributedDataParallel as DDP


from mace.tools import torch_geometric, init_wandb

# do not make mace_scf things called tools and modules, then its easy.
from mace import tools
import mace.modules
from mace.tools.scripts_utils import LRScheduler

# mace_scf replaces data, and some utils
import mace_scf.data
import mace_scf.utils

from mace_scf import electrostatics
from mace_scf.utils.check_args import check_config_conflicts
import mace_scf.utils.run_train_utils
from mace.tools.slurm_distributed import DistributedEnvironment


import warnings
import traceback


def make_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return make_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(make_jsonable(key)): make_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class NoOpLRScheduler:
    """Checkpoint-compatible scheduler used when scheduling is deliberately disabled."""

    def step(self, metrics=None, epoch=None):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


def build_lr_scheduler(optimizer, args):
    if args.optimizer == "schedulefree":
        logging.info(
            "ScheduleFree optimizer selected; LR scheduler options are ignored."
        )
        return NoOpLRScheduler()
    return LRScheduler(optimizer, args)


def main() -> None:
    args = mace_scf.utils.extended_arg_parser().parse_args()
    check_config_conflicts(args)
    tag = tools.get_tag(name=args.name, seed=args.seed)

    if args.distributed:
        try:
            distr_env = DistributedEnvironment()
        except Exception as e:
            raise ValueError(
                f"Error specifying environment for distributed training: {e}"
            )
        world_size = distr_env.world_size
        local_rank = distr_env.local_rank
        rank = distr_env.rank
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print(distr_env)
    else:
        rank = int(0)

    # Setup, logging and seeds
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=rank)
    if args.distributed:
        logging.info(f"Process group initialized: {torch.distributed.is_initialized()}")
        logging.info(f"Processes: {world_size}")
    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)
    logging.info(f"Configuration: {args}")

    # choices to simply this repo:
    compute_energy = True
    assert args.compute_forces
    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": False,
        "stress": args.compute_stress,
        "polarizability": args.compute_polarizability,
    }

    args.key_specification = mace.data.KeySpecification()
    head_dict = next(iter(args.heads.values()))
    args.key_specification.update(
        info_keys=head_dict.get("info_keys", {}),
        arrays_keys=head_dict.get("arrays_keys", {}),
    )
    logging.info("Using the key specifications to parse data:")
    logging.info(args.key_specification)

    # Data preparation
    train_set, valid_set, z_table, atomic_energies, test_collections = load_train_valid_sets_from_xyz(args, args.config_type_weights)
    logging.info(f"Atomic energies: {atomic_energies.tolist()}")
    
    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=valid_set,
        batch_size=args.valid_batch_size,
        sampler=valid_sampler,
        shuffle=False, #(valid_sampler is None),
        drop_last=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    if args.compute_avg_num_neighbors:
        args.avg_num_neighbors = mace.modules.compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {args.avg_num_neighbors}")
    
    atomic_charges = mace_scf.utils.run_train_utils.get_formal_charges(
        args.model, 
        args.formal_charges_from_data, 
        args.atomic_formal_charges, 
        z_table
    )

    # normalization for field feartures
    args.fermi_level_offset = mace_scf.utils.run_train_utils.get_fermi_level_offset(
        train_loader,
        args,
        device,
    )
    logging.info(f"using Fermi level offset {args.fermi_level_offset}")
    args.field_feature_norms = mace_scf.utils.run_train_utils.get_field_feature_norms(
        train_loader,
        args,
        device,
        fermi_level_offset=args.fermi_level_offset,
    )
    logging.info(f"using normalization for field features {args.field_feature_norms}")
    args.atom_density_scaling = mace_scf.utils.run_train_utils.get_atom_density_scaling(
        train_loader,
        args,
        device,
        z_table=z_table,
    )
    logging.info(f"using atom density scaling factors {args.atom_density_scaling}")

    # Build model
    logging.info("Building model")
    logging.info(f"Hidden irreps: {args.hidden_irreps}")
    model = mace_scf.utils.run_train_utils.build_model(
        args=args,
        z_table=z_table,
        atomic_energies=atomic_energies,
        atomic_charges=atomic_charges,
        train_loader=train_loader,
    ).to(device)

    # Optimizer
    param_options = mace_scf.utils.run_train_utils.get_param_options(model, args)
    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    elif args.optimizer == "schedulefree":
        try:
            from schedulefree import adamw_schedulefree
        except ImportError as exc:
            raise ImportError(
                "`schedulefree` is not installed. Please install it via `pip install schedulefree` or `pip install mace-torch[schedulefree]`"
            ) from exc
        _param_options = {k: v for k, v in param_options.items() if (k != "amsgrad" and k != "betas")}
        _param_options["betas"] = (param_options["betas"][0], param_options["betas"][1])
        optimizer = adamw_schedulefree.AdamWScheduleFree(**_param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)
    
    lr_scheduler = build_lr_scheduler(optimizer, args)

    assert not "batch_positions" in dict(model.named_parameters()), "batch_positions should not be a parameter of the model"

    # find most recent epoch
    start_epoch = 0
    if args.restart_latest:
        for stage_number, train_stage in enumerate(args.train_schedule):
            checkpoint_handler_stage = tools.CheckpointHandler(
                directory=args.checkpoints_dir,
                tag=tag+"_"+train_stage["name"],
                keep=args.keep_checkpoints,
            )
            latest_checkpoint_epoch = checkpoint_handler_stage.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
            if latest_checkpoint_epoch is not None:
                start_epoch = latest_checkpoint_epoch
            else:
                break
    else:
        logging.info("restart_latest is False; starting from initialized model and optimizer.")

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    if rank == 0 and args.wandb:
        logging.info("Using Weights and Biases for logging")
        import wandb

        wandb_config = {}
        args_dict = vars(args)

        args_dict_json = json.dumps(make_jsonable(args_dict))
        for key in args.wandb_log_hypers:
            wandb_config[key] = make_jsonable(args_dict[key])
        init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=wandb_config,
            directory=args.wandb_dir,
        )
        wandb.run.summary["params"] = args_dict_json
        if args.wandb_watch in ("gradients", "all") and args.wandb_watch_log_freq is not None:
            # W&B parameter watching uses top-level forward hooks, which do not
            # fire for FixedPoint wrappers because they call model submethods
            # directly. Parameters are logged manually in the training loop.
            wandb.watch(
                model,
                log="gradients",
                log_freq=args.wandb_watch_log_freq,
            )
    

    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    logging.info(f"Optimizer: {optimizer}")


    # for xyzs, option to log performance on test sets during training
    test_data_loaders = None
    if args.log_on_test_sets:
        if not(args.test_file is not None and args.test_file.endswith(".xyz")):
            raise ValueError("logging on test tests obly availble for and xyz test file")

        test_sets = {}
        test_data_loaders = {}

        for name, subset in test_collections:
            test_sets[name] = [
                mace_scf.data.ExtAtomicData.from_config(
                    config, 
                    z_table=z_table, 
                    cutoff=args.r_max, 
                    atomic_multipoles_max_l=args.atomic_multipoles_max_l
                )
                for config in subset
            ]
        
        for test_name, test_set in test_sets.items():
            if len(test_set) == 0:
                continue
            test_loader = torch_geometric.dataloader.DataLoader(
                test_set,
                batch_size=args.valid_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            test_data_loaders[test_name] = test_loader

    # training loop
    for stage_index, train_stage in enumerate(args.train_schedule):
        if train_stage["end"] <= start_epoch:
            continue
        else:
            start_epoch = train_stage["start"]

        stage_name = train_stage["name"]
        loss_fn = mace_scf.electrostatics.loss.WeightedLoss(train_stage["loss"])
        logging.info(loss_fn)
        model_eval_module = mace_scf.utils.make_model_wrapper(
            model=model,
            optimizer=optimizer,
            output_args=output_args,
            fixed_point_training_options=train_stage.get("fixed_point_training_options"),
        )

        if args.distributed:
            model_eval_module = DDP(model_eval_module, device_ids=[local_rank])

        logging.info(f"Setting global learning rate to {train_stage['lr']}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = train_stage["lr"]
            if param_group["name"]+"_lr" in train_stage:
                new_lr = train_stage[param_group["name"]+"_lr"]
                logging.info(f"Setting learning rate to {new_lr} for {param_group['name']}")
                param_group["lr"] = new_lr
        logging.debug(f"Optimizer: {optimizer}")

        checkpoint_handler_stage = tools.CheckpointHandler(
            directory=args.checkpoints_dir,
            tag=tag+"_"+stage_name,
            keep=args.keep_checkpoints,
        )
        if args.restart_latest:
            latest_checkpoint_epoch = checkpoint_handler_stage.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
            if latest_checkpoint_epoch is not None:
                start_epoch = latest_checkpoint_epoch

        # train_loader = get_data_loader(train_dataset, train_stage["fixed_point_training_options"])

        mace_scf.utils.train(  # this is the mace_scf train
            model=model,
            model_eval_wrapper=model_eval_module,
            loss_fn=loss_fn,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loaders=test_data_loaders,
            optimizer=optimizer,
            ema=ema,
            lr_scheduler=lr_scheduler,
            checkpoint_handler=checkpoint_handler_stage,
            eval_interval=args.eval_interval,
            start_epoch=start_epoch,
            end_epoch=train_stage["end"],
            logger=logger,
            patience=args.patience,
            save_all_checkpoints=args.save_all_checkpoints,
            device=device,
            rank=rank,
            train_sampler=train_sampler,
            max_grad_norm=args.clip_grad,
            log_errors=args.error_table,
            log_wandb=args.wandb,
            debug_log_grad_summary=args.debug_log_grad_summary,
            debug_grad_log_frequency=args.wandb_watch_log_freq,
            wandb_watch=args.wandb_watch,
        )

    # Evaluation on test datasets
    logging.info("Computing metrics for training, validation, and test sets")

    # generate the test dataloader
    test_sets = {}
    test_data_loaders = {}

    if args.test_file is not None and args.test_file.endswith(".xyz"):
        for name, subset in test_collections:
            test_sets[name] = [
                mace_scf.data.ExtAtomicData.from_config(
                    config, 
                    z_table=z_table, 
                    cutoff=args.r_max, 
                    atomic_multipoles_max_l=args.atomic_multipoles_max_l
                )
                for config in subset
            ]
    
    for test_name, test_set in test_sets.items():
        if len(test_set) == 0:
            continue
        test_sampler = None
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
        try:
            drop_last = test_set.drop_last
        except AttributeError as e:  # pylint: disable=W0612
            drop_last = False
        test_loader = torch_geometric.dataloader.DataLoader(
            test_set,
            batch_size=args.valid_batch_size,
            shuffle=(test_sampler is None),
            drop_last=drop_last,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        test_data_loaders[test_name] = test_loader
    
    all_data_loaders = {
        "train": train_loader,
        "valid": valid_loader,
    }
    all_data_loaders.update(test_data_loaders)
    
    logging.info("Computing final error tables and saving models")
    stages_with_models = []
    for train_stage in args.train_schedule:
        stage_name = train_stage["name"]
        stage_tag = tag + "_" + stage_name
        loss_fn = mace_scf.electrostatics.loss.WeightedLoss(train_stage["loss"])
        model_eval_module = mace_scf.utils.make_model_wrapper(
            model=model,
            optimizer=optimizer,
            output_args=output_args,
            fixed_point_training_options=train_stage.get("fixed_point_training_options"),
        )
        checkpoint_handler_stage = tools.CheckpointHandler(
            directory=args.checkpoints_dir,
            tag=stage_tag,
            keep=args.keep_checkpoints,
        )
        latest_checkpoint_epoch = checkpoint_handler_stage.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=False,
            device=device,
        )
        if latest_checkpoint_epoch is None:
            logging.warning(f"No model found for stage {stage_name}")
            continue
        logging.info(f"Loaded model from epoch {latest_checkpoint_epoch}")
        model_eval_module.to(device)

        for param in model_eval_module.model.parameters(): 
            # DDP requires that (at least one) model parameter(s) have requires_grad=True,
            # but we don't want to compute gradients during evaluation.
            param.requires_grad = True
        if args.distributed:
            model_eval_module = DDP(model_eval_module, device_ids=[local_rank])
        for param in model_eval_module.parameters():
            param.requires_grad = False
        assert not "batch_positions" in dict(model.named_parameters()), "batch_positions should not be a parameter of the model"

        table = mace_scf.utils.create_error_table(
            table_type=args.error_table,
            all_data_loaders=all_data_loaders,
            model=model,
            model_eval_wrapper=model_eval_module,
            loss_fn=loss_fn,
            log_wandb=args.wandb,
            device=device,
            distributed=args.distributed,
        )
        if rank == 0:
            logging.info("\nFinal error table for stage %s", stage_name)
            logging.info("\n" + str(table))

            # Save entire model before running any extra SCF convergence diagnostics.
            model_path = Path(args.checkpoints_dir) / (stage_tag + ".model")
            logging.info(f"Saving model to {model_path}")
            if args.save_cpu:
                model = model.to("cpu")
            torch.save(model, model_path)
            torch.save(model, Path(args.model_dir) / (args.name + "_" + stage_name + ".model"))
            stages_with_models.append((train_stage, stage_tag))

    if rank == 0 and args.scf_convergence_summary:
        logging.info("Computing SCF convergence summaries")
        for train_stage, stage_tag in stages_with_models:
            stage_name = train_stage["name"]
            checkpoint_handler_stage = tools.CheckpointHandler(
                directory=args.checkpoints_dir,
                tag=stage_tag,
                keep=args.keep_checkpoints,
            )
            latest_checkpoint_epoch = checkpoint_handler_stage.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
            if latest_checkpoint_epoch is None:
                logging.warning(
                    f"No model found for SCF convergence summary stage {stage_name}"
                )
                continue
            logging.info(
                "Loaded model from epoch %s for SCF convergence summary stage %s",
                latest_checkpoint_epoch,
                stage_name,
            )
            model.to(device)
            if not mace_scf.utils.is_fixed_point_model(model):
                continue
            # TODO: implement DDP for this:
            scf_summary = mace_scf.utils.create_scf_convergence_summary(
                model=model,
                all_data_loaders=all_data_loaders,
                output_args=output_args,
                device=device,
                train_stage=train_stage,
                error_table_type=args.error_table,
                diagnostic_batch_size=args.valid_batch_size,
            )
            logging.info("\n" + scf_summary)

    logging.info("Done")


if __name__ == "__main__":
    main()
