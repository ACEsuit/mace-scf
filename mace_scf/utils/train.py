import dataclasses
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from mace.tools import torch_geometric
from mace.tools.checkpoint import CheckpointHandler, CheckpointState
from mace.tools.torch_tools import tensor_dict_to_device, to_numpy
from mace.tools.utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)
import os
from mace.tools.scatter import scatter_sum


def _should_log_grad_summary(opt_step: int, frequency: Optional[int]) -> bool:
    return frequency is not None and frequency > 0 and opt_step % frequency == 0


def _scalar_param_grad_summary(model: torch.nn.Module) -> Dict[str, Any]:
    param_norm_sq = 0.0
    grad_norm_sq = 0.0
    grad_max_abs = 0.0
    nonfinite_grad_count = 0
    none_grad_param_count = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == "batch_positions":
                continue
            param_detached = param.detach()
            param_norm_sq += float(torch.sum(param_detached * param_detached).item())

            grad = param.grad
            if grad is None:
                none_grad_param_count += 1
                continue

            grad_detached = grad.detach()
            finite_mask = torch.isfinite(grad_detached)
            nonfinite_grad_count += int((~finite_mask).sum().item())
            if torch.any(finite_mask):
                finite_grad = grad_detached[finite_mask]
                grad_norm_sq += float(torch.sum(finite_grad * finite_grad).item())
                grad_max_abs = max(
                    grad_max_abs,
                    float(torch.max(torch.abs(finite_grad)).item()),
                )

    param_norm = param_norm_sq**0.5
    grad_norm = grad_norm_sq**0.5
    grad_param_norm_ratio = grad_norm / param_norm if param_norm > 0.0 else np.nan
    return {
        "grad/global_norm": grad_norm,
        "grad/global_max_abs": grad_max_abs,
        "grad/nonfinite_count": nonfinite_grad_count,
        "grad/none_param_count": none_grad_param_count,
        "param/global_norm": param_norm,
        "grad/param_norm_ratio": grad_param_norm_ratio,
    }


def _log_wandb_parameter_histograms(model: torch.nn.Module) -> None:
    import wandb

    histograms = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == "batch_positions":
                continue
            histograms[f"parameters/{name}"] = wandb.Histogram(
                param.detach().cpu().flatten()
            )
    if histograms:
        wandb.log(histograms)


def train(
    model: torch.nn.Module, # TODO: remove?
    model_eval_wrapper,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    end_epoch: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    device: torch.device,
    log_errors: str,
    rank: int,
    save_all_checkpoints: bool = False,
    train_sampler: Optional[DistributedSampler] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    test_loaders: Optional[dict] = None,
    debug_log_grad_summary: bool = False,
    debug_grad_log_frequency: Optional[int] = None,
    wandb_watch: str = "off",
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    keep_last = False

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    epoch = start_epoch
    opt_step = 0
    while epoch <= end_epoch:
        if epoch > start_epoch:
            lr_scheduler.step(metrics=valid_loss)

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()

        for batch in train_loader:
            _, opt_metrics = take_step(
                model=model, # TODO: remove?
                model_eval_wrapper=model_eval_wrapper,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                max_grad_norm=max_grad_norm,
                device=device,
                debug_log_grad_summary=debug_log_grad_summary,
                debug_grad_log_frequency=debug_grad_log_frequency,
                opt_step=opt_step,
            )
            if rank == 0:
                opt_metrics["mode"] = "opt"
                opt_metrics["epoch"] = epoch
                opt_metrics["opt_step"] = opt_step
                logger.log(opt_metrics)
                if log_wandb and debug_log_grad_summary and _should_log_grad_summary(
                    opt_step, debug_grad_log_frequency
                ):
                    import wandb

                    wandb.log(opt_metrics)
                if (
                    log_wandb
                    and wandb_watch in ("parameters", "all")
                    and _should_log_grad_summary(opt_step, debug_grad_log_frequency)
                ):
                    # W&B's native parameter watcher is a top-level forward hook.
                    # FixedPoint training calls model submethods directly, so log
                    # parameter histograms explicitly instead of relying on forward().
                    # TODO: check this works with DDP
                    _log_wandb_parameter_histograms(model)
            opt_step += 1
        if train_sampler is not None:
            torch.distributed.barrier()

        # Validate
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.eval()
        if epoch % eval_interval == 0:
            valid_loss, eval_metrics = evaluate(
                model=model, # TODO: remove?
                model_eval_wrapper=model_eval_wrapper,
                loss_fn=loss_fn,
                ema=ema,
                data_loader=valid_loader,
                device=device,
            )
            eval_metrics["mode"] = "eval"
            eval_metrics["epoch"] = epoch
            logger.log(eval_metrics)
            if test_loaders is not None:
                for name, loader in test_loaders.items():
                    _, test_eval_metrics = evaluate(
                        model=model, # TODO: remove?
                        model_eval_wrapper=model_eval_wrapper,
                        loss_fn=loss_fn,
                        ema=ema,
                        data_loader=loader,
                        device=device,
                    )
                    test_eval_metrics["epoch"] = epoch
                    test_eval_metrics["mode"] = "eval_test"
                    test_eval_metrics["test_name"] = name
                    logger.log(test_eval_metrics)
            
            if rank == 0:
                valid_err_log(
                    valid_loss,
                    eval_metrics,
                    logger,
                    log_errors,
                    epoch,
                )

                if log_wandb:
                    import wandb
                    wandb_log_dict = {
                        "epoch": epoch,
                        "valid_loss": valid_loss,
                        "valid_rmse_e_per_atom": eval_metrics["rmse_e_per_atom"],
                        "valid_rmse_f": eval_metrics["rmse_f"],
                        "valid_mae_f": eval_metrics["mae_f"],
                    }
                    if "rmse_dma" in eval_metrics:
                        wandb_log_dict["valid_rmse_dma"] = eval_metrics["rmse_dma"]
                        wandb_log_dict["valid_mae_dma"] = eval_metrics["mae_dma"]
                    if "rmse_mu_per_atom" in eval_metrics:
                        wandb_log_dict["valid_rmse_mu_per_atom"] = eval_metrics["rmse_mu_per_atom"]
                        wandb_log_dict["valid_mae_mu_per_atom"] = eval_metrics["mae_mu_per_atom"]
                    if "mae_total_charge" in eval_metrics:
                        wandb_log_dict["valid_mae_total_charge"] = eval_metrics["mae_total_charge"]
                        wandb_log_dict["valid_rmse_total_charge"] = eval_metrics["rmse_total_charge"]
                    if "mae_fermi_level" in eval_metrics:
                        wandb_log_dict["valid_mae_fermi_level"] = eval_metrics["mae_fermi_level"]
                        wandb_log_dict["valid_rmse_fermi_level"] = eval_metrics["rmse_fermi_level"]
                    
                    wandb.log(wandb_log_dict)
                if valid_loss >= lowest_loss:
                    if save_all_checkpoints:
                        if ema is not None:
                            with ema.average_parameters():
                                checkpoint_handler.save(
                                    state=CheckpointState(model, optimizer, lr_scheduler),
                                    epochs=epoch,
                                    keep_last=True,
                                )
                        else:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement"
                        )
                        break
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    if ema is not None:
                        with ema.average_parameters():
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=keep_last,
                            )
                            keep_last = False or save_all_checkpoints
                    else:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
        if train_sampler is not None:
            torch.distributed.barrier()
        epoch += 1

    logging.info("Training complete")


def get_attribute(obj, attr_name):
    """ parse a string to access an attribute of an object """
    parts = attr_name.split('.')
    try:
        for part in parts:
            if '[' in part and ']' in part:
                key, index = part.split('[')
                index = int(index[:-1])
                obj = getattr(obj, key)[index]
            else:
                obj = getattr(obj, part)
        if not( type(obj) == torch.nn.Parameter):
            raise AttributeError(f"model.{attr_name} is not a parameter")
    except AttributeError as e:
        raise ValueError(f"gradient debugging: weight {attr_name} was not found") from e
    return obj


def take_step(
    model: torch.nn.Module, # TODO: remove?
    model_eval_wrapper,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    max_grad_norm: Optional[float],
    device: torch.device,
    debug_log_grad_summary: bool = False,
    debug_grad_log_frequency: Optional[int] = None,
    opt_step: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    batch_dict = batch.to_dict()

    # do not set ema when training
    output = model_eval_wrapper(
        batch_dict,
        training=True,
    )
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()
    del output
    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    if debug_log_grad_summary and _should_log_grad_summary(opt_step, debug_grad_log_frequency):
        loss_dict.update(_scalar_param_grad_summary(model))

    if "DEBUG_IMPLICIT_GRADIENTS" in os.environ:
        the_weight = get_attribute(model, os.environ["DEBUG_IMPLICIT_GRADIENTS"])

        the_loss = loss.clone().detach()
        the_gradient = the_weight.grad.clone().detach()

        # re-evaluate with different values
        delta = 1e-3
        initial_value = the_weight.clone().detach()
        initial_value[0] += delta
        the_weight.requires_grad_(False)
        the_weight.copy_(initial_value)
        the_weight.requires_grad_(True)

        output = model_eval_wrapper(
            batch_dict,
            training=True,
        )
        loss_ = loss_fn(pred=output, ref=batch)

        # compute deltas and reset weight
        fd_gradient = (loss_ - the_loss)/delta
        diff_gradient = the_gradient[0]
        error = fd_gradient.detach() - diff_gradient
        frac_error = error / fd_gradient.detach()

        initial_value[0] -= delta
        the_weight.requires_grad_(False)
        the_weight.copy_(initial_value)
        the_weight.requires_grad_(True)
        del output
        del loss_
        del diff_gradient
        del error
        del frac_error

        # re-evaluate with different values
        initial_value = the_weight.clone().detach()
        initial_value[0] += delta*2
        the_weight.requires_grad_(False)
        the_weight.copy_(initial_value)
        the_weight.requires_grad_(True)

        output = model_eval_wrapper(
            batch_dict,
            training=True,
        )
        loss_ = loss_fn(pred=output, ref=batch)

        # compute deltas and reset weight
        fd_gradient1 = 0.5 * (loss_ - the_loss) / delta
        diff_gradient = the_gradient[0]
        error = fd_gradient1.detach() - diff_gradient
        frac_error = error / fd_gradient.detach()
        estimate_of_noise = fd_gradient1.detach() - fd_gradient.detach()
        logging.info(f"true g(2)={fd_gradient1.item():1.4g}+-{abs(estimate_of_noise):1.3g}, diff_gradient={diff_gradient.item():1.10g}, actual error={error.item()}, fractional={frac_error.item()}")

        initial_value[0] -= delta*2
        the_weight.requires_grad_(False)
        the_weight.copy_(initial_value)
        the_weight.requires_grad_(True)
        del output
        del loss_
        del fd_gradient
        del diff_gradient
        del error
        del frac_error
        del the_gradient
    
    if max_grad_norm is not None:
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        grad_norm_before_clip_value = float(grad_norm_before_clip.detach().cpu().item())
        loss_dict["grad_norm_before_clip"] = grad_norm_before_clip_value
        loss_dict["grad_clip_applied"] = grad_norm_before_clip_value > max_grad_norm
    
    if hasattr(model, "batch_positions"):
        del model.batch_positions

    optimizer.step()
    if ema is not None:
        ema.update()
    loss_dict["time"] = time.time() - start_time
    return loss, loss_dict


def evaluate(
    model: torch.nn.Module,
    model_eval_wrapper,
    loss_fn: torch.nn.Module,
    ema: Optional[ExponentialMovingAverage],
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    num_configs = 0
    total_loss = 0.0
    E_computed = False
    delta_es_list = []
    delta_es_per_atom_list = []
    delta_fs_list = []
    Fs_computed = False
    fs_list = []
    stress_computed = False
    delta_stress_list = []
    delta_stress_per_atom_list = []
    virials_computed = False
    delta_virials_list = []
    delta_virials_per_atom_list = []
    Mus_computed = False
    delta_mus_list = []
    delta_mus_per_atom_list = []
    mus_list = []
    dmas_computed = False
    delta_dmas_list = []
    dmas_list = []
    delta_esps_list = []
    esps_list = []
    polarizability_computed = False
    delta_polarizability_list = []
    delta_polarizability_per_atom_list = []
    total_charge_computed = False
    delta_total_charge_list = []
    fermi_level_computed = False
    delta_fermi_level_list = []
    batch = None  # for pylint

    for name, param in model.named_parameters():
        if name == 'batch_positions':
            continue
        param.requires_grad_(False)
        param.grad = None

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model_eval_wrapper(
            batch_dict,
            training=False,
            ema=ema,
        )

        if hasattr(model, "batch_positions"):
            del model.batch_positions
        for name, param in model.named_parameters():
            param.requires_grad_(False)
            param.grad = None

        # avoid memory leaks
        for key in output:
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].detach()
        

        # DDP requires that loss is calculated on GPU (nccl has no CPU backend)
        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()
        num_configs += batch.num_graphs

        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device("cpu"))

        if output.get("energy") is not None and batch.energy is not None:
            E_computed = True
            delta_es_list.append(batch.energy - output["energy"])
            delta_es_per_atom_list.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            Fs_computed = True
            delta_fs_list.append(batch.forces - output["forces"])
            fs_list.append(batch.forces)
        if output.get("stress") is not None and batch.stress is not None:
            stress_computed = True
            delta_stress_list.append(batch.stress - output["stress"])
            delta_stress_per_atom_list.append(
                (batch.stress - output["stress"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("virials") is not None and batch.virials is not None:
            virials_computed = True
            delta_virials_list.append(batch.virials - output["virials"])
            delta_virials_per_atom_list.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("density_coefficients") is not None and batch.total_charge is not None:
            total_charge_computed = True
            total_charge = scatter_sum(
                src=output["density_coefficients"][:,0], index=batch.batch, dim=-1
            )
            delta_total_charge_list.append(batch.total_charge - total_charge)
        if output.get("fermi_level") is not None and batch.fermi_level is not None:
            fermi_level_computed = True
            delta_fermi_level_list.append(batch.fermi_level - output["fermi_level"])
        if output.get("dipole") is not None and batch.dipole is not None:
            dipole_components_to_include = batch.dipole_weight.view(-1, 3) > 0.0
            if torch.any(dipole_components_to_include):
                dipole_differences = (batch.dipole - output["dipole"])
                num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1)
                num_atoms = num_atoms.repeat(1, 3)

                delta_mus_list.append(dipole_differences[dipole_components_to_include])
                delta_mus_per_atom_list.append(
                    dipole_differences[dipole_components_to_include] / num_atoms[dipole_components_to_include]
                )
                mus_list.append(batch.dipole[dipole_components_to_include]) # mus list is len(observations) not len(structures)
        if (
            output.get("density_coefficients") is not None
            and batch.density_coefficients is not None
        ):
            dmas_computed = True
            delta_dmas_list.append(
                batch.density_coefficients - output["density_coefficients"]
            )
            dmas_list.append(batch.density_coefficients)

        if (
            output.get("electrostatic_potentials") is not None
            and batch.electrostatic_potentials is not None
        ):
            esps_computed = True
            delta_esps_list.append(
                batch.electrostatic_potentials - output["electrostatic_potentials"]
            )
            esps_list.append(batch.electrostatic_potentials)
        else:
            esps_computed= False
        if output.get("polarizability") is not None and batch.polarizability is not None:
            polars_to_include = batch.polarizability_weight > 0.0
            if torch.any(polars_to_include):
                polarizability_computed = True
                delta_polarizability_list.append(batch.polarizability[polars_to_include] - output["polarizability"][polars_to_include])
                delta_polarizability_per_atom_list.append(
                    (batch.polarizability - output["polarizability"])[polars_to_include]
                    / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)[polars_to_include]
                )

    Mus_computed = len(delta_mus_list) > 0
    polars_computed = len(delta_polarizability_list) > 0

    avg_loss = total_loss / len(data_loader)

    aux = {
        "loss": avg_loss,
    }

    if E_computed:
        delta_es = to_numpy(torch.cat(delta_es_list, dim=0))
        delta_es_per_atom = to_numpy(torch.cat(delta_es_per_atom_list, dim=0))
        aux["mae_e"] = compute_mae(delta_es)
        aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
        aux["rmse_e"] = compute_rmse(delta_es)
        aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
        aux["q95_e"] = compute_q95(delta_es)
        offset = np.mean(delta_es_per_atom)
        reduced = delta_es_per_atom - offset
        aux["offset_e_per_atom"] = offset
        aux["rmse_spread_e_per_atom"] = compute_rmse(reduced)
        aux["mae_spread_e_per_atom"] = compute_mae(reduced)
    if Fs_computed:
        delta_fs = to_numpy(torch.cat(delta_fs_list, dim=0))
        fs = to_numpy(torch.cat(fs_list, dim=0))
        aux["mae_f"] = compute_mae(delta_fs)
        aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
        aux["rmse_f"] = compute_rmse(delta_fs)
        aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
        aux["q95_f"] = compute_q95(delta_fs)
    if stress_computed:
        delta_stress = to_numpy(torch.cat(delta_stress_list, dim=0))
        delta_stress_per_atom = to_numpy(torch.cat(delta_stress_per_atom_list, dim=0))
        aux["mae_stress"] = compute_mae(delta_stress)
        aux["rmse_stress"] = compute_rmse(delta_stress)
        aux["rmse_stress_per_atom"] = compute_rmse(delta_stress_per_atom)
        aux["q95_stress"] = compute_q95(delta_stress)
    if virials_computed:
        delta_virials = to_numpy(torch.cat(delta_virials_list, dim=0))
        delta_virials_per_atom = to_numpy(torch.cat(delta_virials_per_atom_list, dim=0))
        aux["mae_virials"] = compute_mae(delta_virials)
        aux["rmse_virials"] = compute_rmse(delta_virials)
        aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
        aux["q95_virials"] = compute_q95(delta_virials)
    if Mus_computed:
        delta_mus = to_numpy(torch.cat(delta_mus_list, dim=0))
        delta_mus_per_atom = to_numpy(torch.cat(delta_mus_per_atom_list, dim=0))
        mus = to_numpy(torch.cat(mus_list, dim=0))
        aux["mae_mu"] = compute_mae(delta_mus)
        aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
        aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
        aux["rmse_mu"] = compute_rmse(delta_mus)
        aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
        aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
        aux["q95_mu"] = compute_q95(delta_mus)
    if dmas_computed:
        delta_dmas = to_numpy(torch.cat(delta_dmas_list, dim=0))
        dmas = to_numpy(torch.cat(dmas_list, dim=0))
        aux["mae_dma"] = compute_mae(delta_dmas)
        aux["rel_mae_dma"] = compute_rel_mae(delta_dmas, dmas)
        aux["rmse_dma"] = compute_rmse(delta_dmas)
        aux["rel_rmse_dma"] = compute_rel_rmse(delta_dmas, dmas)
        aux["q95_dma"] = compute_q95(delta_dmas)
        if delta_dmas.shape[0] > 0:
            aux['rmse_charges'] = compute_rmse(delta_dmas[:,0:1])
        if delta_dmas.shape[1] > 1:
            aux['rmse_local_dipoles'] = compute_rmse(delta_dmas[:,1:4])
    if esps_computed:
        delta_esps = to_numpy(torch.cat(delta_esps_list, dim=0))
        esps = to_numpy(torch.cat(esps_list, dim=0))
        aux["mae_esp"] = compute_mae(delta_esps)
        aux["rel_mae_esp"] = compute_rel_mae(delta_esps, esps)
        aux["rmse_esp"] = compute_rmse(delta_esps)
        aux["rel_rmse_esp"] = compute_rel_rmse(delta_esps, esps)
        aux["q95_esp"] = compute_q95(delta_esps)
    if polarizability_computed:
        delta_polarizability = to_numpy(torch.cat(delta_polarizability_list, dim=0))
        delta_polarizability_per_atom = to_numpy(torch.cat(delta_polarizability_per_atom_list, dim=0))
        aux["mae_polarizability"] = compute_mae(delta_polarizability)
        aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
        aux["rmse_polarizability_per_atom"] = compute_rmse(delta_polarizability_per_atom)
        aux["q95_polarizability"] = compute_q95(delta_polarizability)
    if total_charge_computed:
        delta_total_charge = to_numpy(torch.cat(delta_total_charge_list, dim=0))
        aux["mae_total_charge"] = compute_mae(delta_total_charge)
        aux["rmse_total_charge"] = compute_rmse(delta_total_charge)
        aux["q95_total_charge"] = compute_q95(delta_total_charge)
    if fermi_level_computed:
        delta_fermi_level = to_numpy(torch.cat(delta_fermi_level_list, dim=0))
        aux["mae_fermi_level"] = compute_mae(delta_fermi_level)
        aux["rmse_fermi_level"] = compute_rmse(delta_fermi_level)
        aux["q95_fermi_level"] = compute_q95(delta_fermi_level)

    aux["time"] = time.time() - start_time

    for name, param in model.named_parameters():
        param.requires_grad = True

    return avg_loss, aux



def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch,
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    logger.log(eval_metrics)

    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_stress_per_atom={error_stress:.1f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_virials_per_atom={error_virials:.1f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} mDebye"
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_Mu_per_atom={error_mu:.2f} mDebye"
        )
    elif log_errors == "DensityCoefficientsRMSE":
        error_dma = eval_metrics["rmse_dma"] * 1e3
        rel_error_dma = eval_metrics["rel_rmse_dma"]
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_DMA={error_dma:.1f} me, rel_RMSE_DMA={rel_error_dma:.2f} %"
        )
    elif log_errors == "DensityEnergyRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_dma = eval_metrics["rmse_dma"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_DMA={error_dma:.1f} me"
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.6f} meA/atom"
        )
    elif log_errors == "DensityDipoleRMSE":
        error_dma = eval_metrics["rmse_dma"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_DMA={error_dma:.1f} me, RMSE_MU_per_atom={error_mu:.6f} meA/atom"
        )
    elif log_errors == "EnergyDensityDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_dma = eval_metrics["rmse_dma"] * 1e3
        if not "rmse_mu_per_atom" in eval_metrics:
            error_mu = "NO DIPOLES FOUND VALID SET WHEN LOGGING"
        else:
            error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
            error_mu = f"{error_mu:.2f}"
        if not "rmse_polarizability_per_atom" in eval_metrics:
            error_polarizability = "no polarizability found"
        else:
            error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
            error_polarizability = f"{error_polarizability:.2f}"
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_DMA={error_dma:.1f} me, RMSE_Mu_per_atom={error_mu} meA, RMSE_polarizability_per_atom={error_polarizability} me A^2 / V"
        )
    elif log_errors == "EnergyDipolePotentialsRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        if not "rmse_mu_per_atom" in eval_metrics:
            error_mu = "NO DIPOLES FOUND VALID SET WHEN LOGGING"
        else:
            error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
            error_mu = f"{error_mu:.2f}"
        error_esp = eval_metrics["rmse_esp"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_Mu_per_atom={error_mu} meA, RMSE_ESP={error_esp:.1f} mV"
        )
