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
from mace.tools.torch_tools import to_numpy
from mace.tools.utils import MetricsLogger
import os

from mace_scf.utils.eval_metrics import MaceSCFLoss


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
    should_stop = False

    # rank 0 alone decides whether to stop (see below); every other rank
    # just waits for this broadcast flag, mirroring mace.tools.train.train.
    exit_now = torch.zeros(1, device=device) if train_sampler is not None else None

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
                        should_stop = True
                        if exit_now is not None:
                            exit_now.fill_(1)
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
        if exit_now is not None:
            torch.distributed.broadcast(exit_now, src=0)
            should_stop = exit_now.item() == 1
        if should_stop:
            break
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
    for name, param in model.named_parameters():
        if name == 'batch_positions':
            continue
        param.requires_grad_(False)
        param.grad = None

    metrics = MaceSCFLoss(loss_fn=loss_fn).to(device)

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

        # DDP (and MaceSCFLoss's cross-rank all_reduce/all_gather) requires
        # that the loss and per-sample tensors stay on GPU (nccl has no CPU
        # backend).
        metrics.update(batch, output)

    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

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
