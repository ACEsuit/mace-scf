import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
from prettytable import PrettyTable

from mace.modules.loss import is_ddp_enabled
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum
from mace.tools.torch_tools import tensor_dict_to_device
from mace.tools.utils import compute_rel_rmse, compute_rmse

from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_scf import (
    CONSTANT_CHARGE_CONVERGED_TOTAL_CHARGE_TOL,
)
from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions


SCF_SUMMARY_NUM_STEPS = 100
SCF_SUMMARY_TOLERANCE = 1e-7
SCF_SUMMARY_DEFAULT_MIXING_VALUES = (0.2,)
SCF_SUMMARY_INITIAL_DENSITY = "local_guess"
SCF_SUMMARY_INITIAL_FERMI_LEVEL = "from_data"


def is_fixed_point_model(model: torch.nn.Module) -> bool:
    module = model.module if hasattr(model, "module") else model
    return module.__class__.__name__ in ("FixedPoint", "FixedPointCore")


def scf_summary_mixing_values(train_stage: dict) -> Tuple[float, float]:
    mixing = None
    options = train_stage.get("fixed_point_training_options")
    if options is not None and options.scf is not None:
        mixing = float(options.scf.mixing_parameter)

    if mixing is not None and mixing < SCF_SUMMARY_DEFAULT_MIXING_VALUES[0]:
        values = (mixing, 2.0 * mixing)
    else:
        values = SCF_SUMMARY_DEFAULT_MIXING_VALUES
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _rebuild_loader(loader, batch_size: int = 1):
    dataset = loader.dataset
    if is_ddp_enabled():
        # Shard the (unsampled, full) dataset across ranks so the SCF
        # diagnostic parallelizes instead of running entirely on rank 0.
        # A plain stride split (not DistributedSampler) keeps every graph
        # covered exactly once -- DistributedSampler's drop_last=False
        # padding would repeat a few graphs across ranks.
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        indices = list(range(rank, len(dataset), world_size))
        dataset = Subset(dataset, indices)
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )


def _setting_label(constant_charge: bool) -> str:
    return "Constant-charge" if constant_charge else "Constant-Fermi"


def _make_scf_options(constant_charge: bool, mixing: float) -> FixedPointSCFOptions:
    return FixedPointSCFOptions(
        num_scf_steps=SCF_SUMMARY_NUM_STEPS,
        scf_tolerance=SCF_SUMMARY_TOLERANCE,
        mixing_parameter=float(mixing),
        constant_charge=bool(constant_charge),
        use_autograd_forces=True,
        initial_density=SCF_SUMMARY_INITIAL_DENSITY,
        initial_fermi_level=SCF_SUMMARY_INITIAL_FERMI_LEVEL,
    )


def _status_counts(statuses: List[str]) -> Tuple[int, int, int]:
    converged = sum(status == "converged" for status in statuses)
    diverged = sum(status == "diverged" for status in statuses)
    other = len(statuses) - converged - diverged
    return converged, other, diverged


def _percent(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100.0 * count / total


def _steps_summary(steps: List[int]) -> Tuple[str, str]:
    if not steps:
        return "n/a", "n/a"
    values = np.asarray(steps, dtype=float)
    return f"{np.median(values):.0f}", f"{np.percentile(values, 95):.0f}"


def _empty_metrics():
    return {
        "num_converged": 0,
        "delta_e_per_atom": [],
        "delta_f": [],
        "ref_f": [],
        "delta_dma": [],
        "ref_dma": [],
        "delta_mu": [],
        "delta_mu_per_atom": [],
        "ref_mu": [],
        "delta_esp": [],
        "ref_esp": [],
        "delta_polarizability_per_atom": [],
        "delta_charge": [],
    }


def _accumulate_converged_metrics(metrics, batch, output):
    metrics["num_converged"] += int(batch.num_graphs)
    if output.get("energy") is not None and getattr(batch, "energy", None) is not None:
        num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).to(output["energy"].device)
        metrics["delta_e_per_atom"].append(
            ((batch.energy.to(output["energy"].device) - output["energy"]) / num_atoms)
            .detach()
            .cpu()
        )
    if output.get("forces") is not None and getattr(batch, "forces", None) is not None:
        metrics["delta_f"].append(
            (batch.forces.to(output["forces"].device) - output["forces"]).detach().cpu()
        )
        metrics["ref_f"].append(batch.forces.detach().cpu())
    if (
        output.get("density_coefficients") is not None
        and getattr(batch, "density_coefficients", None) is not None
    ):
        metrics["delta_dma"].append(
            (
                batch.density_coefficients.to(output["density_coefficients"].device)
                - output["density_coefficients"]
            )
            .detach()
            .cpu()
        )
        metrics["ref_dma"].append(batch.density_coefficients.detach().cpu())
    if output.get("dipole") is not None and getattr(batch, "dipole", None) is not None:
        dipole_weight = getattr(batch, "dipole_weight", None)
        if dipole_weight is not None:
            include = dipole_weight.view(-1, 3) > 0.0
        else:
            include = torch.ones_like(batch.dipole, dtype=torch.bool)
        if torch.any(include):
            dipole_diff = batch.dipole.to(output["dipole"].device) - output["dipole"]
            num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1)
            num_atoms = num_atoms.repeat(1, 3).to(dipole_diff.device)
            include = include.to(dipole_diff.device)
            metrics["delta_mu"].append(dipole_diff[include].detach().cpu())
            metrics["delta_mu_per_atom"].append(
                (dipole_diff[include] / num_atoms[include]).detach().cpu()
            )
            metrics["ref_mu"].append(batch.dipole[include.cpu()].detach().cpu())
    if (
        output.get("electrostatic_potentials") is not None
        and getattr(batch, "electrostatic_potentials", None) is not None
    ):
        metrics["delta_esp"].append(
            (
                batch.electrostatic_potentials.to(
                    output["electrostatic_potentials"].device
                )
                - output["electrostatic_potentials"]
            )
            .detach()
            .cpu()
        )
        metrics["ref_esp"].append(batch.electrostatic_potentials.detach().cpu())
    if (
        output.get("polarizability") is not None
        and getattr(batch, "polarizability", None) is not None
    ):
        polar_weight = getattr(batch, "polarizability_weight", None)
        if polar_weight is not None:
            include = polar_weight > 0.0
        else:
            include = torch.ones(
                batch.polarizability.shape[0],
                dtype=torch.bool,
                device=batch.polarizability.device,
            )
        if torch.any(include):
            polar_diff = (
                batch.polarizability.to(output["polarizability"].device)
                - output["polarizability"]
            )
            num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            num_atoms = num_atoms.to(polar_diff.device)
            include = include.to(polar_diff.device)
            metrics["delta_polarizability_per_atom"].append(
                (polar_diff[include] / num_atoms[include]).detach().cpu()
            )
    if (
        output.get("density_coefficients") is not None
        and getattr(batch, "total_charge", None) is not None
    ):
        total_charge = scatter_sum(
            src=output["density_coefficients"][:, 0],
            index=batch.batch.to(output["density_coefficients"].device),
            dim=-1,
            dim_size=batch.num_graphs,
        )
        metrics["delta_charge"].append(
            (batch.total_charge.to(total_charge.device) - total_charge).detach().cpu()
        )


def _metric_or_missing(metrics, key):
    if not metrics[key]:
        return "not found"
    values = torch.cat(metrics[key], dim=0).numpy()
    return values


def _format_error_row(name: str, metrics: dict) -> List[str]:
    return _format_error_row_for_type(name, metrics, "PerAtomRMSE")


def _metric_value(metrics, key, rel_key=None):
    values = _metric_or_missing(metrics, key)
    if isinstance(values, str):
        return "not found"
    if rel_key is None:
        return f"{compute_rmse(values) * 1e3:.2f}"
    ref = _metric_or_missing(metrics, rel_key)
    if isinstance(ref, str):
        return "not found"
    return f"{compute_rel_rmse(values, ref):.2f}"


def _format_error_row_for_type(name: str, metrics: dict, table_type: str) -> List[str]:
    n = metrics["num_converged"]
    delta_e = _metric_or_missing(metrics, "delta_e_per_atom")
    delta_f = _metric_or_missing(metrics, "delta_f")
    ref_f = _metric_or_missing(metrics, "ref_f")

    if isinstance(delta_e, str):
        rmse_e = "not found"
    else:
        rmse_e = f"{compute_rmse(delta_e) * 1e3:.2f}"

    if isinstance(delta_f, str):
        rmse_f = "not found"
        rel_rmse_f = "not found"
    else:
        rmse_f = f"{compute_rmse(delta_f) * 1e3:.2f}"
        rel_rmse_f = f"{compute_rel_rmse(delta_f, ref_f):.2f}"

    rmse_dma = _metric_value(metrics, "delta_dma")
    rel_rmse_dma = _metric_value(metrics, "delta_dma", "ref_dma")
    rmse_mu = _metric_value(metrics, "delta_mu_per_atom")
    rel_rmse_mu = _metric_value(metrics, "delta_mu", "ref_mu")
    rmse_esp = _metric_value(metrics, "delta_esp")
    rel_rmse_esp = _metric_value(metrics, "delta_esp", "ref_esp")
    rmse_polar = _metric_value(metrics, "delta_polarizability_per_atom")
    delta_charge = _metric_or_missing(metrics, "delta_charge")
    if isinstance(delta_charge, str):
        rmse_charges = "not found"
    else:
        rmse_charges = f"{compute_rmse(delta_charge) * 1e3:.2f}"

    delta_dma = _metric_or_missing(metrics, "delta_dma")
    if isinstance(delta_dma, str):
        rmse_local_dipoles = "not found"
    elif delta_dma.shape[1] > 1:
        rmse_local_dipoles = f"{compute_rmse(delta_dma[:, 1:4]) * 1e3:.2f}"
    else:
        rmse_local_dipoles = "not found"

    if table_type == "DensityCoefficientsRMSE":
        return [name, n, rmse_dma, rel_rmse_dma, rmse_charges, rmse_local_dipoles]
    if table_type == "DensityEnergyRMSE":
        return [name, n, rmse_e, rmse_f, rel_rmse_f, rmse_dma]
    if table_type == "PerAtomRMSE":
        return [name, n, rmse_e, rmse_f, rel_rmse_f]
    if table_type == "DipoleRMSE":
        return [name, n, rmse_mu, rel_rmse_mu]
    if table_type == "DensityDipoleRMSE":
        return [name, n, rmse_mu, rel_rmse_mu, rmse_dma, rel_rmse_dma]
    if table_type == "EnergyDensityDipoleRMSE":
        return [
            name,
            n,
            rmse_e,
            rmse_f,
            rel_rmse_f,
            rmse_dma,
            rmse_mu,
            rel_rmse_mu,
            rmse_polar,
        ]
    if table_type == "EnergyDipolePotentialsRMSE":
        return [
            name,
            n,
            rmse_e,
            rmse_f,
            rel_rmse_f,
            rmse_mu,
            rel_rmse_mu,
            rmse_esp,
            rel_rmse_esp,
        ]
    return [name, n, rmse_e, rmse_f, rel_rmse_f]


def _build_observables(model, batch_dict, local_state, scf_result, output_args):
    return model.build_observables(
        data=batch_dict,
        local_state=local_state,
        density=scf_result.density,
        fermi_level=scf_result.fermi_level,
        field_feats=scf_result.field_feats,
        training=False,
        compute_force=output_args["forces"],
        compute_virials=output_args.get("virials", False),
        compute_stress=output_args.get("stress", False),
    )


def _evaluate_one_setting(
    model,
    all_data_loaders,
    output_args,
    device,
    scf_options: FixedPointSCFOptions,
    batch_size: int,
):
    runner = FixedPointSCFRunner(scf_options)
    result = {}

    for name, loader in all_data_loaders.items():
        logging.info(
            "SCF convergence summary: evaluating %s, constant_charge=%s, mixing=%s",
            name,
            scf_options.constant_charge,
            scf_options.mixing_parameter,
        )
        statuses = []
        steps = []
        final_changes = []
        metrics = _empty_metrics()

        for batch in _rebuild_loader(loader, batch_size=batch_size):
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            local_state = model.local_part(
                batch_dict,
                compute_force=False,
            )
            initial_density = runner.get_initial_density(local_state, batch_dict)
            initial_fermi = runner.get_initial_fermi(model, local_state, batch_dict)
            scf_result = runner.converge(
                model,
                batch_dict,
                local_state,
                initial_density,
                initial_fermi,
                compute_force=False,
            )

            graph_statuses = _per_graph_statuses(
                batch_dict=batch_dict,
                scf_result=scf_result,
                scf_options=scf_options,
            )
            statuses.extend(graph_statuses)
            steps.extend([int(scf_result.terminated_step)] * batch.num_graphs)
            final_changes.extend(
                [
                    float(value)
                    for value in scf_result.final_avg_abs_change.detach().cpu()
                ]
            )

            if hasattr(model, "batch_positions"):
                del model.batch_positions
            del local_state
            del scf_result

            """ if status == "converged":
                local_state = model.local_part(
                    batch_dict,
                    compute_force=output_args["forces"],
                )
                initial_density = runner.get_initial_density(local_state, batch_dict)
                initial_fermi = runner.get_initial_fermi(model, local_state, batch_dict)
                scf_result = runner.converge(
                    model,
                    batch_dict,
                    local_state,
                    initial_density,
                    initial_fermi,
                    compute_force=output_args["forces"],
                )
                if scf_result.status == "converged":
                    output = _build_observables(
                        model, batch_dict, local_state, scf_result, output_args
                    )
                    for key in output:
                        if isinstance(output[key], torch.Tensor):
                            output[key] = output[key].detach()
                    output = tensor_dict_to_device(output, device=torch.device("cpu"))
                    batch_cpu = batch.cpu()
                    _accumulate_converged_metrics(metrics, batch_cpu, output)
                    del output
                    del batch_cpu
                del local_state
                del scf_result """

            if hasattr(model, "batch_positions"):
                del model.batch_positions
            for param in model.parameters():
                param.requires_grad_(False)
                param.grad = None
            del batch_dict
            del batch

        if is_ddp_enabled():
            # statuses/steps/final_changes are plain Python lists of
            # primitives (not GPU tensors), so all_gather_object is the
            # natural collective here.
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                gathered, (statuses, steps, final_changes)
            )
            statuses = [s for local in gathered for s in local[0]]
            steps = [s for local in gathered for s in local[1]]
            final_changes = [c for local in gathered for c in local[2]]

        result[name] = {
            "statuses": statuses,
            "steps": steps,
            "final_changes": final_changes,
            "metrics": metrics,
        }

    return result


def _per_graph_statuses(
    batch_dict,
    scf_result,
    scf_options: FixedPointSCFOptions,
) -> List[str]:
    num_graphs = batch_dict["ptr"].numel() - 1

    if scf_result.status == "diverged":
        return ["diverged"] * num_graphs

    final_change = scf_result.final_avg_abs_change.detach()
    converged = final_change < scf_options.scf_tolerance

    if scf_options.constant_charge:
        total_charge = scatter_sum(
            src=scf_result.density.detach()[:, 0],
            index=batch_dict["batch"],
            dim=-1,
            dim_size=num_graphs,
        )
        charge_converged = (
            torch.abs(total_charge - batch_dict["total_charge"])
            < CONSTANT_CHARGE_CONVERGED_TOTAL_CHARGE_TOL
        )
        converged = torch.logical_and(converged, charge_converged)

    fallback_status = (
        scf_result.status if scf_result.status != "converged" else "max_steps_reached"
    )
    return [
        "converged" if bool(is_converged) else fallback_status
        for is_converged in converged.detach().cpu()
    ]


def _render_convergence_table(setting_results, mixing_values):
    table = PrettyTable()
    table.field_names = [
        "config_type",
        "n",
        "mixing",
        "converged %",
        "other %",
        "diverged %",
        "median steps",
        "p95 steps",
    ]
    for mixing_i, mixing in enumerate(mixing_values):
        if mixing_i > 0:
            if hasattr(table, "add_divider"):
                table.add_divider()
        rows = setting_results[mixing]
        for name in sorted(rows):
            row = rows[name]
            statuses = row["statuses"]
            converged, other, diverged = _status_counts(statuses)
            median_steps, p95_steps = _steps_summary(row["steps"])
            table.add_row(
                [
                    name,
                    len(statuses),
                    f"{mixing:.3g}",
                    f"{_percent(converged, len(statuses)):.1f}",
                    f"{_percent(other, len(statuses)):.1f}",
                    f"{_percent(diverged, len(statuses)):.1f}",
                    median_steps,
                    p95_steps,
                ]
            )
    return table


def _select_best_mixing(setting_results, mixing_values):
    best = None
    for mixing in mixing_values:
        total = 0
        converged = 0
        for row in setting_results[mixing].values():
            total += len(row["statuses"])
            converged += sum(status == "converged" for status in row["statuses"])
        key = (converged, -float(mixing))
        if best is None or key > best[0]:
            best = (key, mixing, converged, total)
    _, mixing, converged, total = best
    return mixing, converged, total


def _render_error_table(setting_result, table_type: str):
    table = PrettyTable()
    if table_type == "DensityCoefficientsRMSE":
        table.field_names = [
            "config_type",
            "n converged",
            "RMSE DMA / e A^l",
            "rel DMA %",
            "RMSE qs",
            "RMSE dipoles",
        ]
    elif table_type == "DensityEnergyRMSE":
        table.field_names = [
            "config_type",
            "n converged",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE DMA / e A^l",
        ]
    elif table_type == "PerAtomRMSE":
        table.field_names = [
            "config_type",
            "n converged",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "DipoleRMSE":
        table.field_names = [
            "config_type",
            "n converged",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
        ]
    elif table_type == "DensityDipoleRMSE":
        table.field_names = [
            "config_type",
            "n converged",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "RMSE DMA / e A^l",
            "rel DMA %",
        ]
    elif table_type == "EnergyDensityDipoleRMSE":
        table.field_names = [
            "config_type",
            "n converged",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE DMA / e A^l",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "polarizability / me A^2 / V",
        ]
    elif table_type == "EnergyDipolePotentialsRMSE":
        table.field_names = [
            "config_type",
            "n converged",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "RMSE ESP / mV",
            "relative ESP RMSE %",
        ]
    else:
        table.field_names = [
            "config_type",
            "n converged",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    for name in sorted(setting_result):
        table.add_row(
            _format_error_row_for_type(name, setting_result[name]["metrics"], table_type)
        )
    return table


def create_scf_convergence_summary(
    model,
    all_data_loaders: Dict,
    output_args: Dict[str, bool],
    device,
    train_stage: dict,
    error_table_type: str = "PerAtomRMSE",
    diagnostic_batch_size: int = 1,
    mixing_values: Optional[Iterable[float]] = None,
) -> str:
    module = model.module if hasattr(model, "module") else model
    if not is_fixed_point_model(module):
        return "SCF convergence summary skipped: model is not FixedPoint/FixedPointCore."

    for param in module.parameters():
        param.requires_grad_(False)
        param.grad = None

    if mixing_values is None:
        mixing_values = scf_summary_mixing_values(train_stage)
    mixing_values = tuple(float(value) for value in mixing_values)
    diagnostic_batch_size = max(1, int(diagnostic_batch_size))

    lines = [
        "SCF convergence summary",
        f"stage: {train_stage.get('name', 'unknown')}",
        "diagnostic mode: unroll_scf",
        f"num_scf_steps: {SCF_SUMMARY_NUM_STEPS}",
        f"scf_tolerance: {SCF_SUMMARY_TOLERANCE:.1e}",
        f"initial_density: {SCF_SUMMARY_INITIAL_DENSITY}",
        f"initial_fermi_level: {SCF_SUMMARY_INITIAL_FERMI_LEVEL}",
        "mixing values: " + ", ".join(f"{value:.3g}" for value in mixing_values),
        "rows: train, valid, test config types",
        f"diagnostic batch size: {diagnostic_batch_size}",
    ]

    for constant_charge in (False, True):
        charge_label = _setting_label(constant_charge)
        setting_results = {}
        for mixing in mixing_values:
            scf_options = _make_scf_options(
                constant_charge=constant_charge,
                mixing=mixing,
            )
            setting_results[mixing] = _evaluate_one_setting(
                module,
                all_data_loaders,
                output_args,
                device,
                scf_options,
                diagnostic_batch_size,
            )

        lines.append("")
        lines.append(f"{charge_label} convergence")
        lines.append(str(_render_convergence_table(setting_results, mixing_values)))

        """ best_mixing, converged, total = _select_best_mixing(
            setting_results, mixing_values
        )
        percent = _percent(converged, total)
        lines.append("")
        lines.append(f"{charge_label} selected setting for converged-subset errors:")
        lines.append(
            f"mixing={best_mixing:.3g}, converged={converged}/{total} configs "
            f"({percent:.1f}%)"
        )
        lines.append("")
        lines.append(f"{charge_label} errors on converged subset only")
        lines.append(str(_render_error_table(setting_results[best_mixing], error_table_type))) """

    for param in module.parameters():
        param.requires_grad_(True)

    return "\n".join(lines)
