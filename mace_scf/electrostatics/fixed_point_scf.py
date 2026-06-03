"""
SCF convergence functions for FixedPointCore.

Each function drives the self-consistent loop by calling model.scf_step
and handling mixing, detaching, and convergence checking.
"""
import logging
import torch
from mace.tools.scatter import scatter_sum, scatter_mean

from .fixed_point_state import LocalState, SCFState, FixedPointSCFOptions


SCF_DIVERGED_TOTAL_CHARGE = 10.0
CONSTANT_FERMI_DIVERGED_ABS_CHANGE = 1.0
CONSTANT_CHARGE_DIVERGED_ABS_CHANGE = 5.0
CONSTANT_CHARGE_CONVERGED_TOTAL_CHARGE_TOL = 1e-3


def _initial_final_avg_abs_change(reference: torch.Tensor, num_graphs: int) -> torch.Tensor:
    return torch.full(
        (num_graphs,),
        float("inf"),
        dtype=reference.dtype,
        device=reference.device,
    )


def _constant_fermi_converged(avg_abs_change: torch.Tensor, tol: float) -> bool:
    return bool(torch.all(avg_abs_change < tol))


def _constant_fermi_diverged(
    total_charge: torch.Tensor, avg_abs_change: torch.Tensor
) -> bool:
    return bool(
        torch.any(torch.abs(total_charge) > SCF_DIVERGED_TOTAL_CHARGE)
        or torch.any(avg_abs_change > CONSTANT_FERMI_DIVERGED_ABS_CHANGE)
    )


def _constant_charge_converged(
    avg_abs_change: torch.Tensor,
    total_charge: torch.Tensor,
    target_charge: torch.Tensor,
    tol: float,
) -> bool:
    return bool(
        torch.all(avg_abs_change < tol)
        and torch.all(
            torch.abs(total_charge - target_charge)
            < CONSTANT_CHARGE_CONVERGED_TOTAL_CHARGE_TOL
        )
    )


def _constant_charge_diverged(
    total_charge: torch.Tensor, avg_abs_change: torch.Tensor
) -> bool:
    return bool(
        torch.any(torch.abs(total_charge) > SCF_DIVERGED_TOTAL_CHARGE)
        or torch.any(avg_abs_change > CONSTANT_CHARGE_DIVERGED_ABS_CHANGE)
    )


def converge_constant_fermi(
    model,
    data,
    local_state: LocalState,
    initial_density: torch.Tensor,
    initial_fermi_level: torch.Tensor,
    scf_options: FixedPointSCFOptions,
    compute_force: bool = False,
):
    """
    SCF loop at fixed Fermi level (constant_charge=False).

    The Fermi level is provided explicitly via initial_fermi_level; only the
    density is iterated until self-consistency.
    """
    logger = logging.getLogger(__name__)
    num_graphs = data["ptr"].numel() - 1
    num_scf_steps = scf_options.num_scf_steps
    mixing = scf_options.mixing_parameter
    tol = scf_options.scf_tolerance

    if not compute_force:
        num_autograd_steps = 0
    elif scf_options.use_autograd_forces:
        num_autograd_steps = num_scf_steps - 1
    else:
        num_autograd_steps = 1

    # Build Fermi level features once (constant throughout the loop)
    fermi_level_features = model.features_from_fermi_level(
        data["batch"], local_state.positions, initial_fermi_level
    )

    charge_density = initial_density.clone()
    charges_history = [charge_density.clone().detach()]
    terminated_step = -1
    status = "max_steps_reached"
    final_avg_abs_change = _initial_final_avg_abs_change(initial_density, num_graphs)

    field_feats = None
    for step_i in range(num_scf_steps - 1):
        field_dep, field_feats = model.scf_step(
            data, local_state, charge_density, charge_density,
            fermi_level_features,
        )
        new_density = local_state.field_independent_charge_density + field_dep

        Delta_n = new_density - charge_density
        charge_density = charge_density + Delta_n * mixing

        total_charge = scatter_sum(
            src=charge_density[:, 0],
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )

        if step_i < num_scf_steps - num_autograd_steps - 1:
            charge_density = charge_density.detach()
            field_feats = field_feats.detach()

        charges_history.append(charge_density.clone().detach())
        abs_change = torch.mean(
            torch.abs(charges_history[-1] - charges_history[-2]), dim=-1
        )
        avg_abs_change = scatter_mean(
            src=abs_change, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        terminated_step = step_i
        final_avg_abs_change = avg_abs_change

        logger.debug(
            f"step {step_i}, total_charges={total_charge}, "
            f"abs_changes={torch.mean(abs_change)}"
        )
        if _constant_fermi_converged(avg_abs_change, tol):
            status = "converged"
            logger.debug(
                f"SCF converged at step {step_i}, total_charges={total_charge}, "
                f"abs_changes={torch.mean(abs_change)}"
            )
            break

        if _constant_fermi_diverged(total_charge, avg_abs_change):
            status = "diverged"
            logger.debug(
                f"SCF diverged at step {step_i}, total_charges={total_charge}, "
                f"abs_changes={torch.mean(abs_change)}"
            )
            break

    charges_history = torch.stack(charges_history, dim=-1)

    return SCFState(
        density=charge_density,
        fermi_level=initial_fermi_level,
        field_feats=field_feats,
        charges_history=charges_history,
        terminated_step=terminated_step,
        status=status,
        final_avg_abs_change=final_avg_abs_change,
    )


def converge_constant_charge(
    model,
    data,
    local_state: LocalState,
    initial_density: torch.Tensor,
    initial_fermi_level: torch.Tensor,
    scf_options: FixedPointSCFOptions,
    compute_force: bool = False,
):
    """
    SCF loop at fixed total charge (constant_charge=True).

    Uses Fukui-function-based charge redistribution to enforce the
    target total charge at every step while optimizing the Fermi level.
    """
    logger = logging.getLogger(__name__)
    num_graphs = data["ptr"].numel() - 1
    n_nodes = data["batch"].size(0)
    num_scf_steps = scf_options.num_scf_steps
    mixing = scf_options.mixing_parameter
    tol = scf_options.scf_tolerance

    target_charge = data["total_charge"]
    charge_density = initial_density.clone()
    fermi_level = initial_fermi_level.clone()

    field_indep_total_charge = scatter_sum(
        src=local_state.field_independent_charge_density[:, 0],
        index=data["batch"],
        dim=-1,
        dim_size=num_graphs,
    )
    logger.debug(
        f"field indep bit={field_indep_total_charge}, "
        f"data_total_charge={target_charge}"
    )

    charge_density_renormalized = charge_density.clone()
    charges_history = [charge_density.clone().detach()]
    terminated_step = -1
    status = "max_steps_reached"
    final_avg_abs_change = _initial_final_avg_abs_change(initial_density, num_graphs)

    # --- initial Fukui redistribution (before the loop) ---
    node_fermi = torch.index_select(fermi_level, 0, data["batch"])
    node_fermi.requires_grad_(True)
    fermi_level_features = model.features_from_fermi_level_nodewise(
        data["batch"], local_state.positions, node_fermi
    )

    field_dep_contribution, field_feats_0 = model.scf_step(
        data,
        local_state,
        charge_density_in=charge_density,
        total_charges=charge_density,
        fermi_level_features=fermi_level_features,
    )

    dq_dmu = torch.autograd.grad(
        field_dep_contribution[:, 0].sum(),
        node_fermi,
        retain_graph=compute_force,
        create_graph=compute_force,
    )[0]

    _total_charge = scatter_sum(
        src=charge_density[:, 0], index=data["batch"], dim=-1, dim_size=num_graphs
    )
    total_responses = scatter_sum(
        src=dq_dmu, index=data["batch"], dim=-1, dim_size=num_graphs
    )
    fukui_functions = dq_dmu / torch.index_select(
        total_responses, 0, data["batch"]
    )
    charge_deficit = torch.index_select(
        target_charge - _total_charge, 0, data["batch"]
    )
    charge_density_renormalized[:, 0] += fukui_functions * charge_deficit

    # --- SCF loop ---
    field_feats = None
    for step_i in range(num_scf_steps - 1):
        terminate_after_update = False
        node_fermi = torch.index_select(fermi_level, 0, data["batch"])
        node_fermi.requires_grad_(True)
        fermi_level_features = model.features_from_fermi_level_nodewise(
            data["batch"], local_state.positions, node_fermi
        )

        field_dep_contribution, field_feats = model.scf_step(
            data,
            local_state,
            charge_density_in=charge_density_renormalized,
            total_charges=charge_density,
            fermi_level_features=fermi_level_features,
        )
        new_density = local_state.field_independent_charge_density + field_dep_contribution

        Delta_n = new_density - charge_density_renormalized
        charge_density = charge_density_renormalized + Delta_n * mixing
        charge_density_renormalized = charge_density.clone()

        _total_charge = scatter_sum(
            src=charge_density[:, 0],
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )

        # Fukui redistribution
        dq_dmu = torch.autograd.grad(
            field_dep_contribution[:, 0].sum(),
            node_fermi,
            retain_graph=compute_force,
            create_graph=compute_force,
        )[0]

        total_responses = scatter_sum(
            src=dq_dmu, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        fukui_functions = dq_dmu / torch.index_select(
            total_responses, 0, data["batch"]
        )
        charge_deficit = torch.index_select(
            target_charge - _total_charge, 0, data["batch"]
        )
        charge_density_renormalized[:, 0] += fukui_functions * charge_deficit

        if not compute_force:
            charge_density = charge_density.detach()
            field_feats = field_feats.detach()
            fermi_level = fermi_level.detach()
            total_responses = total_responses.detach()
            fukui_functions = fukui_functions.detach()
            charge_deficit = charge_deficit.detach()
            charge_density_renormalized = charge_density_renormalized.detach()

        charges_history.append(charge_density.detach().clone())
        abs_change = torch.mean(
            torch.abs(charges_history[-1] - charges_history[-2]), dim=-1
        )
        avg_abs_change = scatter_mean(
            src=abs_change, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        terminated_step = step_i
        final_avg_abs_change = avg_abs_change

        logger.debug(
            f"step {step_i}, fermi_levels={fermi_level}, "
            f"gradient={total_responses}, total_qs={_total_charge}, "
            f"abs_change={torch.mean(abs_change)}"
        )
        if _constant_charge_converged(
            avg_abs_change, _total_charge, target_charge, tol
        ):
            status = "converged"
            logger.debug(
                f"converged at step {step_i}, "
                f"fermi_levels={fermi_level}, "
                f"total_charges={_total_charge}, "
                f"abs_changes={torch.mean(abs_change)}"
            )
            break

        if _constant_charge_diverged(_total_charge, avg_abs_change):
            status = "diverged"
            terminate_after_update = True
            logger.debug(
                f"SCF diverged at step {step_i}, "
                f"total_charges={_total_charge}, "
                f"abs_changes={torch.mean(abs_change)}"
            )

        # Fermi level Newton update
        small_gradients = torch.abs(total_responses) < 1e-6
        delta_Q = target_charge - _total_charge
        delta_Q[small_gradients] = 1.0

        delta_mu = torch.divide(delta_Q, total_responses)
        delta_mu[small_gradients] = -torch.sign(delta_Q[small_gradients])

        if terminate_after_update or step_i == num_scf_steps - 2:
            fermi_level = fermi_level + delta_mu
            break
        else:
            fermi_level = fermi_level + delta_mu.clamp(-1.0, 1.0)

    charges_history = torch.stack(charges_history, dim=-1)
    return SCFState(
        density=charge_density,
        fermi_level=fermi_level,
        field_feats=field_feats,
        charges_history=charges_history,
        terminated_step=terminated_step,
        status=status,
        final_avg_abs_change=final_avg_abs_change,
    )
