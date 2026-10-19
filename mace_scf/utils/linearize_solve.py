import torch
from mace.tools.scatter import scatter_sum

JACOBIAN_VMAP_CHUNK_SIZE = 512 # reduce incidence of OOM errors. Tuned for 80GB A100 w/o CUEQ.


def _jacobian_from_vmapped_vjps(outputs, inputs, create_graph: bool):
    """Build a dense Jacobian by vmapping VJPs over output basis vectors."""
    output_dim = outputs.numel()
    basis = torch.eye(output_dim, dtype=outputs.dtype, device=outputs.device)

    def vjp_row(output_grad):
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=output_grad,
            retain_graph=True,
            create_graph=create_graph,
        )[0]

    return torch.vmap(vjp_row, chunk_size=JACOBIAN_VMAP_CHUNK_SIZE)(basis)


def linearize_and_solve_density(
    model,
    data,
    local_state,
    solved_density,
    solved_fermi_level,
    constant_charge: bool = False,
    linear_solve: str = "inverse",
):
    """Dense prototype of implicit fixed-point differentiation.

    This is intentionally simple and expensive: it builds the full Jacobian of
    the fixed-point residual with autograd, then solves the linearized residual
    equation directly. In constant-charge mode, the solved Fermi level is added
    to the unknown vector and the total-charge constraint is appended to the
    residual.
    """
    if linear_solve != "inverse":
        raise ValueError(
            "linearize_solve prototype only supports linear_solve='inverse'"
        )

    density_shape = solved_density.shape
    flat_density_star = solved_density.detach().reshape(-1)
    fermi_star = solved_fermi_level.detach().reshape(-1)
    density_size = flat_density_star.numel()
    num_graphs = data["ptr"].numel() - 1

    if constant_charge:
        flat_star = torch.cat([flat_density_star, fermi_star]).clone()
    else:
        flat_star = flat_density_star.clone()
    flat_star.requires_grad_(True)

    def extract_state(flat_state):
        density = flat_state[:density_size].reshape(density_shape)
        if constant_charge:
            fermi_level = flat_state[density_size:]
        else:
            fermi_level = solved_fermi_level
        return density, fermi_level

    def residual_from_flat(flat_state):
        density, fermi_level = extract_state(flat_state)
        fermi_features = model.features_from_fermi_level(
            data["batch"],
            local_state.positions,
            fermi_level,
        )
        field_dep, _ = model.scf_step(
            data=data,
            local_state=local_state,
            charge_density_in=density,
            total_charges=density,
            fermi_level_features=fermi_features,
        )
        density_out = local_state.field_independent_charge_density + field_dep
        density_residual = (density - density_out).reshape(-1)
        if not constant_charge:
            return density_residual

        total_charge = scatter_sum(
            src=density[:, 0],
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )
        charge_residual = total_charge - data["total_charge"]
        return torch.cat([density_residual, charge_residual])

    residual = residual_from_flat(flat_star)
    jacobian = _jacobian_from_vmapped_vjps(residual, flat_star, create_graph=False)

    delta = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
    state_1 = flat_star - delta

    # Second chord step (frozen Jacobian) makes the solution accurate to
    # second order, so mixed derivatives d2x/dr dtheta (force and stress
    # parameter gradients) are exact. Checkable with
    # tests/test_fixed_point_parameter_gradients.py:
    # 1) one step, detached J: wrong (only first derivatives exact).
    # 2) one step, attached J: right only if d2R/dx2 == 0, e.g.
    #    nonlinearity_cls == 'NoNonLinearity'.
    # 3) two steps, attached or detached J: right. Detached (used here)
    #    avoids differentiating through the Jacobian construction,
    #    which halves required memory.
    # Third-order losses (e.g. Hessians) would require more testing.

    delta = torch.linalg.solve(jacobian, residual_from_flat(state_1).unsqueeze(-1)).squeeze(-1)
    solved_state = state_1 - delta

    density, fermi_level = extract_state(solved_state)

    fermi_features = model.features_from_fermi_level(
        data["batch"],
        local_state.positions,
        fermi_level,
    )

    field_dep, field_feats = model.scf_step(
        data=data,
        local_state=local_state,
        charge_density_in=density,
        total_charges=density,
        fermi_level_features=fermi_features,
    )
    del field_dep

    return density, fermi_level, field_feats
