import torch
import warnings

try:
    import torchopt
except ImportError:
    warnings.warn("torchopt not found, implicit differentiation not available")

from mace.tools.scatter import scatter_sum


def make_implicit_scf_module(
    model,
    solved_density,
    solved_fermi_level,
    positions,
    constant_charge,
    num_graphs,
    linear_solve="inverse",
):
    if linear_solve == "inverse":
        ls = torchopt.linear_solve.solve_inv()
    elif linear_solve == "normal_cg":
        ls = torchopt.linear_solve.solve_normal_cg(maxiter=1000, rtol=1e-8, atol=1e-8)
    else:
        raise ValueError(f"Unknown linear_solve method: {linear_solve}")

    class ImplicitSCFModule(
        torchopt.nn.ImplicitMetaGradientModule,
        linear_solve=ls,
    ):
        def __init__(self, model, solved_density, solved_fermi_level,
                     positions, constant_charge, num_graphs):
            super().__init__()
            self.meta_net = model
            self.positions = positions
            self._constant_charge = constant_charge
            self._charge_shape = solved_density.shape
            self._num_graphs = num_graphs

            if constant_charge:
                ext = torch.cat([
                    solved_density.flatten().detach(),
                    solved_fermi_level.detach(),
                ])
            else:
                ext = solved_density.flatten().detach()
            self.extended_charge_density = ext.clone().requires_grad_(True)

        def extract(self):
            n = self._charge_shape.numel()
            if self._constant_charge:
                density = self.extended_charge_density[:n].reshape(self._charge_shape)
                fermi = self.extended_charge_density[n:]
                return density, fermi
            return self.extended_charge_density.reshape(self._charge_shape), None

        def optimality(self, data):
            data_mod = dict(data)
            data_mod["positions"] = self.positions
            # compute_force=False avoids in-place requires_grad_() which is
            # forbidden inside functorch transforms. self.positions already
            # requires_grad as a meta-parameter, so the graph tracks through it.
            local_state = self.meta_net.local_part(data_mod, compute_force=False)

            density, fermi = self.extract()
            if fermi is None:
                fermi = data["fermi_level"]

            fermi_features = self.meta_net.features_from_fermi_level(
                data["batch"], local_state.positions, fermi
            )

            field_dep, _ = self.meta_net.scf_step(
                data_mod, local_state, density, density, fermi_features
            )
            density_out = local_state.field_independent_charge_density + field_dep

            diff_q = density - density_out
            if self._constant_charge:
                total_q = scatter_sum(
                    density[:, 0], data["batch"], dim=-1,
                    dim_size=self._num_graphs,
                )
                diff_f = total_q - data["total_charge"]
                return (torch.cat([diff_q.flatten(), diff_f]),)
            return (diff_q.flatten(),)

        def solve(self, data):
            return self

    return ImplicitSCFModule(
        model=model,
        solved_density=solved_density,
        solved_fermi_level=solved_fermi_level,
        positions=positions,
        constant_charge=constant_charge,
        num_graphs=num_graphs,
    )
