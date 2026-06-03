from typing import Dict, NamedTuple, Optional, Tuple

import torch

from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools.scatter import scatter_sum

from graph_longrange.utils import permute_to_e3nn_convention

from .compiled_kspace import KSpacePlanner
from .e3nn_compile_utils import replace_e3nn_script_codegen_with_fx
from .fixed_point_core import FixedPointCore

try:
    _dynamo_disable = torch.compiler.disable
except AttributeError:
    import torch._dynamo

    _dynamo_disable = torch._dynamo.disable


@_dynamo_disable
def _call_module(module, *args, **kwargs):
    return module(*args, **kwargs)


class CompiledFixedPointOptions(NamedTuple):
    backend: str
    mode: str
    dynamic: bool
    fullgraph: bool
    scope: str
    chunk_size: int


class PreparedFixedPointInputs(NamedTuple):
    core_args: Tuple[torch.Tensor, ...]
    positions: torch.Tensor
    external_field: torch.Tensor
    fermi_level: torch.Tensor
    total_charge: torch.Tensor


class CompiledFixedPointLocalState(NamedTuple):
    node_feats: torch.Tensor
    all_layer_feats: torch.Tensor
    edge_attrs: torch.Tensor
    edge_feats: torch.Tensor
    field_independent_charge_density: torch.Tensor
    energies: torch.Tensor
    electrostatics_plan: Dict[str, torch.Tensor]
    external_field_features: torch.Tensor


class FixedPointSCFCompiledCore(torch.nn.Module):
    def __init__(
        self,
        model: FixedPointCore,
        pbc_handling: str,
        num_scf_steps: int,
        mixing_parameter: float,
        constant_charge: bool,
    ):
        super().__init__()
        if pbc_handling not in ("pbc", "slab"):
            raise ValueError(
                "Compiled FixedPointSCF supports only "
                f"pbc_handling='pbc' or 'slab', got {pbc_handling!r}."
            )
        if num_scf_steps <= 1:
            raise ValueError("Compiled FixedPointSCF requires num_scf_steps > 1.")

        self.pbc_handling = pbc_handling
        self.num_scf_steps = int(num_scf_steps)
        self.mixing_parameter = float(mixing_parameter)
        self.constant_charge = bool(constant_charge)

        self.atomic_numbers = model.atomic_numbers
        self.charges_dim = model.charges_irreps.dim
        self.field_feature_norms = model.field_feature_norms
        self.fermi_level_offset = model.fermi_level_offset
        self.from_ell_max_field_update = getattr(
            model,
            "from_ell_max_field_update",
            9,
        )
        self.add_local_electron_energy = bool(model.add_local_electron_energy)

        self.atomic_energies_fn = model.atomic_energies_fn
        self.node_embedding = model.node_embedding
        self.radial_embedding = model.radial_embedding
        self.spherical_harmonics = model.spherical_harmonics
        self.interactions = model.interactions
        self.products = model.products
        self.readouts = model.readouts
        self.lr_source_maps = model.lr_source_maps
        self.layer_feature_mixer = model.layer_feature_mixer

        self.electric_potential_descriptor = model.electric_potential_descriptor
        self.external_field_contribution = model.external_field_contribution
        self.external_field_contribution_internal = (
            model.external_field_contribution_internal
        )
        self.field_dependent_charges_map = model.field_dependent_charges_map
        replace_e3nn_script_codegen_with_fx(self.field_dependent_charges_map)
        self.local_electron_energy = model.local_electron_energy
        self.coulomb_energy = model.coulomb_energy

    def _center_fermi_level(self, fermi_level: torch.Tensor) -> torch.Tensor:
        return fermi_level - self.fermi_level_offset

    @staticmethod
    def _sum_nodes(
        src: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        dim: int,
    ) -> torch.Tensor:
        if num_graphs == 1:
            if dim in (-1, 0):
                return src.sum(dim=0, keepdim=True)
            if dim == -2:
                return src.sum(dim=0, keepdim=True)
        return scatter_sum(
            src=src,
            index=batch,
            dim=dim,
            dim_size=num_graphs,
        )

    def _compute_geometry_features(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
        )
        edge_attrs = _call_module(
            self.spherical_harmonics,
            permute_to_e3nn_convention(vectors),
        )
        edge_feats, _ = _call_module(
            self.radial_embedding,
            lengths,
            node_attrs,
            edge_index,
            self.atomic_numbers,
        )
        return edge_attrs, edge_feats

    def _compute_external_field_features(
        self,
        batch: torch.Tensor,
        positions: torch.Tensor,
        external_field: torch.Tensor,
    ) -> torch.Tensor:
        efield_potential = torch.zeros(
            (external_field.size(0), 4),
            dtype=positions.dtype,
            device=positions.device,
        )
        efield_potential[:, 1:] = external_field
        return _call_module(
            self.external_field_contribution,
            batch,
            positions,
            efield_potential,
        )

    def _compute_fermi_level_features(
        self,
        batch: torch.Tensor,
        positions: torch.Tensor,
        fermi_level: torch.Tensor,
    ) -> torch.Tensor:
        fermi_potential = torch.zeros(
            (fermi_level.numel(), 4),
            dtype=positions.dtype,
            device=positions.device,
        )
        fermi_potential[:, 0] = self._center_fermi_level(fermi_level)
        return _call_module(
            self.external_field_contribution,
            batch,
            positions,
            fermi_potential,
        )

    def _compute_node_fermi_level_features(
        self,
        batch: torch.Tensor,
        positions: torch.Tensor,
        node_fermi_level: torch.Tensor,
    ) -> torch.Tensor:
        node_potential = torch.zeros(
            (node_fermi_level.numel(), 4),
            dtype=positions.dtype,
            device=positions.device,
        )
        node_potential[:, 0] = self._center_fermi_level(node_fermi_level)
        return self.external_field_contribution_internal(
            batch,
            positions,
            node_potential,
        )

    def _sum_node_scalars(
        self,
        values: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if num_graphs == 1:
            return values.sum(dim=0, keepdim=True)
        return scatter_sum(
            src=values,
            index=batch,
            dim=-1,
            dim_size=num_graphs,
        )

    @staticmethod
    def _replace_monopole(
        density: torch.Tensor,
        monopole: torch.Tensor,
    ) -> torch.Tensor:
        monopole = monopole.unsqueeze(-1)
        if density.shape[1] == 1:
            return monopole
        return torch.cat((monopole, density[:, 1:]), dim=-1)

    def _encode_local_state(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        external_field: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> CompiledFixedPointLocalState:
        num_graphs = ptr.numel() - 1

        node_e0 = _call_module(self.atomic_energies_fn, node_attrs).squeeze(-1)
        e0 = self._sum_nodes(
            src=node_e0,
            batch=batch,
            num_graphs=num_graphs,
            dim=-1,
        )

        node_feats = _call_module(self.node_embedding, node_attrs)
        edge_attrs, edge_feats = self._compute_geometry_features(
            node_attrs=node_attrs,
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
        )

        energies = [e0]
        features = []
        charge_density = torch.zeros(
            (batch.size(0), self.charges_dim),
            device=batch.device,
            dtype=positions.dtype,
        )

        for interaction, product, readout, lr_source_map in zip(
            self.interactions,
            self.products,
            self.readouts,
            self.lr_source_maps,
        ):
            node_feats, sc = _call_module(
                interaction,
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = _call_module(
                product,
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )
            features.append(node_feats.clone())
            node_energies = _call_module(readout, node_feats).squeeze(-1)
            energy = self._sum_nodes(
                src=node_energies,
                batch=batch,
                num_graphs=num_graphs,
                dim=-1,
            )
            energies.append(energy)

            charge_sources = _call_module(
                lr_source_map,
                node_attrs=node_attrs,
                node_feats=node_feats,
            )
            charge_density = charge_density + charge_sources.squeeze(-2)

        all_layer_feats = _call_module(
            self.layer_feature_mixer,
            torch.stack(features, dim=0),
        )
        stacked_energies = torch.stack(energies, dim=-1)

        electrostatics_plan = _call_module(
            self.electric_potential_descriptor.precompute_geometry,
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc.view(-1, 3),
        )
        external_field_features = self._compute_external_field_features(
            batch=batch,
            positions=positions,
            external_field=external_field,
        )

        return CompiledFixedPointLocalState(
            node_feats=node_feats,
            all_layer_feats=all_layer_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            field_independent_charge_density=charge_density,
            energies=stacked_energies,
            electrostatics_plan=electrostatics_plan,
            external_field_features=external_field_features,
        )

    def _scf_step(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
        charge_density: torch.Tensor,
        fermi_level_features: torch.Tensor,
        total_charges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if total_charges is None:
            total_charges = charge_density
        field_feats = self.electric_potential_descriptor.forward_dynamic(
            cache=local_state.electrostatics_plan,
            source_feats=charge_density,
        )
        field_feats = field_feats + local_state.external_field_features
        field_feats = field_feats + fermi_level_features
        field_feats = field_feats / self.field_feature_norms

        field_dep_contribution = self.field_dependent_charges_map(
            node_attrs=node_attrs,
            node_feats=local_state.all_layer_feats,
            edge_attrs=local_state.edge_attrs[:, : self.from_ell_max_field_update],
            edge_feats=local_state.edge_feats,
            edge_index=edge_index,
            potential_features=field_feats,
            local_charges=local_state.field_independent_charge_density,
            total_charges=total_charges,
        )
        return field_dep_contribution, field_feats

    def _run_fixed_step_constant_fermi_scf(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        positions: torch.Tensor,
        fermi_level: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fermi_level_features = self._compute_fermi_level_features(
            batch=batch,
            positions=positions,
            fermi_level=fermi_level,
        )
        charge_density = local_state.field_independent_charge_density.clone()
        field_feats = local_state.external_field_features + fermi_level_features

        for _ in range(self.num_scf_steps - 1):
            field_dep, field_feats = self._scf_step(
                node_attrs=node_attrs,
                edge_index=edge_index,
                local_state=local_state,
                charge_density=charge_density,
                fermi_level_features=fermi_level_features,
            )
            new_density = local_state.field_independent_charge_density + field_dep
            charge_density = charge_density + (
                new_density - charge_density
            ) * self.mixing_parameter

        return charge_density, field_feats

    def _run_fixed_step_constant_fermi_scf_chunk(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        positions: torch.Tensor,
        fermi_level: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
        charge_density: torch.Tensor,
        chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fermi_level_features = self._compute_fermi_level_features(
            batch=batch,
            positions=positions,
            fermi_level=fermi_level,
        )
        field_feats = local_state.external_field_features + fermi_level_features

        for _ in range(chunk_size):
            field_dep, field_feats = self._scf_step(
                node_attrs=node_attrs,
                edge_index=edge_index,
                local_state=local_state,
                charge_density=charge_density,
                fermi_level_features=fermi_level_features,
            )
            new_density = local_state.field_independent_charge_density + field_dep
            charge_density = charge_density + (
                new_density - charge_density
            ) * self.mixing_parameter

        return charge_density, field_feats

    def _constant_charge_redistribution(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        positions: torch.Tensor,
        target_charge: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
        charge_density_for_field: torch.Tensor,
        total_charges_for_update: torch.Tensor,
        fermi_level: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_graphs = target_charge.numel()
        node_fermi = torch.index_select(fermi_level, 0, batch)

        def response_fn(mu):
            mu_features = self._compute_node_fermi_level_features(
                batch=batch,
                positions=positions,
                node_fermi_level=mu,
            )
            field_dep, _ = self._scf_step(
                node_attrs=node_attrs,
                edge_index=edge_index,
                local_state=local_state,
                charge_density=charge_density_for_field,
                fermi_level_features=mu_features,
                total_charges=total_charges_for_update,
            )
            return field_dep[:, 0].sum()

        fermi_level_features = self._compute_node_fermi_level_features(
            batch=batch,
            positions=positions,
            node_fermi_level=node_fermi,
        )
        field_dep_contribution, field_feats = self._scf_step(
            node_attrs=node_attrs,
            edge_index=edge_index,
            local_state=local_state,
            charge_density=charge_density_for_field,
            fermi_level_features=fermi_level_features,
            total_charges=total_charges_for_update,
        )
        new_density = local_state.field_independent_charge_density + field_dep_contribution

        dq_dmu = torch.func.grad(response_fn)(node_fermi)

        total_charge = self._sum_node_scalars(
            values=total_charges_for_update[:, 0],
            batch=batch,
            num_graphs=num_graphs,
        )
        total_responses = self._sum_node_scalars(
            values=dq_dmu,
            batch=batch,
            num_graphs=num_graphs,
        )
        response_per_node = torch.index_select(total_responses, 0, batch)
        fukui_functions = dq_dmu / response_per_node
        charge_deficit = torch.index_select(
            target_charge - total_charge,
            0,
            batch,
        )
        redistributed_charge = (
            total_charges_for_update[:, 0] + fukui_functions * charge_deficit
        )
        redistributed_density = self._replace_monopole(
            total_charges_for_update,
            redistributed_charge,
        )
        return redistributed_density, field_feats, total_charge, total_responses

    def _run_fixed_step_constant_charge_scf(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        positions: torch.Tensor,
        fermi_level: torch.Tensor,
        target_charge: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        charge_density = local_state.field_independent_charge_density.clone()
        fermi_level = fermi_level.clone()

        charge_density_renormalized, field_feats, _, _ = (
            self._constant_charge_redistribution(
                node_attrs=node_attrs,
                edge_index=edge_index,
                batch=batch,
                positions=positions,
                target_charge=target_charge,
                local_state=local_state,
                charge_density_for_field=charge_density,
                total_charges_for_update=charge_density,
                fermi_level=fermi_level,
            )
        )

        for step_i in range(self.num_scf_steps - 1):
            node_fermi = torch.index_select(fermi_level, 0, batch)

            def response_fn(mu):
                mu_features = self._compute_node_fermi_level_features(
                    batch=batch,
                    positions=positions,
                    node_fermi_level=mu,
                )
                field_dep_response, _ = self._scf_step(
                    node_attrs=node_attrs,
                    edge_index=edge_index,
                    local_state=local_state,
                    charge_density=charge_density_renormalized,
                    fermi_level_features=mu_features,
                    total_charges=charge_density,
                )
                return field_dep_response[:, 0].sum()

            fermi_level_features = self._compute_node_fermi_level_features(
                batch=batch,
                positions=positions,
                node_fermi_level=node_fermi,
            )
            field_dep, field_feats = self._scf_step(
                node_attrs=node_attrs,
                edge_index=edge_index,
                local_state=local_state,
                charge_density=charge_density_renormalized,
                fermi_level_features=fermi_level_features,
                total_charges=charge_density,
            )
            new_density = local_state.field_independent_charge_density + field_dep
            charge_density = charge_density_renormalized + (
                new_density - charge_density_renormalized
            ) * self.mixing_parameter

            dq_dmu = torch.func.grad(response_fn)(node_fermi)
            num_graphs = target_charge.numel()
            total_charge = self._sum_node_scalars(
                values=charge_density[:, 0],
                batch=batch,
                num_graphs=num_graphs,
            )
            total_responses = self._sum_node_scalars(
                values=dq_dmu,
                batch=batch,
                num_graphs=num_graphs,
            )
            fukui_functions = dq_dmu / torch.index_select(total_responses, 0, batch)
            charge_deficit = torch.index_select(
                target_charge - total_charge,
                0,
                batch,
            )
            redistributed_charge = (
                charge_density[:, 0] + fukui_functions * charge_deficit
            )
            charge_density_renormalized = self._replace_monopole(
                charge_density,
                redistributed_charge,
            )

            small_gradients = torch.abs(total_responses) < 1.0e-6
            delta_q = target_charge - total_charge
            safe_delta_q = torch.where(
                small_gradients,
                torch.ones_like(delta_q),
                delta_q,
            )
            safe_responses = torch.where(
                small_gradients,
                torch.ones_like(total_responses),
                total_responses,
            )
            delta_mu = safe_delta_q / safe_responses
            delta_mu = torch.where(
                small_gradients,
                -torch.sign(delta_q),
                delta_mu,
            )
            if step_i < self.num_scf_steps - 2:
                delta_mu = delta_mu.clamp(-1.0, 1.0)
            fermi_level = fermi_level + delta_mu

        return charge_density, fermi_level, field_feats

    def _run_fixed_step_constant_charge_scf_chunk(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        positions: torch.Tensor,
        target_charge: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
        charge_density: torch.Tensor,
        charge_density_renormalized: torch.Tensor,
        fermi_level: torch.Tensor,
        clamp_delta_mu: torch.Tensor,
        chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        field_feats = local_state.external_field_features

        for step_i in range(chunk_size):
            node_fermi = torch.index_select(fermi_level, 0, batch)

            def response_fn(mu):
                mu_features = self._compute_node_fermi_level_features(
                    batch=batch,
                    positions=positions,
                    node_fermi_level=mu,
                )
                field_dep_response, _ = self._scf_step(
                    node_attrs=node_attrs,
                    edge_index=edge_index,
                    local_state=local_state,
                    charge_density=charge_density_renormalized,
                    fermi_level_features=mu_features,
                    total_charges=charge_density,
                )
                return field_dep_response[:, 0].sum()

            fermi_level_features = self._compute_node_fermi_level_features(
                batch=batch,
                positions=positions,
                node_fermi_level=node_fermi,
            )
            field_dep, field_feats = self._scf_step(
                node_attrs=node_attrs,
                edge_index=edge_index,
                local_state=local_state,
                charge_density=charge_density_renormalized,
                fermi_level_features=fermi_level_features,
                total_charges=charge_density,
            )
            new_density = local_state.field_independent_charge_density + field_dep
            charge_density = charge_density_renormalized + (
                new_density - charge_density_renormalized
            ) * self.mixing_parameter

            dq_dmu = torch.func.grad(response_fn)(node_fermi)
            num_graphs = target_charge.numel()
            total_charge = self._sum_node_scalars(
                values=charge_density[:, 0],
                batch=batch,
                num_graphs=num_graphs,
            )
            total_responses = self._sum_node_scalars(
                values=dq_dmu,
                batch=batch,
                num_graphs=num_graphs,
            )
            fukui_functions = dq_dmu / torch.index_select(total_responses, 0, batch)
            charge_deficit = torch.index_select(
                target_charge - total_charge,
                0,
                batch,
            )
            redistributed_charge = (
                charge_density[:, 0] + fukui_functions * charge_deficit
            )
            charge_density_renormalized = self._replace_monopole(
                charge_density,
                redistributed_charge,
            )

            small_gradients = torch.abs(total_responses) < 1.0e-6
            delta_q = target_charge - total_charge
            safe_delta_q = torch.where(
                small_gradients,
                torch.ones_like(delta_q),
                delta_q,
            )
            safe_responses = torch.where(
                small_gradients,
                torch.ones_like(total_responses),
                total_responses,
            )
            delta_mu = safe_delta_q / safe_responses
            delta_mu = torch.where(
                small_gradients,
                -torch.sign(delta_q),
                delta_mu,
            )
            clamped_delta_mu = delta_mu.clamp(-1.0, 1.0)
            should_clamp = clamp_delta_mu[step_i].to(dtype=torch.bool)
            delta_mu = torch.where(should_clamp, clamped_delta_mu, delta_mu)
            fermi_level = fermi_level + delta_mu

        return charge_density, charge_density_renormalized, fermi_level, field_feats

    def _compute_coulomb_energy(
        self,
        density: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.coulomb_energy.forward(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=density,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc.view(-1, 3),
        )

    @staticmethod
    def _compute_total_dipole(
        density: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if num_graphs == 1:
            dipole = (positions * density[:, :1]).sum(dim=0, keepdim=True)
            if density.shape[1] > 1:
                dipole = dipole + density[:, 1:4].sum(dim=0, keepdim=True)[:, [2, 0, 1]]
            return dipole
        dipole = scatter_sum(
            src=positions * density[:, :1],
            index=batch.unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )
        if density.shape[1] > 1:
            dipole_p = scatter_sum(
                src=density[:, 1:4],
                index=batch,
                dim=-2,
                dim_size=num_graphs,
            )
            dipole = dipole + dipole_p[:, [2, 0, 1]]
        return dipole

    def _compute_final_observables(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        external_field: torch.Tensor,
        fermi_level: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
        density: torch.Tensor,
        field_feats: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        num_graphs = ptr.numel() - 1
        total_energy = torch.sum(local_state.energies, dim=-1)

        if self.add_local_electron_energy:
            local_q_e = _call_module(
                self.local_electron_energy,
                node_attrs=node_attrs,
                node_feats=local_state.node_feats,
                edge_attrs=local_state.edge_attrs[:, : self.from_ell_max_field_update],
                edge_feats=local_state.edge_feats,
                edge_index=edge_index,
                field_feats=field_feats,
                charges_0=local_state.field_independent_charge_density,
                charges_induced=density,
            )
            electron_energy = self._sum_nodes(
                src=local_q_e,
                batch=batch,
                num_graphs=num_graphs,
                dim=-1,
            )
            total_energy = total_energy + electron_energy
        else:
            electron_energy = torch.zeros_like(total_energy)

        dipole = self._compute_total_dipole(
            density,
            positions,
            batch,
            num_graphs,
        )
        electrostatic_energy = self._compute_coulomb_energy(
            density=density,
            positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
        )
        field_energy = torch.sum(external_field * dipole, dim=-1)
        total_energy = total_energy + electrostatic_energy + field_energy

        return (
            total_energy,
            density,
            fermi_level,
            dipole,
            electrostatic_energy,
            electron_energy,
            local_state.field_independent_charge_density,
            field_feats,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        cell: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        external_field: torch.Tensor,
        fermi_level: torch.Tensor,
        total_charge: torch.Tensor,
        head: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del cell, head
        local_state = self._encode_local_state(
            node_attrs=node_attrs,
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
            batch=batch,
            ptr=ptr,
            volume=volume,
            pbc=pbc,
            external_field=external_field,
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
        )
        if self.constant_charge:
            density, fermi_level, field_feats = (
                self._run_fixed_step_constant_charge_scf(
                    node_attrs=node_attrs,
                    edge_index=edge_index,
                    batch=batch,
                    positions=positions,
                    fermi_level=fermi_level,
                    target_charge=total_charge,
                    local_state=local_state,
                )
            )
        else:
            density, field_feats = self._run_fixed_step_constant_fermi_scf(
                node_attrs=node_attrs,
                edge_index=edge_index,
                batch=batch,
                positions=positions,
                fermi_level=fermi_level,
                local_state=local_state,
            )
        return self._compute_final_observables(
            node_attrs=node_attrs,
            positions=positions,
            edge_index=edge_index,
            batch=batch,
            ptr=ptr,
            volume=volume,
            pbc=pbc,
            external_field=external_field,
            fermi_level=fermi_level,
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            local_state=local_state,
            density=density,
            field_feats=field_feats,
        )


class FixedPointSCFCompiledRegion(torch.nn.Module):
    def __init__(
        self,
        core: FixedPointSCFCompiledCore,
        include_observables: bool,
        constant_charge: bool,
    ):
        super().__init__()
        self.core = core
        self.include_observables = bool(include_observables)
        self.constant_charge = bool(constant_charge)

    def forward(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        external_field: torch.Tensor,
        fermi_level: torch.Tensor,
        total_charge: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.constant_charge:
            density, fermi_level, field_feats = (
                self.core._run_fixed_step_constant_charge_scf(
                    node_attrs=node_attrs,
                    edge_index=edge_index,
                    batch=batch,
                    positions=positions,
                    fermi_level=fermi_level,
                    target_charge=total_charge,
                    local_state=local_state,
                )
            )
        else:
            density, field_feats = self.core._run_fixed_step_constant_fermi_scf(
                node_attrs=node_attrs,
                edge_index=edge_index,
                batch=batch,
                positions=positions,
                fermi_level=fermi_level,
                local_state=local_state,
            )
        if not self.include_observables:
            dummy_energy = density.new_zeros((ptr.numel() - 1,))
            dummy_dipole = density.new_zeros((ptr.numel() - 1, 3))
            return (
                dummy_energy,
                density,
                fermi_level,
                dummy_dipole,
                dummy_energy,
                dummy_energy,
                local_state.field_independent_charge_density,
                field_feats,
            )

        return self.core._compute_final_observables(
            node_attrs=node_attrs,
            positions=positions,
            edge_index=edge_index,
            batch=batch,
            ptr=ptr,
            volume=volume,
            pbc=pbc,
            external_field=external_field,
            fermi_level=fermi_level,
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            local_state=local_state,
            density=density,
            field_feats=field_feats,
        )


class FixedPointSCFCompiledChunk(torch.nn.Module):
    def __init__(
        self,
        core: FixedPointSCFCompiledCore,
        constant_charge: bool,
        chunk_size: int,
    ):
        super().__init__()
        self.core = core
        self.constant_charge = bool(constant_charge)
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        fermi_level: torch.Tensor,
        total_charge: torch.Tensor,
        local_state: CompiledFixedPointLocalState,
        charge_density: torch.Tensor,
        charge_density_renormalized: torch.Tensor,
        clamp_delta_mu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.constant_charge:
            return self.core._run_fixed_step_constant_charge_scf_chunk(
                node_attrs=node_attrs,
                edge_index=edge_index,
                batch=batch,
                positions=positions,
                target_charge=total_charge,
                local_state=local_state,
                charge_density=charge_density,
                charge_density_renormalized=charge_density_renormalized,
                fermi_level=fermi_level,
                clamp_delta_mu=clamp_delta_mu,
                chunk_size=self.chunk_size,
            )

        density, field_feats = self.core._run_fixed_step_constant_fermi_scf_chunk(
            node_attrs=node_attrs,
            edge_index=edge_index,
            batch=batch,
            positions=positions,
            fermi_level=fermi_level,
            local_state=local_state,
            charge_density=charge_density,
            chunk_size=self.chunk_size,
        )
        return density, density, fermi_level, field_feats


class CompiledFixedPointEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        pbc_handling: str,
        num_scf_steps: int,
        mixing_parameter: float,
        constant_charge: bool,
        compile_options: Optional[CompiledFixedPointOptions] = None,
    ):
        if not isinstance(model, FixedPointCore):
            raise NotImplementedError(
                "Compiled fixed-point evaluation is currently implemented "
                "only for FixedPointCore models."
            )
        if pbc_handling not in ("pbc", "slab"):
            raise ValueError(
                "Compiled fixed-point evaluation supports only "
                f"pbc_handling='pbc' or 'slab', got {pbc_handling!r}."
            )
        if num_scf_steps <= 1:
            raise ValueError("Compiled fixed-point evaluation requires num_scf_steps > 1.")

        self.model = model
        self.pbc_handling = pbc_handling
        self.num_scf_steps = int(num_scf_steps)
        self.mixing_parameter = float(mixing_parameter)
        self.constant_charge = bool(constant_charge)
        self.kspace_planner = KSpacePlanner(model.kspace_cutoff)
        self.compile_scope = "none" if compile_options is None else compile_options.scope
        if self.compile_scope not in (
            "none",
            "scf",
            "scf_observables",
            "scf_chunk",
            "full",
        ):
            raise ValueError(
                "compile scope must be one of 'none', 'scf', 'scf_observables', "
                f"'scf_chunk', or 'full', got {self.compile_scope!r}."
            )
        self.compile_chunk_size = 0 if compile_options is None else int(
            compile_options.chunk_size
        )
        if self.compile_scope == "scf_chunk":
            if self.compile_chunk_size <= 0:
                raise ValueError("compile_chunk_size must be positive for scf_chunk.")
            self._validate_chunk_num_scf_steps(self.num_scf_steps)

        core = FixedPointSCFCompiledCore(
            model=model,
            pbc_handling=pbc_handling,
            num_scf_steps=self.num_scf_steps,
            mixing_parameter=self.mixing_parameter,
            constant_charge=self.constant_charge,
        )
        self.core = core
        if compile_options is None:
            self.compiled_core = core
            self.compiled_region = None
            self.compiled_chunk = None
        else:
            try:
                if compile_options.scope == "full":
                    self.compiled_core = torch.compile(
                        core,
                        backend=compile_options.backend,
                        mode=compile_options.mode,
                        dynamic=compile_options.dynamic,
                        fullgraph=compile_options.fullgraph,
                    )
                    self.compiled_region = None
                    self.compiled_chunk = None
                elif compile_options.scope == "scf_chunk":
                    self.compiled_core = core
                    self.compiled_region = None
                    chunk = FixedPointSCFCompiledChunk(
                        core=core,
                        constant_charge=self.constant_charge,
                        chunk_size=self.compile_chunk_size,
                    )
                    self.compiled_chunk = torch.compile(
                        chunk,
                        backend=compile_options.backend,
                        mode=compile_options.mode,
                        dynamic=compile_options.dynamic,
                        fullgraph=compile_options.fullgraph,
                    )
                else:
                    self.compiled_core = core
                    self.compiled_chunk = None
                    region = FixedPointSCFCompiledRegion(
                        core=core,
                        include_observables=compile_options.scope == "scf_observables",
                        constant_charge=self.constant_charge,
                    )
                    self.compiled_region = torch.compile(
                        region,
                        backend=compile_options.backend,
                        mode=compile_options.mode,
                        dynamic=compile_options.dynamic,
                        fullgraph=compile_options.fullgraph,
                    )
            except (AttributeError, RuntimeError) as exc:
                raise RuntimeError(
                    "Failed to initialize torch.compile for "
                    "FixedPointSCF compiled evaluation. Check that the installed "
                    "PyTorch/Python combination supports torch.compile."
                ) from exc

    def _validate_chunk_num_scf_steps(self, num_scf_steps: int) -> None:
        if num_scf_steps <= 1:
            raise ValueError("Compiled fixed-point evaluation requires num_scf_steps > 1.")
        if (num_scf_steps - 1) % self.compile_chunk_size != 0:
            raise ValueError(
                "compile_scope='scf_chunk' requires (num_scf_steps - 1) "
                "to be an exact multiple of compile_chunk_size."
            )

    def set_num_scf_steps(self, num_scf_steps: int) -> None:
        num_scf_steps = int(num_scf_steps)
        if self.compile_scope == "scf_chunk":
            self._validate_chunk_num_scf_steps(num_scf_steps)
            self.num_scf_steps = num_scf_steps
            return
        raise ValueError(
            "Changing num_scf_steps after construction is supported only for "
            "compile_scope='scf_chunk'."
        )

    def _prepare_inputs(self, data: Dict[str, torch.Tensor]) -> PreparedFixedPointInputs:
        positions = data["positions"].detach().clone().requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        external_field = data.get("external_field")
        if external_field is None:
            external_field = torch.zeros(
                (num_graphs, 3),
                dtype=positions.dtype,
                device=positions.device,
            )
        else:
            external_field = external_field.view(num_graphs, 3)

        fermi_level = data.get("fermi_level")
        if fermi_level is None:
            fermi_level = self.model.fermi_level_offset.expand(num_graphs).clone()
        else:
            fermi_level = fermi_level.view(num_graphs)
        if self.constant_charge:
            fermi_level = fermi_level.detach().clone().requires_grad_(True)

        total_charge = data.get("total_charge")
        if total_charge is None:
            total_charge = torch.zeros(
                (num_graphs,),
                dtype=positions.dtype,
                device=positions.device,
            )
        else:
            total_charge = total_charge.view(num_graphs)

        head = data.get("head")
        if head is None:
            head = torch.zeros(
                (num_graphs,),
                dtype=torch.long,
                device=positions.device,
            )
        else:
            head = head.to(device=positions.device, dtype=torch.long).view(-1)

        cell = data["cell"].view(-1, 3, 3)
        volume = data["volume"].view(-1)
        pbc = data["pbc"].view(-1, 3)
        plan = self.kspace_planner.get_plan(
            cell=cell,
            rcell=data["rcell"].view(-1, 3, 3),
            pbc_handling=self.pbc_handling,
        )

        core_args = (
            data["node_attrs"],
            positions,
            data["edge_index"],
            data["shifts"],
            data["batch"],
            data["ptr"],
            cell,
            volume,
            pbc,
            external_field,
            fermi_level,
            total_charge,
            head,
            plan.k_vectors,
            plan.k_norm2,
            plan.k_vector_batch,
            plan.k0_mask,
        )
        return PreparedFixedPointInputs(
            core_args=core_args,
            positions=positions,
            external_field=external_field,
            fermi_level=fermi_level,
            total_charge=total_charge,
        )

    def evaluate(self, data: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        prepared = self._prepare_inputs(data)
        if self.compiled_chunk is not None:
            self._validate_chunk_num_scf_steps(self.num_scf_steps)
            (
                node_attrs,
                positions,
                edge_index,
                shifts,
                batch,
                ptr,
                cell,
                volume,
                pbc,
                external_field,
                fermi_level,
                total_charge,
                head,
                k_vectors,
                k_norm2,
                k_vector_batch,
                k0_mask,
            ) = prepared.core_args
            del cell, head

            local_state = self.core._encode_local_state(
                node_attrs=node_attrs,
                positions=positions,
                edge_index=edge_index,
                shifts=shifts,
                batch=batch,
                ptr=ptr,
                volume=volume,
                pbc=pbc,
                external_field=external_field,
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
            )
            charge_density = local_state.field_independent_charge_density.clone()
            charge_density_renormalized = charge_density

            if self.constant_charge:
                charge_density_renormalized, field_feats, _, _ = (
                    self.core._constant_charge_redistribution(
                        node_attrs=node_attrs,
                        edge_index=edge_index,
                        batch=batch,
                        positions=positions,
                        target_charge=total_charge,
                        local_state=local_state,
                        charge_density_for_field=charge_density,
                        total_charges_for_update=charge_density,
                        fermi_level=fermi_level,
                    )
                )
            else:
                field_feats = local_state.external_field_features

            num_chunks = (self.num_scf_steps - 1) // self.compile_chunk_size
            for chunk_i in range(num_chunks):
                clamp_delta_mu = torch.ones(
                    (self.compile_chunk_size,),
                    dtype=torch.bool,
                    device=positions.device,
                )
                if self.constant_charge and chunk_i == num_chunks - 1:
                    clamp_delta_mu[-1] = False
                (
                    charge_density,
                    charge_density_renormalized,
                    fermi_level,
                    field_feats,
                ) = self.compiled_chunk(
                    node_attrs,
                    positions,
                    edge_index,
                    batch,
                    fermi_level,
                    total_charge,
                    local_state,
                    charge_density,
                    charge_density_renormalized,
                    clamp_delta_mu,
                )

            (
                energy,
                density_coefficients,
                fermi_level,
                dipole,
                electrostatic_energy,
                electron_energy,
                field_independent_charge_density,
                electrostatic_features,
            ) = self.core._compute_final_observables(
                node_attrs=node_attrs,
                positions=positions,
                edge_index=edge_index,
                batch=batch,
                ptr=ptr,
                volume=volume,
                pbc=pbc,
                external_field=external_field,
                fermi_level=fermi_level,
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
                local_state=local_state,
                density=charge_density,
                field_feats=field_feats,
            )
        elif self.compiled_region is None:
            (
                energy,
                density_coefficients,
                fermi_level,
                dipole,
                electrostatic_energy,
                electron_energy,
                field_independent_charge_density,
                electrostatic_features,
            ) = self.compiled_core(*prepared.core_args)
        else:
            (
                node_attrs,
                positions,
                edge_index,
                shifts,
                batch,
                ptr,
                cell,
                volume,
                pbc,
                external_field,
                fermi_level_input,
                total_charge,
                head,
                k_vectors,
                k_norm2,
                k_vector_batch,
                k0_mask,
            ) = prepared.core_args
            del cell, head
            local_state = self.core._encode_local_state(
                node_attrs=node_attrs,
                positions=positions,
                edge_index=edge_index,
                shifts=shifts,
                batch=batch,
                ptr=ptr,
                volume=volume,
                pbc=pbc,
                external_field=external_field,
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
            )
            (
                energy,
                density_coefficients,
                fermi_level,
                dipole,
                electrostatic_energy,
                electron_energy,
                field_independent_charge_density,
                electrostatic_features,
            ) = self.compiled_region(
                node_attrs,
                positions,
                edge_index,
                batch,
                ptr,
                volume,
                pbc,
                external_field,
                fermi_level_input,
                total_charge,
                k_vectors,
                k_norm2,
                k_vector_batch,
                k0_mask,
                local_state,
            )
            if self.compile_scope == "scf":
                (
                    energy,
                    density_coefficients,
                    fermi_level,
                    dipole,
                    electrostatic_energy,
                    electron_energy,
                    field_independent_charge_density,
                    electrostatic_features,
                ) = self.core._compute_final_observables(
                    node_attrs=node_attrs,
                    positions=positions,
                    edge_index=edge_index,
                    batch=batch,
                    ptr=ptr,
                    volume=volume,
                    pbc=pbc,
                    external_field=external_field,
                    fermi_level=fermi_level,
                    k_vectors=k_vectors,
                    k_norm2=k_norm2,
                    k_vector_batch=k_vector_batch,
                    k0_mask=k0_mask,
                    local_state=local_state,
                    density=density_coefficients,
                    field_feats=electrostatic_features,
                )
        forces = -torch.autograd.grad(
            energy.sum(),
            prepared.positions,
            retain_graph=False,
            create_graph=False,
        )[0]
        charges_history = density_coefficients.unsqueeze(-1).expand(
            *density_coefficients.shape,
            self.num_scf_steps,
        )
        return {
            "energy": energy,
            "forces": forces,
            "stress": None,
            "density_coefficients": density_coefficients,
            "charges_history": charges_history,
            "fermi_level": fermi_level,
            "external_field": prepared.external_field,
            "dipole": dipole,
            "electrostatic_energy": electrostatic_energy,
            "electrostatic_features": electrostatic_features,
            "electron_energy": electron_energy,
            "field_independent_charge_density": field_independent_charge_density,
            "esps": None,
            "esps_dft": None,
            "polarizability": None,
        }


def build_compiled_fixed_point_evaluator(
    model: torch.nn.Module,
    pbc_handling: str,
    num_scf_steps: int,
    mixing_parameter: float,
    constant_charge: bool = False,
    backend: str = "inductor",
    mode: str = "default",
    dynamic: bool = True,
    fullgraph: bool = False,
    enabled: bool = True,
    scope: str = "scf",
    chunk_size: int = 1,
) -> CompiledFixedPointEvaluator:
    compile_options = None
    if enabled:
        compile_options = CompiledFixedPointOptions(
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
            scope=scope,
            chunk_size=chunk_size,
        )
    return CompiledFixedPointEvaluator(
        model=model,
        pbc_handling=pbc_handling,
        num_scf_steps=num_scf_steps,
        mixing_parameter=mixing_parameter,
        constant_charge=constant_charge,
        compile_options=compile_options,
    )
