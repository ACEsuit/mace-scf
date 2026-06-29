from typing import Dict, NamedTuple, Optional, Tuple

import torch

from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools.scatter import scatter_sum

from graph_longrange.utils import permute_to_e3nn_convention

from .compiled_kspace import KSpacePlanner
from .localsources import LocalSplitCharges, LocalCharges
from .utils import compute_polarization

try:
    _dynamo_disable = torch.compiler.disable
except AttributeError:
    import torch._dynamo

    _dynamo_disable = torch._dynamo.disable


@_dynamo_disable
def _call_module(module, *args, **kwargs):
    return module(*args, **kwargs)


class CompiledLocalSourceOptions(NamedTuple):
    backend: str
    mode: str
    dynamic: bool
    fullgraph: bool


class PreparedLocalSourceInputs(NamedTuple):
    core_args: Tuple[torch.Tensor, ...]
    positions: torch.Tensor
    external_field: torch.Tensor


class LocalSplitChargesCompiledCore(torch.nn.Module):
    def __init__(self, model: LocalSplitCharges, pbc_handling: str):
        super().__init__()
        if pbc_handling not in ("pbc", "slab"):
            raise ValueError(
                "Compiled LocalSplitCharges supports only "
                f"pbc_handling='pbc' or 'slab', got {pbc_handling!r}."
            )
        self.pbc_handling = pbc_handling

        self.atomic_numbers = model.atomic_numbers
        self.charges_dim = model.charges_irreps.dim
        self.atomic_energies_fn = model.atomic_energies_fn
        self.node_embedding = model.node_embedding
        self.radial_embedding = model.radial_embedding
        self.spherical_harmonics = model.spherical_harmonics
        self.interactions = model.interactions
        self.products = model.products
        self.readouts = model.readouts
        self.oxidation_state_mixer = model.oxidation_state_mixer
        self.formal_charges = model.formal_charges
        self.lr_source_maps = model.lr_source_maps
        self.coulomb_energy = model.coulomb_energy

    def _compute_geometry_features(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return vectors, lengths, edge_attrs, edge_feats

    def _compute_atomic_base_energy(
        self,
        node_attrs: torch.Tensor,
        node_heads: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_e0_heads = _call_module(self.atomic_energies_fn, node_attrs)
        node_e0 = torch.gather(
            node_e0_heads,
            dim=1,
            index=node_heads.view(-1, 1),
        ).squeeze(-1)
        e0 = scatter_sum(
            src=node_e0,
            index=batch,
            dim=-1,
            dim_size=num_graphs,
        )
        return e0, node_e0

    def _initialize_density_and_formal_charge_terms(
        self,
        node_attrs: torch.Tensor,
        charges: torch.Tensor,
        positions: torch.Tensor,
        lengths: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        charge_density = torch.zeros(
            (batch.size(0), self.charges_dim),
            device=batch.device,
            dtype=positions.dtype,
        )
        edge_fluxes = torch.zeros_like(lengths)
        formal_charges = _call_module(self.formal_charges, node_attrs, charges)
        charge_density[:, 0] += formal_charges
        formal_charge_dipole = scatter_sum(
            src=positions * formal_charges.unsqueeze(-1),
            index=batch.unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )
        return charge_density, edge_fluxes, formal_charges, formal_charge_dipole

    def _run_local_interactions_and_sources(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        vectors: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        node_heads: torch.Tensor,
        formal_charges: torch.Tensor,
        charge_density: torch.Tensor,
        edge_fluxes: torch.Tensor,
        energies: list,
        node_energies_list: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for interaction, product, readout, charge_map in zip(
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
            node_energies = _call_module(readout, node_feats, node_heads)
            node_energies = torch.gather(
                node_energies,
                dim=1,
                index=node_heads.view(-1, 1),
            ).squeeze(-1)
            energy = scatter_sum(
                src=node_energies,
                index=batch,
                dim=-1,
                dim_size=num_graphs,
            )
            energies.append(energy)
            node_energies_list.append(node_energies)

            multipoles_contr, edge_fluxes_contr = _call_module(
                charge_map,
                node_attrs=node_attrs,
                node_formal_charges=formal_charges,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                edge_vectors=vectors,
                batch=batch,
                num_graphs=num_graphs,
            )
            edge_fluxes = edge_fluxes + edge_fluxes_contr
            charge_density = charge_density + multipoles_contr.squeeze(-2)

        contributions = torch.stack(energies, dim=-1)
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)
        return contributions, node_energy, charge_density, edge_fluxes

    @staticmethod
    def _compute_split_charge_dipole(
        charge_density: torch.Tensor,
        edge_fluxes: torch.Tensor,
        vectors: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        formal_charge_dipole: torch.Tensor,
    ) -> torch.Tensor:
        total_dipole = compute_polarization(
            charge_density,
            edge_fluxes.squeeze(-1),
            vectors,
            edge_index,
            batch,
            num_graphs,
        )
        return total_dipole + formal_charge_dipole

    def _compute_electrostatic_and_field_energy(
        self,
        charge_density: torch.Tensor,
        total_dipole: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        external_field: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> torch.Tensor:
        electro_energy = self.coulomb_energy.forward(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=charge_density,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc.view(-1, 3),
        )
        field_energy = torch.sum(total_dipole * external_field, dim=-1)
        return electro_energy + field_energy

    @staticmethod
    def _compute_total_dipole(
        charge_density: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        dipole = scatter_sum(
            src=positions * charge_density[:, :1],
            index=batch.unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )
        if charge_density.shape[1] > 1:
            dipole_p = scatter_sum(
                src=charge_density[:, 1:4],
                index=batch,
                dim=-2,
                dim_size=num_graphs,
            )
            dipole = dipole + dipole_p[:, [2, 0, 1]]
        return dipole

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
        charges: torch.Tensor,
        external_field: torch.Tensor,
        head: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del cell
        num_graphs = ptr.numel() - 1
        node_heads = head[batch]

        node_feats = _call_module(self.node_embedding, node_attrs)
        vectors, lengths, edge_attrs, edge_feats = self._compute_geometry_features(
            node_attrs=node_attrs,
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
        )
        e0, node_e0 = self._compute_atomic_base_energy(
            node_attrs=node_attrs,
            node_heads=node_heads,
            batch=batch,
            num_graphs=num_graphs,
        )
        (
            charge_density,
            edge_fluxes,
            formal_charges,
            formal_charge_dipole,
        ) = self._initialize_density_and_formal_charge_terms(
            node_attrs=node_attrs,
            charges=charges,
            positions=positions,
            lengths=lengths,
            batch=batch,
            num_graphs=num_graphs,
        )

        node_feats = _call_module(
            self.oxidation_state_mixer,
            node_attrs,
            node_feats,
            formal_charges,
        )
        (
            contributions,
            node_energy,
            charge_density,
            edge_fluxes,
        ) = self._run_local_interactions_and_sources(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            vectors=vectors,
            batch=batch,
            num_graphs=num_graphs,
            node_heads=node_heads,
            formal_charges=formal_charges,
            charge_density=charge_density,
            edge_fluxes=edge_fluxes,
            energies=[e0],
            node_energies_list=[node_e0],
        )

        total_energy = torch.sum(contributions, dim=-1)
        total_dipole = self._compute_split_charge_dipole(
            charge_density=charge_density,
            edge_fluxes=edge_fluxes,
            vectors=vectors,
            edge_index=edge_index,
            batch=batch,
            num_graphs=num_graphs,
            formal_charge_dipole=formal_charge_dipole,
        )
        total_energy = total_energy + self._compute_electrostatic_and_field_energy(
            charge_density=charge_density,
            total_dipole=total_dipole,
            positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
            external_field=external_field,
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
        )
        return total_energy, node_energy, charge_density, total_dipole


class LocalChargesCompiledCore(torch.nn.Module):
    def __init__(self, model: LocalCharges, pbc_handling: str):
        super().__init__()
        if pbc_handling not in ("pbc", "slab"):
            raise ValueError(
                "Compiled LocalCharges supports only "
                f"pbc_handling='pbc' or 'slab', got {pbc_handling!r}."
            )
        self.pbc_handling = pbc_handling

        self.atomic_numbers = model.atomic_numbers
        self.charges_dim = model.charges_irreps.dim
        self.atomic_energies_fn = model.atomic_energies_fn
        self.node_embedding = model.node_embedding
        self.radial_embedding = model.radial_embedding
        self.spherical_harmonics = model.spherical_harmonics
        self.interactions = model.interactions
        self.products = model.products
        self.readouts = model.readouts
        self.lr_source_maps = model.lr_source_maps
        self.coulomb_energy = model.coulomb_energy

    def _compute_geometry_features(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return edge_attrs, edge_feats, lengths

    def _compute_atomic_base_energy(
        self,
        node_attrs: torch.Tensor,
        node_heads: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_e0_heads = _call_module(self.atomic_energies_fn, node_attrs)
        node_e0 = torch.gather(
            node_e0_heads,
            dim=1,
            index=node_heads.view(-1, 1),
        ).squeeze(-1)
        e0 = scatter_sum(
            src=node_e0,
            index=batch,
            dim=-1,
            dim_size=num_graphs,
        )
        return e0, node_e0

    def _run_local_interactions_and_sources(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        node_heads: torch.Tensor,
        charge_density: torch.Tensor,
        energies: list,
        node_energies_list: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for interaction, product, readout, charge_map in zip(
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
            node_energies = _call_module(readout, node_feats, node_heads)
            node_energies = torch.gather(
                node_energies,
                dim=1,
                index=node_heads.view(-1, 1),
            ).squeeze(-1)
            energy = scatter_sum(
                src=node_energies,
                index=batch,
                dim=-1,
                dim_size=num_graphs,
            )
            energies.append(energy)
            node_energies_list.append(node_energies)

            charge_sources = _call_module(
                charge_map,
                node_feats=node_feats,
                node_attrs=node_attrs,
            )
            charge_density = charge_density + charge_sources.squeeze(-2)

        contributions = torch.stack(energies, dim=-1)
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)
        return contributions, node_energy, charge_density

    def _compute_electrostatic_and_field_energy(
        self,
        charge_density: torch.Tensor,
        total_dipole: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        external_field: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> torch.Tensor:
        electro_energy = self.coulomb_energy.forward(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=charge_density,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc.view(-1, 3),
        )
        field_energy = torch.sum(total_dipole * external_field, dim=-1)
        return electro_energy + field_energy

    @staticmethod
    def _compute_total_dipole(
        charge_density: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        dipole = scatter_sum(
            src=positions * charge_density[:, :1],
            index=batch.unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )
        if charge_density.shape[1] > 1:
            dipole_p = scatter_sum(
                src=charge_density[:, 1:4],
                index=batch,
                dim=-2,
                dim_size=num_graphs,
            )
            dipole = dipole + dipole_p[:, [2, 0, 1]]
        return dipole

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
        charges: torch.Tensor,
        external_field: torch.Tensor,
        head: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del cell, charges
        num_graphs = ptr.numel() - 1
        node_heads = head[batch]

        node_feats = _call_module(self.node_embedding, node_attrs)
        edge_attrs, edge_feats, _ = self._compute_geometry_features(
            node_attrs=node_attrs,
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
        )
        e0, node_e0 = self._compute_atomic_base_energy(
            node_attrs=node_attrs,
            node_heads=node_heads,
            batch=batch,
            num_graphs=num_graphs,
        )
        charge_density = torch.zeros(
            (batch.size(0), self.charges_dim),
            device=batch.device,
            dtype=positions.dtype,
        )
        contributions, node_energy, charge_density = (
            self._run_local_interactions_and_sources(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                batch=batch,
                num_graphs=num_graphs,
                node_heads=node_heads,
                charge_density=charge_density,
                energies=[e0],
                node_energies_list=[node_e0],
            )
        )

        total_energy = torch.sum(contributions, dim=-1)
        total_dipole = self._compute_total_dipole(
            charge_density,
            positions,
            batch,
            num_graphs,
        )
        total_energy = total_energy + self._compute_electrostatic_and_field_energy(
            charge_density=charge_density,
            total_dipole=total_dipole,
            positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
            external_field=external_field,
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
        )
        return total_energy, node_energy, charge_density, total_dipole


class CompiledLocalSourceEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        pbc_handling: str,
        compile_options: Optional[CompiledLocalSourceOptions] = None,
    ):
        if not isinstance(model, (LocalSplitCharges, LocalCharges)):
            raise NotImplementedError(
                "Compiled local-source evaluation is currently implemented "
                "only for LocalSplitCharges and LocalCharges."
            )
        if pbc_handling not in ("pbc", "slab"):
            raise ValueError(
                "Compiled local-source evaluation supports only "
                f"pbc_handling='pbc' or 'slab', got {pbc_handling!r}."
            )
        self.model = model
        self.pbc_handling = pbc_handling
        self.kspace_planner = KSpacePlanner(model.kspace_cutoff)
        if isinstance(model, LocalSplitCharges):
            core = LocalSplitChargesCompiledCore(model, pbc_handling)
        else:
            core = LocalChargesCompiledCore(model, pbc_handling)
        self.core = core
        if compile_options is None:
            self.compiled_core = core
        else:
            try:
                self.compiled_core = torch.compile(
                    core,
                    backend=compile_options.backend,
                    mode=compile_options.mode,
                    dynamic=compile_options.dynamic,
                    fullgraph=compile_options.fullgraph,
                )
            except (AttributeError, RuntimeError) as exc:
                raise RuntimeError(
                    "Failed to initialize torch.compile for "
                    f"{type(core).__name__}. Check that the "
                    "installed PyTorch/Python combination supports "
                    "torch.compile."
                ) from exc

    def _prepare_inputs(self, data: Dict[str, torch.Tensor]) -> PreparedLocalSourceInputs:
        positions = data["positions"].detach().clone().requires_grad_(True)
        num_nodes = positions.size(0)
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

        charges = data.get("charges")
        if charges is None:
            charges = torch.zeros(
                (num_nodes,),
                dtype=positions.dtype,
                device=positions.device,
            )

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
            charges,
            external_field,
            head,
            plan.k_vectors,
            plan.k_norm2,
            plan.k_vector_batch,
            plan.k0_mask,
        )
        return PreparedLocalSourceInputs(
            core_args=core_args,
            positions=positions,
            external_field=external_field,
        )

    def evaluate(self, data: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        prepared = self._prepare_inputs(data)
        energy, node_energy, density_coefficients, dipole = self.compiled_core(
            *prepared.core_args
        )
        forces = -torch.autograd.grad(
            energy.sum(),
            prepared.positions,
            retain_graph=False,
            create_graph=False,
        )[0]
        return {
            "energy": energy,
            "node_energy": node_energy,
            "forces": forces,
            "stress": None,
            "density_coefficients": density_coefficients,
            "external_field": prepared.external_field,
            "fermi_level": None,
            "dipole": dipole,
            "polarizability": None,
        }


def build_compiled_local_source_evaluator(
    model: torch.nn.Module,
    pbc_handling: str,
    backend: str = "inductor",
    mode: str = "reduce-overhead",
    dynamic: bool = False,
    fullgraph: bool = False,
    enabled: bool = True,
) -> CompiledLocalSourceEvaluator:
    compile_options = None
    if enabled:
        compile_options = CompiledLocalSourceOptions(
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )
    return CompiledLocalSourceEvaluator(
        model=model,
        pbc_handling=pbc_handling,
        compile_options=compile_options,
    )
