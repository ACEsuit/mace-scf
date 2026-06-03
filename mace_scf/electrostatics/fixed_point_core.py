from typing import Any, Callable, Dict, List, Optional, Type
import numpy as np
import torch
from e3nn import o3
import logging

from mace.tools.scatter import scatter_sum
from mace.modules.utils import (
    get_edge_vectors_and_lengths,
    get_outputs,
)

from mace.modules import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
)

from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.gto_utils import (
    gto_basis_kspace_cutoff,
    DisplacedGTOExternalFieldBlock,
    GTOInternalFieldtoFeaturesBlock,
)
from graph_longrange.utils import permute_to_e3nn_convention

from .field_blocks import (
    ScaledEnvironmentDependentSourceBlock,
    MultiLayerFeatureMixer,
    GeneralNonLinearBiasReadoutBlock,
)
from .utils import compute_total_charge_dipole
from .fixed_point_state import LocalState


class FixedPointCore(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atom_density_scaling: np.ndarray,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        field_feature_max_l: int = 0,
        field_feature_widths: List[float] = [1.0],
        field_si: bool = False,
        include_electrostatic_self_interaction: bool = False,
        add_local_electron_energy: bool = False,
        quadrupole_feature_corrections: bool = False,
        return_electrostatic_potentials: bool = False,
        heads: Optional[List[str]] = None,
        field_feature_norms: Optional[np.ndarray] = None,
        field_norm_factor: Optional[float] = 1.0,
        pbc_handling: str = "mixed_periodic",
        fermi_level_offset: float = 0.0,
        *,
        fixedpoint_update_config: Dict[str, Any],
        field_readout_config: Dict[str, Any],
        use_linear_local_charges: bool = False,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        self.register_buffer(
            "fermi_level_offset",
            torch.tensor(float(fermi_level_offset), dtype=torch.get_default_dtype()),
        )
        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
            [atomic_multipoles_smearing_width] + field_feature_widths,
            max(atomic_multipoles_max_l, field_feature_max_l),
        )
        self.register_buffer(
            "kspace_cutoff",
            torch.tensor(kspace_cutoff, dtype=torch.get_default_dtype()),
        )

        if field_feature_norms is not None:
            assert len(field_feature_norms) == len(field_feature_widths) * (
                field_feature_max_l + 1
            ), f"{len(field_feature_widths) * (field_feature_max_l+1)}, {len(field_feature_norms)}"
        else:
            field_feature_norms = [1.0] * len(field_feature_widths) * (
                field_feature_max_l + 1
            )
        field_feature_norms_expanded = []
        for ll in range(field_feature_max_l + 1):
            for jj in range(len(field_feature_widths)):
                field_feature_norms_expanded += [
                    field_feature_norms[ll * len(field_feature_widths) + jj]
                ] * (2 * ll + 1)
        self.register_buffer(
            "field_feature_norms",
            torch.tensor(field_feature_norms_expanded, dtype=torch.get_default_dtype()),
        )

        if heads is None:
            heads = ["Default"]
        if len(heads) != 1:
            raise ValueError(
                f"FixedPoint only supports a single head, got heads={heads}"
            )
        self.heads = heads
        fixedpoint_update_config = dict(fixedpoint_update_config)
        field_readout_config = dict(field_readout_config)

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps(
            [(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
        )

        # Electrostatic field features
        self.electric_potential_descriptor = GTOElectrostaticFeatures(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            feature_max_l=field_feature_max_l,
            feature_smearing_widths=field_feature_widths,
            kspace_cutoff=self.kspace_cutoff,
            include_self_interaction=field_si,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
            integral_normalization="receiver",
            pbc_handling=pbc_handling,
        )

        self.charges_irreps = o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        lr_sh_irreps = o3.Irreps.spherical_harmonics(field_feature_max_l)
        self.potential_irreps = (lr_sh_irreps * len(field_feature_widths)).sort()[
            0
        ].simplify()

        # Density readout blocks
        if use_linear_local_charges:
            self.lr_source_maps = torch.nn.ModuleList(
                [
                    ScaledEnvironmentDependentSourceBlock(
                        irreps_in=hidden_irreps,
                        max_l=atomic_multipoles_max_l,
                        atom_density_scaling=atom_density_scaling,
                    )
                ]
            )
        else:
            self.lr_source_maps = torch.nn.ModuleList(
                [
                    GeneralNonLinearBiasReadoutBlock(
                        irreps_in=hidden_irreps,
                        MLP_irreps=MLP_irreps,
                        irreps_out=self.charges_irreps,
                        gate=gate,
                    )
                ]
            )

        for i in range(num_interactions - 1):
            hidden_irreps_out = hidden_irreps

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                    )
                )
            else:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps, o3.Irreps(f"{len(heads)}x0e")
                    )
                )
            if use_linear_local_charges:
                self.lr_source_maps.append(
                    ScaledEnvironmentDependentSourceBlock(
                        irreps_in=hidden_irreps,
                        max_l=atomic_multipoles_max_l,
                        atom_density_scaling=atom_density_scaling,
                    )
                )
            else:
                self.lr_source_maps.append(
                    GeneralNonLinearBiasReadoutBlock(
                        irreps_in=hidden_irreps,
                        MLP_irreps=MLP_irreps,
                        irreps_out=self.charges_irreps,
                        gate=gate,
                    )
                )

        # Field-dependent charge update block
        lr_source_cls = fixedpoint_update_config.pop("type")
        max_ell_field_update = 2
        field_update_sh_irreps = o3.Irreps.spherical_harmonics(max_ell_field_update)
        self.from_ell_max_field_update = (max_ell_field_update + 1) ** 2
        field_interaction_irreps = (
            (field_update_sh_irreps * num_features).sort()[0].simplify()
        )
        self.field_dependent_charges_map = lr_source_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=field_update_sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=field_interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            potential_irreps=self.potential_irreps,
            charges_irreps=self.charges_irreps,
            num_elements=num_elements,
            field_norm_factor=field_norm_factor,
            atom_density_scaling=atom_density_scaling,
            **fixedpoint_update_config,
        )

        # Local electron energy readout
        self.add_local_electron_energy = add_local_electron_energy
        field_readout_cls = field_readout_config.pop("type")
        self.local_electron_energy = field_readout_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=field_update_sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=field_interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            potential_irreps=self.potential_irreps,
            charges_irreps=self.charges_irreps,
            num_elements=num_elements,
            **field_readout_config,
        )

        # External field blocks
        field_feat_norm_mode = "receiver"
        self.external_field_contribution = DisplacedGTOExternalFieldBlock(
            field_feature_max_l, field_feature_widths, field_feat_norm_mode
        )
        self.external_field_contribution_internal = GTOInternalFieldtoFeaturesBlock(
            field_feature_max_l, field_feature_widths, field_feat_norm_mode
        )

        # Coulomb energy
        self.coulomb_energy = GTOElectrostaticEnergy(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            kspace_cutoff=self.kspace_cutoff,
            include_self_interaction=include_electrostatic_self_interaction,
            pbc_handling=pbc_handling,
        )
        self.return_electrostatic_potentials = return_electrostatic_potentials

        self.layer_feature_mixer = MultiLayerFeatureMixer(
            node_feats_irreps=hidden_irreps,
            num_interactions=num_interactions,
        )

    # ------------------------------------------------------------------ #
    #  Core decomposed methods                                            #
    # ------------------------------------------------------------------ #

    def center_fermi_level(self, fermi_level: torch.Tensor) -> torch.Tensor:
        return fermi_level - self.fermi_level_offset

    def uncenter_fermi_level(self, centered_fermi_level: torch.Tensor) -> torch.Tensor:
        return centered_fermi_level + self.fermi_level_offset

    def features_from_fermi_level(
        self,
        batch: torch.Tensor,
        positions: torch.Tensor,
        fermi_level: torch.Tensor,
    ) -> torch.Tensor:
        fermi_potential = torch.zeros(
            (fermi_level.numel(), 4), dtype=positions.dtype, device=positions.device
        )
        fermi_potential[:, 0] = self.center_fermi_level(fermi_level)
        return self.external_field_contribution(batch, positions, fermi_potential)

    def features_from_fermi_level_nodewise(
        self,
        batch: torch.Tensor,
        positions: torch.Tensor,
        node_fermi_level: torch.Tensor,
    ) -> torch.Tensor:
        node_ext = torch.zeros(
            (node_fermi_level.numel(), 4),
            dtype=positions.dtype,
            device=positions.device,
        )
        node_ext[:, 0] = self.center_fermi_level(node_fermi_level)
        return self.external_field_contribution_internal(batch, positions, node_ext)

    def local_part(
        self,
        data: Dict[str, torch.Tensor],
        compute_force: bool = False,
    ) -> LocalState:
        """Run the MACE backbone and prepare all SCF-independent state."""
        if not hasattr(self, "from_ell_max_field_update"):
            self.from_ell_max_field_update = 9

        positions = data["positions"]
        if compute_force:
            positions.requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"]).squeeze(-1)
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions,
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(permute_to_e3nn_convention(vectors))
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # K-space grid
        k_vectors, k_vectors_norms_squared, k_vectors_batch, k0_mask = (
            compute_k_vectors_flat(
                self.kspace_cutoff,
                data["cell"].view(-1, 3, 3),
                data["rcell"].view(-1, 3, 3),
            )
        )

        # Interaction layers
        energies = [e0]
        features = []
        charge_density = torch.zeros(
            (data["batch"].size(-1), self.charges_irreps.dim),
            device=data["batch"].device,
            dtype=torch.get_default_dtype(),
        )

        for interaction, product, readout, lr_source_map in zip(
            self.interactions, self.products, self.readouts, self.lr_source_maps
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            features.append(node_feats.clone())
            node_energies = readout(node_feats).squeeze(-1)
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )
            energies.append(energy)

            charge_sources = lr_source_map(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
            )
            charge_density += charge_sources.squeeze(-2)

        all_layer_feats = self.layer_feature_mixer(torch.stack(features, dim=0))
        stacked_energies = torch.stack(energies, dim=-1)

        # Precompute geometry for electrostatics
        electrostatics_cache = self.electric_potential_descriptor.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=k_vectors_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k0_mask,
            node_positions=positions,
            batch=data["batch"],
            volume=data["volume"],
            pbc=data["pbc"].view(-1, 3),
        )

        # Precompute external E-field features (no Fermi level)
        efield_potential = torch.zeros(
            (num_graphs, 4), dtype=positions.dtype, device=positions.device
        )
        efield_potential[:, 1:] = data["external_field"]
        external_field_features = self.external_field_contribution(
            data["batch"], positions, efield_potential
        )

        return LocalState(
            node_feats=node_feats,
            all_layer_feats=all_layer_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            field_independent_charge_density=charge_density,
            positions=positions,
            energies=stacked_energies,
            electrostatics_cache=electrostatics_cache,
            external_field_features=external_field_features,
            k_vectors=k_vectors,
            k_vectors_norms_squared=k_vectors_norms_squared,
            k_vectors_batch=k_vectors_batch,
            k0_mask=k0_mask,
        )

    def scf_step(
        self,
        data: Dict[str, torch.Tensor],
        local_state: LocalState,
        charge_density_in: torch.Tensor,
        total_charges: torch.Tensor,
        fermi_level_features: torch.Tensor,
    ):
        """
        One SCF iteration: compute electrostatic field from charge_density_in,
        add external contributions, and produce the field-dependent charge
        contribution (delta).

        Args:
            charge_density_in: density used to compute the internal electrostatic field.
            total_charges: current charge estimate passed to the update block.
            fermi_level_features: pre-built field features from the Fermi level,
                constructed by the caller using external_field_contribution or
                external_field_contribution_internal.

        Returns:
            (field_dep_contribution, field_feats): the field-dependent charge
                update and the electrostatic field features.
        """
        field_feats = self.electric_potential_descriptor.forward_dynamic(
            cache=local_state.electrostatics_cache,
            source_feats=charge_density_in,
        )
        field_feats += local_state.external_field_features
        field_feats += fermi_level_features
        field_feats /= self.field_feature_norms

        field_dep_contribution = self.field_dependent_charges_map(
            node_attrs=data["node_attrs"],
            node_feats=local_state.all_layer_feats,
            edge_attrs=local_state.edge_attrs[:, : self.from_ell_max_field_update],
            edge_feats=local_state.edge_feats,
            edge_index=data["edge_index"],
            potential_features=field_feats,
            local_charges=local_state.field_independent_charge_density,
            total_charges=total_charges,
        )
        return field_dep_contribution, field_feats

    def build_observables(
        self,
        data: Dict[str, torch.Tensor],
        local_state: LocalState,
        density: torch.Tensor,
        fermi_level: torch.Tensor,
        field_feats: torch.Tensor,
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Compute all output quantities from the converged SCF density."""
        num_graphs = data["ptr"].numel() - 1
        positions = local_state.positions

        # Backbone energy contributions
        total_energy = torch.sum(local_state.energies, dim=-1)

        # Local electron energy
        local_q_e = self.local_electron_energy(
            node_attrs=data["node_attrs"],
            node_feats=local_state.node_feats,
            edge_attrs=local_state.edge_attrs[:, : self.from_ell_max_field_update],
            edge_feats=local_state.edge_feats,
            edge_index=data["edge_index"],
            field_feats=field_feats,
            charges_0=local_state.field_independent_charge_density,
            charges_induced=density,
        )
        le_total = scatter_sum(
            src=local_q_e, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        if self.add_local_electron_energy:
            total_energy = total_energy + le_total
        else:
            le_total = torch.zeros_like(le_total)

        # Charge and dipole
        total_charge, total_dipole = compute_total_charge_dipole(
            density, positions, data["batch"], num_graphs
        )

        # Coulomb energy
        electro_energy = self.coulomb_energy(
            k_vectors=local_state.k_vectors,
            k_norm2=local_state.k_vectors_norms_squared,
            k_vector_batch=local_state.k_vectors_batch,
            k0_mask=local_state.k0_mask,
            source_feats=density,
            node_positions=positions,
            batch=data["batch"],
            volume=data["volume"],
            pbc=data["pbc"].view(-1, 3),
        )
        total_energy = total_energy + electro_energy + torch.sum(
            data["external_field"] * total_dipole, dim=-1
        )

        # Electrostatic potentials
        if self.return_electrostatic_potentials:
            esps = self.electric_potential_descriptor.compute_esps(
                cache=local_state.electrostatics_cache,
                source_feats=density,
                pbc=data["pbc"].view(-1, 3),
            ).view(-1, 1)
            esps_dft = self.electric_potential_descriptor.compute_esps(
                cache=local_state.electrostatics_cache,
                source_feats=data["density_coefficients"],
                pbc=data["pbc"].view(-1, 3),
            ).view(-1, 1)
            node_external_fields = torch.index_select(
                data["external_field"], 0, data["batch"]
            )
            esps += torch.sum(
                node_external_fields * positions, dim=-1, keepdim=True
            )
            esps_dft += torch.sum(
                node_external_fields * positions, dim=-1, keepdim=True
            )
        else:
            esps = None
            esps_dft = None

        # Forces
        displacement = torch.zeros(
            (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
        )
        forces, _, _, _, _ = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "forces": forces,
            "contributions": local_state.energies,
            "density_coefficients": density,
            "charges_history": None,
            "fermi_level": fermi_level,
            "external_field": data["external_field"],
            "dipole": total_dipole,
            "electrostatic_energy": electro_energy,
            "electrostatic_features": field_feats,
            "electron_energy": le_total,
            "field_independent_charge_density": local_state.field_independent_charge_density,
            "esps": esps,
            "esps_dft": esps_dft,
        }
