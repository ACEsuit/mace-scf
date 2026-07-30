""" implementation of local source models, in which no scf or minimization is needed """
import torch
from typing import Callable, Dict, List, Optional, Type, Tuple
from e3nn.util.jit import compile_mode
from e3nn import o3
from scipy.constants import pi
import numpy as np

from mace.tools.scatter import scatter_sum
from mace.modules.utils import (
    get_edge_vectors_and_lengths,
    get_symmetric_displacement,
)
from e3nn.io import CartesianTensor

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.utils import permute_to_e3nn_convention

from mace.modules import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
)
from .bonded_blocks import (
    PerSpeciesFormalChargesBlock,
    PerAtomFormalChargesBlock,
    static_bond_transfer_blocks,
    oxidation_state_mixers,
)
from .field_blocks import (
    EnvironmentDependentSourceBlock,
    LinearPolarizabilityReadoutBlock
)

from .fourier_density import FourierDensityFittingBlock
from .utils import (
    compute_total_charge_dipole,
    compute_polarization,
    get_outputs,
)


class _LocalSourceModelBase(torch.nn.Module):
    @staticmethod
    def _make_readout(
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        heads: List[str],
        is_last: bool,
        use_linear_final_readout: bool,
    ) -> torch.nn.Module:
        output_irreps = o3.Irreps(f"{len(heads)}x0e")
        if is_last and not use_linear_final_readout:
            return NonLinearReadoutBlock(
                hidden_irreps,
                (len(heads) * MLP_irreps).simplify(),
                gate,
                output_irreps,
                len(heads),
            )
        return LinearReadoutBlock(hidden_irreps, output_irreps)

    def _init_local_model(
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
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        use_linear_final_readout: bool = False,
    ) -> None:
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["Default"]
        self.heads = heads

        self.hidden_irreps = hidden_irreps
        self.node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        self.node_feats_irreps = o3.Irreps(
            [(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=self.node_attr_irreps, irreps_out=self.node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        self.edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        self.sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        self.interaction_irreps = (self.sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            self.sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=self.node_attr_irreps,
            node_feats_irreps=self.node_feats_irreps,
            edge_attrs_irreps=self.sh_irreps,
            edge_feats_irreps=self.edge_feats_irreps,
            target_irreps=self.interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        use_sc_first = "Residual" in str(interaction_cls_first)
        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])
        self.readouts = torch.nn.ModuleList(
            [LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))]
        )

        for layer_idx in range(1, num_interactions):
            inter = interaction_cls(
                node_attrs_irreps=self.node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=self.sh_irreps,
                edge_feats_irreps=self.edge_feats_irreps,
                target_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            self.products.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=self.interaction_irreps,
                    target_irreps=hidden_irreps,
                    correlation=correlation,
                    num_elements=num_elements,
                    use_sc=True,
                )
            )
            self.readouts.append(
                self._make_readout(
                    hidden_irreps=hidden_irreps,
                    MLP_irreps=MLP_irreps,
                    gate=gate,
                    heads=heads,
                    is_last=layer_idx == num_interactions - 1,
                    use_linear_final_readout=use_linear_final_readout,
                )
            )

    def _init_electrostatics(
        self,
        kspace_cutoff_factor: float,
        atomic_multipoles_max_l: int,
        atomic_multipoles_smearing_width: float,
        include_electrostatic_self_interaction: bool = False,
        pbc_handling: str = "mixed_periodic",
    ) -> None:
        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
            [atomic_multipoles_smearing_width], atomic_multipoles_max_l
        )
        self.register_buffer(
            "kspace_cutoff",
            torch.tensor(kspace_cutoff, dtype=torch.get_default_dtype()),
        )
        self.coulomb_energy = GTOElectrostaticEnergy(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=include_electrostatic_self_interaction,
            pbc_handling=pbc_handling,
        )

    def _compute_coulomb_and_field_energy(
        self,
        charge_density: torch.Tensor,
        total_dipole: torch.Tensor,
        data: Dict[str, torch.Tensor],
        cell: torch.Tensor,
        rcell: torch.Tensor,
        volume: torch.Tensor,
        external_field: torch.Tensor,
    ) -> torch.Tensor:
        k_vectors, k_vectors_norms_squared, k_vectors_batch, k0_mask = (
            compute_k_vectors_flat(
                self.kspace_cutoff, cell.view(-1, 3, 3), rcell.view(-1, 3, 3)
            )
        )
        electro_energy = self.coulomb_energy(
            k_vectors=k_vectors,
            k_norm2=k_vectors_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k0_mask,
            source_feats=charge_density,
            node_positions=data["positions"],
            batch=data["batch"],
            volume=volume,
            pbc=data["pbc"].view(-1, 3),
        )
        field_energy = torch.sum(total_dipole * external_field, dim=-1)
        return electro_energy + field_energy

@compile_mode("script")
class LocalSplitCharges(_LocalSourceModelBase):
    """fit charges from local features
    NOTE: this does not inherit from MACE. This is because MACE sets the hidden irreps at the last
    layer to be scalar, whereas for this models we don't want this. To change this, you need to
    copy almost all of MACE.__init__, so I just replaced it."""

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
        formal_charges_from_data: bool,  # this must be explicit
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        atomic_formal_charges: Optional[np.ndarray] = None,
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        include_electrostatic_self_interaction: bool = False,
        pbc_handling: str = "mixed_periodic",
        static_bond_transfer_block: str = "NoFieldSymmetricPredictionSourceBlock",
        oxidation_state_mixer: str = "NoOxidationStateMixer",
        oxidation_state_range: Optional[Tuple] = (-4., 4.),
        heads: Optional[List[str]] = None,
        compute_polarizability: bool = False,
        use_linear_final_readout: bool = False,
        rho_fftn_mode: Optional[str] = None,
    ):
        super().__init__()
        self._init_local_model(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            correlation=correlation,
            gate=gate,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            heads=heads,
            use_linear_final_readout=use_linear_final_readout,
        )

        # embedding of oxidation states
        self.oxidation_state_mixer = oxidation_state_mixers[oxidation_state_mixer](
            node_feats_irreps=self.node_feats_irreps,
            oxidation_state_range=oxidation_state_range,
        )

        # charge prediction
        if formal_charges_from_data:
            self.formal_charges = PerAtomFormalChargesBlock()
        else:
            self.formal_charges = PerSpeciesFormalChargesBlock(atomic_formal_charges)

        self.charges_irreps = o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        static_bond_transfer_block_cls = static_bond_transfer_blocks[static_bond_transfer_block]
        self.lr_source_maps = torch.nn.ModuleList(
            [
                static_bond_transfer_block_cls(
                    node_feats_irreps=self.hidden_irreps,
                    edge_attrs_irreps=self.sh_irreps,
                    edge_feats_irreps=self.edge_feats_irreps,
                    target_irreps=self.interaction_irreps,
                    max_l=atomic_multipoles_max_l,
                    num_elements=num_elements,
                )
                for _ in range(num_interactions)
            ]
        )

        # polarizability
        if compute_polarizability: 
            self.polarizability_readouts = torch.nn.ModuleList(
                [
                    LinearPolarizabilityReadoutBlock(
                        irreps_in=self.hidden_irreps,
                    )
                    for _ in range(num_interactions)
                ]
            )
        change_of_basis = CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis
        self.register_buffer(
            "change_of_basis", torch.tensor(change_of_basis)
        )

        self._init_electrostatics(
            kspace_cutoff_factor=kspace_cutoff_factor,
            atomic_multipoles_max_l=atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=atomic_multipoles_smearing_width,
            include_electrostatic_self_interaction=include_electrostatic_self_interaction,
            pbc_handling=pbc_handling,
        )

        # Fourier density fitting: shared block, see fourier_density.py.
        if rho_fftn_mode is not None:
            self.fourier_density_block: Optional[FourierDensityFittingBlock] = (
                FourierDensityFittingBlock(
                    mode=rho_fftn_mode,
                    atomic_multipoles_max_l=atomic_multipoles_max_l,
                    atomic_multipoles_smearing_width=atomic_multipoles_smearing_width,
                    kspace_cutoff=float(self.kspace_cutoff.item()),
                )
            )
        else:
            self.fourier_density_block = None

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        external_field: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )

        # Setup
        if not training:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True

        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        cell = data["cell"].clone().view(-1,3,3)

        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=cell.clone(),
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )
            cell.requires_grad_(True)
        
        rcell = 2*pi*torch.linalg.inv_ex(cell.mT)[0]
        volume = torch.linalg.det(cell.view(-1, 3, 3)).abs()

        # Atomic energies
        node_e0_heads = self.atomic_energies_fn(data["node_attrs"])
        node_e0 = torch.gather(node_e0_heads, dim=1, index=node_heads.view(-1,1)).squeeze(-1)
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs, n_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(permute_to_e3nn_convention(vectors))
        edge_feats, _ = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        if external_field is None:
            external_field = data['external_field']

        # density
        charge_density = torch.zeros(
            (data["batch"].size(-1), self.charges_irreps.dim),
            device=data["batch"].device,
            dtype=torch.get_default_dtype(),
        )

        # fixed formal charges
        edge_fluxes = torch.zeros_like(lengths)  # [n_edges, 1]
        charge_density[:, 0] += self.formal_charges(data["node_attrs"], data["charges"])
        FQ = self.formal_charges(data["node_attrs"], data["charges"]) 
        formal_charge_dipole_contribution = scatter_sum(
            src=data["positions"] * FQ.unsqueeze(-1), 
            index=data["batch"].unsqueeze(-1), 
            dim=0, 
            dim_size=num_graphs
        )

        # mix oxidation states into embedding
        node_feats = self.oxidation_state_mixer(data["node_attrs"], node_feats, FQ)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        polarizabilities = []
        for layer_i, (interaction, product, readout, charge_map) in enumerate(zip(
            self.interactions, self.products, self.readouts, self.lr_source_maps
        )):
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
            node_energies = readout(node_feats, node_heads)
            node_energies = torch.gather(node_energies, dim=1, index=node_heads.view(-1,1)).squeeze(-1)
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

            # charges
            multipoles_contr, edge_fluxes_contr = charge_map(
                node_attrs=data["node_attrs"],
                node_formal_charges=FQ,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                edge_vectors=vectors,
                batch=data["batch"],
                num_graphs=num_graphs
            )  # [n_node, 1, (source_l+1)**2]

            edge_fluxes += edge_fluxes_contr
            charge_density += multipoles_contr.squeeze(-2)

            # polarizabilities
            if hasattr(self, "polarizability_readouts"):
                polarizabilities.append(self.polarizability_readouts[layer_i](node_feats))

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # also get the polarization
        # this function sums up the atomic dipoles and the interatomic fluxes
        polarization = compute_polarization(
            charge_density,
            edge_fluxes.squeeze(-1), 
            vectors,
            data["edge_index"],
            data["batch"],
            num_graphs
        )
        polarization += formal_charge_dipole_contribution
        total_dipole = polarization

        total_energy += self._compute_coulomb_and_field_energy(
            charge_density=charge_density,
            total_dipole=total_dipole,
            data=data,
            cell=cell,
            rcell=rcell,
            volume=volume,
            external_field=external_field,
        )

        # cartesian polarization
        if hasattr(self, "polarizability_readouts"):
            stacked = torch.stack(polarizabilities, dim=-1)
            atomic_polarizabilities = torch.sum(stacked, dim=-1)
            polarizability = scatter_sum(
                src=atomic_polarizabilities,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )
            # we are using condon-shotley pc, but e3nn doesnt, so permutation is needed. 
            cart_polarizability = torch.einsum("ijk,...i->...jk", self.change_of_basis, polarizability)[:,[2,0,1],:]
            cart_polarizability = cart_polarizability[:,:,[2,0,1]]
        else:
            cart_polarizability = None

        # Outputs
        forces, virials, stress, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        # Fourier density observables (optional). Delegated to the shared
        # block; see fourier_density.py.
        rho_fftn_out = None
        rho_fftn_dft_out = None
        rho_fftn_k2_out = None
        rho_fftn_mask_out = None
        rho_fftn_grid_shape_out = None
        if self.fourier_density_block is not None:
            fd = self.fourier_density_block(
                kspace_cutoff=self.kspace_cutoff,
                cell=cell,
                rcell=rcell,
                volume=volume,
                positions=data["positions"],
                batch=data["batch"],
                density_pred=charge_density,
                density_ref=data["density_coefficients"],
                data_rho_fftn=data.get("rho_fftn"),
                data_rho_fftn_shape=data.get("rho_fftn_shape"),
            )
            rho_fftn_out = fd["rho_fftn"]
            rho_fftn_dft_out = fd["rho_fftn_dft"]
            rho_fftn_k2_out = fd["k_vectors_normed_squared"]
            rho_fftn_mask_out = fd["k_vectors_mask"]
            rho_fftn_grid_shape_out = fd["k_vectors_grid_shape"]

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "density_coefficients": charge_density,
            "external_field": external_field,
            "fermi_level": None,
            "dipole" : total_dipole,
            "polarizability": cart_polarizability,
            "rho_fftn": rho_fftn_out,
            "rho_fftn_dft": rho_fftn_dft_out,
            "k_vectors_normed_squared": rho_fftn_k2_out,
            "k_vectors_mask": rho_fftn_mask_out,
            "k_vectors_grid_shape": rho_fftn_grid_shape_out,
        }










@compile_mode("script")
class LocalCharges(_LocalSourceModelBase):
    """fit charges from local features, non-conserving"""

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
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        include_electrostatic_self_interaction: bool = False,
        pbc_handling: str = "mixed_periodic",
        heads: Optional[List[str]] = None,
    ):
        super().__init__()
        self._init_local_model(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            correlation=correlation,
            gate=gate,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            heads=heads,
        )

        self.charges_irreps = o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        self.lr_source_maps = torch.nn.ModuleList(
            [
                EnvironmentDependentSourceBlock(
                    irreps_in=self.hidden_irreps,
                    max_l=atomic_multipoles_max_l,
                )
                for _ in range(num_interactions)
            ]
        )
        self._init_electrostatics(
            kspace_cutoff_factor=kspace_cutoff_factor,
            atomic_multipoles_max_l=atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=atomic_multipoles_smearing_width,
            include_electrostatic_self_interaction=include_electrostatic_self_interaction,
            pbc_handling=pbc_handling,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        external_field: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )

        # Setup
        if not training:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True

        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        cell = data["cell"].clone().view(-1,3,3)

        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=cell.clone(),
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )
            cell.requires_grad_(True)
        
        rcell = 2*pi*torch.linalg.inv_ex(cell.mT)[0]
        volume = torch.linalg.det(cell.view(-1, 3, 3)).abs()

        # Atomic energies
        node_e0_heads = self.atomic_energies_fn(data["node_attrs"])
        node_e0 = torch.gather(node_e0_heads, dim=1, index=node_heads.view(-1,1)).squeeze(-1)
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs, n_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(permute_to_e3nn_convention(vectors))
        edge_feats, _ = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        if external_field is None:
            external_field = data['external_field']


        # density
        charge_density = torch.zeros(
            (data["batch"].size(-1), self.charges_irreps.dim),
            device=data["batch"].device,
            dtype=torch.get_default_dtype(),
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        for interaction, product, readout, charge_map in zip(
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
            node_energies = readout(node_feats, node_heads)
            node_energies = torch.gather(node_energies, dim=1, index=node_heads.view(-1,1)).squeeze(-1)
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

            # charges
            charge_sources = charge_map(
                node_feats=node_feats,
                node_attrs=data["node_attrs"],
            )  # [n_node, 1, (source_l+1)**2]

            charge_density += charge_sources.squeeze(-2)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        _, total_dipole = compute_total_charge_dipole(
            charge_density,
            data["positions"],
            data["batch"],
            num_graphs
        )
        total_energy += self._compute_coulomb_and_field_energy(
            charge_density=charge_density,
            total_dipole=total_dipole,
            data=data,
            cell=cell,
            rcell=rcell,
            volume=volume,
            external_field=external_field,
        )

        # Outputs
        forces, virials, stress, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "density_coefficients": charge_density,
            "dipole": total_dipole,
            "external_field": external_field,
            "fermi_level": None,
            "polarizability": None,
        }






@compile_mode("script")
class FixedChargeBaselinedMACE(_LocalSourceModelBase):

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
        formal_charges_from_data: bool,  # this must be explicit
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        atomic_formal_charges: Optional[np.ndarray] = None,
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_smearing_width: float = 1.0,
        include_electrostatic_self_interaction: bool = False,
        pbc_handling: str = "mixed_periodic",
        heads: Optional[List[str]] = None,
        use_linear_final_readout: bool = False,
    ):
        super().__init__()
        self._init_local_model(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            correlation=correlation,
            gate=gate,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            heads=heads,
            use_linear_final_readout=use_linear_final_readout,
        )

        # charge prediction
        if formal_charges_from_data:
            self.formal_charges = PerAtomFormalChargesBlock()
        else:
            self.formal_charges = PerSpeciesFormalChargesBlock(atomic_formal_charges)

        self._init_electrostatics(
            kspace_cutoff_factor=kspace_cutoff_factor,
            atomic_multipoles_max_l=0,
            atomic_multipoles_smearing_width=atomic_multipoles_smearing_width,
            include_electrostatic_self_interaction=include_electrostatic_self_interaction,
            pbc_handling=pbc_handling,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        external_field: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )

        if not training:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True

        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        cell = data["cell"].clone().view(-1, 3, 3)
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=cell.clone(),
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )
            cell.requires_grad_(True)

        rcell = 2 * pi * torch.linalg.inv_ex(cell.mT)[0]
        volume = torch.linalg.det(cell.view(-1, 3, 3)).abs()

        node_e0_heads = self.atomic_energies_fn(data["node_attrs"])
        node_e0 = torch.gather(node_e0_heads, dim=1, index=node_heads.view(-1, 1)).squeeze(-1)
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(permute_to_e3nn_convention(vectors))
        edge_feats, _ = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if external_field is None:
            external_field = data["external_field"]

        # density
        charge_density = torch.zeros(
            (data["batch"].size(-1), 1),
            device=data["batch"].device,
            dtype=torch.get_default_dtype(),
        )

        # fixed formal charges
        charge_density[:, 0] += self.formal_charges(data["node_attrs"], data["charges"])

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
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
            node_energies = readout(node_feats, node_heads)
            node_energies = torch.gather(
                node_energies, dim=1, index=node_heads.view(-1, 1)
            ).squeeze(-1)
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # also dipole
        _, formal_charge_polarization = compute_total_charge_dipole(
            charge_density,
            data["positions"],
            data["batch"],
            num_graphs
        )

        total_energy += self._compute_coulomb_and_field_energy(
            charge_density=charge_density,
            total_dipole=formal_charge_polarization,
            data=data,
            cell=cell,
            rcell=rcell,
            volume=volume,
            external_field=external_field,
        )

        # Outputs
        forces, virials, stress, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "density_coefficients": charge_density,
            "external_field": external_field,
            "fermi_level": None,
            "dipole": formal_charge_polarization,
            "polarizability": None,
        }
