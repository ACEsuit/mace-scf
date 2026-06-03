from typing import Callable, Dict, List, Optional, Type
import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
import logging
from scipy.constants import e, epsilon_0, pi

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

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.utils import permute_to_e3nn_convention

from .utils import compute_total_charge_dipole


class AutogradHardness(torch.nn.Module):
    def __init__(self, coulomb_energy, kspace_cutoff):
        super().__init__()
        self.coulomb_energy = coulomb_energy
        self.kspace_cutoff = kspace_cutoff

    def forward(self, enegs, hardness, data):
        qeq_device = data["positions"].device
        all_charges = []
        positions = data["positions"]
        data_batch = data["batch"]
        unique_batch_indices = torch.unique(data_batch)
        for id_batch_index, batch_index in enumerate(unique_batch_indices):
            mol_cell = data["cell"][id_batch_index * 3 : id_batch_index * 3 + 3]
            mol_pbc = data["pbc"][id_batch_index * 3 : id_batch_index * 3 + 3]
            molecule_indices = torch.where(data_batch == batch_index)
            mol_pos = positions[molecule_indices]
            mol_enegs = enegs[molecule_indices]
            mol_hardness = hardness[molecule_indices]
            qeq_dtype = mol_enegs.dtype
            num_atoms = mol_pos.shape[0]
            mol_charges = torch.ones_like(
                mol_enegs, requires_grad=True, device=qeq_device
            )
            rcell = 2 * pi * torch.linalg.inv_ex(mol_cell.mT)[0]
            volume = torch.linalg.det(mol_cell.view(-1, 3, 3)).abs()
            k_vectors, k_vectors_norms_squared, k_vectors_batch, k0_mask = (
                compute_k_vectors_flat(
                    self.kspace_cutoff,
                    mol_cell.view(-1, 3, 3),
                    rcell.view(-1, 3, 3),
                )
            )
            single_graph_batch = torch.zeros(
                num_atoms, device=mol_pos.device, dtype=torch.int64
            )
            electro_energy = self.coulomb_energy(
                k_vectors=k_vectors,
                k_norm2=k_vectors_norms_squared,
                k_vector_batch=k_vectors_batch,
                k0_mask=k0_mask,
                source_feats=mol_charges.unsqueeze(-1),
                node_positions=mol_pos,
                batch=single_graph_batch,
                volume=volume,
                pbc=mol_pbc.view(-1, 3),
            )
            f_by_q = torch.autograd.grad(
                electro_energy, inputs=mol_charges, create_graph=True
            )[0]

            def differentiate_fn(x):
                return torch.autograd.grad(
                    f_by_q, mol_charges, x, create_graph=True, retain_graph=True
                )

            In = torch.eye(len(mol_charges), device=qeq_device, dtype=qeq_dtype)
            jacobian = torch.vmap(differentiate_fn, randomness="same", chunk_size=16)(
                In
            )[0]

            A = jacobian + torch.diag(mol_hardness)
            A = torch.cat(
                (
                    A,
                    torch.ones((num_atoms, 1), device=qeq_device, dtype=qeq_dtype),
                ),
                dim=1,
            )
            A = torch.cat(
                (
                    A,
                    torch.ones((1, num_atoms + 1), device=qeq_device, dtype=qeq_dtype),
                ),
                dim=0,
            )
            A[-1, -1] = A.new_tensor(0.0)

            b = -1 * mol_enegs
            total_q = (
                data["total_charge"][id_batch_index]
                if isinstance(data["total_charge"], torch.Tensor)
                else data["total_charge"]
            )
            if isinstance(total_q, torch.Tensor):
                total_q = total_q.reshape(1).to(device=qeq_device, dtype=qeq_dtype)
            else:
                total_q = torch.tensor([total_q], device=qeq_device, dtype=qeq_dtype)
            b = torch.cat((b, total_q), dim=0)

            charges_solved = torch.linalg.solve(A, b)[:-1]
            all_charges.append(charges_solved)

        return torch.cat(all_charges)


class NonPBC(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.pi = torch.tensor(pi)
        self.sqrt2 = torch.sqrt(torch.tensor(2.0))
        self.epsilon_0 = torch.tensor(epsilon_0)
        self.e = torch.tensor(e)
        self.alpha = alpha

    def forward(self, enegs, hardness, data):
        positions = data["positions"]
        data_batch = data["batch"]
        qeq_device = positions.device

        batched_charges = []
        unique_batch_indices = torch.unique(data_batch)
        for batch_index in unique_batch_indices:
            molecule_indices = torch.where(data_batch == batch_index)
            mol_pos = positions[molecule_indices]
            mol_enegs = enegs[molecule_indices]
            num_atoms = mol_pos.shape[0]

            diff = mol_pos[:, None, :] - mol_pos[None, :, :]
            allRij = torch.linalg.norm(diff, dim=-1)
            eye = torch.eye(num_atoms, device=qeq_device)
            Rij = allRij + eye
            antieye = 1.0 - eye
            invRij = torch.reciprocal(Rij)
            Vscreen = torch.erf(0.5 * Rij / self.alpha) * invRij
            self_term = (1 / (torch.sqrt(self.pi) * self.alpha)) * torch.ones(
                Rij.shape[0], device=qeq_device
            )
            A = Vscreen * antieye + torch.diag(self_term)
            A = 1 / (4 * self.pi * self.epsilon_0) * A * self.e * 1e10
            A = A + torch.diag(hardness[molecule_indices])
            A = torch.cat(
                (A, torch.ones((num_atoms, 1), device=positions.device)), dim=1
            )
            A = torch.cat((A, torch.ones((1, num_atoms + 1), device=qeq_device)), dim=0)
            A[-1, -1] = 0
            b = -1 * mol_enegs
            b = torch.cat(
                (
                    b,
                    torch.tensor(
                        [data["total_charge"][batch_index]], device=qeq_device
                    ),
                ),
                dim=0,
            )
            charges = torch.linalg.solve(A, b)[:-1]

            batched_charges.append(charges)
        return torch.cat(batched_charges, dim=0)


class QEqClass(torch.nn.Module):
    def __init__(
        self,
        qeq_coulomb_energy,
        kspace_cutoff,
        atomic_multipoles_smearing_width=1.0,
        qeq_charges_option="autograd",
    ):
        super().__init__()
        self.alpha = atomic_multipoles_smearing_width
        if qeq_charges_option is None:
            qeq_charges_option = "autograd"
        if qeq_charges_option == "nonPBC":
            self.qeq_charges = NonPBC(atomic_multipoles_smearing_width)
        elif qeq_charges_option == "autograd":
            self.qeq_charges = AutogradHardness(
                qeq_coulomb_energy, kspace_cutoff=kspace_cutoff
            )
        else:
            raise ValueError(
                "qeq_charges_option must be one of {'autograd', 'nonPBC'}, "
                f"got {qeq_charges_option!r}"
            )

    def forward(self, enegs, hardness, data):
        batched_charges = self.qeq_charges(enegs, hardness, data)
        return batched_charges


@compile_mode("script")
class MACEQEq(torch.nn.Module):
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
        field_feature_widths: List[float] = [1.0],
        include_electrostatic_self_interaction: bool = True,
        pbc_handling: str = "mixed_periodic",
        heads: Optional[List[str]] = None,
        train_hardness: bool = False,
        read_enegs: bool = False,
        read_hardness: bool = False,
        default_hardness: float = 2.0,
        qeq_charges: str = "autograd",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "alpha",
            (
                torch.sqrt(torch.tensor(2))
                * torch.tensor(
                    atomic_multipoles_smearing_width,
                    dtype=torch.get_default_dtype(),
                )
            ),
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
            [atomic_multipoles_smearing_width] + field_feature_widths,
            atomic_multipoles_max_l,
        )
        self.register_buffer(
            "kspace_cutoff",
            torch.tensor(kspace_cutoff, dtype=torch.get_default_dtype()),
        )
        if heads is None:
            heads = ["Default"]
        self.heads = heads
        self.train_hardness = train_hardness
        self.read_enegs = read_enegs
        self.read_hardness = read_hardness
        self.default_hardness = default_hardness

        logger = logging.getLogger(__name__)
        logger.info(
            "MACEQEq: trainable hardness %s, read_hardness %s, default_hardness %s, read enegs %s",
            self.train_hardness,
            self.read_hardness,
            self.default_hardness,
            self.read_enegs,
        )

        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
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

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
        )

        self.enegs_readouts = torch.nn.ModuleList()
        self.enegs_readouts.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
        )

        self.hardness_readouts = torch.nn.ModuleList()
        self.hardness_readouts.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
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
                self.enegs_readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                    )
                )
                self.hardness_readouts.append(
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
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
                )
                self.enegs_readouts.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
                )
                self.hardness_readouts.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{len(heads)}x0e"))
                )

        self.coulomb_energy = GTOElectrostaticEnergy(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=include_electrostatic_self_interaction,
            pbc_handling=pbc_handling,
        )

        self.QEq = QEqClass(
            atomic_multipoles_smearing_width=atomic_multipoles_smearing_width,
            qeq_charges_option=qeq_charges,
            kspace_cutoff=kspace_cutoff,
            qeq_coulomb_energy=self.coulomb_energy,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if compute_displacement:
            raise ValueError("MACEQEq does not support compute_displacement=True")

        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        num_atoms_arange = torch.arange(
            data["positions"].shape[0], device=data["positions"].device
        )
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        if training:
            for p in self.parameters():
                p.requires_grad = True

        positions = data["positions"]

        if compute_force:
            positions.requires_grad_(True)
        else:
            positions.requires_grad_(False)

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

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

        energies = [e0]
        enegs_list = []
        hardness_list = []
        if self.read_enegs:
            enegs = data["enegs"]
        else:
            enegs = torch.zeros_like(data["enegs"], device=edge_attrs.device)
        read_hardness = getattr(
            self, "read_hardness", getattr(self, "read_hardnes", False)
        )
        if read_hardness:
            hardness = data["hardness"]
        else:
            hardness = self.default_hardness * torch.ones_like(
                data["hardness"], device=edge_attrs.device
            )
        for (
            interaction,
            product,
            readout,
            enegs_readout,
            hardness_readout,
        ) in zip(
            self.interactions,
            self.products,
            self.readouts,
            self.enegs_readouts,
            self.hardness_readouts,
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_enegs = enegs_readout(node_feats).squeeze(-1)
            enegs_list.append(node_enegs)

            if self.train_hardness:
                node_hardness = hardness_readout(node_feats).squeeze(-1)
                hardness_list.append(node_hardness)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
        enegs_field = torch.sum(
            data["positions"] * data["external_field"][data["batch"]], dim=1
        )
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        stacked_enegs = torch.stack(enegs_list, dim=0)
        enegs = enegs + torch.sum(stacked_enegs, dim=0)
        # The convention on fields is that field is E = grad V.
        enegs = enegs + enegs_field
        if self.train_hardness:
            stacked_hardness = torch.stack(hardness_list, dim=0)
            hardness = hardness + torch.sum(stacked_hardness, dim=0)
        batched_charges = self.QEq(enegs, hardness, data)
        site_energy = torch.zeros_like(total_energy)
        qchi = torch.multiply(batched_charges, enegs)
        qnu = torch.multiply(torch.pow(batched_charges, 2), hardness)
        site_energy += scatter_sum(qchi, data["batch"], dim=0, dim_size=num_graphs)
        site_energy += 0.5 * scatter_sum(qnu, data["batch"], dim=0, dim_size=num_graphs)

        cell = data["cell"].clone().view(-1, 3, 3)
        rcell = 2 * pi * torch.linalg.inv_ex(cell.mT)[0]
        volume = torch.linalg.det(cell.view(-1, 3, 3)).abs()
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
            source_feats=batched_charges.unsqueeze(-1),
            node_positions=data["positions"],
            batch=data["batch"],
            volume=volume,
            pbc=data["pbc"].view(-1, 3),
        )
        qeq_energy = site_energy + electro_energy
        total_energy = total_energy + qeq_energy
        _, total_dipole = compute_total_charge_dipole(
            batched_charges.unsqueeze(-1),
            data["positions"],
            data["batch"],
            num_graphs,
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
            "enegs": enegs,
            "hardness": hardness,
            "energy": total_energy,
            "qeq_energy": qeq_energy,
            "forces": forces,
            "dipole": total_dipole,
            "density_coefficients": batched_charges.unsqueeze(-1),
        }


autogradHardness = AutogradHardness

# For legacy code
MaceQEq = MACEQEq
