import torch
from typing import Dict, List, Tuple, Optional
from mace.tools.scatter import scatter_sum
from scipy.constants import c, e
import scipy
import numpy as np
from e3nn.io import CartesianTensor
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.modules.utils import (
    compute_forces,
    compute_hessians_vmap,
)


def get_change_of_basis() -> torch.Tensor:
    return CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis


def spherical_to_cartesian(t: torch.Tensor):
    """
    Convert spherical notation to cartesian notation
    """
    change_of_basis = get_change_of_basis().to(t.device)
    return torch.einsum("ijk,...i->...jk", change_of_basis, t)


def compute_fixed_charge_dipole(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) / (1e-11 / c / e)  # [N_atoms,3]
    return scatter_sum(
        src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )  # [N_graphs,3]


def compute_total_charge_dipole(
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
):
    dipole_contribution = positions * density_coefficients[:,:1]

    dipole = scatter_sum(
        src=dipole_contribution, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )

    if density_coefficients.shape[1] > 1:
        dipole_p = scatter_sum(
            src=density_coefficients[...,1:4], index=batch, dim=-2, dim_size=num_graphs
        )
        dipole = dipole + dipole_p[...,[2,0,1]] # CS phase convention

    total_charge = scatter_sum(
        src=density_coefficients[:,0], index=batch, dim=-1#, dim_size=num_graphs
    )

    return total_charge, dipole


def compute_forces_virials_cellstress(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials, cell_virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement, cell],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if cell_virials is not None:
        # For a strain `cell -> cell @ (I + eps)` the chain rule gives the
        # virial contribution of the cell gradient as dE/deps = cell^T @ (dE/dcell).
        cell = cell.view(-1, 3, 3)
        cell_virials = torch.matmul(cell.transpose(-1, -2), cell_virials)
        cell_virials = 0.5 * (cell_virials + cell_virials.transpose(-1, -2))
        virials = virials + cell_virials

    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    displacement: Optional[torch.Tensor],
    vectors: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_hessian: bool = False,
    compute_edge_forces: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = compute_forces_virials_cellstress(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=(training or compute_hessian or compute_edge_forces),
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(
                energy=energy,
                positions=positions,
                training=(training or compute_hessian or compute_edge_forces),
            ),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    if compute_hessian:
        assert forces is not None, "Forces must be computed to get the hessian"
        hessian = compute_hessians_vmap(forces, positions)
    else:
        hessian = None
    if compute_edge_forces and vectors is not None:
        edge_forces = compute_forces(
            energy=energy,
            positions=vectors,
            training=(training or compute_hessian),
        )
        if edge_forces is not None:
            edge_forces = -1 * edge_forces  # Match LAMMPS sign convention
    else:
        edge_forces = None
    return forces, virials, stress, hessian, edge_forces


def compute_polarization(
    density_coefficients: torch.Tensor,
    edge_fluxes: torch.Tensor,
    edge_vectors: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
):
    # flux piece
    edge_dipoles = edge_fluxes.unsqueeze(-1) * edge_vectors
    sender, receiver = edge_index
    total_flux = scatter_sum(
        src=edge_dipoles, index=batch[sender], dim=-2, dim_size=num_graphs
    )

    #print("charges piece:", total_flux)

    # dipole piece
    if density_coefficients.shape[1] > 1:
        dipole_p = scatter_sum(
            src=density_coefficients[...,1:4], index=batch, dim=-2, dim_size=num_graphs
        )
        #print("dipoles piece:", dipole_p[...,[2,0,1]])
        total_flux = total_flux + dipole_p[...,[2,0,1]] # CS phase convention
    #print("added: ", total_flux)

    return total_flux


def compute_coulomb_energy(
    partial_charges: torch.Tensor, data: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Compute the coulomb energy of a system of partial charges"""
    # compute the pairwise distances
    # compute the distances, accounting for pbc
    posn = data["positions"]
    batch_indices = data["batch"]

    output_energies = []
    k_e = 14.399645478425668

    for idx in torch.unique(batch_indices):
        # get the positions of the atoms in the current molecule
        molecule_mask = batch_indices == idx
        positions = posn[molecule_mask]
        molecule_partial_charges = partial_charges[molecule_mask]
        # iterate over each molecule in the batch

        # are the distance accounting for pbc? No

        distances = torch.cdist(positions, positions)
        # put ones on the diagonal to avoid dividing by zero
        distances = distances + torch.eye(distances.shape[0], device=distances.device)

        # change all distances greater than the cutoff to infinity, use a 1 angstrom cutoff
        # compute the coulomb energy
        potential = (
            k_e
            * torch.outer(molecule_partial_charges, molecule_partial_charges)
            * (torch.erf(distances / 1) / distances)
        )

        potential = torch.triu(potential, diagonal=1)
        # sum the values to get the total energy
        potential_energy = torch.sum(potential)
        # print("potential energy", potential_energy)
        # print(potential)
        # print(coulomb_energy)
        # print("final potential", coulomb_energy)
        output_energies.append(potential_energy)

    output_energies = torch.stack(output_energies)  # [n_graphs])
    return output_energies


@compile_mode("script")
class undo_reshape(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        batch, _, _ = tensor.shape
        out = []
        for mul, d in zip(self.muls, self.dims):
            out.append(tensor[:, :, ix:ix + d].reshape(batch, -1))
            ix += d
        out = torch.cat(out, dim=-1)
        return out


def compute_effective_index(
    indices: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Stack indices to shape (num_indices, N)
    indices_stack = torch.stack(indices, dim=0)  # Shape: (num_indices, N)

    # Transpose to get combinations per element
    index_combinations = indices_stack.t()  # Shape: (N, num_indices)

    # Find unique combinations and get inverse indices
    unique_combinations, inverse_indices = torch.unique(
        index_combinations, dim=0, return_inverse=True
    )

    return inverse_indices, unique_combinations
