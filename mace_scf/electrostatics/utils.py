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


def resize_fft(F: torch.Tensor, new_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Resize a 3D FFT array `F` to `new_shape` by zero-padding or truncating
    each dimension symmetrically around DC. Ported from FourierFitting
    (mace-tools/macetools/electrostatics/utils.py) for the rho_fftn "direct"
    reference mode.
    """
    old_shape = F.shape
    F_shifted = torch.fft.fftshift(F)
    resized = torch.zeros(new_shape, dtype=F.dtype, device=F.device)

    def compute_indices(old_N, new_N):
        if new_N <= old_N:
            start_old = old_N // 2 - new_N // 2
            start_new = 0
            size = new_N
        else:
            start_old = 0
            start_new = new_N // 2 - old_N // 2
            size = old_N
        return start_old, start_new, size

    slices_old = []
    slices_new = []
    for i in range(3):
        start_old, start_new, size = compute_indices(old_shape[i], new_shape[i])
        slices_old.append(slice(start_old, start_old + size))
        slices_new.append(slice(start_new, start_new + size))

    resized[slices_new[0], slices_new[1], slices_new[2]] = \
        F_shifted[slices_old[0], slices_old[1], slices_old[2]]

    scale = (new_shape[0] * new_shape[1] * new_shape[2]) / (
        old_shape[0] * old_shape[1] * old_shape[2]
    )
    return torch.fft.ifftshift(resized) * scale


# NOTE: the installed graph_longrange.kspace.compute_k_vectors no longer
# supports the full_grid=True path (it was dropped when the flat/sparse layout
# `compute_k_vectors_flat` was introduced). The Fourier-fitting loss needs the
# dense, per-graph, full-FFT-grid layout so that a DFT reference `rho_fftn`
# loaded from data can be zero-padded / truncated onto the model's k-grid via
# `resize_fft`. Re-implemented locally here to keep the port contained to
# mace-scf. Mirror of BAK_repo/FourierFitting/graph_longrange/kspace.py
# (full_grid=True branch).
def compute_k_vectors_dense_full_grid(
    cutoff: torch.Tensor,
    cell_vectors: torch.Tensor,     # [n_graphs, 3, 3]
    r_cell_vectors: torch.Tensor,   # [n_graphs, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Dense full-grid k-vectors for Fourier fitting.

    Returns
    -------
    k_vectors : [n_graphs, n_kvec, 3]
    k_vectors_normed_squared : [n_graphs, n_kvec]
    mask : [n_graphs, n_kvec]           True where the k-vector is inside the cutoff sphere
    grid_shape : [2*n1max, 2*n2max, 2*n3max]  the FFT grid shape used to enumerate k-vectors
    """
    device = cell_vectors.device
    dtype = r_cell_vectors.dtype

    # Determine per-axis max integer index from cutoff.
    norms = torch.norm(cell_vectors, dim=-1)                          # [n_graphs, 3]
    normed_lattice_vectors = cell_vectors / norms.unsqueeze(-1)
    dot_products = torch.einsum(
        "bij,bij->bi", r_cell_vectors, normed_lattice_vectors
    )                                                                 # [n_graphs, 3]
    max_ns = torch.ceil(cutoff * torch.pow(dot_products, -1)).to(torch.int64)
    max_max_ns = torch.max(max_ns, dim=0).values                      # [3]
    n1max = int(max_max_ns[0].item())
    n2max = int(max_max_ns[1].item())
    n3max = int(max_max_ns[2].item())

    grid_shape = [2 * n1max, 2 * n2max, 2 * n3max]

    # Enumerate all k-vectors on the fftfreq grid (with factors of 2 to match
    # the half-sphere convention used by the old full_grid=False path).
    int_kvecs = torch.cartesian_prod(
        torch.fft.fftfreq(2 * n1max, device=device) * 2 * n1max,
        torch.fft.fftfreq(2 * n2max, device=device) * 2 * n2max,
        torch.fft.fftfreq(2 * n3max, device=device) * 2 * n3max,
    ).to(dtype)

    k_vectors = torch.einsum("ni,bij->bnj", int_kvecs, r_cell_vectors)
    k_vectors_normed_squared = torch.einsum("bni,bni->bn", k_vectors, k_vectors)
    mask = k_vectors_normed_squared.le(cutoff * cutoff)
    return k_vectors, k_vectors_normed_squared, mask, grid_shape


def assemble_rho_fftn_dense_grid(
    source_feats: torch.Tensor,     # [n_nodes, m_dim]  (single sigma; matches charges_irreps)
    node_positions: torch.Tensor,   # [n_nodes, 3]
    k_vectors: torch.Tensor,        # [n_graph, max_k, 3]
    basis_fs: torch.Tensor,         # [n_graph, max_k, n_sigma(=1), m_dim, 2]  from GTOBasis
    volume: torch.Tensor,           # [n_graph]
    batch: torch.Tensor,            # [n_nodes]
) -> torch.Tensor:
    """Assemble rho(k) on the dense per-graph k-grid used by the Fourier
    fitting loss. Mirror of graph_longrange.features.assemble_fourier_series_batch
    but for the dense [n_graph, max_k, ...] layout instead of the flat
    [n_k_total, ...] layout, so it lines up with resize_fft output.

    Returns rho(k) with shape [n_graph, max_k, 2] (real, imag stacked).
    """
    import math
    n_graph = k_vectors.size(0)
    max_k = k_vectors.size(1)

    # Per-node k-vectors and basis (broadcast from graph → node)
    kv_per_node = torch.index_select(k_vectors, 0, batch)   # [n_nodes, max_k, 3]
    inner = (kv_per_node * node_positions.unsqueeze(1)).sum(-1)  # [n_nodes, max_k]
    cos_i = torch.cos(inner)
    sin_i = torch.sin(inner)

    basis_per_node = torch.index_select(basis_fs.squeeze(-3), 0, batch)  # [n_nodes, max_k, m_dim, 2]
    br = basis_per_node[..., 0]     # [n_nodes, max_k, m_dim]
    bi = basis_per_node[..., 1]

    coef = source_feats                                     # [n_nodes, m_dim]
    contrib_r = (coef.unsqueeze(1) * (br * cos_i.unsqueeze(-1) + bi * sin_i.unsqueeze(-1))).sum(-1)
    contrib_i = (coef.unsqueeze(1) * (bi * cos_i.unsqueeze(-1) - br * sin_i.unsqueeze(-1))).sum(-1)

    rho_r = scatter_sum(contrib_r, batch, dim=0, dim_size=n_graph)  # [n_graph, max_k]
    rho_i = scatter_sum(contrib_i, batch, dim=0, dim_size=n_graph)

    rho = torch.stack([rho_r, rho_i], dim=-1)
    scale = (2 * math.pi) ** 3
    return rho * scale / volume.view(-1, 1, 1)


def build_rho_fftn_dft_from_data(
    data_rho_fftn: torch.Tensor,        # [sum_i N_i]  flat complex, concatenated over batch
    data_rho_fftn_shape: torch.Tensor,  # [n_graph, 3] long
    grid_shape: List[int],              # target FFT grid shape used by the model's dense k-grid
) -> torch.Tensor:
    """Resize each per-graph raw DFT rho_fftn onto the model's k-grid and
    stack into [n_graph, prod(grid_shape), 2] real+imag flat layout. Used by
    the "direct" reference mode.
    """
    n_graph = int(data_rho_fftn_shape.size(0))
    total = grid_shape[0] * grid_shape[1] * grid_shape[2]
    out = torch.zeros(
        (n_graph, total, 2),
        dtype=data_rho_fftn.real.dtype if data_rho_fftn.is_complex() else data_rho_fftn.dtype,
        device=data_rho_fftn.device,
    )
    offset = 0
    for i in range(n_graph):
        shape = tuple(int(s) for s in data_rho_fftn_shape[i].tolist())
        n_i = shape[0] * shape[1] * shape[2]
        cs = data_rho_fftn[offset:offset + n_i].reshape(shape)
        offset += n_i
        resized = resize_fft(cs, (grid_shape[0], grid_shape[1], grid_shape[2]))
        out[i, :, 0] = resized.real.flatten()
        out[i, :, 1] = resized.imag.flatten()
    return out


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
        cell_virials = 0.5 * (cell_virials + cell_virials.transpose(-1, -2))
        cell = cell.view(-1, 3, 3)
        cell_virials *= cell
        virials += cell_virials

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
