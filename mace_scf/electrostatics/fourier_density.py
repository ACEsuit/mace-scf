"""Shared Fourier density fitting block used by both FixedPointCore and
LocalSplitCharges.

Encapsulates the parts that were previously duplicated in each model:
  * the dense-grid GTOBasis (normalize="none" to match the BAK_repo
    ref_fourier path in localsources.py / fixed_point.py)
  * the dense per-graph k-vector pass (see utils.compute_k_vectors_dense
    _full_grid — re-implemented locally because the installed graph_longrange
    dropped the full_grid=True variant)
  * predicted rho(k) via assemble_rho_fftn_dense_grid
  * reference rho(k) either from ground-truth multipoles ("from_multipoles")
    or from raw DFT fftn resized onto the model grid ("direct")

The forward returns a dict with:
    rho_fftn                 [n_graph, max_k, 2]
    rho_fftn_dft             [n_graph, max_k, 2]
    k_vectors_normed_squared [n_graph, max_k]
    k_vectors_mask           [n_graph, max_k] bool
    k_vectors_grid_shape     [3] long
"""
from typing import Dict, Optional

import torch

from graph_longrange.gto_utils import GTOBasis

from .utils import (
    assemble_rho_fftn_dense_grid,
    build_rho_fftn_dft_from_data,
    compute_k_vectors_dense_full_grid,
)


class FourierDensityFittingBlock(torch.nn.Module):
    _ALLOWED_MODES = ("from_multipoles", "direct")

    def __init__(
        self,
        mode: str,
        atomic_multipoles_max_l: int,
        atomic_multipoles_smearing_width: float,
        kspace_cutoff: float,
    ):
        super().__init__()
        if mode not in self._ALLOWED_MODES:
            raise ValueError(
                f"mode must be one of {self._ALLOWED_MODES}; got {mode!r}"
            )
        self.mode: str = mode
        self.density_basis = GTOBasis(
            max_l=atomic_multipoles_max_l,
            sigmas=[atomic_multipoles_smearing_width],
            kspace_cutoff=float(kspace_cutoff),
            normalize="none",
        )

    @torch.jit.ignore
    def _direct_reference(
        self,
        data_rho_fftn: torch.Tensor,
        data_rho_fftn_shape: torch.Tensor,
        grid_shape,
        target_shape,
    ) -> torch.Tensor:
        flat = build_rho_fftn_dft_from_data(
            data_rho_fftn=data_rho_fftn,
            data_rho_fftn_shape=data_rho_fftn_shape,
            grid_shape=grid_shape,
        )
        return flat.view(target_shape)

    def forward(
        self,
        kspace_cutoff: torch.Tensor,
        cell: torch.Tensor,
        rcell: torch.Tensor,
        volume: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        density_pred: torch.Tensor,
        density_ref: torch.Tensor,
        data_rho_fftn: Optional[torch.Tensor] = None,
        data_rho_fftn_shape: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        k_dense, k2_dense, k_mask_dense, grid_shape = compute_k_vectors_dense_full_grid(
            kspace_cutoff, cell.view(-1, 3, 3), rcell.view(-1, 3, 3)
        )
        k0_mask_dense = (k2_dense == 0).to(k2_dense.dtype)
        basis_fs = self.density_basis(k_dense, k2_dense, k0_mask_dense)

        rho_pred = assemble_rho_fftn_dense_grid(
            source_feats=density_pred,
            node_positions=positions,
            k_vectors=k_dense,
            basis_fs=basis_fs,
            volume=volume.view(-1),
            batch=batch,
        )

        if self.mode == "from_multipoles":
            rho_dft = assemble_rho_fftn_dense_grid(
                source_feats=density_ref,
                node_positions=positions,
                k_vectors=k_dense,
                basis_fs=basis_fs,
                volume=volume.view(-1),
                batch=batch,
            )
        else:  # "direct" — TorchScript-unsafe because of complex tensors + Python loop;
               # gated behind @torch.jit.ignore.
            assert data_rho_fftn is not None and data_rho_fftn_shape is not None, (
                "rho_fftn_mode='direct' requires data['rho_fftn'] and "
                "data['rho_fftn_shape'] to be populated"
            )
            rho_dft = self._direct_reference(
                data_rho_fftn, data_rho_fftn_shape, grid_shape, rho_pred.shape
            )

        return {
            "rho_fftn": rho_pred,
            "rho_fftn_dft": rho_dft,
            "k_vectors_normed_squared": k2_dense,
            "k_vectors_mask": k_mask_dense,
            "k_vectors_grid_shape": torch.tensor(grid_shape, dtype=torch.long),
        }
