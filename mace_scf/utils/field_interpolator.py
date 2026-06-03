import numpy as np
import torch
import warnings

from graph_longrange.features import (
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
    compute_coulomb_factor,
)
from graph_longrange.gto_utils import GTOBasis, gto_basis_kspace_cutoff
from graph_longrange.kspace import (
    compute_k_vectors_flat,
    evaluate_fourier_series_at_points_flat,
)
from graph_longrange.utils import FIELD_CONSTANT
from graph_longrange.jellium_energy import _slab_fourier_series_flat, _smooth_slab_density

from contextlib import contextmanager


@contextmanager
def use_dtype(dtype=torch.float64):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


class _BaseInterpolator:
    def __init__(
        self,
        sigma=2.0, 
        multipoles_max_l=1, 
        kspace_cutoff_factor=None,
        kspace_cutoff=None,
        device='cpu',
        subtract_total_charge=False,
    ):
        warnings.warn('currently only works for slabs with z-axis as the non-periodic axis.')
        cutoff_given = kspace_cutoff is not None
        factor_given = kspace_cutoff_factor is not None
        assert cutoff_given != factor_given, "Either provide a cutoff factor or a cutoff"
        if factor_given:
            kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
                [sigma],
                multipoles_max_l,
            )
        self.kspace_cutoff = kspace_cutoff
        self.max_l = multipoles_max_l
        self.sigma = sigma
        self.device = device
        self.dtype = torch.float64
        self.subtract_total_charge = subtract_total_charge

        with use_dtype(self.dtype):
            self.density_basis = GTOBasis(
                max_l=multipoles_max_l,
                sigmas=[sigma],
                kspace_cutoff=self.kspace_cutoff,
                normalize="multipoles",
            ).to(self.device)

    @staticmethod
    def _total_charge_dipole(
        multipoles: torch.Tensor, node_positions: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_charge = torch.zeros(
            1, dtype=multipoles.dtype, device=multipoles.device
        )
        total_charge.index_add_(0, batch, multipoles[:, 0])
        dipole_q = torch.zeros(
            (1, 3), dtype=multipoles.dtype, device=multipoles.device
        )
        dipole_q.index_add_(0, batch, node_positions * multipoles[:, 0].unsqueeze(-1))
        if multipoles.shape[-1] > 1:
            dipole_p = torch.zeros_like(dipole_q)
            dipole_p.index_add_(0, batch, multipoles[:, 1:4])
            dipole_q = dipole_q + dipole_p[:, [2, 0, 1]]
        return total_charge, dipole_q

    def _prepare_inputs(self, atoms, atomic_multipoles, coords):
        output_shape = coords.shape[:-1]
        sample_points = torch.tensor(
            coords, dtype=self.dtype, device=self.device
        ).reshape(-1, 3)

        node_positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device
        ).reshape(-1, 3)
        cell = torch.tensor(
            np.asarray(atoms.get_cell().array),
            dtype=self.dtype,
            device=self.device,
        ).reshape(1, 3, 3)
        pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=self.device).reshape(
            1, 3
        )
        batch = torch.zeros(
            node_positions.shape[0], dtype=torch.long, device=self.device
        )
        volume = torch.det(cell)

        multipoles = torch.tensor(
            atomic_multipoles, dtype=self.dtype, device=self.device
        ).reshape(node_positions.size(0), -1)
        return (
            output_shape,
            sample_points,
            node_positions,
            cell,
            pbc,
            batch,
            volume,
            multipoles,
        )

    def _compute_kspace(self, cell):
        rcell = 2 * np.pi * torch.linalg.inv(cell).transpose(-1, -2)
        return compute_k_vectors_flat(
            cutoff=self.kspace_cutoff, cell_vectors=cell, r_cell_vectors=rcell
        )

    def _compute_density(
        self,
        multipoles,
        cosines,
        sines,
        density_basis_fs,
        volume_per_k,
    ):
        return assemble_fourier_series_batch(
            source_feats=multipoles,
            cosines=cosines,
            sines=sines,
            density_basis_fs=density_basis_fs,
            volume_per_k=volume_per_k,
        )

    def _augment_density(
        self,
        density,
        k_vectors,
        k_norm2,
        k0_mask,
        volume_per_k,
    ):
        return density

    def _compute_total_dipole(
        self,
        multipoles,
        node_positions,
        batch,
    ):
        return self._total_charge_dipole(multipoles, node_positions, batch)

    def _correction_field_z(self, total_dipole, volume, pbc):
        A = FIELD_CONSTANT / (4 * np.pi)
        correction_field_z = A * 4 * np.pi * total_dipole[:, 2] / volume
        correction_field_z = correction_field_z * (~pbc[:, 2]).to(total_dipole.dtype)
        return correction_field_z

    def __call__(self, atoms, atomic_multipoles, external_field, fermi_level, coords):
        assert coords.shape[-1] == 3
        assert len(external_field.shape) == 1
        assert type(fermi_level) == float

        (
            output_shape,
            sample_points,
            node_positions,
            cell,
            pbc,
            batch,
            volume,
            multipoles,
        ) = self._prepare_inputs(atoms, atomic_multipoles, coords)

        if not torch.all(pbc == torch.tensor([True, True, False], device=pbc.device)):
            raise ValueError("PotentialInterpolator only supports pbc = TTF.")

        if self.subtract_total_charge:
            total_charge, _ = self._total_charge_dipole(multipoles, node_positions, batch)
            multipoles[:, 0] -= total_charge[0] / multipoles.shape[0]

        k_vectors, k_norm2, k_vector_batch, k0_mask = self._compute_kspace(cell)

        self._volume = volume
        self._total_charge = multipoles[:, 0].sum()

        inner_products = torch.matmul(k_vectors, node_positions.t())
        k_mask = k_vector_batch[:, None] == batch[None, :]
        k_mask_f = k_mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * k_mask_f
        sines = torch.sin(inner_products) * k_mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        volume_per_k = volume.reshape(-1)[k_vector_batch]
        density = self._compute_density(
            multipoles=multipoles,
            cosines=cosines,
            sines=sines,
            density_basis_fs=density_basis_fs,
            volume_per_k=volume_per_k,
        )
        density = self._augment_density(
            density=density,
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k0_mask=k0_mask,
            volume_per_k=volume_per_k,
        )
        k_factor_coulomb = compute_coulomb_factor(k_norm2, k0_mask)
        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=k_factor_coulomb,
        )

        _, total_dipole = self._compute_total_dipole(
            multipoles, node_positions, batch
        )
        correction_field_z = self._correction_field_z(total_dipole, volume, pbc)

        sample_batch = torch.zeros(
            sample_points.shape[0], dtype=torch.long, device=self.device
        )

        samples_density = evaluate_fourier_series_at_points_flat(
            k_vectors=k_vectors,
            k_vector_batch=k_vector_batch,
            fourier_coefficients=density,
            sample_points=sample_points,
            sample_batch=sample_batch,
            k0_mask=k0_mask,
        )
        samples_potential = evaluate_fourier_series_at_points_flat(
            k_vectors=k_vectors,
            k_vector_batch=k_vector_batch,
            fourier_coefficients=potential,
            sample_points=sample_points,
            sample_batch=sample_batch,
            k0_mask=k0_mask,
        ) + fermi_level

        spread_field_z = correction_field_z[sample_batch]
        samples_potential_corrected = (
            samples_potential
            + spread_field_z * sample_points[:, 2]
            + external_field[2] * sample_points[:, 2]
            + fermi_level
        )

        samples_density = samples_density.cpu().detach().numpy().reshape(output_shape)
        samples_potential = samples_potential.cpu().detach().numpy().reshape(output_shape)
        samples_potential_corrected = samples_potential_corrected.cpu().detach().numpy().reshape(output_shape)

        return samples_density, samples_potential, samples_potential_corrected


class PotentialInterpolator(_BaseInterpolator):
    def __init__(
        self,
        sigma=2.0,
        multipoles_max_l=1,
        kspace_cutoff_factor=None,
        kspace_cutoff=None,
        device='cpu',
        subtract_total_charge=False,
    ):
        super().__init__(
            sigma=sigma,
            multipoles_max_l=multipoles_max_l,
            kspace_cutoff_factor=kspace_cutoff_factor,
            kspace_cutoff=kspace_cutoff,
            device=device,
            subtract_total_charge=subtract_total_charge,
        )


class PotentialInterpolatorJellium(_BaseInterpolator):
    def __init__(
        self,
        slab_bounds,
        sigma=2.0,
        multipoles_max_l=1,
        kspace_cutoff_factor=None,
        kspace_cutoff=None,
        device='cpu',
        subtract_total_charge=False,
    ):
        super().__init__(
            sigma=sigma,
            multipoles_max_l=multipoles_max_l,
            kspace_cutoff_factor=kspace_cutoff_factor,
            kspace_cutoff=kspace_cutoff,
            device=device,
            subtract_total_charge=subtract_total_charge,
        )
        slab_lower = float(slab_bounds[0])
        slab_upper = float(slab_bounds[1])
        if slab_upper <= slab_lower:
            raise ValueError("Jellium slab bounds are invalid (upper <= lower).")
        self.register_buffer("slab_lower", torch.tensor(slab_lower))
        self.register_buffer("slab_upper", torch.tensor(slab_upper))

        with use_dtype(self.dtype):
            self.smoothing_basis = GTOBasis(
                max_l=0,
                sigmas=[sigma],
                kspace_cutoff=self.kspace_cutoff,
                normalize="multipoles",
            ).to(self.device)

    def _augment_density(
        self,
        density,
        k_vectors,
        k_norm2,
        k0_mask,
        volume_per_k,
    ):
        total_charge = -self._total_charge
        lower = self.slab_lower
        upper = self.slab_upper
        slab_density = _slab_fourier_series_flat(
            total_charge=total_charge,
            lower=lower,
            upper=upper,
            k_vectors=k_vectors,
            k0_mask=k0_mask,
            volume=self._volume,
        )
        smoothing_basis_fs = self.smoothing_basis(k_vectors, k_norm2, k0_mask)
        cosines_one = torch.ones(
            (k_vectors.size(0), 1), device=k_vectors.device, dtype=k_vectors.dtype
        )
        sines_one = torch.zeros_like(cosines_one)
        ones = torch.ones((1, 1), device=k_vectors.device, dtype=k_vectors.dtype)
        smoothing_density = assemble_fourier_series_batch(
            source_feats=ones,
            cosines=cosines_one,
            sines=sines_one,
            density_basis_fs=smoothing_basis_fs,
            volume_per_k=volume_per_k,
        )
        return density + _smooth_slab_density(
            slab_density=slab_density,
            smoothing_density=smoothing_density,
            volume_per_k=volume_per_k,
        )

    def _compute_total_dipole(
        self,
        multipoles,
        node_positions,
        batch,
    ):
        total_charge, total_dipole = self._total_charge_dipole(
            multipoles, node_positions, batch
        )
        z_center = 0.5 * (self.slab_lower + self.slab_upper)
        slab_com = torch.zeros(
            (1, 3), device=node_positions.device, dtype=node_positions.dtype
        )
        slab_com[0, 2] = z_center
        total_dipole = total_dipole - slab_com * total_charge.unsqueeze(-1)
        return total_charge, total_dipole


class PotentialInterpolatorPostProcessor:
    def __init__(self, calc, sigma, **kwargs):
        self.calc = calc
        if 'kspace_cutoff' in kwargs:
            kspace_cutoff = kwargs.pop('kspace_cutoff')
        else:
            kspace_cutoff = calc.model.coulomb_energy.kspace_cutoff
        self.base_interpolator = PotentialInterpolator(
            sigma=sigma,
            multipoles_max_l=calc.model.coulomb_energy.density_max_l,
            kspace_cutoff=kspace_cutoff,
            **kwargs
        )

    def __call__(self, atoms, coords):
        atomic_multipoles = self.calc.results['density_coefficients']
        external_field = self.calc.results['external_field']
        fermi_level = self.calc.results['fermi_level']
        return self.base_interpolator(
            atoms, atomic_multipoles, external_field, fermi_level, coords
        )
