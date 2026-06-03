"""ASE calculator wrapping ``graph_longrange.energy.GTOElectrostaticEnergy``.

Evaluates the Coulomb energy of a user-supplied set of atom-centred GTO
multipoles (and an optional homogeneous external field). No ML model is
involved.

Units are fixed throughout the repo to ``e / eV / Angstrom`` and this
calculator is no exception: there are no unit-conversion knobs, and passing
``energy_units_to_eV`` / ``length_units_to_A`` (or similar) raises
``TypeError``.
"""

from typing import Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from scipy.constants import pi

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.kspace import compute_k_vectors_flat
from mace.modules.utils import get_symmetric_displacement

def _load_electrostatics_utils():
    """Load ``mace_scf/electrostatics/utils.py`` without importing the
    parent package, because the broader ``mace_scf.electrostatics`` package
    has unrelated import issues we don't want to drag in here.
    """
    import importlib.util
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "electrostatics", "utils.py")
    spec = importlib.util.spec_from_file_location(
        "mace_scf._coulomb_electrostatics_utils", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_eu = _load_electrostatics_utils()
compute_forces_virials_cellstress = _eu.compute_forces_virials_cellstress
compute_total_charge_dipole = _eu.compute_total_charge_dipole


_FORBIDDEN_UNIT_KWARGS = ("energy_units_to_eV", "length_units_to_A")


class GTOCoulombCalculator(Calculator):
    """Coulomb energy of GTO multipoles, exposed as an ASE calculator."""

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "dipole",
        "partial_charges",
        "partial_dipoles",
    ]

    def __init__(
        self,
        max_l: int,
        smearing_width: float,
        kspace_cutoff_factor: float = 1.5,
        pbc_handling: str = "mixed_periodic",
        include_self_interaction: bool = False,
        multipoles_key: str = "multipoles",
        external_field_key: str = "external_field",
        device: str = "cpu",
        default_dtype: str = "float64",
        **kwargs,
    ):
        offenders = [k for k in _FORBIDDEN_UNIT_KWARGS if k in kwargs]
        if offenders:
            raise TypeError(
                f"GTOCoulombCalculator does not accept unit-conversion "
                f"kwargs {offenders}. The whole repo uses e / eV / Angstrom; "
                f"convert inputs to those units before calling the calculator."
            )

        Calculator.__init__(self, **kwargs)

        if default_dtype == "float64":
            self._dtype = torch.float64
        elif default_dtype == "float32":
            self._dtype = torch.float32
        else:
            raise ValueError(f"Unknown default_dtype: {default_dtype!r}")

        self.device = torch.device(device)
        self.max_l = int(max_l)
        self.density_dim = (self.max_l + 1) ** 2
        self.smearing_width = float(smearing_width)
        self.pbc_handling = pbc_handling
        self.multipoles_key = multipoles_key
        self.external_field_key = external_field_key

        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
            [self.smearing_width], self.max_l
        )
        self.kspace_cutoff = float(kspace_cutoff)
        self.coulomb_energy = GTOElectrostaticEnergy(
            density_max_l=self.max_l,
            density_smearing_width=self.smearing_width,
            kspace_cutoff=self.kspace_cutoff,
            include_self_interaction=include_self_interaction,
            pbc_handling=pbc_handling,
        ).to(self.device).to(self._dtype)

        self._multipoles: Optional[np.ndarray] = None
        self._external_field: Optional[np.ndarray] = None

    def set_multipoles(self, multipoles: np.ndarray) -> None:
        """Provide the per-atom multipole coefficients to use on the next call.

        Shape ``(n_atoms, (max_l + 1) ** 2)``. The ordering follows the CS
        phase convention used elsewhere in mace_scf (see
        ``docs/concepts/atomic_multipoles.md``): for ``max_l=1`` the columns
        are ``[q, p_y, p_z, p_x]``.
        """
        arr = np.asarray(multipoles)
        if arr.ndim != 2 or arr.shape[1] != self.density_dim:
            raise ValueError(
                f"multipoles must have shape (n_atoms, {self.density_dim}), "
                f"got {arr.shape}."
            )
        self._multipoles = arr.astype(np.float64, copy=True)

    def set_external_field(self, external_field: np.ndarray) -> None:
        arr = np.asarray(external_field).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(
                f"external_field must have 3 components, got shape {arr.shape}."
            )
        self._external_field = arr.astype(np.float64, copy=True)

    def _resolve_multipoles(self, atoms) -> np.ndarray:
        if self._multipoles is not None:
            arr = self._multipoles
        elif self.multipoles_key in atoms.arrays:
            arr = np.asarray(atoms.arrays[self.multipoles_key])
        else:
            raise ValueError(
                "No multipoles provided. Call calc.set_multipoles(...) or "
                f"set atoms.arrays[{self.multipoles_key!r}]."
            )
        if arr.shape != (len(atoms), self.density_dim):
            raise ValueError(
                f"multipoles shape {arr.shape} does not match "
                f"(n_atoms={len(atoms)}, density_dim={self.density_dim})."
            )
        return arr

    def _resolve_external_field(self, atoms) -> np.ndarray:
        if self._external_field is not None:
            return self._external_field
        if self.external_field_key in atoms.info:
            arr = np.asarray(atoms.info[self.external_field_key]).reshape(-1)
            if arr.shape != (3,):
                raise ValueError(
                    f"atoms.info[{self.external_field_key!r}] must have 3 "
                    f"components, got shape {arr.shape}."
                )
            return arr.astype(np.float64, copy=True)
        return np.zeros(3, dtype=np.float64)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, system_changes=system_changes)

        properties = properties or ["energy"]
        compute_stress = "stress" in properties

        multipoles_np = self._resolve_multipoles(atoms)
        external_field_np = self._resolve_external_field(atoms)
        n_atoms = len(atoms)
        num_graphs = 1

        positions = torch.tensor(
            np.asarray(atoms.positions),
            dtype=self._dtype,
            device=self.device,
            requires_grad=True,
        )
        cell = torch.tensor(
            np.asarray(atoms.cell),
            dtype=self._dtype,
            device=self.device,
        ).view(1, 3, 3).clone()
        pbc = torch.tensor(
            np.asarray(atoms.pbc), dtype=torch.bool, device=self.device
        ).view(1, 3)
        batch = torch.zeros(n_atoms, dtype=torch.long, device=self.device)

        # Match _LocalSourceModelBase.forward: build a symmetric strain
        # variable wired into positions and cell so autograd produces the
        # correct stress. No edges in this calculator -> empty edge_index
        # and unit_shifts; get_symmetric_displacement just returns empty
        # shifts in that case.
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        unit_shifts = torch.zeros((0, 3), dtype=self._dtype, device=self.device)
        positions, _, displacement = get_symmetric_displacement(
            positions=positions,
            unit_shifts=unit_shifts,
            cell=cell.clone(),
            edge_index=edge_index,
            num_graphs=num_graphs,
            batch=batch,
        )
        cell.requires_grad_(True)

        rcell = 2 * pi * torch.linalg.inv_ex(cell.mT)[0]
        volume = torch.linalg.det(cell.view(-1, 3, 3)).abs()

        k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
            self.kspace_cutoff,
            cell.view(-1, 3, 3),
            rcell.view(-1, 3, 3),
        )

        multipoles_t = torch.tensor(
            multipoles_np, dtype=self._dtype, device=self.device
        )
        external_field_t = torch.tensor(
            external_field_np, dtype=self._dtype, device=self.device
        ).view(1, 3)

        coulomb_e = self.coulomb_energy(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=multipoles_t,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

        _, total_dipole = compute_total_charge_dipole(
            density_coefficients=multipoles_t,
            positions=positions,
            batch=batch,
            num_graphs=num_graphs,
        )
        field_energy = torch.sum(total_dipole * external_field_t, dim=-1)
        energy = coulomb_e + field_energy  # [num_graphs]

        forces, _, stress = compute_forces_virials_cellstress(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            training=False,
            compute_stress=compute_stress,
        )

        energy_val = float(energy.detach().cpu().item())
        partial_charges = multipoles_np[:, 0].copy()
        if multipoles_np.shape[1] > 1:
            partial_dipoles = multipoles_np[:, [3, 1, 2]].copy()
        else:
            partial_dipoles = np.zeros((n_atoms, 3))

        results = {
            "energy": energy_val,
            "free_energy": energy_val,
            "forces": forces.detach().cpu().numpy(),
            "dipole": total_dipole.detach().cpu().numpy()[0],
            "partial_charges": partial_charges,
            "partial_dipoles": partial_dipoles,
            "external_field": external_field_np.copy(),
        }
        if compute_stress:
            results["stress"] = full_3x3_to_voigt_6_stress(
                stress.detach().cpu().numpy()[0]
            )
        self.results = results
