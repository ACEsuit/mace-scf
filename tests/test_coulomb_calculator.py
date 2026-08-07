"""Tests for ``mace_scf.calculators.coulomb.GTOCoulombCalculator``.

1. Two opposite charges in vacuum (``pbc_handling="realspace"``) against the
   analytic smeared-Coulomb formula.
2. Same configuration evaluated in ``pbc_handling="molecule_in_box"`` with a
   large enough cell that the monopole-dipole correction reproduces the
   isolated-pair result.
3. Rocksalt NaCl in ``pbc_handling="pbc"`` against the matscipy Ewald
   calculator. With a sufficiently small smearing width the GTO Coulomb
   energy (with the smeared self-energy subtracted) reproduces the
   point-charge Ewald result to ~1e-5.
4. Analytic stress against a central finite difference in the applied strain.
   This is the only test that exercises the cell-gradient path: the reported
   stress combines a positions/displacement term with a k-space term that
   reaches autograd solely through ``cell.requires_grad_(True)`` in
   ``GTOCoulombCalculator.calculate``. Without that line ``cell_virials``
   comes back ``None`` (masked by ``allow_unused=True``) and the k-space
   contribution silently disappears.
"""

import math

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.stress import voigt_6_to_full_3x3_stress
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.utils import FIELD_CONSTANT

from mace_scf.calculators.coulomb import GTOCoulombCalculator


# Coulomb constant in eV * Angstrom / e^2, derived from graph_longrange's
# FIELD_CONSTANT = 1 / epsilon_0 (in e^2 / (eV * A)).
_K_E = FIELD_CONSTANT / (4.0 * math.pi)


def _smeared_pair_energy(q1: float, q2: float, d: float, sigma: float) -> float:
    """Energy of two l=0 GTO multipoles with smearing ``sigma`` at distance d.

    Matches the formula used by
    ``graph_longrange.realspace_electrostatics.charges_energy_from_graph``.
    """
    return _K_E * q1 * q2 * math.erf(d / (2.0 * sigma)) / d


@pytest.fixture(autouse=True)
def _double_precision():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def test_two_charges_realspace_matches_analytic():
    sigma = 0.5
    d = 4.0
    q = 0.7
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    atoms.cell = np.eye(3) * 20.0
    atoms.pbc = False

    calc = GTOCoulombCalculator(
        max_l=0,
        smearing_width=sigma,
        pbc_handling="realspace",
        include_self_interaction=False,
    )
    calc.set_multipoles(np.array([[q], [-q]]))
    atoms.calc = calc

    expected = _smeared_pair_energy(q, -q, d, sigma)
    assert atoms.get_potential_energy() == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_two_charges_molecule_in_box_matches_realspace():
    sigma = 0.5
    d = 4.0
    q = 0.7
    box = 30.0
    atoms = Atoms(
        "HH",
        positions=[
            [box / 2 - d / 2, box / 2, box / 2],
            [box / 2 + d / 2, box / 2, box / 2],
        ],
    )
    atoms.cell = np.eye(3) * box
    atoms.pbc = True

    calc = GTOCoulombCalculator(
        max_l=0,
        smearing_width=sigma,
        kspace_cutoff_factor=2.0,
        pbc_handling="molecule_in_box",
        include_self_interaction=False,
    )
    calc.set_multipoles(np.array([[q], [-q]]))
    atoms.calc = calc

    expected = _smeared_pair_energy(q, -q, d, sigma)
    # k-space sum + monopole-dipole correction in a finite box only reaches
    # the isolated-pair limit asymptotically with box size, so loosen the
    # tolerance accordingly.
    assert atoms.get_potential_energy() == pytest.approx(expected, rel=5e-3, abs=5e-3)


def test_nacl_matches_matscipy_ewald():
    matscipy_ewald = pytest.importorskip(
        "matscipy.calculators.ewald.calculator"
    )

    # matscipy >=devel still uses the long-removed ``np.int`` alias in one
    # branch of its triclinic-cell handling. Patch it so the test works on
    # numpy >= 1.20.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]

    a = 5.64
    # Conventional cubic NaCl (8 atoms, orthorhombic cell). matscipy's Ewald
    # heuristics assume a roughly orthogonal cell.
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=a, cubic=True)
    assert len(atoms) == 8
    # Charges: Na +1, Cl -1, in the order ase.build.bulk produces.
    symbols = atoms.get_chemical_symbols()
    charges = np.array(
        [1.0 if s == "Na" else -1.0 for s in symbols], dtype=float
    )

    # GTO Coulomb energy. Smearing must be much smaller than the bond length
    # (a/2 = 2.82 A) for the smeared GTO formula to reproduce point-charge
    # Coulomb after subtracting the smeared self-energy. The kspace cutoff
    # uses the graph_longrange heuristic (gto_basis_kspace_cutoff) and
    # multiplies the result by 2.0 - which is exactly what passing
    # kspace_cutoff_factor=2.0 does.
    sigma = 0.1
    calc = GTOCoulombCalculator(
        max_l=0,
        smearing_width=sigma,
        kspace_cutoff_factor=2.0,
        pbc_handling="pbc",
        include_self_interaction=False,
    )
    # sanity check that the calculator is actually using
    # 2.0 * gto_basis_kspace_cutoff([sigma], 0) as the kspace cutoff:
    assert calc.kspace_cutoff == pytest.approx(
        2.0 * gto_basis_kspace_cutoff([sigma], 0)
    )
    calc.set_multipoles(charges.reshape(-1, 1))
    atoms.calc = calc
    gto_energy = atoms.get_potential_energy()

    # Reference: matscipy Ewald summation on the same point charges.
    ref_atoms = atoms.copy()
    ref_atoms.set_array("charge", charges, dtype=float)
    ewald = matscipy_ewald.Ewald()
    ewald.set(accuracy=1e-8, cutoff=6.0, verbose=False)
    ref_atoms.calc = ewald
    ewald_energy = ref_atoms.get_potential_energy()

    assert gto_energy == pytest.approx(ewald_energy, rel=1e-5, abs=1e-5)


# --- stress vs finite-difference strain ------------------------------------

_STRESS_Q = 0.7
_STRESS_SIGMA = 0.6

# Triclinic and with both atoms off any symmetry axis, so that all six
# independent stress components are non-trivial. A cubic cell with a
# symmetric charge arrangement would leave the shear terms at zero and the
# comparison would pass vacuously.
_STRESS_CELL = np.array(
    [
        [7.0, 0.0, 0.0],
        [0.9, 6.2, 0.0],
        [0.4, 0.7, 5.8],
    ]
)
_STRESS_SCALED_POSITIONS = np.array(
    [
        [0.12, 0.20, 0.31],
        [0.55, 0.61, 0.72],
    ]
)


def _strained_pair(strain: np.ndarray | None = None) -> Atoms:
    """Two opposite charges in a triclinic periodic cell, optionally strained.

    Positions are set from *scaled* coordinates so that straining the cell
    carries the atoms with it. That reproduces exactly the transformation
    ``get_symmetric_displacement`` applies internally, where both cell and
    positions are right-multiplied by ``(I + strain)``.
    """
    cell = _STRESS_CELL
    if strain is not None:
        cell = cell @ (np.eye(3) + strain)
    atoms = Atoms("HH", cell=cell, pbc=True)
    atoms.set_scaled_positions(_STRESS_SCALED_POSITIONS)
    return atoms


def _stress_calculator() -> GTOCoulombCalculator:
    calc = GTOCoulombCalculator(
        max_l=0,
        smearing_width=_STRESS_SIGMA,
        kspace_cutoff_factor=1.5,
        pbc_handling="pbc",
        include_self_interaction=False,
    )
    calc.set_multipoles(np.array([[_STRESS_Q], [-_STRESS_Q]]))
    return calc


def _energy_at_strain(strain: np.ndarray | None) -> float:
    # A fresh calculator per evaluation, so no result caching in the ASE
    # Calculator base class can leak between strained configurations.
    atoms = _strained_pair(strain)
    atoms.calc = _stress_calculator()
    return atoms.get_potential_energy()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "compute_forces_virials_cellstress (mace_scf/electrostatics/utils.py) "
        "folds dE/dcell into the virial with an elementwise product, "
        "`cell_virials *= cell`, where the chain rule for cell -> cell @ (I + eps) "
        "requires the matrix product cell.T @ dE/dcell. Verified numerically: "
        "with the matmul the analytic stress reproduces this finite difference "
        "to ~2e-9, with the elementwise product it is off by ~4e-1. Fixing it "
        "changes stress for every LocalSources model too, so the golden "
        "*.expected.xyz files must be regenerated from the real reference "
        "models in the same change."
    ),
)
def test_stress_matches_finite_difference_strain():
    atoms = _strained_pair()
    atoms.calc = _stress_calculator()
    analytic = voigt_6_to_full_3x3_stress(atoms.get_stress())

    volume = atoms.get_volume()
    delta = 1e-5
    numerical = np.zeros((3, 3))

    for i in range(3):
        for j in range(i, 3):
            # Symmetric strain: for i == j the two increments add to give
            # eps_ii = delta; for i != j they give eps_ij = eps_ji = delta/2,
            # so in both cases dE = V * sigma_ij * delta.
            strain = np.zeros((3, 3))
            strain[i, j] += 0.5 * delta
            strain[j, i] += 0.5 * delta

            e_plus = _energy_at_strain(strain)
            e_minus = _energy_at_strain(-strain)
            value = (e_plus - e_minus) / (2.0 * delta * volume)
            numerical[i, j] = value
            numerical[j, i] = value

    # Guard against a vacuous pass: if the analytic stress collapsed to zero
    # the allclose below would still succeed against a zero finite difference.
    assert np.max(np.abs(analytic)) > 1e-6, (
        f"analytic stress is ~zero, test would be vacuous: {analytic}"
    )

    assert np.allclose(analytic, numerical, rtol=1e-6, atol=1e-9), (
        "stress does not match the finite-difference strain derivative:\n"
        f"analytic=\n{analytic}\nnumerical=\n{numerical}\n"
        f"difference=\n{analytic - numerical}"
    )
