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
"""

import math

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
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
