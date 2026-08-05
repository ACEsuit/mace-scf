"""Finite-difference validation of the autograd stress of the LocalSplitCharges model.

The analytic stress is assembled in compute_forces_virials_cellstress, which
converts the cell gradient of the k-space electrostatic energy into a virial.
A random-weight model exercises that code path just as well as a trained one,
so the test builds a small model on the fly and needs no model fixture files.

The triclinic cell is essential: an incorrect cell-gradient conversion can
reproduce the correct normal stress on a diagonal cell while corrupting the
shear components, so cubic-only agreement is not sufficient.
"""

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from ase.stress import voigt_6_to_full_3x3_stress
from e3nn import o3

import mace.modules
import mace.tools
from mace.tools import torch_tools

from mace_scf.calculators.localsources import MACELocalSplitCharges
from mace_scf.electrostatics import LocalSplitCharges
from tests.utils import disable_e3nn_codegen, seed_torch

FORMAL_CHARGES_KEY = "formal_oxidation_states"
FINITE_DIFFERENCE_STEP = 1e-6
STRESS_ABSOLUTE_TOLERANCE = 1e-9  # eV/A^3; observed agreement is ~3e-12
MINIMUM_STRESS_SIGNAL = 1e-6  # eV/A^3; guards against a trivially zero stress

PERIODIC_IMAGE_INVARIANCE_TOLERANCE = 1e-12  # eV/A^3; observed agreement is ~1e-17

CUBIC_CELL = np.diag([7.0, 7.0, 7.0])
TRICLINIC_CELL = np.array(
    [
        [7.0, 0.0, 0.0],
        [1.3, 6.8, 0.0],
        [0.9, -1.1, 6.9],
    ]
)

# Integer lattice translations applied per atom. The two molecules are split
# across different periodic images (atom 2 relative to atoms 0-1, atoms 4-5
# relative to atom 3), so bonded edges cross image boundaries.
PERIODIC_IMAGE_SHIFTS = np.array(
    [
        [1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [-1, 1, 0],
        [0, 0, 1],
        [200, 0, -1],
    ]
)


def build_random_local_split_charges_model(
    seed: int, atomic_numbers: list[int]
) -> LocalSplitCharges:
    torch_tools.set_default_dtype("float64")
    seed_torch(seed)
    z_table = mace.tools.get_atomic_number_table_from_zs(atomic_numbers)
    interaction_cls = mace.modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    with disable_e3nn_codegen():
        return LocalSplitCharges(
            r_max=3.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls,
            num_interactions=2,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("8x0e+8x1o"),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.array([1.0] * len(z_table)),
            avg_num_neighbors=10.0,
            atomic_numbers=z_table.zs,
            correlation=2,
            formal_charges_from_data=True,
            gate=mace.modules.gate_dict["silu"],
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            include_electrostatic_self_interaction=True,
            pbc_handling="pbc",
        )


@pytest.fixture(scope="module")
def local_split_charges_calculator(tmp_path_factory):
    model = build_random_local_split_charges_model(seed=42, atomic_numbers=[1, 8])
    model_path = tmp_path_factory.mktemp("models") / "random_local_split_charges.model"
    torch.save(model, model_path)
    return MACELocalSplitCharges(
        model_path=str(model_path),
        device="cpu",
        formal_charges_key=FORMAL_CHARGES_KEY,
        pbc_handling="pbc",
    )


def periodic_water_dimer(cell: np.ndarray) -> Atoms:
    positions = np.array(
        [
            [3.10, 3.05, 3.12],
            [3.65, 3.80, 2.95],
            [2.31, 3.36, 3.55],
            [4.85, 5.10, 4.60],
            [5.25, 4.35, 5.05],
            [4.25, 5.55, 5.22],
        ]
    )
    atoms = Atoms(symbols="OHHOHH", positions=positions, cell=cell, pbc=True)
    atoms.arrays[FORMAL_CHARGES_KEY] = np.array([-2.0, 1.0, 1.0, -2.0, 1.0, 1.0])
    return atoms


@pytest.mark.parametrize(
    "cell", [CUBIC_CELL, TRICLINIC_CELL], ids=["cubic", "triclinic"]
)
def test_autograd_stress_matches_finite_difference_stress(
    local_split_charges_calculator, cell
):
    atoms = periodic_water_dimer(cell)
    atoms.calc = local_split_charges_calculator

    autograd_stress = voigt_6_to_full_3x3_stress(atoms.get_stress(voigt=True))
    finite_difference_stress = voigt_6_to_full_3x3_stress(
        local_split_charges_calculator.calculate_numerical_stress(
            atoms, d=FINITE_DIFFERENCE_STEP, voigt=True
        )
    )

    print(finite_difference_stress)
    print(autograd_stress)

    off_diagonal = finite_difference_stress - np.diag(np.diag(finite_difference_stress))
    assert np.abs(np.diag(finite_difference_stress)).max() > MINIMUM_STRESS_SIGNAL
    assert np.abs(off_diagonal).max() > MINIMUM_STRESS_SIGNAL

    np.testing.assert_allclose(
        autograd_stress,
        finite_difference_stress,
        rtol=0.0,
        atol=STRESS_ABSOLUTE_TOLERANCE,
        err_msg="autograd stress deviates from finite-difference stress",
    )


@pytest.mark.parametrize(
    "cell", [CUBIC_CELL, TRICLINIC_CELL], ids=["cubic", "triclinic"]
)
def test_stress_invariant_under_periodic_image_shifts(
    local_split_charges_calculator, cell
):
    """Atoms translated into other periodic images are the same physical system,
    so the stress must not change, and autograd must still match finite
    differences with molecules split across image boundaries."""
    reference_atoms = periodic_water_dimer(cell)
    reference_atoms.calc = local_split_charges_calculator
    reference_stress = voigt_6_to_full_3x3_stress(
        reference_atoms.get_stress(voigt=True)
    )

    shifted_atoms = periodic_water_dimer(cell)
    shifted_atoms.positions = shifted_atoms.positions + PERIODIC_IMAGE_SHIFTS @ cell
    shifted_atoms.calc = local_split_charges_calculator

    autograd_stress = voigt_6_to_full_3x3_stress(shifted_atoms.get_stress(voigt=True))
    finite_difference_stress = voigt_6_to_full_3x3_stress(
        local_split_charges_calculator.calculate_numerical_stress(
            shifted_atoms, d=FINITE_DIFFERENCE_STEP, voigt=True
        )
    )

    np.testing.assert_allclose(
        autograd_stress,
        reference_stress,
        rtol=0.0,
        atol=PERIODIC_IMAGE_INVARIANCE_TOLERANCE,
        err_msg="stress changed when atoms were moved to other periodic images",
    )
    np.testing.assert_allclose(
        autograd_stress,
        finite_difference_stress,
        rtol=0.0,
        atol=STRESS_ABSOLUTE_TOLERANCE,
        err_msg="autograd stress deviates from finite-difference stress",
    )
