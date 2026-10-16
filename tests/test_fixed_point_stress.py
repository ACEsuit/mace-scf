"""Stress of FixedPointCore through the training wrapper vs finite differences.

The finite-difference stress comes from ase: it strains the cell and the
positions by cell -> cell @ (I + eps) and takes sigma = (1/V) dE/deps from
central differences of the converged energy.

The triclinic cell is essential: an incorrect cell-gradient conversion can give
the correct normal stress on a diagonal cell but wrong shear components.
"""

from functools import cache

import numpy as np
import pytest
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.fd import calculate_numerical_stress

from tests.utils import (
    CUBIC_CELL,
    TRICLINIC_CELL,
    copy_batch,
    dataset_from_atoms,
    make_fixed_point_model_random,
    make_fixed_point_wrapper,
    periodic_water_dimer,
    wrap_loader,
)

UPDATE_SCALE = 0.2  # as in test_fixed_point_parameter_gradients.py: the SCF converges
FINITE_DIFFERENCE_STEP = 1e-5
STRESS_ABSOLUTE_TOLERANCE = 1e-8  # eV/A^3; observed agreement is ~6e-12
MINIMUM_STRESS_SIGNAL = 1e-6  # eV/A^3; guards against a trivially zero stress
FORCE_ABSOLUTE_TOLERANCE = 1e-10

MODELS = {
    "linear": "NoNonLinearity",
    "nonlinear": "MLPNonLinearity",
}
CELLS = {"cubic": CUBIC_CELL, "triclinic": TRICLINIC_CELL}


class _WrapperEnergy(Calculator):
    """ase calculator that returns the converged energy of a wrapper."""

    implemented_properties = ["energy"]

    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        output = self.wrapper(_batch(self.atoms), training=False)
        self.results["energy"] = output["energy"].item()


def _batch(atoms):
    dataset = dataset_from_atoms([atoms], cutoff=3.0, atomic_multipoles_max_l=1)
    return next(wrap_loader(dataset, batch_size=1))


@cache
def _model(model_kind):
    # The wrappers use lr=0, so tests can share one model per kind.
    torch.set_default_dtype(torch.float64)
    return make_fixed_point_model_random(
        seed=0, nonlinearity_cls=MODELS[model_kind], update_scale=UPDATE_SCALE
    )


@cache
def _finite_difference_stress(model_kind, cell_name, energy_mode, constant_charge):
    """unroll_scf and linearize_solve have the same energy, so they share one result."""
    atoms = periodic_water_dimer(CELLS[cell_name])
    atoms.calc = _WrapperEnergy(
        make_fixed_point_wrapper(_model(model_kind), energy_mode, constant_charge)
    )
    return calculate_numerical_stress(
        atoms, eps=FINITE_DIFFERENCE_STEP, voigt=False, force_consistent=False
    )


def _check_stress(model_kind, cell_name, mode, constant_charge):
    wrapper = make_fixed_point_wrapper(
        _model(model_kind), mode, constant_charge, stress=True
    )
    output = wrapper(_batch(periodic_water_dimer(CELLS[cell_name])), training=True)
    autograd_stress = output["stress"].detach().numpy()[0]
    energy_mode = "direct" if mode == "direct" else "unroll_scf"
    finite_difference_stress = _finite_difference_stress(
        model_kind, cell_name, energy_mode, constant_charge
    )
    print(
        f"\nmax_abs_stress_error="
        f"{np.abs(autograd_stress - finite_difference_stress).max():.2e} "
        f"max_abs_stress={np.abs(finite_difference_stress).max():.2e}"
    )

    assert np.abs(np.diag(finite_difference_stress)).max() > MINIMUM_STRESS_SIGNAL
    if cell_name == "triclinic":
        off_diagonal = finite_difference_stress - np.diag(np.diag(finite_difference_stress))
        assert np.abs(off_diagonal).max() > MINIMUM_STRESS_SIGNAL
    np.testing.assert_allclose(
        autograd_stress,
        finite_difference_stress,
        rtol=0.0,
        atol=STRESS_ABSOLUTE_TOLERANCE,
        err_msg="autograd stress deviates from finite-difference stress",
    )


@pytest.mark.parametrize("cell_name", CELLS)
@pytest.mark.parametrize("model_kind", MODELS)
def test_direct_stress_matches_finite_difference(model_kind, cell_name):
    _check_stress(model_kind, cell_name, "direct", constant_charge=False)


@pytest.mark.parametrize("constant_charge", [False, True], ids=["cf", "cc"])
@pytest.mark.parametrize("cell_name", CELLS)
@pytest.mark.parametrize("model_kind", MODELS)
@pytest.mark.parametrize("mode", ["unroll_scf", "linearize_solve"])
def test_scf_stress_matches_finite_difference(mode, model_kind, cell_name, constant_charge):
    _check_stress(model_kind, cell_name, mode, constant_charge)


@pytest.mark.parametrize("mode", ["direct", "unroll_scf", "linearize_solve"])
def test_forces_unchanged_by_stress(mode):
    model = _model("nonlinear")
    batch = _batch(periodic_water_dimer(TRICLINIC_CELL))
    forces = {}
    for stress in (False, True):
        wrapper = make_fixed_point_wrapper(model, mode, constant_charge=True, stress=stress)
        forces[stress] = wrapper(copy_batch(batch), training=True)["forces"].detach()
    torch.testing.assert_close(
        forces[True], forces[False], rtol=0.0, atol=FORCE_ABSOLUTE_TOLERANCE
    )


def test_implicit_mode_with_stress_raises():
    with pytest.raises(NotImplementedError):
        make_fixed_point_wrapper(_model("linear"), "implicit", True, stress=True)


PERIODIC_IMAGE_TOLERANCE = 1e-10  # eV/A^3 and eV; limited by the SCF tolerance
# Integer lattice translations per atom, as in test_lsc_stress_finite_difference.py:
# both molecules are split across periodic images.
PERIODIC_IMAGE_SHIFTS = np.array(
    [[1, 0, 0], [1, 0, 0], [0, -1, 0], [-1, 1, 0], [0, 0, 1], [200, 0, -1]]
)


def _energy_and_stress(wrapper, atoms):
    output = wrapper(_batch(atoms), training=True)
    return output["energy"].item(), output["stress"].detach().numpy()[0]


def _assert_same_energy_and_stress(reference, other, message):
    np.testing.assert_allclose(other[0], reference[0], rtol=0.0, atol=PERIODIC_IMAGE_TOLERANCE,
                               err_msg=f"energy {message}")
    np.testing.assert_allclose(other[1], reference[1], rtol=0.0, atol=PERIODIC_IMAGE_TOLERANCE,
                               err_msg=f"stress {message}")


@pytest.mark.parametrize("mode", ["direct", "linearize_solve"])
def test_stress_invariant_under_periodic_image_shifts(mode):
    """Atoms moved into other periodic images are the same system. The stress
    at the reference positions is checked against finite differences above."""
    wrapper = make_fixed_point_wrapper(_model("nonlinear"), mode, True, stress=True)
    shifted = periodic_water_dimer(TRICLINIC_CELL)
    shifted.positions = shifted.positions + PERIODIC_IMAGE_SHIFTS @ TRICLINIC_CELL
    _assert_same_energy_and_stress(
        _energy_and_stress(wrapper, periodic_water_dimer(TRICLINIC_CELL)),
        _energy_and_stress(wrapper, shifted),
        "changed when atoms were moved to other periodic images",
    )
