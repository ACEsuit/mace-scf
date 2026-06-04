# Visualising charge density and electrostatic potential

`mace_scf.utils.field_interpolator` provides helpers that take a set of
atom-centred GTO multipoles (either supplied directly or read off a trained
MACE_SCF calculator) and evaluate the resulting smeared charge density and
electrostatic potential on a user-specified set of sample points. This is
useful for plotting line cuts and slices of the electrostatic potential a
model has learned.

Two interfaces are provided:

- `PotentialInterpolator`: stand-alone evaluator. You pass the atoms, the
  multipoles, an optional external field, and the sample points.
- `PotentialInterpolatorPostProcessor`: thin wrapper around an ASE
  calculator (e.g. `MACEPolarizable`) that pulls multipoles, external
  field, and Fermi level off the calculator's last `results` dict.

> **PBC restriction.** Both interpolators currently assume periodic
> boundaries in $x$ and $y$ and open boundaries in $z$ (a slab with $z$ as
> the non-periodic axis). The constructor emits a warning to that effect,
> and `__call__` raises if `atoms.pbc != [True, True, False]`.

## Input Requirements

The direct `PotentialInterpolator` interface expects:

- `atoms`: an ASE `Atoms` object with Cartesian positions in Angstrom, a
  cell, and `atoms.pbc` exactly equal to `[True, True, False]`;
- `atomic_multipoles`: a numpy array with shape
  `(n_atoms, (multipoles_max_l + 1) ** 2)`;
- `external_field`: a length-3 numpy array using the same sign and units as
  the MACE_SCF calculator external-field inputs;
- `fermi_level`: a Python `float` scalar offset added to the potential;
- `coords`: Cartesian sample points in Angstrom, with shape `(..., 3)`.

For `multipoles_max_l=0`, `atomic_multipoles` has one column containing
monopole coefficients. For `multipoles_max_l=1`, it has four columns:
one monopole coefficient and three dipole-like density coefficients in the
same ordering used by the model density coefficients. The ordering follows
the convention described in [atomic multipoles](../concepts/atomic_multipoles.md).

The interpolator is intended for visualisation. The `sigma` value is the
Gaussian smearing width used for plotting the density and potential, and it
does not have to match the smearing used during model training. Smaller
values produce sharper features and usually need a larger `kspace_cutoff` or
`kspace_cutoff_factor`; larger values produce smoother line cuts and slices.

If `subtract_total_charge=True`, the net monopole charge is uniformly removed
from the supplied multipoles before the Fourier sum is evaluated. This is
useful for visualising nearly neutral slabs when small residual charge errors
would otherwise dominate the long-range potential.

## Outputs

Each call returns three numpy arrays with the same leading shape as the
input `coords` (minus the trailing length-3 axis):

| return value | meaning |
| --- | --- |
| `density` | smeared GTO charge density $\rho(\mathbf r)$ |
| `potential_uncorrected` | electrostatic potential from the periodic Fourier sum alone (plus `fermi_level` offset) |
| `potential` | the same potential plus the slab dipole correction *and* the contribution of the external field, $V_\mathrm{corr}(\mathbf r) = V(\mathbf r) + (E^\mathrm{corr}_z + E^\mathrm{ext}_z) z + \mathrm{const}$ |

The slab dipole correction is the standard
$E^\mathrm{corr}_z = (1/\varepsilon_0) p_z / V$ term that removes the
artificial macroscopic field a periodic image of a charged or polarised
slab would otherwise impose.

## Using `PotentialInterpolator` directly

Given an `Atoms` object with a slab cell (PBC `[T, T, F]`) and a numpy array
of per-atom multipoles, e.g. from a previous calculation stored in
`atoms.arrays`:

```python
import numpy as np
import torch
from copy import deepcopy

from mace_scf.utils.field_interpolator import PotentialInterpolator

torch.set_default_dtype(torch.float64)

interpolator = PotentialInterpolator(
    sigma=1.0,                  # GTO smearing width (Angstrom)
    multipoles_max_l=1,         # 0 for monopoles only, 1 for monopoles + dipoles
    kspace_cutoff_factor=1.5,   # multiplies graph_longrange's heuristic
                                # (~1.0 is loose, ~2.0 is tight)
    device="cpu",
    subtract_total_charge=True, # uniformly subtract any residual net charge
                                # before evaluating the Fourier sum
)

# Atoms object: must have pbc exactly equal to [True, True, False].
atoms = ...

# Line of Cartesian sample points along z through (x=0, y=0), in Angstrom.
N = 100
z = np.linspace(-40.0, 40.0, N).reshape((N, 1))
coords = np.hstack([np.zeros_like(z), np.zeros_like(z), z])

# Multipoles attached to the Atoms object. Shape (n_atoms, (max_l + 1) ** 2),
# in the density-coefficient ordering documented in concepts/atomic_multipoles.md.
multipoles = deepcopy(atoms.arrays["atoms_in_molecules_atomic_multipoles"])

density, potential_uncorr, potential = interpolator(
    atoms,
    atomic_multipoles=multipoles,
    external_field=atoms.info["homogeneous_field"],  # shape (3,)
    fermi_level=0.0,                                  # Python float scalar offset
    coords=coords,
)
```

## Using `PotentialInterpolatorPostProcessor` with a model

If you have a trained MACE_SCF calculator, the post-processor wraps it so you
don't have to extract the multipoles, external field, and Fermi level
yourself:

```python
import ase.io
import numpy as np

from mace_scf.calculators.fixedpoint_scf import MACEPolarizable
from mace_scf.utils.field_interpolator import PotentialInterpolatorPostProcessor

config = ase.io.read("path/to/configs.xyz", index="-1")

calc = MACEPolarizable(
    model_path="path/to/fit.model",
    device="cpu",
    external_field_key="external_field",
    fermi_level_key="e_fermi",
    ignore_nonconverged=False,  # set True if you ran in fixed-step SCF mode
)
config.calc = calc

# Choose the external field for this evaluation.
config.info["external_field"] = np.array([0.0, 0.0, 0.05])

# Cartesian sample points: a line along z through (0, 0), in Angstrom.
N = 100
z = np.linspace(-40.0, 40.0, N).reshape((N, 1))
coords = np.hstack([np.zeros_like(z), np.zeros_like(z), z])

interpolator = PotentialInterpolatorPostProcessor(calc, sigma=2.0, device="cpu")

# Trigger an SCF / forward pass so calc.results is populated.
config.get_potential_energy()

density, potential_uncorr, potential = interpolator(config, coords)
```

The post-processor pulls `density_coefficients`, `external_field`, and
`fermi_level` out of `calc.results` and forwards them to a
`PotentialInterpolator` configured with the same `multipoles_max_l` and
`kspace_cutoff` as the model's own `coulomb_energy` module. The only
hyperparameter you set is the smearing width `sigma` used for the
visualisation density (which can differ from the model's smearing — a
larger `sigma` gives smoother line cuts).

## Jellium variant

`PotentialInterpolatorJellium` is the analogue for models that include a
jellium background. It takes an additional `slab_bounds=(z_lower, z_upper)`
argument that locates the jellium region, and internally adds the smoothed
jellium counter-charge density to the Fourier series before evaluating the
potential. The call signature is otherwise identical to
`PotentialInterpolator`.
