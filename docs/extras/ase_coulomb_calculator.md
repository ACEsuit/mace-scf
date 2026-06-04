# `GTOCoulombCalculator`: standalone Coulomb energy as an ASE calculator

`mace_scf.calculators.coulomb.GTOCoulombCalculator` is a thin ASE
`Calculator` wrapping `graph_longrange.energy.GTOElectrostaticEnergy`. It
evaluates the Coulomb energy (and forces / stress / dipole) of a
user-supplied set of atom-centred smeared GTO multipoles, with no ML model
involved.

Given atom-centred multipole coefficients $p_{i,lm}$ (see
[atomic multipoles](../concepts/atomic_multipoles.md)) the calculator
returns

$$
E = E_{\mathrm{Coulomb}}[\{p_{i,lm}\}] + \mathbf{d}_{\mathrm{tot}}\cdot\mathbf{E}_{\mathrm{ext}},
$$

Note that the sign of the external field is such that the field is the gradient of the electrostatic potential, not negative the gradient. This is consistent with several DFT codes. 

## Units

Inputs and outputs are in **e / eV / Å**.

## Construction

```python
from mace_scf.calculators.coulomb import GTOCoulombCalculator

calc = GTOCoulombCalculator(
    max_l=0,
    smearing_width=0.5,
    kspace_cutoff_factor=1.5,      # multiplies graph_longrange's heuristic
    pbc_handling="mixed_periodic", # "realspace" / "pbc" / "slab" /
                                   # "molecule_in_box" / "mixed_periodic" /
                                   # "auto"
    include_self_interaction=False,
    multipoles_key="multipoles",
    external_field_key="external_field",
    device="cpu",
    default_dtype="float64",
)
```

The reciprocal-space cutoff used internally is

```python
kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
    [smearing_width], max_l
)
```

so `kspace_cutoff_factor` is a multiplier on the `graph_longrange` heuristic.

`pbc_handling` is passed straight through to `GTOElectrostaticEnergy` — see
[boundary conditions](../concepts/boundary_conditions.md) for the meaning of
each mode.

## Supplying multipoles and an external field

ASE's `Atoms.get_potential_energy()` does not forward keyword arguments to
`Calculator.calculate`, so per-call inputs go through one of two channels:

1. **Setter methods on the calculator** (the typical interface):

   ```python
   calc.set_multipoles(np.array([[+1.0], [-1.0]]))  # shape (n_atoms, (max_l+1)**2)
   calc.set_external_field(np.array([0.0, 0.0, 0.01]))
   ```

   `set_multipoles` always takes a fixed numpy array. If you want to change
   the multipoles each MD step, call `set_multipoles` from your integration
   loop. It does **not** accept a callable.

2. **Attached to the `Atoms` object**, useful when the data already lives on
   a trajectory:

   ```python
   atoms.arrays["multipoles"] = np.array([[+1.0], [-1.0]])
   atoms.info["external_field"] = np.array([0.0, 0.0, 0.01])
   ```

   The keys (`multipoles`, `external_field`) are configurable via
   `multipoles_key` and `external_field_key`.

Explicit setters take precedence over `atoms` attributes. If no multipoles
are supplied via either channel, `calculate` raises.

Multipoles follow the CS phase ordering used throughout mace_scf (see
[atomic multipoles](../concepts/atomic_multipoles.md)). For `max_l=1` the
four columns are `[q, p_y, p_z, p_x]`. The `dipole` result returned by the
calculator is the *total* dipole including the position-weighted monopole
contribution, computed the same way as in the trained models.

## Outputs

```
energy, free_energy   total Coulomb + field energy (eV)
forces                (n_atoms, 3) (eV / A)
stress                Voigt-6 stress (eV / A^3), only when requested
dipole                (3,) total dipole (e * A)
partial_charges       (n_atoms,)
partial_dipoles       (n_atoms, 3)
external_field        (3,)
```

## Compatability with other Ewald Sums

You can find an example comparing this calculator to `Ewald` class in the `matscipy` ([`matscipy.calculators.ewald.Ewald`](https://libatoms.github.io/matscipy/generated/matscipy.calculators.ewald.calculator.html)) in `./tests/test_coulomb_calculator.py`. 

Note that to get argeement you need to use a small smearing width, since matscipy treats point charges instead of Gaussian smeared charges. You also need `include_self_interaction=False` and a sufficiently large kspace cutoff such as `kspace_cutoff_factor=2.0`.