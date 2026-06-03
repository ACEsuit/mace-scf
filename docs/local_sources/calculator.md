# Local-Source ASE Calculators

The local-source ASE calculators are implemented in
`mace_scf/calculators/localsources.py`.

These calculators wrap one-shot local-source models. They predict source
coefficients and evaluate electrostatics once; there is no SCF loop.

## Classes

- `MACELocalCharges`
- `MACELocalSplitCharges`
- `MACEFixedChargeBaseline`

## Minimal Example

```python
from ase.io import read
from mace_scf.calculators.localsources import MACELocalCharges

atoms = read("input.xyz")

calc = MACELocalCharges(
    model_path="nonpol.model",
    device="cuda",
    pbc_handling="slab",
)

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Model-Specific Examples

`MACELocalSplitCharges` can read per-atom formal charges when the model was
trained to use them:

```python
from mace_scf.calculators.localsources import MACELocalSplitCharges

calc = MACELocalSplitCharges(
    model_path="local_symmetric_charges.model",
    device="cuda",
    formal_charges_key="formal_oxidation_states", # not needed if you trained with per-species formal charges
    external_field_key="homogeneous_field",
    pbc_handling="slab",
)
```

`MACEFixedChargeBaseline` uses the fixed-charge information stored in the
trained model:

```python
from mace_scf.calculators.localsources import MACEFixedChargeBaseline

calc = MACEFixedChargeBaseline(
    model_path="fixed_charge_baseline.model",
    device="cuda",
    pbc_handling="pbc",
)
```

## Boundary Handling

`pbc_handling` selects the electrostatic boundary treatment used by the
calculator:

- `realspace`
- `pbc`
- `slab`
- `molecule_in_box`
- `mixed_periodic`

See [boundary conditions](../concepts/boundary_conditions.md) for the meaning
of these modes, including the distinction between `realspace` and
`molecule_in_box`.

The calculator setting can differ from the value used during training. This is
sometimes useful for testing, but it changes the electrostatic semantics of the
calculation. For production use, choose a calculator `pbc_handling` value that
matches the boundary treatment intended during training.

Common choices:

```python
# Bulk periodic system
calc = MACELocalCharges(
    model_path="nonpol.model",
    device="cuda",
    pbc_handling="pbc",
)

# Slab system
calc = MACELocalSplitCharges(
    model_path="local_symmetric_charges.model",
    device="cuda",
    pbc_handling="slab",
)

# Non-periodic molecule or cluster, open-boundary electrostatics
calc = MACELocalCharges(
    model_path="nonpol.model",
    device="cuda",
    pbc_handling="realspace",
)

# Molecule or cluster treated in a periodic box with correction
calc = MACELocalCharges(
    model_path="nonpol.model",
    device="cuda",
    pbc_handling="molecule_in_box",
)

# Mixed dataset containing multiple boundary-condition classes
calc = MACELocalSplitCharges(
    model_path="local_symmetric_charges.model",
    device="cuda",
    pbc_handling="mixed_periodic",
)
```

## Results

After calculation, `calc.results` includes:

- `energy`
- `free_energy`
- `forces`
- `stress`
- `partial_charges`
- `partial_dipoles`
- `density_coefficients`
- `dipole`
- `external_field`
- `fermi_level`
- optional `polarizability`

`density_coefficients` follow the atomic multipole ordering described in
[atomic multipoles](../concepts/atomic_multipoles.md). `partial_dipoles` are
Cartesian. `external_field` is the value loaded into the model input batch.

## Torch Compilation

