# Fixed-Point SCF ASE Calculator

The fixed-point ASE calculator is implemented in
`mace_scf/calculators/fixedpoint_scf.py`.

Use this page as an API reference for constructing the calculator. For the SCF
concepts behind `num_scf_steps`, `mixing_parameter`, Fermi levels, initial
states, and restarts, read [using fixed-point SCF models](using.md) first.

## Class

Use:

```python
from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
```

`MACEPolarizable` is still available as a compatibility alias, but new code
should use `MACEFixedPointSCF`.

## Minimal Example

For a neutral constant-charge calculation with no applied external field, the
calculator can be constructed with only the model path, device, boundary
condition handling, and SCF options:

```python
from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF

calc = MACEFixedPointSCF(
    model_path="fit.model",
    device="cuda",
    pbc_handling="slab",
    scf_options={
        "num_scf_steps": 100,
        "scf_tolerance": 1.0e-6,
        "mixing_parameter": 0.2,
        "constant_charge": True,
        "use_autograd_forces": True,
    },
)

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

If `atoms.info["total_charge"]` is absent, the calculator uses `0.0`. If
`atoms.info["external_field"]` is absent, the calculator uses a zero external
field. If `atoms.info["initial_fermi_level"]` is absent, the calculator uses a
zero Fermi-level input.

## Constructor Arguments

Required arguments:

- `model_path`
  path to the saved fixed-point model;
- `device`
  device passed through the usual MACE/PyTorch device handling, such as
  `"cpu"` or `"cuda"`.

Common optional arguments:

- `scf_options`
  dictionary containing SCF runtime options;
- `pbc_handling`
  boundary-condition mode used by the model electrostatics, for example
  `"mixed_periodic"`, `"slab"`, `"pbc"`, `"realspace"`, or
  `"molecule_in_box"`;
- `total_charge_key`
  name of the `atoms.info` field containing the requested total charge;
- `external_field_key`
  name of the `atoms.info` field containing the applied homogeneous field;
- `fermi_level_key`
  name of the `atoms.info` field containing the Fermi-level input;
- `ignore_nonconverged`
  suppresses the non-convergence warning when the final density change is
  above `scf_tolerance`. Divergence and large constant-charge errors can still
  raise errors;
- `save_full_scf_history`
  stores the full density history in `calc.results["full_scf_history"]`.

Less common unit and dtype arguments:

- `energy_units_to_eV`
- `length_units_to_A`
- `default_dtype`

Leave these at their defaults unless the saved model uses units or precision
that differ from the normal ASE-facing workflow.

## SCF Options

The calculator accepts these `scf_options` keys:

```python
scf_options={
    "num_scf_steps": 100,
    "scf_tolerance": 1.0e-5,
    "mixing_parameter": 0.2,
    "constant_charge": True,
    "use_autograd_forces": True,
}
```

These are runtime options only. Architecture hyperparameters such as field
feature widths, update blocks, density smearing widths, and nonlocal energy
readout settings are already stored in the saved model and are not passed to
the calculator.

You do not need to include `initial_density` or `initial_fermi_level` in calculator
`scf_options`. The calculator uses the model's local density guess for the initial multipoles. For the Fermi-level, the calculator reads the Fermi-level from
the configured `fermi_level_key`. In constant charge simualtions this is the initial Fermi-level, and in constant chemical potential mode this is the fixed Fermi-level.

## Input Keys

The calculator maps fields from `atoms.info` and `atoms.arrays` into the model
input batch. The default key names are:

```python
total_charge_key = "total_charge"
external_field_key = "external_field"
fermi_level_key = "initial_fermi_level"
atomic_multipoles_key = "initial_density_coefficients"
```

You only need to pass these key arguments when your `Atoms` objects use
different names.

For example, if your input file stores the applied field as
`atoms.info["homogeneous_field"]`, pass:

```python
calc = MACEFixedPointSCF(
    model_path="fit.model",
    device="cuda",
    external_field_key="homogeneous_field",
)
```

The meaning of the fields is:

- `total_charge_key`
  used in constant-charge mode. If the field is absent, the target charge
  defaults to `0.0`;
- `external_field_key`
  applied homogeneous field. If absent, the field defaults to zero;
- `fermi_level_key`
  Fermi-level input. In constant-Fermi mode this is the controlled Fermi level.
  In constant-charge mode it is the initial Fermi-level guess. If absent, it
  defaults to zero;
- `atomic_multipoles_key`
  maps an optional atom array into `density_coefficients`. The standard
  calculator does not require this for normal energy and force calculations
  because it starts from the model's local density guess. In the current
  calculator API, this is not the usual way to set the SCF initial density.

## Arguments You Usually Do Not Need

For ordinary ASE use, do not pass:

- `atomic_multipoles_key`
  unless your workflow deliberately stores multipoles in the input file;
- `external_field_key`
  if there is no applied field or the default key name is already correct;
- `fermi_level_key`
  if a zero Fermi-level input is acceptable or the default key name is already
  correct;
- `total_charge_key`
  if the default key name is already correct. For neutral calculations, you
  usually do not need to put `total_charge` in `atoms.info` at all;

## Constant Charge And Constant Fermi Level

For constant-charge calculations:

```python
atoms.info["total_charge"] = 0.0

calc = MACEFixedPointSCF(
    model_path="fit.model",
    device="cuda",
    scf_options={
        "constant_charge": True,
        "num_scf_steps": 100,
        "scf_tolerance": 1.0e-6,
        "mixing_parameter": 0.2,
        "use_autograd_forces": True,
    },
)
```

For constant-Fermi calculations:

```python
atoms.info["initial_fermi_level"] = -4.8

calc = MACEFixedPointSCF(
    model_path="fit.model",
    fermi_level_key="initial_fermi_level",
    device="cuda",
    scf_options={
        "constant_charge": False,
        "num_scf_steps": 100,
        "scf_tolerance": 1.0e-6,
        "mixing_parameter": 0.2,
        "use_autograd_forces": True,
    },
)
```

In constant-charge mode, `fermi_level_key` supplies an initial guess. In
constant-Fermi mode, it supplies the fixed Fermi level.

## Results

After a calculation, `calc.results` includes:

- `energy`
- `free_energy`
- `forces`
- `density_coefficients`
- `dipole`
- `fermi_level`
- `external_field`
- `electrostatic_energy`
- `electron_energy`
- `convergence_history`
- `num_scf_steps`
- `electrostatic_features`
- `electrostatic_potentials`, if the model returns them
- `full_scf_history`, if `save_full_scf_history=True`

The most useful diagnostic is usually:

```python
calc.results["convergence_history"]
```

It records the mean absolute density change between SCF steps. Inspect this
when a calculation is slow, oscillatory, or non-converged.

## Developer Restart Options

The calculator has restart hooks:

- `scf_restart`
- `restart_density`
- `restart_fermi_level`

These reuse state from the previous call to the same calculator object. They
are currently developer-facing options and are not recommended as the default
user workflow. Validate a model without restarts before relying on restarted
calculations.
