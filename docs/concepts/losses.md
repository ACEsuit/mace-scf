# Losses

MACE_SCF training stages use loss dictionaries inside `train_schedule`. These
loss dictionaries are more explicit than the default MACE examples because the
models may train on charge, multipoles, dipoles, fields, electrostatic
potentials, and SCF diagnostics in addition to energies and forces.

## Loss Dictionaries

A training stage can define a `loss` block:

```yaml
train_schedule:
  0:
    name: stage1
    loss:
      atomic_multipoles: 100.0
      dipole_per_atom: 1.0
      energy_per_atom: 1.0
      forces: 100.0
```

Each key names a loss term. Each value is the weight applied to that term.

## Available Loss Names

Basic loss terms:
- `energy_per_atom`
- `forces`
- `stress` (currently only supported for local-source models)

new losses useful for electrostatic models:
- `atomic_multipoles`
- `total_charge`
- `total_charge_per_atom`
- `dipole`
- `dipole_per_atom`
- `polarizability`
- `fermi_level`
- `fermi_level_per_atom`
- `esps` electrostatic potential on atoms. Note that the reference (truth) value of the ESP is computed from the DFT derived atomic charges or multipoles. ESPs computed directly from DFT are not yet supported, but will be in the future.

There are also more losses for experimenting with things like net intermolecular forces:
- `cluster_virial`
- `cluster_virial_per_atom`
- `molecular_forces`


Not every loss is valid or useful for every model. For example, fermi level training only makes sense for fixed-point SCF training, `polarizability` requires a
model and dataset with polarizability outputs, and `esps` or `field_features`
require the relevant electrostatic-potential or field-feature data paths.
The model-specific training pages describe the common choices for each model family.

## Inspecting Loss Balance

During training, you can inspect the loss breakdown. If a metric is not
improving, the breakdown can show whether a loss term has too little or too
much weight relative to the other targets.

The per-batch breakdown is written to the debug log. It looks like:

```text
DEBUG: loss breakdown: forces: 7.496695865065772e-06, dipole_per_atom: 9.02030930287152e-06, esps: 0.00014263704012035686, energy_per_atom: 0.0001251799228796405, total_charge_per_atom: 5.479098701744614e-08, 
```

Use this line to check whether the weighted terms are on comparable scales, or
whether one term dominates the optimization.

There also a script which can be used to plot this during a training run:
```bash
python scripts/plot_batch_losses.py logs/<fit_name>_debug.log
```
This script accepts the optional arguments `--min_epoch` and `--max_epoch`.