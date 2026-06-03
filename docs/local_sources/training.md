# Training Local-Source Models

The three local-source models are trained through the same entry point,
`scripts/run_train.py`, and use the same basic data/config structure. The main
differences are the model selected with `--model`, whether formal charges are
needed, and which loss terms are physically meaningful.

This page covers local-source-specific training. The general config format is
described in [data and config files](../concepts/data_and_config_files.md), and
the density-coefficient conventions are described in
[atomic multipoles](../concepts/atomic_multipoles.md).

## Minimal LocalCharges Example

A minimal `LocalCharges` config can train on energies, forces, dipoles, and
atomic multipoles:

```yaml
heads:
  default:
    info_keys:
      energy: REF_energy
      dipole: REF_dipole
      external_field: external_field
      total_charge: total_charge
    arrays_keys:
      forces: REF_forces
      atomic_multipoles: REF_atomic_multipoles
train_schedule:
  0:
    name: stage1
    start: 0
    end: 99
    loss:
      atomic_multipoles: 100.0
      dipole: 1.0
      energy_per_atom: 1.0
      forces: 100.0
    lr: 0.01
  1:
    name: stage2
    start: 100
    end: 199
    loss:
      dipole: 10.0
      energy_per_atom: 1000.0
      forces: 100.0
    lr: 0.001
```
Note that you can mix and match loss functions between training stages. A matching training command is:

```bash
python scripts/run_train.py \
    --name nonpol_example \
    --config config.yaml \
    --train_file train.xyz \
    --valid_fraction 0.2 \
    --test_file train.xyz \
    --E0s average \
    --model LocalCharges \
    --hidden_irreps '64x0e + 64x1o' \
    --r_max 6.0 \
    --batch_size 8 \
    --valid_batch_size 8 \
    --eval_interval 2 \
    --error_table PerAtomRMSE \
    --device cuda \
    --default_dtype float64 \
    --atomic_multipoles_max_l 1 \
    --atomic_multipoles_smearing_width 1.5 \
    --electrostatic_pbc_method mixed_periodic \
    --restart_latest \
    --save_cpu
```

Replace the `REF_*` keys with the actual extxyz keys in the dataset. At the
start of training, check the logged data-loading summary to make sure all
intended quantities were found.

## Common Parameters

These settings are shared by local-source models:

- `--model`
  selects the architecture. Local-source choices are `LocalCharges`,
  `LocalSplitCharges`, and `FixedChargeBaselinedMACE`.
- `--atomic_multipoles_max_l`
  selects the highest multipole order. See
  [atomic multipoles](../concepts/atomic_multipoles.md).
- `--atomic_multipoles_smearing_width`
  sets the Gaussian smearing width. See
  [atomic multipoles](../concepts/atomic_multipoles.md).
- `--electrostatic_pbc_method`
  selects the electrostatic boundary handling used during training. See
  [boundary conditions](../concepts/boundary_conditions.md).


For homogeneous datasets, use a fixed boundary mode such as `pbc`, `slab`, or
`realspace`. Use `mixed_periodic` only for intentionally mixed datasets.

## Reccomended Losses

As discussed in [losses](../concepts/data_and_config_files.md), you can compose a loss from many different options. As well as `energy_per_atom` and `forces`, with these models you can also use:

- `atomic_multipoles`, when reference density coefficients are available
- `dipole_per_atom` or `dipole`, when reference dipoles are available

These aren't strictly required for training, but you can fit on them. For the LocalCharges model, its reccomended to also train on `total_charge_per_atom` or `total_charge`, since the total charge is not correct by construction. This doesn't apply to the LCS or fixed charge models.

## Settings for LocalCharges

`LocalCharges` does not require any additiaon settings beyond the general settings in *concepts*.

## Settings for LocalSplitCharges

`LocalSplitCharges` requires formal charges. These formal charges define
the conserved total charge, while the model learns local charge transfers on
top.

There are two ways to provide them.

### Fixed Per-Species Formal Charges

Use `--atomic_formal_charges` when each element has one formal charge. For example, for training on liquid water one would use:

```bash
--atomic_formal_charges "{1: 1.0, 8: -2.0}"
```

The dictionary keys are atomic numbers. The model construction code expects a
charge for every element in the training `z_table`.

Use this mode when formal oxidation states are fixed by chemistry and do not
vary between configurations.

### Per-Atom Formal Charges From Data

Use `--formal_charges_from_data` when formal charges vary by atom or by
configuration:

```bash
--formal_charges_from_data
```

The config must then map an array key for charges:

```yaml
heads:
  default:
    arrays_keys:
      charges: formal_charges
```

Use this mode when the dataset already contains per-atom oxidation states or
when the same element can appear in more than one formal charge state.

### Settings for Polarizability Output

`LocalSplitCharges` can add a direct polarizability readout with:

```bash
--compute_polarizability
```

To train this output, the config must map a `polarizability` info key and the
loss should include `polarizability`. The local-source calculator exposes
`polarizability` for `MACELocalSplitCharges` models that were trained with
this readout.

## FixedChargeBaselinedMACE

`FixedChargeBaselinedMACE` uses fixed formal monopoles as the long-range charge
density. It does not learn atomic multipoles or charge transfer. This model currently supports formal charges specified per species. Use `--atomic_formal_charges` as described above.
