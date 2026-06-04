# Training Fixed-Point SCF Models

The main training entry point is `scripts/run_train.py`. Datasets, losses, and
training stages are specified in `config.yaml`; see
[data and config files](../concepts/data_and_config_files.md).

Fixed-point training has two layers of configuration:

- architecture hyperparameters, which define the density representation, field
  features, fixed-point update rule, and energy readouts;
- training hyperparameters, which define how the model is trained in each
  schedule stage.

Before setting up a training schedule, read
[using fixed-point SCF models](using.md). This page does not repeat those SCF
definitions.

## Architecture Hyperparameters

Architecture hyperparameters are command-line arguments to
`scripts/run_train.py`. They define the model and should usually be fixed
across all stages of a training schedule.

### Density Representation

The fixed-point model represents charge density with atom-centred Gaussian
multipoles. The main density and electrostatics arguments are:

- `--atomic_multipoles_max_l`
- `--atomic_multipoles_smearing_width`
- `--kspace_cutoff_factor`
- `--electrostatic_pbc_method`
- `--include_electrostatic_self_interaction`

See [atomic multipoles](../concepts/atomic_multipoles.md) and
[boundary conditions](../concepts/boundary_conditions.md).

### Field Features

The fixed-point update uses atom-centred electrostatic field features to describe $\mathbf{v}_{\text{eff}}(\mathbf{r})$ rather
than the raw potential. The main field-feature arguments are:

- `--field_feature_max_l`
  maximum angular order;
- `--field_feature_widths`
  Gaussian widths used to sample the potential, for example `"[1.5, 3.0]"`;
- `--field_feature_norms`
  `"None"`, `"average"`, or an explicit list with length
  `len(field_feature_widths) * (field_feature_max_l + 1)`;
- `--include_field_si`
  include self-interaction-like terms in the field-feature path;

Example:

```bash
--field_feature_max_l=1 \
--field_feature_widths="[1.5, 3.0]" \
--field_feature_norms="[25.0,25.0,1.0,1.0]" \
```

For `field_feature_max_l=1` and two widths, the explicit normalization list has
four entries: two for `l=0` and two for `l=1`.

With:

```bash
--field_feature_norms=average
```

the training setup estimates one RMS field-feature scale for each
`(angular order, width)` pair from the training set. For example, with
`field_feature_max_l=1` and two widths, it returns four normalizers in the same
order expected by the explicit list.

Average norm estimation uses the same electrostatic feature machinery as
`FixedPointCore`. If your data doesn't have any `atomic_multipoles`, this functionality will not work.

### Fixed-Point Update Rule

The update rule maps local geometry features and electrostatic field features
to a field-dependent density correction. It is configured with
`--fixedpoint_update_config`, which can be set in the `config.yaml`.

Default:

```yaml
fixedpoint_update_config:
  type: OneBodyVariableUpdate
  potential_embedding_cls: BiasedLinearPotentialEmbedding
  nonlinearity_cls: NoNonLinearity
```

You should choose `type` from:

- OneBodyVariableUpdate
- ManyBodyUpdate

The manybody update is greatly more expensive, since it needs to be iterated as the SCF converges. We generally haven't seem substantial improvements with the many body update, and it is mostly useful for experimentation.

The `nonlinearity_cls` can currently take valuees:

- "NoNonLinearity"
- "MLPNonLinearity"

One-body linear updates are the simplest option. Nonlinear and many-body-style
updates add flexibility at higher complexity.

The optional `--use_linear_local_charges` flag switches the field-independent
local density readout to a scaled linear local-source block.
`--atom_density_scaling` provides the species-dependent scaling for that path.

`--atom_density_scaling` accepts:

- `"None"`
  use a scale of `1.0` for every element;
- an explicit dictionary, for example `"{1: 0.5, 8: 2.0}"`;
- `"average"`
  compute one species scale from the training-set reference atomic multipoles.

The `"average"` mode computes the RMS over all selected multipole components
up to `atomic_multipoles_max_l`, separately for each element, and returns the
scales in the model's atomic-number table order. Configs without reference
`atomic_multipoles` are excluded from the average using the stored property
weights, so missing multipoles do not lower the scale by entering as zeros.

### Nonlocal Energy Term

The fixed-point density is not obtained by minimizing a variational energy
functional. The model therefore has a separate density-dependent energy
readout, called `local_electron_energy` in `FixedPointCore`.

This term is configured with `--field_readout_config`.

Default:

```python
{"type": "StrictQuadraticFieldEnergyReadout"}
```

Example:

```bash
--field_readout_config "{'type': 'StrictQuadraticFieldEnergyReadout'}"
```

Registered readout names include:

- `NullFieldReadout`
- `StrictQuadraticFieldEnergyReadout`
- `OneBodyMLPFieldReadout`
- `ManyBodyChargesReadout`
- `ManyBodyChargesFieldReadout`

## Training Method

The training methods and fixed-point formulation are described in the
fixed-point [paper](https://arxiv.org/abs/2603.14700).

Current recommended starting schedule:

1. train with direct training, `mode: direct`, for about 100
   epochs;
2. continue with `mode: unroll_scf`, using about 10 SCF steps and
   `mixing_parameter: 0.3`.

The first stage trains the response map at the reference density. The second
stage trains the model closer to inference, where it must run its own SCF loop.

### Stage Options

Each `train_schedule` stage can include a `fixed_point_training_options` block:

```yaml
fixed_point_training_options:
  mode: unroll_scf
  scf:
    num_scf_steps: 10
    constant_charge: true
    mixing_parameter: 0.3
    initial_density: from_data
    initial_fermi_level: from_data
    use_autograd_forces: true
```

The training keys are:

- `mode`. This can take values
  - `direct`
  - `unroll_scf`
  - `implicit`
  - `linearize_solve`
- `scf`
  nested SCF settings used by `unroll_scf` and `implicit`.

The nested `scf` keys are:

- `num_scf_steps`
- `scf_tolerance`
- `constant_charge`
- `mixing_parameter`
- `initial_density`
- `initial_fermi_level`
- `use_autograd_forces`

`mode: direct` does not run an SCF loop and should not include an `scf` block.
For `mode: unroll_scf` and `mode: implicit`, the `scf` block must explicitly
set `num_scf_steps` and `mixing_parameter`.

See [using fixed-point SCF models](using.md) for the meaning of these SCF
settings. Training-specific notes are given below.

Other training arguments, such as `--field_block_weight_decay` and
`--local_charges_weight_decay`, control optimizer parameter groups.

### Example Schedule

```yaml
train_schedule:
  0:
    name: direct
    start: 0
    end: 99
    loss:
      atomic_multipoles: 100.0
      total_charge_per_atom: 1000.0
      dipole_per_atom: 1000000.0
      energy_per_atom: 10.0
      forces: 100.0
    lr: 0.01
    fixed_point_training_options:
      mode: direct
  1:
    name: unroll
    start: 100
    end: 110
    loss:
      atomic_multipoles: 100.0
      total_charge_per_atom: 1000.0
      dipole_per_atom: 1000000.0
      energy_per_atom: 1000.0
      forces: 100.0
    lr: 0.001
    fixed_point_training_options:
      mode: unroll_scf
      scf:
        num_scf_steps: 10
        constant_charge: true
        mixing_parameter: 0.3
        initial_density: from_data
        initial_fermi_level: from_data
        use_autograd_forces: true
```

Use this as a starting point. Adjust losses, learning rates, and SCF settings
for the dataset and model stability.

## Training Modes

### Direct Training

```yaml
mode: direct
```

This is direct training in the terminology of the paper. The model is not
iterated to self-consistency. The wrapper uses reference `atomic_multipoles`
and `fermi_level` from the data, applies one update, and compares the result to
the reference density and other targets.

This mode is computationally inexpensive and stable, making it useful at the
start of training. Its main limitation is the mismatch with inference: a model
can fit the direct objective but still be unstable when run as an SCF model.

### Unrolled SCF Training

Repo name:

```yaml
mode: unroll_scf
```

This mode runs the SCF loop during training and differentiates through the
unrolled iterations. The loss is applied to the model's own SCF result.

The cost grows with `num_scf_steps`, and long unrolls can be memory-intensive.
The current practical continuation stage is about 10 steps with
`mixing_parameter: 0.3`.

`initial_density: from_data` and `initial_fermi_level: from_data` are useful
when moving from direct training to unrolled SCF. `local_guess` and `zero` are
more inference-like tests.

### Implicit Differentiation

Repo name:

```yaml
mode: implicit
```

Implicit differentiation treats the converged SCF density as the solution of a
fixed-point equation, rather than as the output of a fixed number of unrolled
iterations. Conceptually, gradients are computed while enforcing the condition
that the fixed-point equation remains solved. Please see [implicit_diff](./implicit_diff.md) for details.

This method requires the `torchopt` package, which can be difficult to install and get working. Please ensure you run the tests (`tests/test_fixed_point_wrapper_training_modes.py`) after installing torchopt to check you are getting the right numbers.

### Linearize and Solve

Repo name:

```yaml
mode: linearize_solve
```

The `linearize_solve` method is equilivant to implicit differentiation, but we find it to be generally better behaved. This method works by iterating to the solution during in training, then, at the solution, the update rule is linearized and the output predicted by solving the resulting linear problem. This is equivalent to linearizing at the solition to create a quadratic QEq or linear polarizable force type model, which can then be solved by matrix inversion. Please see [implicit_diff](./implicit_diff.md) for details.

This method **does not require** `torchopt` or any other dependencies.

## Losses

Common fixed-point losses include:

- `atomic_multipoles`
- `total_charge_per_atom`
- `dipole_per_atom`
- `fermi_level_per_atom`
- `energy_per_atom`
- `forces`
- `esps`, if electrostatic potentials are returned and labelled
- `field_features`, for field-feature supervision workflows
- `fixedpoint_scf_stability`, for stability-oriented training experiments

For direct fixed-point training, `atomic_multipoles` is usually central. For
SCF stages, energy, forces, charge, dipole, and density losses are applied to
the model's own SCF result.