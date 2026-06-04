# Evaluating Fixed-Point SCF Models

Use `scripts/eval_fixed_point.py` to label extxyz files with predictions from a
fixed-point model. Read [Using fixed-point SCF models](using.md) first if the
SCF settings or initial-state options are unfamiliar.

The script has two modes:

- `scf`
  run a self-consistent solve and write the converged outputs;
- `direct`
  apply one direct fixed-point update using reference density coefficients and
  Fermi levels from the input file.

## SCF Evaluation

Typical SCF evaluation:

```bash
python scripts/eval_fixed_point.py \
    --mode scf \
    --configs input.xyz \
    --model fit_stage1.model \
    --output evaluated.xyz \
    --device cuda \
    --batch_size 8 \
    --pbc_handling slab \
    --fermi_level_key VBM_plus_avg_v \
    --external_field_key homogeneous_field \
    --atomic_multipoles_key AIMS_atom_multipoles \
    --total_charge_key total_charge \
    --scf_options "{'num_scf_steps': 100, 'mixing_parameter': 0.2, 'scf_tolerance': 1e-6, 'constant_charge': True, 'use_autograd_forces': True}"
```

`--mode scf` is the default. It starts from a local density guess unless
`--initial_density from_data` is given. It starts from a zero Fermi-level
contribution unless `--initial_fermi_level from_data` is given.

Use data-provided initial states only when the corresponding keys are present
for every configuration:

```bash
python scripts/eval_fixed_point.py \
    --mode scf \
    --configs input.xyz \
    --model fit_stage1.model \
    --output evaluated.xyz \
    --atomic_multipoles_key AIMS_atom_multipoles \
    --fermi_level_key VBM_plus_avg_v \
    --initial_density from_data \
    --initial_fermi_level from_data
```

These options choose the initial SCF state. They do not choose constant-charge
or constant-Fermi evaluation; that is controlled by `constant_charge` inside
`--scf_options`.

The older flags `--scf_restart_multipoles` and `--scf_restart_fermi_level` are
deprecated aliases for `--initial_density from_data` and
`--initial_fermi_level from_data`.

## Direct Fixed-Point Evaluation

Direct fixed-point evaluation applies one update from reference multipoles and
a reference Fermi level. This is useful for inspecting the direct fixed-point
training target, not for production inference.

```bash
python scripts/eval_fixed_point.py \
    --mode direct \
    --configs input.xyz \
    --model fit_stage1.model \
    --output fixed_point_eval.xyz \
    --device cuda \
    --pbc_handling mixed_periodic \
    --atomic_multipoles_key AIMS_atom_multipoles \
    --fermi_level_key VBM_plus_avg_v \
    --external_field_key homogeneous_field \
    --total_charge_key total_charge
```

In `direct` mode, the input file must contain the multipoles named by
`--atomic_multipoles_key` and the Fermi levels named by `--fermi_level_key`.
SCF-only settings such as `num_scf_steps`, `mixing_parameter`, and
`--scf_history` are not used by the direct update.

## Boundary Conditions

`--pbc_handling` selects the electrostatic boundary-condition treatment:

- `realspace`
- `pbc`
- `slab`
- `molecule_in_box`
- `mixed_periodic`
- `auto`

Use the same boundary-condition setting used for the intended evaluation
workflow. For mixed datasets, `mixed_periodic` and `auto` are the most useful
options. See [Boundary conditions](../concepts/boundary_conditions.md) for the
meaning of these modes.

## Output Keys

With the default `MACE_` prefix, the script writes:

- `MACE_energy`
- `MACE_forces`, unless `--dont_compute_force` is used
- `MACE_density_coefficients`
- `MACE_dipole`
- `MACE_fermi_level`
- `MACE_electrostatic_energy`
- `MACE_electron_energy`
- `MACE_num_scf_steps`
- `MACE_electrostatic_features`

If the model returns electrostatic potentials, the script also writes
`MACE_esps`.

SCF mode can also write convergence diagnostics:

- `scf_convergence`
- `scf_convergence_relative`
- `scf_charge_history`

Control these with:

```bash
--scf_history none
--scf_history absolute_change
--scf_history full_history
```

Direct mode does not write SCF history diagnostics and reports
`MACE_num_scf_steps` as `1`.

Use `--info_prefix` to change the `MACE_` prefix for model outputs. The
`scf_*` diagnostic keys are intentionally not prefixed.

## Notes

`--compute_stress` is not supported by the fixed-point evaluation script.

Use `--dont_compute_force` when only scalar outputs are needed. This skips
writing `MACE_forces`.

Use `scripts/vis_fixed_point_scf.py` when you need an extxyz trajectory of the
SCF iterations for diagnosing slow, oscillatory, or divergent convergence.
