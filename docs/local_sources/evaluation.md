# Evaluating Local-Source Models

Use `scripts/eval_local_charges.py` to label extxyz files with predictions from
local-source models.

## Command

Typical usage:

```bash
python scripts/eval_local_charges.py \
    --model split_charge_example_stage1.model \
    --configs input.xyz \
    --output labelled.xyz \
    --batch_size 16 \
    --device cuda \
    --pbc_handling slab \
    --external_field_key homogeneous_field \
    --fermi_level_key VBM
```

### formal charges per atom

As described in [training](training.md), one can train a split charge model where the oxidation state is an input instead of being fixed per species. If you have trained such a model, you also need to specify the oxidation state per atom when evaluating the model. You can do this with `--formal_charges_key`.

## Outputs

With the default `MACE_` prefix, the script writes outputs such as:

- `MACE_energy`
- `MACE_forces`
- `MACE_density_coefficients`
- `MACE_dipole`
- optional `MACE_polarizability`
- optional model contribution keys