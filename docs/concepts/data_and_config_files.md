# Data And Config Files

MACE_SCF uses MACE data loading, but the electrostatic models add extra data
keys and training schedule structure. Most training examples split settings
between a YAML config file and command-line arguments.

## Training Script and Config Files

We specify the data processing and many training settings in a `config.yaml` file as well as a call to `run_train.py`, there are very many additional options copmared to MACE, so its reccomended to go over the docs before trying some fits.

An example training script and `config.yaml` for the Local Symmetric Charge MACE model would be:

`train.sh`:
```bash
python scripts/run_train.py \
    --name="split_charge_example" \
    --config=config.yaml \
    --train_file=$train \
    --valid_fraction=0.2 \
    --test_file=$train \
    --E0s=average \
    --model="LocalSplitCharges" \
    --hidden_irreps='64x0e + 64x1o' \
    --r_max=6.0 \
    --eval_interval=2 \
    --batch_size=8 \
    --valid_batch_size=2 \
    --error_table="PerAtomRMSE" \
    --amsgrad \
    --device=cuda \
    --atomic_multipoles_max_l=1 \
    --atomic_multipoles_smearing_width=1.5 \
    --atomic_formal_charges="{13: 0.0}" \
    --default_dtype="float64" \
    --restart_latest \
    --save_cpu
```

`config.yaml`:
```yaml
heads:
  revPBEd3:
    info_keys:
      energy: AIMS_energy
      dipole: AIMS_dipole
      polarizability: AIMS_polarizability
      fermi_level: VBM
      external_field: homogeneous_field
      total_charge: total_charge
    arrays_keys:
      forces: AIMS_forces
      atomic_multipoles: AIMS_atom_multipoles
train_schedule:
  0:
    name: stage1
    start: 0
    end: 99
    loss:
      atomic_multipoles: 100.0
      dipole_per_atom: 1.0
      energy_per_atom: 1.0
      forces: 100.0
    lr: 0.01
  1:
    name: stage2
    start: 100
    end: 199
    loss:
      atomic_multipoles: 100.0
      dipole_per_atom: 1.0
      energy_per_atom: 1000.0
      forces: 100.0
    lr: 0.001
```

There are many new options in `train.sh`, the most important ones relate to handling periodicity and specifying the desciptio of the charge density in the model (atomic multipole order, smearing width, etc). You can find more information in [atomic_multipoles](atomic_multipoles.md) and [boundary_conditions](boundary_conditions.md).

The `config.yaml` mostly covers the data which is read from input xyz files, and the loss weights and training schedule, and is explained below.

## Data Keys

The keys used to read data from train/test xyz files must be specified in the `heads` dictionary.

The `info_keys` subsection refers to anything in `atoms.info`.

```yaml
heads:
  LDA:
    info_keys:
      energy: AIMS_energy
      dipole: AIMS_dipole
      polarizability: AIMS_polarizability
      fermi_level: dft_fermi_level
      external_field: homogeneous_field
      total_charge: total_charge
```

and `arrays_keys` is anything in `atoms.arrays`:

```yaml
    arrays_keys:
      forces: AIMS_forces
      atomic_multipoles: AIMS_atom_multipoles
```

**You don't need to specify everything**: if you are only training on energy and forces, just ommit the other fields.

> ### Check if your data is loaded correctly
> At the start of each training run, a line like this is printed in the log file:
> ```text
> INFO: Total Training set [energy: 219, dipole components: 219, fermi_level: 219, total_charge: 219, external_field: 219, stress: 0, head: 219, forces: 219, atomic_multipoles: 219]
> ```
> Always check that these numbers agree with what you intended to load from the
> xyz file.

## Config Dipole Weight

MACE-style extxyz files can carry per-configuration weights using keys of the
form `config_<property>_weight`. MACE_SCF uses this for component-wise dipole
masking.

If a config contains the dipole key specified in `heads.info_keys.dipole`, it must
also contain an explicit `config_dipole_weight`. 

`config_dipole_weight` must be a 3-vector in Cartesian component order. It
controls which dipole components contribute to dipole losses and error metrics. You need to set this differently depending on what the dipole physically means in the dataset:

- for molecular clusters where all dipole components are meaningful, `config_dipole_weight="1.0 1.0 1.0"`
- for slabs where only the surface-normal dipole is meaningful, `config_dipole_weight="0.0 0.0 1.0"`

If you have periodic configurations, the dipole is generally not a valid
quantity unless it is computed via the modern theory of polarization, with
care taken to handle branch cuts. To ignore the dipole on such a config,
set `config_dipole_weight="0.0 0.0 0.0"` (or a partial mask). Configs that
do not have the dipole info key at all are automatically given a zero
dipole weight by the data loader and so do not contribute to dipole losses
or metrics.

## train_schedule

`train_schedule` defines one or more training stages. Each stage can set:

- `name`
- `start` and `end` epochs
- `loss`
- `lr`
- more detailed options for SCF models (e.g. `fixed_point_training_options`) when relevant

For example:

```yaml
train_schedule:
  0:
    name: stage1
    start: 0
    end: 99
    loss:
      atomic_multipoles: 100.0
      energy_per_atom: 10.0
      forces: 100.0
    lr: 0.01
  1:
    name: stage2
    start: 100
    end: 200
    loss:
      energy_per_atom: 1000.0
      forces: 100.0
    lr: 0.01
```

You can build whatever loss you want and change it between different stages, you can even add or remove components of the loss, as done in the example above. The loss can be composed from the available [loss functions](losses.md). 

The training schedule generalizes the `_stagetwo` syntax in MACE.
