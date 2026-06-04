# Welcome

MACE_SCF extends the MACE machine-learning interatomic potential architecture
with electrostatic source models and self-consistent field models.


Compared with a standard MACE workflow, MACE_SCF introduces several extra
concepts that users need to set deliberately: parameters for atom-centred multipoles, electrostatic boundary condition handling, custom training schedules, and
electrostatics-specific losses.

## Key Concepts

There are several concepts and API choices which are used in all the different models in MACE_SCF, and which aren't present in MACE. The links below are therefore useful to read before training or using any models:

- [Data and config files](concepts/data_and_config_files.md).
  Compared to MACE, we use a more expressive and explicit configuration file to control data extraction and training schedules. Please read this before training any models.
- [Losses](concepts/losses.md). Similarly, the way a loss function is defined, differes from MACE.
- [Atomic multipoles](concepts/atomic_multipoles.md). All the models in this repo use atomic multipole moments to represent a charge density. This page
  covers multipole defintion, smearing width, coefficient ordering, and where
  multipoles appear in datasets and model outputs.
- [Boundary conditions](concepts/boundary_conditions.md). Electrostatic calculations differ based on whether you are evaluating a cluster, a periodic system, or a slab.
  This page explains how we handle this and what you need to be aware of. You can also find information about things like molecule and slab correction terms.

## Model Families

MACE_SCF currently has two main electrostatic model families.

**local-source models** predict charge-like or multipole-like source
coefficients directly from local atomic environments. These coefficients are
passed once to the electrostatics backend to compute Coulomb energies, field
energies, dipoles, and related quantities. Currently, we provide two architectures: **Local Charge MACE** and **Local Symmetric Charge MACE**. For more information, see:

- [local-source models](local_sources/index.md)

You can also train various self-consistent MACE models. Currently, one can train **MACE-QEq**, **FixedPointSCF** and **EnergyFunctionalSCF** models. For details on all these models, please see our preprint cited in the README. Documentation on how to use these models can be found here:

- [Fixed-Point SCF models](fixed_point_scf/index.md)
- Energy Functional SCF models
- MACE-QEq

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

concepts/index
local_sources/index
fixed_point_scf/index
extras/index
```
