# Atomic Multipoles

All models in this repo use atom-centred multipole moments to represent a
coarse-grained charge density. They appear in datasets, model inputs, model
outputs, losses, calculators, and evaluation scripts.

## Density Coefficients

MACE_SCF often refers to predicted multipoles as `density_coefficients`. These
coefficients describe smeared atom-centred multipoles, not point charges. The
electrostatics backend uses them to compute Coulomb energies, field features,
dipoles, and potentials.

The backend for these operations is `graph_longrange`. MACE_SCF models predict
the atom-centred coefficients; `graph_longrange` evaluates the electrostatic
quantities associated with the resulting smooth density.

## Gaussian Multipole Density

The coarse-grained charge density is represented as a sum of Gaussian type
orbital multipoles:

$$
\rho(\mathbf r) = \sum_{i,lm} p_{i,lm}\,\phi_{lm}(\mathbf r-\mathbf r_i).
$$

Here $i$ indexes atoms and $(l,m)$ indexes spherical multipole components.
$l=0$ is charge, $l=1$ is dipole, and higher $l$ values correspond to higher
multipoles. The basis functions are normalized so that each $\phi_{lm}$ carries
one unit of the corresponding multipole moment:

$$
\phi_{lm}(\mathbf r) = C\,Y_{lm}(\hat{\mathbf r})\,|\mathbf r|^l
    \exp(-|\mathbf r|^2 / 2\sigma^2).
$$

The electrostatic energy is the Coulomb energy of this smooth density:

$$
E^\mathrm{LR} = \frac{1}{2}
    \iint
    \frac{\rho(\mathbf r)\rho(\mathbf r')}
         {4\pi\epsilon_0|\mathbf r-\mathbf r'|}
    d\mathbf r\,d\mathbf r'.
$$

## Parameters

There are two important parameters which are passed to `run_train.py`.

- `--atomic_multipoles_max_l=1`
  selects the highest multipole order represented in the density coefficients. $l=0$ means charges only, $l=1$ means charges and dipoles, and so on. In practice $l=1$ is usually the best choice, and some features are not implemented for $l\geq2$. 
- `--atomic_multipoles_smearing_width=1.5`
  sets the Gaussian width of the multipoles in Angstrom, denoted $\sigma$ in the equations above.

## Coefficient Ordering

For charge-plus-dipole atomic multipoles, the coefficient order follows the
Condon-Shortley phase convention:

```text
[Q, d_y, d_z, d_x]
```

Here `Q` is the atomic charge, and `d_x`, `d_y`, and `d_z` are the Cartesian
components of the atomic dipole. Note that the stored coefficient order is **not
Cartesian order** for the dipole.

On the other hand, outputs from models which are explicitly named `dipole` or `partial_dipoles` (as opposed to `density_coefficients`) are Cartesian:

```text
[d_x, d_y, d_z]
```

This is also important when reading atomic multipoles from an xyz file. The coede expects density coefficients in the xyz file to be Condon Shortly phase convention.