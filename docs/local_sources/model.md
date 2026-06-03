# Local-Source Model Architectures

Local-source models are electrostatic MACE models without a self-consistent
field loop. They predict, or otherwise define, atom-centred source coefficients
from local information and then evaluate the long-range electrostatic energy
once.

The total energy has the form:

$$
E = E^\mathrm{SR} + E^\mathrm{LR} + E^\mathrm{field}.
$$

$E^\mathrm{SR}$ is a normal MACE-style short-range energy, written as a sum of
atomic readouts from the message-passing features. $E^\mathrm{LR}$ is the
Coulomb energy of a smooth charge density built from atom-centred multipoles.
$E^\mathrm{field}$ is the coupling of the predicted dipole or polarization to
an external homogeneous field.

The shared Gaussian multipole density representation is described in
[atomic multipoles](../concepts/atomic_multipoles.md). The local-source models
differ in how the coefficients $p_{i,lm}$ are obtained.

## Implemented Architectures

- [LocalCharges](nonpolarizable.md)
  directly predicts all local density coefficients from MACE node features.
- [LocalSplitCharges](local_symmetric_charges.md)
  starts from formal charges and predicts conservative local charge transfers
  between neighbouring atoms, plus local higher multipoles.
- [FixedChargeBaselinedMACE](fixed_charge_baseline.md)
  uses fixed formal monopoles as the long-range density and learns only the
  MACE short-range correction on top.

## Physical Tradeoffs

`LocalCharges` is the simplest learned-density architecture. It can represent
local charge and dipole response, but the total charge is not guaranteed by
construction unless it is constrained through the loss or data.

`LocalSplitCharges` builds charge conservation into the architecture. Its
learned charge transfer is antisymmetric over local directed edges, so the sum
of learned transfer charges cancels exactly and the total charge is fixed by
the supplied formal charges. This makes it better suited to systems where
oxidation states are meaningful and polarization should be compatible with
periodic translations.

`FixedChargeBaselinedMACE` is the most constrained architecture. It is useful
as a physically interpretable baseline when fixed oxidation-state charges are a
reasonable approximation, or when one wants to separate the effect of adding a
fixed long-range Coulomb term from learning a flexible charge model.

All three models use the same electrostatic boundary handling modes described
in [boundary conditions](../concepts/boundary_conditions.md), and the same
density-coefficient conventions described in
[atomic multipoles](../concepts/atomic_multipoles.md).
