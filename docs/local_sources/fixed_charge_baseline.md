# FixedChargeBaselinedMACE

`FixedChargeBaselinedMACE` is the fixed-charge local-source baseline. It does
not learn charge or multipole coefficients. Instead, it uses fixed formal
monopoles as the long-range charge density and learns a MACE short-range energy
on top.

## Architecture

Each atom is assigned a formal charge `q_i^{(0)}`. The charges can be fixed per
species or read from a per-atom array in the dataset.

The density coefficients are monopoles only:

$$
p_{i,00} = q_i^{(0)}.
$$

No local charge-transfer block is used, and no atomic dipoles or higher
multipoles are learned by this architecture.

The total energy is:

$$
E = E^\mathrm{SR}_\mathrm{MACE}
    + E^\mathrm{LR}\left[\{q_i^{(0)}\}\right]
    + E^\mathrm{field}.
$$

The short-range part is learned from MACE readouts. The long-range part is the
Coulomb energy of the fixed smeared formal charges. The field term couples the
formal-charge dipole to the external field.

## Physical Properties

This model has the strongest built-in charge constraint: the charge density is
fixed by the formal charges and cannot adapt to the environment.

That makes the electrostatic term easy to interpret, but it also limits the
model. Any environment-dependent polarization, charge transfer, screening, or
atomic dipole response must be absorbed indirectly into the learned short-range
MACE energy, if it can be represented there at all.

## Use Cases

Use `FixedChargeBaselinedMACE` when:

- fixed oxidation-state charges are a reasonable approximation;
- a simple electrostatic baseline is needed;
- you want to compare learned charge models against a fixed-charge Coulomb
  reference;
- the main goal is to add a transparent long-range ionic term to a local MACE
  model.

Avoid it when environment-dependent charge redistribution or polarization is a
central part of the target physics. In those cases, use
`LocalSplitCharges`, `LocalCharges`, or a fixed-point SCF model instead.
