# LocalCharges

`LocalCharges` is the direct local-charge architecture. In the paper
description it corresponds to `MACE-LocalCharges`.

Despite the class name, the model can predict monopoles and higher multipoles.
The name means that the model does not perform a self-consistent polarization
or field-response solve. It predicts the density coefficients once from local
geometric features.

## Architecture

The model first runs the usual MACE local message-passing stack. Let
`h_i^{(s)}` denote the equivariant node features on atom `i` after interaction
layer `s`.

For each layer, an equivariant linear map reads local multipoles from the node
features. In the notation used in the paper, for layer 1 and layer 2:

$$
p_{i,lm} = \sum_k W_{k} h^{(1)}_{i,klm}
    + \sum_k W'_{k} h^{(2)}_{i,klm}.
$$

The implementation generalizes this pattern over the configured number of
interaction layers. Each layer has an `EnvironmentDependentSourceBlock`, and
the predicted source contributions are summed:

$$
p_i = \sum_s p_i^{(s)}.
$$

The same message-passing features also feed ordinary MACE energy readouts. The
final energy is the sum of the short-range MACE energy, the Coulomb energy of
the predicted smooth multipole density, and the external-field coupling.

## Physical Properties

`LocalCharges` is local in its density prediction. The density coefficients
on an atom are functions of the local environment inside the MACE cutoff; after
they are predicted, long-range electrostatics is evaluated explicitly.

The architecture is equivariant because the multipole readout maps equivariant
node features to spherical multipole coefficients. It can therefore predict
charges, dipoles, and higher multipoles in a rotation-compatible way.

The main limitation is that total charge is not conserved by construction. If
the model predicts monopoles, the sum of those monopoles is whatever the local
readouts produce. Charge conservation can be encouraged by training on
`total_charge` or atomic multipoles, but it is not a hard architectural
constraint.

## Use Cases

Use `LocalCharges` when:

- a direct local density prediction is sufficient;
- the dataset includes reliable atomic multipole targets;
- exact charge conservation is not essential, or can be handled through loss
  terms;
- a simple one-shot electrostatic baseline is desired before using a more
  constrained charge-transfer architecture.

Avoid it when exact total charge conservation is a requirement of the
simulation. In that case, `LocalSplitCharges` is usually the better
local-source architecture.
