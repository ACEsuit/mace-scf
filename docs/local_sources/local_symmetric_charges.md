# LocalSplitCharges

`LocalSplitCharges` is the local split-charge architecture. It combines
formal oxidation-state charges with learned local charge transfers and optional
local higher multipoles.

The motivation is to keep the useful flexibility of a learned local density
model while enforcing a physically important constraint: the learned charge
transfer should not change the total charge of the system.

## Energy And Density

As with the other local-source models, the total energy is decomposed into a
short-range MACE energy and a long-range electrostatic energy:

$$
E = E^\mathrm{SR} + E^\mathrm{LR} + E^\mathrm{field}.
$$

The long-range term is computed from the smooth Gaussian multipole density
described in [local-source model architectures](model.md).

## Formal Charges

Each atom starts with a formal charge `q_i^{(0)}`. These formal charges can be
fixed per species or read from a per-atom array in the dataset.

The formal charges define the total charge of the system:

$$
Q_\mathrm{tot} = \sum_i q_i^{(0)}.
$$

The learned charge-transfer part is constructed so that it sums to zero, so it
cannot change this total.

## Local Charge Transfer

Instead of directly predicting the charge on each atom, the model predicts a
directed charge transfer `p_{ij}` on each local edge between neighbouring atoms
`i` and `j`. The final monopole on atom `i` is

$$
q_i = q_i^{(0)}
    + \sum_{j \in \mathcal{NN}(i)} (p_{ij} - p_{ji}).
$$

Every learned transfer appears once with a plus sign and once with a minus
sign across the system. Therefore

$$
\sum_i \left(q_i - q_i^{(0)}\right) = 0,
$$

up to numerical precision. This is the key physical constraint of the model.

The transfer values are functions of the local node features of the atoms and
their relative displacement. In the implementation, the default static transfer
block builds equivariant edge features from sender/receiver node features,
spherical harmonics of the edge vector, and radial edge features, then maps
those edge features to scalar transfers.

## Higher Multipoles

When `atomic_multipoles_max_l > 0`, the model can also predict local higher
multipoles such as atomic dipoles. These are read from local node features in
the same equivariant density basis used by the other local-source models.

The default split-charge block zeroes the directly predicted monopole from the
local multipole readout and replaces it with the conservative transfer charge.
This keeps the charge-conservation property while still allowing local dipoles
and higher multipoles.

## Polarization

The split-charge construction gives a natural polarization expression. The
model polarization includes local atomic multipoles, formal-charge positions,
and charge-transfer contributions along local bonds:

$$
\mathbf P^\mathrm{MLIP} =
  \sum_i \left(\mathbf p_i + q_i^{(0)} \mathbf r_i\right)
  +
  \sum_i \sum_{j \in \mathcal{NN}(i)}
    \left(p_{ij}\mathbf r_{ij} + p_{ji}\mathbf r_{ji}\right).
$$

When the formal charges correspond to oxidation states, this polarization is
translation invariant in the intended periodic sense and changes correctly when
an ion is moved through a periodic cell boundary.

This is important for calculations in a homogeneous applied electric field.
The field coupling can be written in terms of the model polarization, so the
energy changes consistently when an ion crosses a periodic cell boundary. This
is the same physical requirement that appears in the modern theory of
polarization: the absolute polarization in a periodic system is only defined
modulo a polarization quantum, but changes in polarization under continuous
atomic motion must be well defined. The formal-charge part of the
`LocalSplitCharges` polarization gives the correct branch change for
integer oxidation-state transport through the cell.

The same polarization expression can also be differentiated with respect to
atomic positions to obtain Born-effective-charge-like response tensors:

$$
Z^{*}_{i,\alpha\beta}
  \propto
  \frac{\partial P_\alpha}{\partial r_{i,\beta}}.
$$

The proportionality factor depends on the unit convention used for
polarization and cell volume. In practice, this provides a route to extracting
charge-response information from the trained local split-charge model.

## Optional Polarizability Readout

The implementation can optionally add a polarizability readout. This readout
maps local equivariant features to atomic polarizability contributions and sums
them over atoms. This is separate from an SCF polarization response; it is a
direct readout from the local-source model.

## Use Cases

Use `LocalSplitCharges` when:

- formal oxidation states are meaningful;
- exact total charge conservation is important;
- the model should represent local charge transfer rather than arbitrary local
  monopoles;
- periodic polarization behaviour matters;
- charges or multipoles may be latent variables trained indirectly from
  energies and forces, or directly from reference multipoles when available.

The model is less appropriate when formal charges are not well defined, or when
the relevant charge redistribution is strongly nonlocal and cannot be captured
by local charge transfers within the cutoff.
