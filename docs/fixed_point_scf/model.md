# Fixed-Point SCF Model

The fixed-point SCF model is a self-consistent electrostatic MACE model. It
does not predict the final charge density in one forward pass. Instead, it
predicts how the atom-centred multipole density should respond to an electrostatic potential. 

The model is designed for systems where long-range electrostatic induction,
charge redistribution, applied fields, or charge/Fermi-level control are part
of the physics. It is more expensive and more delicate to train than a
local-source model, but it can represent effects that a one-shot local density
prediction cannot.

## Core Idea

The model represents the coarse-grained charge density with the same
atom-centred Gaussian multipoles described in
[atomic multipoles](../concepts/atomic_multipoles.md). The fixed-point
equation can be summarized as:

$$
\mathbf p^{(t+1)} =
  F_\mathrm{ML}
  \left(
    \{(z_i,\mathbf r_i)\}_i,
    \mathbf v_\mathrm{eff}[\mathbf p^{(t)}]
  \right).
$$

Here $\mathbf p$ is the vector of atomic multipole coefficients and
$\mathbf v_\mathrm{eff}$ is an atom-centred representation of the effective
electrostatic potential. That potential contains the Coulomb potential from the
current density at step $t$, any applied external field, and the Fermi level
term:

$$
v_\mathrm{eff}[\mathbf p](\mathbf r) =
  \int
  \frac{\rho(\mathbf r')}{|\mathbf r-\mathbf r'|}
  d\mathbf r' + v_\mathrm{app}(\mathbf r) + \mu.
$$

This is analogous in spirit to the self-consistency cycle in electronic
structure theory, but it is not a Kohn-Sham diagonalization. The learned
function $F_\mathrm{ML}$ replaces the difficult part of the response: given a
geometry and an effective potential, it predicts an updated coarse-grained
charge density.

## Why A Fixed Point?

In a local-source model, an atom's density coefficients are predicted from the
local geometry and then electrostatics is evaluated once. That is efficient and
often useful, but the predicted density does not explicitly respond to the
long-range electrostatic environment.

In the fixed-point model, the density and electrostatic potential are coupled:

1. a trial density generates a long-range potential;
2. the potential is projected into atom-centred electrostatic features;
3. the model predicts an updated density from geometry and those features;
4. the process is repeated until the density stops changing.

This lets electrostatic induction and charge redistribution be mediated through
the same self-consistent loop. It also gives a natural way to control total
charge by changing the Fermi level.

## Charge Control And Fermi Level

The Fermi level $\mu$ is treated as a Lagrange-multiplier-like term added to
the effective potential. There are two evaluation modes:

- constant Fermi level:
  keep $\mu$ fixed and iterate the density. The resulting total charge is the
  model's response to that Fermi level.
- constant charge:
  adjust the effective Fermi-level contribution during the SCF loop so the
  final density matches a requested total charge.

Constant-charge mode is usually the more natural choice for atomistic
calculations with a specified total charge. Constant-Fermi mode can be useful
for workflows closer to a grand-canonical electronic setting.

The model must learn the correct response of total charge to the Fermi level.
This is not guaranteed by architecture alone, so datasets and losses must
include the charge-state information needed to teach that behaviour.

## Architecture Overview

The current implementation is `FixedPointCore` in
`mace_scf/electrostatics/fixed_point_core.py`. The code is organized around
three conceptual methods.

### local_part

`local_part` computes all state that does not depend on the iterated SCF
density:

- MACE-style geometry features;
- local short-range energy contributions;
- a field-independent local density contribution;
- mixed layer features used by the fixed-point update function;
- electrostatics geometry caches;
- external-field features that do not depend on the Fermi level.

This field-independent density is used both as an initial local guess and as
the baseline density to which the field-dependent update is added.

### scf_step

`scf_step` performs one fixed-point update. It:

1. computes electrostatic field features from the current trial density;
2. adds external-field features;
3. adds Fermi-level features;
4. normalizes the field features;
5. predicts the field-dependent density contribution.

The updated density is:

$$
\mathbf p_\mathrm{new} =
  \mathbf p_\mathrm{local}
  +
  \Delta \mathbf p_\mathrm{field}.
$$

The fixed-point update block can be one-body, nonlinear one-body, or more
many-body-like depending on the configured `fixedpoint_update_config`. In all
cases, the update is a learned map from local geometry and electrostatic
features to a density response.

### build_observables

`build_observables` computes final properties from the converged or truncated
SCF density:

- total energy;
- forces;
- final density coefficients;
- final Fermi level;
- total dipole;
- Coulomb energy;
- local electron-energy contribution;
- electrostatic field features;
- optional electrostatic potentials.

The energy is computed after the density has been chosen by the fixed-point
loop. This is important: the current fixed-point architecture is not a
variational energy minimization over density. The density is the solution of a
learned fixed-point equation, and the energy is a separate readout built from
that final density.

## Electrostatic Features

The effective potential is converted into atom-centred features by projecting
it onto Gaussian basis functions around each atom. These features are analogous
to long-range equivariant descriptors: they summarize the electrostatic
environment around each atom in a rotation-compatible basis.

Important settings include:

- `field_feature_max_l`
  controls the angular order of the electrostatic features.
- `field_feature_widths`
  controls the Gaussian widths used to sample the potential.
- `include_field_si`
  controls whether self-interaction-like contributions are included in the
  field-feature pathway.

These are separate from the density settings
`atomic_multipoles_max_l` and `atomic_multipoles_smearing_width`, which control
the representation of the charge density itself.

## Energy Terms

The final energy contains several pieces:

- local MACE energy contributions from the geometry backbone;
- optional local electron-energy or nonlocal density-dependent energy terms;
- Coulomb energy of the final smooth multipole density;
- coupling of the final dipole to the applied external field.

The additional density-dependent energy term is useful because the model is not
variational in the density. It can be interpreted either as a correction for
missing variational energy contributions or simply as an energy readout that
depends on the final self-consistent density.

## Training Data Requirements

The fixed-point model has stricter data requirements than local-source models.

As discussed in [training](training.md), there are several different ways to train a fixed point model. For direct training, `mode: direct`, the
training batch must contain reference atomic multipoles and reference Fermi
levels. The wrapper uses the reference density to build the electrostatic
features and trains the model so that one fixed-point update reproduces the
reference density. In other words, the DFT-like density is treated as an input
to the training step, not just as a target.

For SCF training modes such as `unroll_scf` and `implicit`, the model is run
closer to inference: an initial density is chosen, the SCF loop is converged or
unrolled, and losses are applied to the model's resulting solution. These modes
still commonly need reference multipoles for density losses, and they usually
need `total_charge`, `fermi_level`, and `external_field` keys to define the SCF
problem and initial guesses.

Common fixed-point data fields are:

- `atomic_multipoles`
  reference density coefficients, required for direct fixed-point training and
  for density losses;
- `fermi_level`
  required for direct fixed-point training and useful as an SCF initial guess
  or constant-Fermi target;
- `total_charge`
  required for constant-charge SCF;
- `external_field`
  required when applied fields are part of the model input;
- `energy`, `forces`, and `dipole`
  common observable targets;
- `electrostatic_potentials` or related potential labels,
  required only for workflows that train potential-like outputs.

The exact keys are mapped through `config.yaml`, as described in
[data and config files](../concepts/data_and_config_files.md).

## Training And Inference Mismatch

Direct fixed-point training is convenient because it avoids solving the SCF
problem during early training. It trains the update function at the reference
density, using the electrostatic features generated from that density.

This does not guarantee that the model will converge to a stable and accurate
solution when started from its own initial guess. During inference, errors can
compound through the SCF loop, and the model may find a density different from
the reference density used during direct training.

For this reason, fixed-point workflows often use staged training: direct
training first, followed by unrolled or implicit SCF training once the update
function is reasonable. The later stages train the model closer to how it will
be evaluated.

## Use Cases

The fixed-point SCF model is appropriate when:

- long-range electrostatic induction is important;
- charge transfer or polarization should respond to the global electrostatic
  environment;
- applied fields are part of the target physics;
- total charge or Fermi level should be controlled explicitly;
- a one-shot local-source model is too limited.

It is less attractive when:

- a simple local density prediction is sufficient;
- reference multipoles or Fermi levels are unavailable for direct training;
- the overhead and convergence risks of an SCF loop are not justified;
- the intended calculation needs a strictly variational density optimization
  rather than a learned fixed-point equation.

## Main Outputs

Fixed-point models can return:

- final energy and forces;
- final `density_coefficients`;
- final `fermi_level`;
- total dipole;
- electrostatic energy;
- local electron energy;
- electrostatic field features;
- SCF charge history;
- optional electrostatic potentials.

The SCF charge history is useful for diagnosing convergence, oscillation, and
divergence.
