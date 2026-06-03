# Boundary Conditions

Long-range energy contributions, field features, and electrostatic potentials
depend on the boundary conditions used to interpret the atomic multipoles. The
same density coefficients can give different electrostatic results under
open-boundary, fully periodic, slab, or molecule-in-box treatments.

## Explicit Boundary Handling

MACE_SCF uses explicit electrostatic boundary modes:

- `realspace`
  open-boundary real-space evaluation.
- `pbc`
  periodic evaluation with no slab or molecule correction.
- `slab`
  periodic evaluation with slab correction. This is only supported when the $z$-direction is the non-periodic direction. 
- `molecule_in_box`
  periodic evaluation with molecule correction.
- `mixed_periodic` (**default**)
  periodic evaluation with per-graph correction branching for mixed datasets.
- `auto` 
  run time branching between realspace and periodic implementations. 

Calculators and evaluation scripts generally expose this as `pbc_handling`.
The training script exposes the command-line option
`--electrostatic_pbc_method="mixed_periodic"`.

## Atoms.pbc Versus pbc_handling

`Atoms.pbc` still describes the boundary condition stored with each structure.
However, the electrostatic evaluator is not always chosen directly from
`Atoms.pbc` at runtime.

For homogeneous datasets or MD runs, choose an explicit fixed mode:

- bulk `TTT`: `pbc`
- slab `TTF`: `slab`
- non-periodic molecule or cluster: `realspace`
- molecule treated in a periodic box with correction: `molecule_in_box`

Choosing a fixed mode can give faster models, since there is less
runtime branching based on the data. It's also more compatible with `torch.compile`. For mixed datasets containing more than one
boundary-condition class, use `mixed_periodic`. This mode will run a periodic calculation, but also examine the `pbc` field of the atoms object and add the relevant corrections for molecule or slab configs.

## Reciprocal-Space Cutoff Factor

Periodic electrostatics uses a reciprocal-space cutoff. MACE_SCF computes a
heuristic cutoff from the multipole order and Gaussian smearing width, then
multiplies it by `kspace_cutoff_factor` to get the actual cutoff.

In training this is exposed as:

```bash
--kspace_cutoff_factor 1.25
```

The parser default is currently `1.25`.

Reducing this factor can make periodic training or evaluation faster, but it
can also make energies and forces noisier. Increasing it is useful when
periodic systems show numerical noise or insufficient reciprocal-space
convergence. The relevant scale depends on the smearing width, multipole order,
cell size, and boundary mode. Generally, the default value is sufficient, but
if you want to quickly run fast tests you can consider reducing this.

## Molecule-In-Box Corrections And Cells

`molecule_in_box` and the non-periodic branch of `mixed_periodic` still use the
periodic electrostatics machinery, then apply correction terms appropriate for
a finite object in a box. Depending on the system and correction path, these
corrections remove leading finite-size artefacts such as monopole or dipole
terms from the periodic calculation.

Because the base calculation is still periodic, non-periodic cluster or
molecule configurations must still have a meaningful `cell` in the extxyz file
or data source. The structure should also have `pbc=FFF`; otherwise the code
will treat it as a periodic solid, slab, or other periodic system rather than a
finite cluster.

The cell should be large enough that the periodic image interactions are small
before correction. A typical practical choice for clusters with total unsigned
charge no larger than 2 electrons is a cubic cell of roughly 30-40 Angstrom. Larger
or more highly charged systems may need a larger box.

This matters during both training and evaluation. A model trained on
non-periodic configurations without cells, with too-small cells, or with
incorrect `pbc` flags can learn or report inconsistent electrostatic energies
and forces.

## Realspace Versus Molecule-In-Box

`realspace` and `molecule_in_box` are both used for non-periodic systems, but
they are not interchangeable.

`realspace` uses an open-boundary evaluator directly. There is no reciprocal
space periodic calculation and no molecule-in-box correction term. This is different from `molecule_in_box`, and hence you will get small differences. In large cells, with a very converged kspace cutoff, the two evaluations will be close to identical.

## Self-Interaction Settings

Two settings are related but distinct:

- `include_electrostatic_self_interaction=True`
  controls self-interaction terms in the electrostatic energy.
- `include_field_si=False`
  controls self-interaction-like contributions in field-feature construction.

For the electrostatic energy, self-interaction means the energy of a smeared
atomic multipole on an atom interacting with itself. Including or excluding
this term changes the energy definition used by the model.

For field features, self-interaction means the projection of the field generated
by an atom's own multipoles onto the field features received by that same atom.
This changes the feature input to field-coupled models, but it is not the same
setting as energy self-interaction. For the non-scf models, this flag has no effect.

### Low-Density Periodic Config Check

When reading `.xyz` training data, the loader checks for configs with
`pbc="T T T"` and a very large volume per atom. This catches a common setup
mistake where a cluster in a large box is accidentally treated as fully
periodic.

The check is controlled by:

- `--low_density_pbc_max_volume_per_atom`
  maximum allowed volume per atom for fully periodic configs, default `100.0`;
- `--allow_low_density_pbc`
  bypass the check when a low-density fully periodic calculation is intentional.

The error message reports the split, config index, config type, cell volume,
number of atoms, and volume per atom. Non-periodic and partially periodic
configs are not affected.