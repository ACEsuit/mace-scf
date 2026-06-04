# MACE-SCF

[![Documentation Status](https://readthedocs.org/projects/mace-scf/badge/?version=latest)](https://mace-scf.readthedocs.io/en/latest/?badge=latest)

MACE_SCF extends the MACE machine-learning interatomic potential architecture
to create charge aware models. The repo provides a sandbox for developing self-consistent MLIPs, as in our preprint on the [design space of electrostatic self-consistent MLIPs](https://arxiv.org/abs/2603.14700).

Compared with a standard MACE workflow, MACE_SCF introduces several new
concepts that users need to set deliberately: atom-centred multipole density
coefficients, electrostatic boundary handling, custom training schedules, and
electrostatics-specific losses. Please read the documentation before trainnig or using any models, since they may not behave at all like local MACE models.

## Development Status

We are still adding some key features to the repo. Non-SCF models like the local split charge MACE can be used, but SCF models are still lacking documentation. For results in the preprint, the **Energy Functional SCF** model is still being cleaned up.

### Requirements

Please install your own version of pytorch. Note that newer version of pytorch (>=2.5) are needed for torch compilation. However, the implicit differentation code provided by `torchopt` seems to be stable with older pytorch versions. See the documentation on implicit differentiation for more information.

Packages related to mace:
- MACE_SCF is tied to `mace==v0.3.14` for now. please checkout this mace version. 
- we use [graph_longrange](https://github.com/WillBaldwin0/graph_electrostatics) for electrostatic calculations of things like electrostatic energy, electric potential and to provide embeddings like compensating jellium slabs.

## Model Families

`mace_scf` provides numerous different electrostatic models, and there is not a reccomended one-size-fits-all archtiecture. The documentation explainsthe model types and trade-offs in more detail.

### Non-SCF Electrostatic models

Two easy to use, reliable models are provided which include elecrtostatic corrections without needing any SCF cycle. These models can often be trained on only total energy and force data. Further details can be found in the documentation.

- **Local Charge (LC-MACE)** provides a way to fit atomic multipole moments from mace descriptors, and use them to compute energetic corrections, dipole moments and more. 
- **Local Split Charge (LSC-MACE)** also predicts charge and multipoles moments from local geometry, but does so in a way that conserves local charge flow. This means certain totla properties are correct by construction, and it also means this model is compatible with the modern theory of polarization. 

### SCF model sandbox

MACE_SCF also contains a sandbox for self-consistent electrostatic models. These methods are more flexible, but they also introduce more choices in model setup, losses, SCF settings and training schedules. Please read the SCF documentation before trying to train or evaluate these models.

- **MACE-QEq** is the charge-equilibration style route discussed in the preprint, similar to many existing MLIP architectures. This part of the repo is still being cleaned up and documented.
- **FixedPointSCF** is the main fixed-point self-consistent model family currently reccomended. The model performs a Kohn-Sham-like SCF cycle to predict at a set of atomic multipole momenets. See `docs/fixed_point_scf/`.
- **EnergyFunctionalSCF** is the alternative approach presented in the preprint, and is also under active cleanup and documentation, and should be treated as experimental for now.

The SCF code in this repo is can be used as a sandbox and contains the implementations used in the arXiv paper. These models can be trained and evaluated, but they are not yet intended as frictionless out-of-the-box workflows.

## Citations

Self Consistent Models
```bibtex
@misc{baldwin2026designspaceselfconsistentelectrostatic,
      title={Design Space of Self--Consistent Electrostatic Machine Learning Interatomic Potentials}, 
      author={William J. Baldwin and Ilyes Batatia and Martin Vondrák and Johannes T. Margraf and Gábor Csányi},
      year={2026},
      eprint={2603.14700},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2603.14700}, 
}
```
Split Charge Model Example Usage
```bibtex
@misc{parker2026falsemetallizationshortrangedmachine,
      title={False Metallization in Short-Ranged Machine Learned Interatomic Potentials}, 
      author={Isaac J. Parker and Mandy J. Hoffmann and William J. Baldwin and Shuang Han and Srishti Gupta and Kara D. Fong and Angelos Michaelides and Christoph Schran and Sandip De and Gábor Csányi},
      year={2026},
      eprint={2603.04228},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2603.04228}, 
}
```
MACE-QEq:
```bibtex
@misc{VondrakMACEQEq,
      title={Integrating Charge Equilibration with Equivariant Machine-Learning Interatomic Potentials}, 
      author={Martin Vondr\'{a}k and William J. Baldwin and G\'{a}bor Cs\'{a}nyi and Karsten Reuter and Johannes T. Margraf},
      journal = {ChemRxiv},
      number = {0224},
      year = {2026},
      doi = {10.26434/chemrxiv.15000377/v1},
      URL = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15000377/v1},
      eprint = {https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15000377/v1}
}
```
