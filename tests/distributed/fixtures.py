"""Shared synthetic model/batch fixtures for distributed-adjacent tests.

Used by tests/test_used_parameters.py (single-process invariant checks that
DDP's find_unused_parameters=False depends on) and by tests/distributed/'s
own gloo-based tests, so both build the same small, deterministic
FixedPointCore model and the same sharded batches instead of each
reimplementing it.
"""

import numpy as np
import torch
from ase.io import read
from e3nn import o3

import mace.modules
import mace.tools
from mace_scf import electrostatics
from mace_scf.electrostatics import field_blocks

from ..paths import reference_config
from ..utils import dataset_from_atoms, disable_e3nn_codegen, seed_torch

CONFIGS_PATH = reference_config("mixed_test_configs.xyz")
OUTPUT_ARGS = {"forces": True, "virials": False, "stress": False}
SEED = 5
ATOMIC_MULTIPOLES_MAX_L = 1
WORLD_SIZE = 2
SHARD_SIZE = 2
NUM_CONFIGS = WORLD_SIZE * SHARD_SIZE

# A representative FixedPointSCFOptions-shaped dict for tests that need an
# scf-mode (unroll_scf/implicit/linearize_solve) train schedule.
SCF_OPTIONS = {
    "num_scf_steps": 30,
    "constant_charge": False,
    "mixing_parameter": 0.2,
    "initial_density": "from_data",
    "initial_fermi_level": "from_data",
    "use_autograd_forces": True,
}


def build_model(device="cpu"):
    """Deterministic, small, randomly initialised FixedPointCore."""
    seed_torch(SEED)
    z_table = mace.tools.get_atomic_number_table_from_zs([1, 8])
    with disable_e3nn_codegen():
        model = electrostatics.FixedPointCore(
            r_max=4.5,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=mace.modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=mace.modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=1,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("4x0e+4x1o"),
            atomic_energies=np.array([-12.674624, -2041.039790]),
            avg_num_neighbors=10.0,
            atomic_numbers=z_table.zs,
            correlation=3,
            gate=mace.modules.gate_dict["silu"],
            MLP_irreps=o3.Irreps("16x0e"),
            radial_MLP=[64, 64, 64],
            radial_type="bessel",
            atom_density_scaling=np.ones(len(z_table)),
            kspace_cutoff_factor=1.0,
            atomic_multipoles_max_l=ATOMIC_MULTIPOLES_MAX_L,
            atomic_multipoles_smearing_width=1.5,
            field_feature_max_l=1,
            field_feature_widths=[1.5],
            include_electrostatic_self_interaction=True,
            add_local_electron_energy=True,
            fixedpoint_update_config={
                "type": field_blocks.OneBodyVariableUpdate,
                "potential_embedding_cls": field_blocks.BiasedLinearPotentialEmbedding,
                "nonlinearity_cls": field_blocks.NoNonLinearity,
            },
            field_readout_config={
                "type": field_blocks.StrictQuadraticFieldEnergyReadout
            },
            pbc_handling="mixed_periodic",
        )
    return model.to(device)


def shard_batches(model, n_shards=WORLD_SIZE, shard_size=SHARD_SIZE):
    """n_shards Batch objects, sharding n_shards*shard_size fixture configs.

    The fixture provides multipoles, total_charge, and a fermi level;
    deterministic synthetic energies and forces are added so the loss
    exercises the upstream mace energy/forces components too.
    """
    num_configs = n_shards * shard_size
    atoms_list = read(str(CONFIGS_PATH), index=":")[:num_configs]
    if len(atoms_list) < num_configs:
        # Fixture file is small; repeat configs deterministically to fill shards.
        atoms_list = (atoms_list * (num_configs // len(atoms_list) + 1))[:num_configs]
    rng = np.random.default_rng(SEED)
    for atoms in atoms_list:
        atoms.info["fake_energy"] = float(rng.normal())
        atoms.arrays["fake_forces"] = rng.normal(size=(len(atoms), 3))
    # Note: mace_scf.data.update_keyspec_from_kwargs doesn't support a
    # total_charge_key mapping, so total_charge falls back to its 0.0
    # default regardless of the fixture file's real values -- fine here,
    # since a nonzero predicted charge against a 0.0 target still produces
    # a nonzero loss gradient (all this file's callers need).
    dataset = dataset_from_atoms(
        atoms_list,
        cutoff=float(model.r_max),
        atomic_multipoles_key="some_multipoles",
        fermi_level_key="the_VBM",
        energy_key="fake_energy",
        forces_key="fake_forces",
        atomic_multipoles_max_l=ATOMIC_MULTIPOLES_MAX_L,
    )
    shards = [
        dataset[start : start + shard_size]
        for start in range(0, num_configs, shard_size)
    ]
    batches = []
    for shard in shards:
        loader = mace.tools.torch_geometric.dataloader.DataLoader(
            dataset=shard, batch_size=shard_size, shuffle=False, drop_last=False
        )
        batches.append(next(iter(loader)))
    return batches


def make_loss_fn():
    """The production training loss (must stay collective-free, see loss.py)."""
    from mace_scf.electrostatics.loss import WeightedLoss

    return WeightedLoss(
        {
            "energy_per_atom": {"weight": 10.0},
            "forces": {"weight": 100.0},
            "atomic_multipoles": {"weight": 100.0},
            "total_charge_per_atom": {"weight": 1000.0},
        }
    )
