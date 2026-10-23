"""Checks that every model parameter gets a gradient, for each training
mode/loss combination listed below (MODE_BRANCHES x LOSSES).

Why this matters: our distributed training uses PyTorch's
DistributedDataParallel with `find_unused_parameters=False`. That's a
performance setting that assumes *every* parameter gets a gradient on
*every* backward pass. If some parameter is silently never used, training
either hangs or crashes with a "parameters that were not used in producing
loss" error -- and whether that happens can depend on which mode/loss
combination is active, so every combination needs checking.

A parameter counts as "used" if `param.grad is not None` after
`backward()` (PyTorch's own definition -- a zero-valued gradient still
counts as used).

If a test case here fails, it means a parameter really did go unused for
that combination. Don't just relax the assertion --
`find_unused_parameters=False` is no longer safe until that's fixed.
"""

import numpy as np
import pytest
import torch

import mace.modules
import mace.tools
from e3nn import o3

from mace_scf import electrostatics
from mace_scf.electrostatics import field_blocks
from mace_scf.electrostatics.fixed_point_state import (
    FixedPointSCFOptions,
    FixedPointTrainingOptions,
)
from mace_scf.electrostatics.loss import WeightedLoss
from mace_scf.utils import model_training_wrappers
from mace_scf.utils.model_training_wrappers import FixedPointWrapper

from . import fixtures
from ..utils import disable_e3nn_codegen, seed_torch


LOSSES = {
    # The loss weights used in a real training run
    # (examples/al_fit_multi_gpu/config_quick.yaml).
    "full": {
        "atomic_multipoles": {"weight": 100.0},
        "total_charge_per_atom": {"weight": 1000.0},
        "energy_per_atom": {"weight": 10.0},
        "forces": {"weight": 100.0},
    },
    # A loss that doesn't mention density/charge outputs at all. This is
    # the case most likely to catch a parameter that's only reachable
    # through those outputs, since nothing else would use it.
    "energy_only": {"energy_per_atom": {"weight": 10.0}},
}

# Each entry: (fixed-point mode, whether to force the data-dependent
# fallback to unroll_scf).
MODE_BRANCHES = [
    ("direct", "normal"),
    ("unroll_scf", "normal"),
    ("implicit", "normal"),
    ("implicit", "fallback"),
    ("linearize_solve", "normal"),
    ("linearize_solve", "fallback"),
]


DEFAULT_SCF_OPTIONS = FixedPointSCFOptions(
    num_scf_steps=100,
    scf_tolerance=1e-9,
    constant_charge=False,
    mixing_parameter=0.5,
    initial_density="from_data",
    initial_fermi_level="from_data",
    use_autograd_forces=True,
)


def build_wrapper(model, mode, scf_options=DEFAULT_SCF_OPTIONS):
    return FixedPointWrapper(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        output_args=fixtures.OUTPUT_ARGS,
        training_options=FixedPointTrainingOptions(
            mode=mode,
            scf=None if mode == "direct" else scf_options,
            linear_solve="inverse",
        ),
    )


def unused_parameters(model, wrapper, loss_fn, batch):
    """Names of parameters that received no gradient, DDP's definition."""
    model.zero_grad(set_to_none=True)
    output = wrapper(batch.to_dict(), training=True)
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()
    unused = {
        name
        for name, param in model.named_parameters()
        # torchopt registers batch_positions onto the model during implicit
        # differentiation; it is not a real parameter and DDP never sees it.
        if name != "batch_positions" and param.grad is None
    }
    if hasattr(model, "batch_positions"):
        del model.batch_positions
    return unused


def force_scf_divergence(wrapper):
    """Make the SCF report divergence, taking the fallback to unroll_scf.

    The fallbacks are data-dependent and not otherwise reachable on demand.
    """
    real_converge = wrapper._runner.converge

    def diverged(*args, **kwargs):
        return real_converge(*args, **kwargs)._replace(
            status="diverged", terminated_step=3
        )

    wrapper._runner.converge = diverged


@pytest.fixture(scope="module")
def fixture_model():
    return fixtures.build_model()


@pytest.fixture(scope="module")
def batches(fixture_model):
    return fixtures.shard_batches(fixture_model)


@pytest.mark.parametrize("loss_name", sorted(LOSSES))
@pytest.mark.parametrize("mode,branch", MODE_BRANCHES)
def test_no_unused_parameters(fixture_model, batches, loss_name, mode, branch, monkeypatch):
    if mode == "implicit":
        pytest.importorskip("torchopt")
    loss_fn = WeightedLoss(LOSSES[loss_name])

    if branch == "fallback" and mode == "linearize_solve":
        def failing_solve(*args, **kwargs):
            raise RuntimeError("forced linear solve failure")

        monkeypatch.setattr(
            model_training_wrappers, "linearize_and_solve_density", failing_solve
        )

    # Check every batch, not just one: a parameter that's unused only for
    # some batches must still be caught, since DDP requires every
    # parameter to get a gradient on every backward pass, not just some.
    for batch in batches:
        wrapper = build_wrapper(fixture_model, mode)
        if branch == "fallback" and mode == "implicit":
            force_scf_divergence(wrapper)
        assert unused_parameters(fixture_model, wrapper, loss_fn, batch) == set()


def build_production_shaped_model():
    """Closer to a real fit than the fixture model.

    Two interactions and two field-feature widths, as in
    examples/al_fit_multi_gpu/submit_4gpu.sh.
    """
    seed_torch(fixtures.SEED)
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
            num_interactions=2,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("8x0e+8x1o"),
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
            atomic_multipoles_max_l=fixtures.ATOMIC_MULTIPOLES_MAX_L,
            atomic_multipoles_smearing_width=1.5,
            field_feature_max_l=1,
            field_feature_widths=[1.5, 3.0],
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
    return model


@pytest.fixture(scope="module")
def production_model():
    return build_production_shaped_model()


@pytest.fixture(scope="module")
def production_batch(production_model):
    return fixtures.shard_batches(production_model)[0]


@pytest.mark.parametrize("initial_density", ["from_data", "local_guess"])
@pytest.mark.parametrize("mode", ["unroll_scf", "linearize_solve"])
def test_no_unused_parameters_production_shape(
    production_model, production_batch, mode, initial_density
):
    """The SCF options that change which code paths the density flows through.

    `local_guess` in particular reaches a readout that `from_data` bypasses.
    """
    scf_options = FixedPointSCFOptions(
        num_scf_steps=100,
        scf_tolerance=1e-9,
        constant_charge=True,
        mixing_parameter=0.5,
        initial_density=initial_density,
        initial_fermi_level="from_data",
        use_autograd_forces=True,
    )
    wrapper = build_wrapper(production_model, mode, scf_options)
    loss_fn = WeightedLoss(LOSSES["energy_only"])
    assert unused_parameters(production_model, wrapper, loss_fn, production_batch) == set()
