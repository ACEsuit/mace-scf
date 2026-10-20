"""Charge response to the Fermi level in the FixedPoint field-update block.

In constant-charge SCF (fixed_point_scf.converge_constant_charge) the Fermi level mu enters
the update block only through the l=0 field features, and dq_i/dmu drives both the Fukui
redistribution and the Newton step on mu. These tests pin down how q_i depends on mu.
"""

import copy
import functools
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

import mace.modules
import mace.tools
import mace.tools.torch_geometric
from mace.tools import torch_tools
from mace.tools.scatter import scatter_sum

import mace_scf.electrostatics as electrostatics
from mace_scf.electrostatics import field_blocks
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions
from tests.paths import reference_model, require_file
from tests.utils import dataset_from_atoms, disable_e3nn_codegen, seed_torch


FORMAL_CHARGES_KEY = "formal_oxidation_states"

# Two water molecules far enough apart to keep the cell sparse.
ATOMIC_NUMBERS = [1, 8]


def water_pair_atoms() -> Atoms:
    atoms = Atoms(
        "OH2OH2",
        positions=[
            [1.0, 1.0, 1.0],
            [1.9, 1.0, 1.0],
            [0.7, 1.9, 1.0],
            [4.0, 4.0, 4.0],
            [4.9, 4.0, 4.0],
            [3.7, 4.9, 4.0],
        ],
        cell=np.diag([7.0, 7.0, 7.0]),
        pbc=True,
    )
    atoms.arrays[FORMAL_CHARGES_KEY] = np.array([-2.0, 1.0, 1.0, -2.0, 1.0, 1.0])
    return atoms


def build_batch(atoms=None, n_graphs: int = 2, cutoff: float = 3.0):
    """A batched data dict with `formal_oxidation_states` present."""
    # The dataset tensors take the current default dtype, so pin it here as well as in
    # build_model: otherwise whichever of the two a test calls first decides the dtype,
    # and calling build_batch first yields float32 data against a float64 model.
    torch_tools.set_default_dtype("float64")

    if atoms is None:
        atoms = water_pair_atoms()
    dataset = dataset_from_atoms(
        [atoms] * n_graphs,
        cutoff=cutoff,
        charges_key=FORMAL_CHARGES_KEY,
        atomic_multipoles_max_l=1,
    )
    loader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=n_graphs, shuffle=False
    )
    return next(iter(loader))


def build_fixed_point_core(seed: int = 4, num_polynomial_cutoff: int = 5, **overrides):
    """A small random-weight FixedPointCore."""
    torch_tools.set_default_dtype("float64")
    seed_torch(seed)
    z_table = mace.tools.get_atomic_number_table_from_zs(ATOMIC_NUMBERS)
    interaction_cls = mace.modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ]
    # Overrides replace the defaults below (e.g. field_feature_widths or
    # fixedpoint_update_config) rather than being passed alongside them.
    kwargs = dict(
        r_max=3.0,
        num_bessel=8,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=3,
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls,
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
        atomic_multipoles_max_l=1,
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
        pbc_handling="pbc",
    )
    kwargs.update(overrides)
    with disable_e3nn_codegen():
        return electrostatics.FixedPointCore(**kwargs)


# A trained FixedPoint model for the checks on real weights. Defaults to the regression
# fixture; point MACE_SCF_FIXEDPOINT_MODEL at any other FixedPoint .model to use that.
TRAINED_MODEL_PATH = Path(
    os.environ.get(
        "MACE_SCF_FIXEDPOINT_MODEL", str(reference_model("fixedpoint_onebodylinear"))
    )
)
# cuequivariance models need a GPU: run with MACE_DEVICE=cuda.
DEVICE = os.environ.get("MACE_DEVICE", "cpu")
MU_STEPS = (-2, -1, 0, 1, 2)
FIELD_FEATURE_WIDTHS = [1.5, 3.0]
# l=0 then l=1 norms for FIELD_FEATURE_WIDTHS, as used by fit_matpes_fixedpoint_vanilla.
MATPES_FIELD_FEATURE_NORMS = [16.85128938, 11.29358665, 0.45036201, 0.25651795]


def _update_config(cls, **options):
    return {
        "type": cls,
        "potential_embedding_cls": field_blocks.BiasedLinearPotentialEmbedding,
        "nonlinearity_cls": field_blocks.NoNonLinearity,
        **options,
    }


@functools.lru_cache(maxsize=None)
def _base_model():
    """A small FixedPointCore with the default update block. Building one takes minutes on
    CPU, so it is built once; other variants are copies with a different update block."""
    return build_fixed_point_core(field_feature_widths=FIELD_FEATURE_WIDTHS)


def _with_update_block(model, block):
    model = copy.deepcopy(model)
    model.field_dependent_charges_map = block
    return model.to(DEVICE)


def _trained_model():
    require_file(TRAINED_MODEL_PATH, "Trained FixedPoint model")
    torch_tools.set_default_dtype("float64")
    model = torch.load(TRAINED_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = model.to(torch.float64)
    block = model.field_dependent_charges_map
    if not (
        type(block) is field_blocks.OneBodyVariableUpdate
        and type(block.nonlinearity) is field_blocks.NoNonLinearity
    ):
        pytest.skip(
            f"{TRAINED_MODEL_PATH.name} uses {type(block).__name__} with "
            f"{type(getattr(block, 'nonlinearity', None)).__name__}, not the linear default"
        )
    dataset = dataset_from_atoms(
        [water_pair_atoms()] * 2,
        cutoff=float(model.r_max),
        z_table=mace.tools.AtomicNumberTable([int(z) for z in model.atomic_numbers]),
        atomic_multipoles_max_l=int(model.coulomb_energy.density_max_l),
    )
    loader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False
    )
    return model, next(iter(loader)).to(DEVICE).to_dict()


def _charges_at_fermi_level(model, data, local_state, density, fermi_level):
    """q_i(mu) for one update at fixed input density, as converge_constant_charge does it."""
    node_fermi = torch.index_select(fermi_level, 0, data["batch"])
    fermi_features = model.features_from_fermi_level_nodewise(
        data["batch"], local_state.positions, node_fermi
    )
    field_dep, _ = model.scf_step(
        data,
        local_state,
        charge_density_in=density,
        total_charges=density,
        fermi_level_features=fermi_features,
    )
    return field_dep[:, 0]


def _assert_charge_linear_in_fermi_level(model, data):
    """q_i(mu) is affine (zero second differences) with the same dq_i/dmu at every mu.
    Returns dq_i/dmu per atom."""
    model.eval()
    local_state = model.local_part(data)
    density = local_state.field_independent_charge_density.detach()
    num_graphs = data["ptr"].numel() - 1
    mu0 = model.fermi_level_offset.expand(num_graphs).clone()

    def charges(mu):
        return _charges_at_fermi_level(model, data, local_state, density, mu)

    for delta in (0.1, 1.0):
        q = torch.stack([charges(mu0 + t * delta).detach() for t in MU_STEPS])
        scale = 1.0 + q.abs().max()
        second_differences = q[2:] - 2 * q[1:-1] + q[:-2]
        assert torch.allclose(
            second_differences, torch.zeros_like(second_differences), atol=1e-10 * scale
        ), f"q(mu) not affine for step {delta}: max |d2q| = {second_differences.abs().max()}"

    atoms = torch.arange(data["batch"].numel(), device=data["batch"].device)
    # Each q_i depends only on its own graph's mu: dq_i/dmu is that column of the Jacobian.
    slopes = torch.stack(
        [torch.autograd.functional.jacobian(charges, mu0 + t)[atoms, data["batch"]] for t in MU_STEPS]
    )
    assert torch.allclose(slopes, slopes[:1].expand_as(slopes), atol=1e-10, rtol=1e-10)

    dq_dmu = slopes[0]
    per_graph = scatter_sum(src=dq_dmu, index=data["batch"], dim=0, dim_size=num_graphs)
    print(
        f"\ndq_i/dmu per atom: {dq_dmu.tolist()}\n"
        f"fraction > 0: {(dq_dmu > 0).double().mean().item():.2f}, "
        f"per-graph sum: {per_graph.tolist()}"
    )
    return dq_dmu


def test_existing_block_charge_is_linear_in_mu_random_weights():
    model = _base_model().to(DEVICE)
    _assert_charge_linear_in_fermi_level(model, build_batch().to(DEVICE).to_dict())


def test_existing_block_charge_is_linear_in_mu_trained_model():
    # The default device matters for cuequivariance's naive kernels, which create
    # index tensors at call time.
    with torch.device(DEVICE):
        model, data = _trained_model()
        _assert_charge_linear_in_fermi_level(model, data)


# ---------------------------------------------------------------------------------------
# OneBodyMonotoneChargeUpdate
# ---------------------------------------------------------------------------------------

# softness_max: None is softplus, a number is softness_max * sigmoid
SOFTNESS_MAX = (None, 1.0, 2.0)
SOFTNESS_INIT = {None: 0.3, 1.0: 0.5, 2.0: 1.0}


def _block(cls, seed=4, **options):
    """An update block with the irreps of _base_model's, built directly (fast)."""
    ref = _base_model().field_dependent_charges_map
    seed_torch(seed)
    with disable_e3nn_codegen():
        block = cls(
            node_attrs_irreps=ref.node_attrs_irreps,
            node_feats_irreps=ref.node_feats_irreps,
            edge_attrs_irreps=ref.edge_attrs_irreps,
            edge_feats_irreps=ref.edge_feats_irreps,
            target_irreps=ref.target_irreps,
            hidden_irreps=ref.hidden_irreps,
            avg_num_neighbors=ref.avg_num_neighbors,
            potential_irreps=ref.potential_irreps,
            charges_irreps=ref.charges_irreps,
            field_norm_factor=float(ref.field_norm_factor),
            potential_embedding_cls=field_blocks.BiasedLinearPotentialEmbedding,
            nonlinearity_cls=field_blocks.NoNonLinearity,
            **options,
        )
    return block.to(DEVICE)


def _monotone_block(softness_max, seed=4):
    """A monotone block with W and b away from their initial values."""
    block = _block(
        field_blocks.OneBodyMonotoneChargeUpdate,
        seed,
        softness_max=softness_max,
        softness_init=SOFTNESS_INIT[softness_max],
    )
    with torch.no_grad():
        block.softness_linear.weight.normal_()
        block.softness_bias.normal_()
    return block


def _block_inputs(block, num_nodes=7, seed=0):
    generator = torch.Generator().manual_seed(seed)
    elements = torch.randint(0, block.node_attrs_irreps.dim, (num_nodes,), generator=generator)
    node_attrs = torch.nn.functional.one_hot(elements, block.node_attrs_irreps.dim).double()
    node_feats = torch.randn(num_nodes, block.node_feats_irreps.dim, generator=generator).double()
    potentials = torch.randn(num_nodes, block.potential_irreps.dim, generator=generator).double()
    return node_attrs.to(DEVICE), node_feats.to(DEVICE), potentials.to(DEVICE)


def _update(block, node_attrs, node_feats, potentials):
    unused = torch.empty(0, device=potentials.device)
    return block(node_attrs, node_feats, unused, unused, unused, potentials, unused, unused)


@pytest.mark.parametrize("softness_max", SOFTNESS_MAX)
def test_monotone_block_is_linear_in_v0_with_slope_minus_softness(softness_max):
    """Moving only the l=0 features moves q0 linearly, by exactly -sum S dv, whatever the
    l>=1 features are."""
    high = float("inf") if softness_max is None else softness_max
    for seed in (4, 5, 6):
        block = _monotone_block(softness_max, seed=seed)
        node_attrs, node_feats, potentials = _block_inputs(block, seed=seed)
        n0 = block.num_scalar_potentials
        softness = block.softness(node_attrs, node_feats).detach()
        assert torch.all(softness > 0.0) and torch.all(softness < high)

        direction = torch.zeros_like(potentials)
        direction[:, :n0] = torch.randn_like(potentials[:, :n0])
        expected_step = -(softness * direction[:, :n0]).sum(-1)
        for higher in (potentials[:, n0:], 10.0 * torch.randn_like(potentials[:, n0:])):
            base = torch.cat([potentials[:, :n0], higher], dim=-1)
            q = torch.stack(
                [_update(block, node_attrs, node_feats, base + t * direction)[:, 0] for t in MU_STEPS]
            ).detach()
            second_differences = q[2:] - 2 * q[1:-1] + q[:-2]
            assert torch.allclose(second_differences, torch.zeros_like(second_differences), atol=1e-10)
            assert torch.allclose(q[1:] - q[:-1], expected_step.expand_as(q[1:]), atol=1e-10)


@pytest.mark.parametrize("softness_max", SOFTNESS_MAX)
def test_monotone_block_initial_softness(softness_max):
    block = _block(
        field_blocks.OneBodyMonotoneChargeUpdate,
        softness_max=softness_max,
        softness_init=SOFTNESS_INIT[softness_max],
    )
    node_attrs, node_feats, _ = _block_inputs(block)
    softness = block.softness(node_attrs, node_feats)
    # W h_i spreads S around softness_init, which sets the bias b
    assert not torch.allclose(softness, softness[0, 0].expand_as(softness))
    with torch.no_grad():
        block.softness_linear.weight.zero_()
    softness = block.softness(node_attrs, node_feats)
    assert torch.allclose(softness, torch.full_like(softness, SOFTNESS_INIT[softness_max]))


@pytest.mark.parametrize(
    "softness_max, softness_init",
    [(0.0, 0.1), (-1.0, 0.1), (None, 0.0), (None, -1.0), (1.0, 1.0), (2.0, 2.5)],
)
def test_monotone_block_rejects_bad_options(softness_max, softness_init):
    with pytest.raises(ValueError):
        _block(
            field_blocks.OneBodyMonotoneChargeUpdate,
            softness_max=softness_max,
            softness_init=softness_init,
        )


@pytest.mark.parametrize("softness_max", SOFTNESS_MAX)
def test_monotone_block_keeps_parent_dipoles(softness_max):
    block = _monotone_block(softness_max)
    parent = _block(field_blocks.OneBodyVariableUpdate, seed=99)
    parent.load_state_dict(
        {k: v for k, v in block.state_dict().items() if not k.startswith("softness")}
    )
    node_attrs, node_feats, potentials = _block_inputs(block)
    ours = _update(block, node_attrs, node_feats, potentials)
    theirs = _update(parent, node_attrs, node_feats, potentials)
    assert torch.allclose(ours[:, 1:], theirs[:, 1:], atol=1e-12)
    assert not torch.allclose(ours[:, 0], theirs[:, 0])


@pytest.mark.parametrize("softness_max", SOFTNESS_MAX)
def test_monotone_model_charge_decreases_with_mu(softness_max):
    """End to end through features_from_fermi_level_nodewise + scf_step, as in
    converge_constant_charge: q_i is affine in mu with dq_i/dmu < 0 on every atom."""
    model = _with_update_block(_base_model(), _monotone_block(softness_max))
    dq_dmu = _assert_charge_linear_in_fermi_level(model, build_batch().to(DEVICE).to_dict())
    assert torch.all(dq_dmu < 0), dq_dmu


@functools.lru_cache(maxsize=None)
def _normalised_monotone_model():
    """Built through fixedpoint_update_config, with the MatPES fits' feature norms."""
    return build_fixed_point_core(
        field_feature_widths=FIELD_FEATURE_WIDTHS,
        field_feature_norms=MATPES_FIELD_FEATURE_NORMS,
        fixedpoint_update_config=_update_config(field_blocks.OneBodyMonotoneChargeUpdate),
    )


@pytest.mark.parametrize("softness_max", SOFTNESS_MAX)
def test_monotone_model_constant_charge_scf_runs(softness_max):
    """Constant-charge SCF converges from the default softness_init on the default
    features (own Coulomb potential excluded), with the l=0 feature norms of the MatPES
    FixedPoint fits. The norms matter: with norms of 1 the own-site l=0 feature per unit
    charge is the 7 A cell's Madelung term (about -16.5), and the SCF diverges for S above
    about 0.035."""
    model = _with_update_block(
        _normalised_monotone_model(),
        _block(field_blocks.OneBodyMonotoneChargeUpdate, softness_max=softness_max),
    )
    model.eval()
    data = build_batch().to(DEVICE).to_dict()
    runner = FixedPointSCFRunner(
        FixedPointSCFOptions(
            num_scf_steps=100,
            scf_tolerance=1e-8,
            mixing_parameter=0.3,
            constant_charge=True,
            initial_density="local_guess",
            initial_fermi_level="zero",
        )
    )
    local_state = model.local_part(data)
    result = runner.converge(
        model=model,
        data=data,
        local_state=local_state,
        initial_density=runner.get_initial_density(local_state, data),
        initial_fermi_level=runner.get_initial_fermi(model, local_state, data),
    )
    total_charge = scatter_sum(
        src=result.density[:, 0],
        index=data["batch"],
        dim=0,
        dim_size=data["total_charge"].numel(),
    )
    assert torch.all(torch.isfinite(result.density))
    assert result.status == "converged", (result.status, result.terminated_step)
    assert torch.allclose(total_charge, data["total_charge"], atol=1e-3), total_charge


def _field_block_groups(model):
    from types import SimpleNamespace

    from mace_scf.utils.run_train_utils import get_param_options

    args = SimpleNamespace(
        model="FixedPointCore",
        weight_decay=1e-8,
        field_block_weight_decay=0.1,
        local_charges_weight_decay=0.0,
        lr=0.01,
        amsgrad=False,
        beta=0.9,
        beta_two=0.999,
    )
    groups = get_param_options(model, args)["params"]
    return [g for g in groups if g["name"] == "field_dependent_charges_map"]


def test_softness_bias_is_not_weight_decayed():
    model = _with_update_block(_base_model(), _monotone_block(None))
    block = model.field_dependent_charges_map
    groups = _field_block_groups(model)
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0]
    assert groups[1]["params"] == [block.softness_bias]
    decayed = {id(p) for p in groups[0]["params"]}
    others = {id(p) for name, p in block.named_parameters() if name != "softness_bias"}
    assert decayed == others


def test_default_block_param_group_unchanged():
    model = _base_model()
    groups = _field_block_groups(model)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.1
    assert {id(p) for p in groups[0]["params"]} == {
        id(p) for p in model.field_dependent_charges_map.parameters()
    }
