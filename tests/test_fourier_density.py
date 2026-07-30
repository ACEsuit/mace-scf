"""Tests for the Fourier density fitting port.

Covers:
  * WeightedFourierDensity + weighted_mean_squared_fourier_density (loss)
  * resize_fft round-trip and scaling
  * compute_k_vectors_dense_full_grid basic properties
  * assemble_rho_fftn_dense_grid: single-atom sanity, batched consistency
  * FourierDensityFittingBlock in "from_multipoles" mode: pred == ref when
    density_pred == density_ref
  * FourierDensityFittingBlock in "direct" mode: shapes line up
  * ExtAtomicData rho_fftn / rho_fftn_shape data plumbing round-trip
  * update_keyspec_from_kwargs picks up rho_fftn_key
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from ase import Atoms

from mace.data import KeySpecification
from mace.data.utils import config_from_atoms
from mace.tools import AtomicNumberTable

from mace_scf.data import ExtAtomicData, update_keyspec_from_kwargs
from mace_scf.electrostatics.fourier_density import FourierDensityFittingBlock
from mace_scf.electrostatics.loss import (
    _LOSS_FUNCTIONS,
    WeightedFourierDensity,
    weighted_mean_squared_fourier_density,
)
from mace_scf.electrostatics.utils import (
    assemble_rho_fftn_dense_grid,
    build_rho_fftn_dft_from_data,
    compute_k_vectors_dense_full_grid,
    resize_fft,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _cubic_cell(L: float, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cell = (torch.eye(3, dtype=dtype) * L).unsqueeze(0)
    rcell = 2 * math.pi * torch.linalg.inv(cell.mT)
    volume = torch.linalg.det(cell).abs()
    return cell, rcell, volume


# ---------------------------------------------------------------------------
# resize_fft
# ---------------------------------------------------------------------------


def test_resize_fft_shape_and_scale():
    F = torch.randn(4, 4, 4, dtype=torch.complex128)
    F2 = resize_fft(F, (6, 6, 6))
    assert F2.shape == (6, 6, 6)
    F_back = resize_fft(F2, (4, 4, 4))
    assert F_back.shape == (4, 4, 4)

    # DC (zero-freq) preservation of the padded version. After zero-padding
    # around the shifted-DC center, the DC bin (index 0 after unshift) is
    # scaled by new_prod/old_prod so the *unnormalized* integral is preserved.
    # Round-trip DC through pad→crop should return the original DC exactly.
    torch.testing.assert_close(F_back[0, 0, 0], F[0, 0, 0], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_k_vectors_dense_full_grid
# ---------------------------------------------------------------------------


def test_dense_full_grid_shapes_and_k0():
    cell, rcell, _ = _cubic_cell(15.0)
    cutoff = torch.tensor(1.5, dtype=cell.dtype)
    k, k2, mask, grid = compute_k_vectors_dense_full_grid(cutoff, cell, rcell)

    assert k.dim() == 3 and k.shape[-1] == 3
    assert k2.shape == k.shape[:2]
    assert mask.shape == k2.shape
    assert len(grid) == 3
    assert all(g > 0 for g in grid)

    # k=0 is at index 0 by fftfreq convention.
    assert torch.equal(k[0, 0], torch.zeros(3, dtype=cell.dtype))
    assert k2[0, 0].item() == 0.0
    # Inside-cutoff mask includes k=0 and excludes at least the max-|k| corner.
    assert bool(mask[0, 0].item()) is True
    assert bool(mask.all()) is False, "expected some k outside the cutoff"


# ---------------------------------------------------------------------------
# assemble_rho_fftn_dense_grid
# ---------------------------------------------------------------------------


def _basis_ones(n_graph: int, n_k: int, m_dim: int, dtype=torch.float64) -> torch.Tensor:
    """basis_fs = [n_graph, n_k, 1, m_dim, 2] with real part 1, imag 0."""
    b = torch.zeros(n_graph, n_k, 1, m_dim, 2, dtype=dtype)
    b[..., 0] = 1.0
    return b


def test_assemble_rho_dc_bin_sums_to_total_charge():
    """At k=0 (all zero k-vectors), rho(k) is just sum_i coef_i * basis(0), so
    if basis is real=1 and coef contains only l=0 (monopole), rho_real reduces
    to the total charge times (2pi)^3 / volume, and rho_imag == 0.
    """
    cell, _, volume = _cubic_cell(10.0)
    n_graph, n_k, m_dim = 1, 3, 1
    k_vectors = torch.zeros(n_graph, n_k, 3, dtype=torch.float64)
    positions = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 4.0, 0.0]], dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    coef = torch.tensor([[0.7], [-0.3]], dtype=torch.float64)  # total = 0.4
    basis_fs = _basis_ones(n_graph, n_k, m_dim)

    rho = assemble_rho_fftn_dense_grid(
        source_feats=coef,
        node_positions=positions,
        k_vectors=k_vectors,
        basis_fs=basis_fs,
        volume=volume,
        batch=batch,
    )
    expected_real = 0.4 * (2 * math.pi) ** 3 / volume.item()
    torch.testing.assert_close(rho[0, 0, 0].item(), expected_real, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(rho[0, 0, 1].item(), 0.0, rtol=0, atol=1e-12)


def test_assemble_rho_batched_equals_per_graph():
    """rho(k) for a batched call must agree with independent per-graph calls."""
    L = 12.0
    cell, _, volume = _cubic_cell(L)
    # Two graphs, 2 and 3 atoms respectively.
    positions_a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 2.0]], dtype=torch.float64)
    positions_b = torch.tensor(
        [[0.5, 0.5, 0.5], [-1.0, 0.0, 3.0], [2.0, 1.0, 0.0]], dtype=torch.float64
    )
    coef_a = torch.tensor([[1.0], [-0.5]], dtype=torch.float64)
    coef_b = torch.tensor([[0.3], [0.2], [-0.1]], dtype=torch.float64)

    n_k = 5
    torch.manual_seed(0)
    kvals = torch.randn(n_k, 3, dtype=torch.float64) * 0.3

    batch_positions = torch.cat([positions_a, positions_b], dim=0)
    batch_index = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    batch_coef = torch.cat([coef_a, coef_b], dim=0)
    volume2 = torch.stack([volume.squeeze(), volume.squeeze()])

    # Same k-vectors for both graphs (rank-2 batched layout).
    k_vectors = kvals.unsqueeze(0).expand(2, n_k, 3).contiguous()
    basis_fs = _basis_ones(2, n_k, 1)

    rho_batched = assemble_rho_fftn_dense_grid(
        source_feats=batch_coef,
        node_positions=batch_positions,
        k_vectors=k_vectors,
        basis_fs=basis_fs,
        volume=volume2,
        batch=batch_index,
    )

    # Per-graph reference (n_graph=1 calls).
    rho_a = assemble_rho_fftn_dense_grid(
        source_feats=coef_a,
        node_positions=positions_a,
        k_vectors=k_vectors[:1],
        basis_fs=basis_fs[:1],
        volume=volume,
        batch=torch.zeros(2, dtype=torch.long),
    )
    rho_b = assemble_rho_fftn_dense_grid(
        source_feats=coef_b,
        node_positions=positions_b,
        k_vectors=k_vectors[:1],
        basis_fs=basis_fs[:1],
        volume=volume,
        batch=torch.zeros(3, dtype=torch.long),
    )

    torch.testing.assert_close(rho_batched[0], rho_a[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(rho_batched[1], rho_b[0], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _fake_pred_ref(n_graph=2, n_k=8, seed=0):
    torch.manual_seed(seed)
    k2 = torch.rand(n_graph, n_k, dtype=torch.float64) * 3.0
    k2[:, 0] = 0.0
    mask = torch.ones(n_graph, n_k, dtype=torch.bool)
    rho_pred = torch.randn(n_graph, n_k, 2, dtype=torch.float64)
    rho_ref = rho_pred + 0.1 * torch.randn_like(rho_pred)
    return {
        "rho_fftn": rho_pred,
        "rho_fftn_dft": rho_ref,
        "k_vectors_normed_squared": k2,
        "k_vectors_mask": mask,
    }


def test_weighted_mean_squared_fourier_density_zero_when_equal():
    pred = _fake_pred_ref()
    pred["rho_fftn_dft"] = pred["rho_fftn"].clone()
    loss = weighted_mean_squared_fourier_density(None, pred)
    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=loss.dtype))


@pytest.mark.parametrize("form", ["poly", "exponential", "gaussian"])
def test_weighted_fourier_density_finite_and_positive(form):
    pred = _fake_pred_ref()
    loss = WeightedFourierDensity(form=form, alpha=1.5, k_min=0.5)(None, pred)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_loss_registry_has_new_entries():
    assert "rho_fftn" in _LOSS_FUNCTIONS
    assert "fourier_density" in _LOSS_FUNCTIONS


# ---------------------------------------------------------------------------
# FourierDensityFittingBlock
# ---------------------------------------------------------------------------


def _make_batch_of_two(dtype=torch.float64):
    """Two graphs of 2 & 3 atoms in a 10 Å cubic cell each; monopole density."""
    L = 10.0
    cell = (torch.eye(3, dtype=dtype) * L).unsqueeze(0).expand(2, 3, 3).contiguous()
    rcell = 2 * math.pi * torch.linalg.inv(cell.mT)
    volume = torch.linalg.det(cell).abs()

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0], [1.5, 0.0, 0.0],           # graph 0
            [0.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5],  # graph 1
        ],
        dtype=dtype,
    )
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    # (max_l+1)^2 = 1 for max_l=0 (monopoles)
    density = torch.tensor([[0.5], [-0.5], [0.3], [-0.1], [-0.2]], dtype=dtype)
    return cell, rcell, volume, positions, batch, density


def test_fourier_density_block_from_multipoles_matches_when_equal():
    """With mode='from_multipoles', if density_pred == density_ref, then
    rho_fftn == rho_fftn_dft exactly."""
    cell, rcell, volume, positions, batch, density = _make_batch_of_two()

    block = FourierDensityFittingBlock(
        mode="from_multipoles",
        atomic_multipoles_max_l=0,
        atomic_multipoles_smearing_width=1.0,
        kspace_cutoff=1.0,
    ).to(torch.float64)

    out = block(
        kspace_cutoff=torch.tensor(1.0, dtype=torch.float64),
        cell=cell,
        rcell=rcell,
        volume=volume,
        positions=positions,
        batch=batch,
        density_pred=density,
        density_ref=density,
    )
    assert set(out.keys()) == {
        "rho_fftn",
        "rho_fftn_dft",
        "k_vectors_normed_squared",
        "k_vectors_mask",
        "k_vectors_grid_shape",
    }
    torch.testing.assert_close(out["rho_fftn"], out["rho_fftn_dft"])
    assert out["k_vectors_normed_squared"].shape == out["rho_fftn"].shape[:2]
    assert out["k_vectors_mask"].dtype == torch.bool
    assert out["k_vectors_grid_shape"].shape == (3,)


def test_fourier_density_block_invalid_mode():
    with pytest.raises(ValueError):
        FourierDensityFittingBlock(
            mode="nope",
            atomic_multipoles_max_l=0,
            atomic_multipoles_smearing_width=1.0,
            kspace_cutoff=1.0,
        )


def test_fourier_density_block_direct_shapes():
    """direct mode: check the reference reshape lines up with the model grid."""
    cell, rcell, volume, positions, batch, density = _make_batch_of_two()

    block = FourierDensityFittingBlock(
        mode="direct",
        atomic_multipoles_max_l=0,
        atomic_multipoles_smearing_width=1.0,
        kspace_cutoff=1.0,
    ).to(torch.float64)

    # First figure out what grid_shape the block will use, so we can produce
    # a "raw DFT" reference with a *different* shape (5x5x5) and confirm
    # resize_fft is exercised.
    _, _, _, grid_shape_used = compute_k_vectors_dense_full_grid(
        torch.tensor(1.0, dtype=torch.float64), cell, rcell
    )

    raw_shape = (5, 5, 5)
    rho_a = torch.randn(*raw_shape, dtype=torch.complex128)
    rho_b = torch.randn(*raw_shape, dtype=torch.complex128)
    flat = torch.cat([rho_a.flatten(), rho_b.flatten()], dim=0)
    shapes = torch.tensor([list(raw_shape), list(raw_shape)], dtype=torch.long)

    out = block(
        kspace_cutoff=torch.tensor(1.0, dtype=torch.float64),
        cell=cell,
        rcell=rcell,
        volume=volume,
        positions=positions,
        batch=batch,
        density_pred=density,
        density_ref=density,
        data_rho_fftn=flat,
        data_rho_fftn_shape=shapes,
    )
    assert out["rho_fftn"].shape == out["rho_fftn_dft"].shape
    n_k_expected = grid_shape_used[0] * grid_shape_used[1] * grid_shape_used[2]
    assert out["rho_fftn"].shape[-2] == n_k_expected


def test_build_rho_fftn_dft_from_data_offset_layout():
    """Two graphs with different raw grid sizes → offsets computed correctly."""
    grid_shape = [6, 6, 6]
    shape_a = (4, 4, 4)
    shape_b = (5, 5, 5)
    rho_a = torch.randn(*shape_a, dtype=torch.complex128)
    rho_b = torch.randn(*shape_b, dtype=torch.complex128)
    flat = torch.cat([rho_a.flatten(), rho_b.flatten()], dim=0)
    shapes = torch.tensor([list(shape_a), list(shape_b)], dtype=torch.long)

    out = build_rho_fftn_dft_from_data(flat, shapes, grid_shape)
    assert out.shape == (2, 6 * 6 * 6, 2)
    # Both graphs must have finite output.
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ExtAtomicData data plumbing
# ---------------------------------------------------------------------------


def test_ext_atomic_data_rho_fftn_from_config():
    L = 6.0
    atoms = Atoms(
        symbols="OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[L, L, L],
        pbc=True,
    )
    rho = np.random.RandomState(42).randn(2, 4, 4, 4)
    atoms.info["rho_fftn_key_data"] = rho
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = np.zeros((3, 3))

    keyspec = KeySpecification()
    keyspec = update_keyspec_from_kwargs(
        keyspec, {"rho_fftn_key": "rho_fftn_key_data"}
    )
    cfg = config_from_atoms(atoms, key_specification=keyspec)

    z_table = AtomicNumberTable([1, 8])
    d = ExtAtomicData.from_config(cfg, z_table=z_table, cutoff=5.0, atomic_multipoles_max_l=0)

    assert d.rho_fftn is not None
    assert d.rho_fftn_shape is not None
    assert d.rho_fftn_shape.shape == (1, 3)
    assert tuple(d.rho_fftn_shape[0].tolist()) == (4, 4, 4)
    assert d.rho_fftn.numel() == 4 * 4 * 4
    assert d.rho_fftn.is_complex()


def test_ext_atomic_data_no_rho_fftn_fallback():
    """When rho_fftn isn't provided, ExtAtomicData still populates something
    minimal so downstream code doesn't have to check for None everywhere."""
    L = 6.0
    atoms = Atoms(
        symbols="OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[L, L, L],
        pbc=True,
    )
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = np.zeros((3, 3))

    keyspec = KeySpecification()
    cfg = config_from_atoms(atoms, key_specification=keyspec)
    z_table = AtomicNumberTable([1, 8])
    d = ExtAtomicData.from_config(cfg, z_table=z_table, cutoff=5.0, atomic_multipoles_max_l=0)

    assert d.rho_fftn is not None
    assert d.rho_fftn_shape is not None
    assert d.rho_fftn.is_complex()


def test_update_keyspec_includes_rho_fftn_key():
    keyspec = KeySpecification()
    keyspec = update_keyspec_from_kwargs(
        keyspec, {"rho_fftn_key": "MY_RHO_KEY"}
    )
    # Whatever the attribute name is, the key should now be discoverable.
    # KeySpecification exposes info_keys as a mapping; check for the raw value.
    joined = str(getattr(keyspec, "info_keys", "")) + str(vars(keyspec))
    assert "MY_RHO_KEY" in joined


# ---------------------------------------------------------------------------
# End-to-end: LocalSplitCharges with rho_fftn_mode="from_multipoles"
# ---------------------------------------------------------------------------


def _build_localsplit_model(rho_fftn_mode, atomic_multipoles_max_l=0):
    """Small random-initialized LocalSplitCharges suitable for a smoke test.

    Mirrors the shape of mace-scf's other integration tests (see
    tests/utils.py::make_polarizable_model_random) but keeps the model tiny.
    """
    import inspect

    import mace.modules
    import mace.tools
    from e3nn import o3
    from graph_longrange.energy import GTOElectrostaticEnergy
    from tests.utils import disable_e3nn_codegen, water_configs

    # mace-scf calls GTOElectrostaticEnergy(..., pbc_handling=...), which was
    # added to graph_longrange after the version installed on this machine.
    # Skip the integration test rather than surface a pre-existing env
    # mismatch that is unrelated to this port.
    if "pbc_handling" not in inspect.signature(GTOElectrostaticEnergy.__init__).parameters:
        pytest.skip(
            "graph_longrange in this env is too old (GTOElectrostaticEnergy has "
            "no pbc_handling arg); LocalSplitCharges cannot instantiate."
        )

    from mace_scf.electrostatics.localsources import LocalSplitCharges

    torch.manual_seed(0)
    np.random.seed(0)
    atoms = water_configs()[0]
    z_table = mace.tools.get_atomic_number_table_from_zs(sorted(set(atoms.get_atomic_numbers())))
    atomic_energies = np.array([1.0] * len(z_table))
    with disable_e3nn_codegen():
        model = LocalSplitCharges(
            r_max=3.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=mace.modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=mace.modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps("4x0e+4x1o"),
            MLP_irreps=o3.Irreps("8x0e"),
            atomic_energies=atomic_energies,
            avg_num_neighbors=5.0,
            atomic_numbers=z_table.zs,
            correlation=2,
            gate=mace.modules.gate_dict["silu"],
            formal_charges_from_data=True,
            radial_type="bessel",
            kspace_cutoff_factor=0.75,
            atomic_multipoles_max_l=atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=1.5,
            include_electrostatic_self_interaction=True,
            rho_fftn_mode=rho_fftn_mode,
        )
    return model, atoms, z_table


def _batch_from_atoms(atoms_list, z_table, atomic_multipoles_max_l=0):
    from ase.atoms import Atoms
    import mace.data
    import mace.tools
    import mace_scf.data as md

    keyspec = mace.data.KeySpecification()
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    # Populate charges so ExtAtomicData.from_config has something for
    # formal_charges_from_data=True.
    for a in atoms_list:
        a.arrays.setdefault("charges", np.zeros(len(a)))

    configs = mace.data.config_from_atoms_list(atoms_list, key_specification=keyspec)
    dataset = [
        md.ExtAtomicData.from_config(
            c,
            z_table=z_table,
            cutoff=3.0,
            atomic_multipoles_max_l=atomic_multipoles_max_l,
        )
        for c in configs
    ]
    loader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False
    )
    return next(iter(loader)).to_dict()


def test_localsplit_forward_emits_rho_fftn_from_multipoles():
    model, atoms, z_table = _build_localsplit_model(rho_fftn_mode="from_multipoles")
    data = _batch_from_atoms([atoms], z_table)

    out = model(data, training=False)
    for key in (
        "rho_fftn",
        "rho_fftn_dft",
        "k_vectors_normed_squared",
        "k_vectors_mask",
        "k_vectors_grid_shape",
    ):
        assert key in out and out[key] is not None, f"missing observable {key}"

    n_graph = data["ptr"].numel() - 1
    assert out["rho_fftn"].shape[0] == n_graph
    assert out["rho_fftn"].shape[-1] == 2
    assert out["rho_fftn"].shape == out["rho_fftn_dft"].shape
    assert out["k_vectors_normed_squared"].shape == out["rho_fftn"].shape[:2]
    assert out["k_vectors_mask"].shape == out["k_vectors_normed_squared"].shape
    assert out["k_vectors_mask"].dtype == torch.bool
    assert out["k_vectors_grid_shape"].shape == (3,)

    # Loss should be finite and differentiable through the model.
    weight_factor = 1 / out["k_vectors_normed_squared"].clamp_min(1e-8) ** 0.5
    weight_factor[:, 0] = 10.0
    weight_factor = weight_factor.clip(0.0, 10.0)
    loss = ((out["rho_fftn"] - out["rho_fftn_dft"]) ** 2 * weight_factor.unsqueeze(-1)).mean()
    assert torch.isfinite(loss)


def test_localsplit_disabled_leaves_rho_fftn_none():
    """rho_fftn_mode=None must not touch the outputs."""
    model, atoms, z_table = _build_localsplit_model(rho_fftn_mode=None)
    data = _batch_from_atoms([atoms], z_table)
    out = model(data, training=False)
    assert out["rho_fftn"] is None
    assert out["rho_fftn_dft"] is None
    assert out["k_vectors_normed_squared"] is None
