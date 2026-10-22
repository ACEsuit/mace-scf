"""LocalSplitCharges' oxidation_state_range must reach its charge-transfer blocks.

The chebyshev basis used to encode formal charges is bounded only inside the range,
so a formal charge of +7 (found in MatPES) reached T_8(1.75) ~ 5e3, giving charge 
transfers of ~1e3 e at random initialisation.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

import mace.modules
from mace_scf.electrostatics import LocalSplitCharges

RANGE = (-8.0, 8.0)


def _model(static_bond_transfer_block="OxidationDependentSymmetricPredictionSourceBlock"):
    torch.set_default_dtype(torch.float64)
    interaction = mace.modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
    return LocalSplitCharges(
        r_max=4.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction,
        interaction_cls_first=interaction,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=8.0,
        atomic_numbers=[1, 8],
        correlation=2,
        formal_charges_from_data=True,
        gate=torch.nn.functional.silu,
        atomic_multipoles_max_l=1,
        static_bond_transfer_block=static_bond_transfer_block,
        oxidation_state_mixer="SumOxidationStateMixer",
        oxidation_state_range=RANGE,
    )


def _encoding(block, q: float) -> torch.Tensor:
    """The Chebyshev features the block computes for a formal charge q."""
    x = (torch.tensor([[q]]) - block.oxidation_state_range[0]) * block.factor - 1.0
    return block.oxidation_basis(x)


def test_transfer_blocks_use_the_model_range():
    model = _model()
    torch.testing.assert_close(
        model.oxidation_state_mixer.ox_embedding.oxidation_state_range, torch.tensor(RANGE)
    )
    for block in model.lr_source_maps:
        torch.testing.assert_close(block.oxidation_state_range, torch.tensor(RANGE))


@pytest.mark.parametrize("q", [-8.0, -7.0, -4.5, 0.0, 4.5, 7.0, 8.0])
def test_encoding_stays_bounded_across_the_range(q):
    """Chebyshev polynomials are bounded by 1 only on [-1, 1]; with the old (-4, 4) range a
    formal charge of 7 gave T_8 ~ 5e3."""
    for block in _model().lr_source_maps:
        assert _encoding(block, q).abs().max().item() <= 1.0 + 1e-12


def test_blocks_without_an_oxidation_embedding_still_build():
    model = _model("NoFieldSymmetricPredictionSourceBlock")
    assert not hasattr(model.lr_source_maps[0], "oxidation_state_range")


def test_transfer_blocks_require_the_range():
    """No silent (-4, 4) default: a caller that forgets the range must fail loudly."""
    from mace_scf.electrostatics.bonded_blocks import static_bond_transfer_blocks

    irreps = o3.Irreps("8x0e + 8x1o")
    for cls in static_bond_transfer_blocks.values():
        with pytest.raises(TypeError, match="oxidation_state_range"):
            cls(
                node_feats_irreps=irreps,
                edge_attrs_irreps=o3.Irreps.spherical_harmonics(2),
                edge_feats_irreps=o3.Irreps("8x0e"),
                target_irreps=irreps,
                max_l=1,
                num_elements=2,
            )


def _collections(train_charges=(), valid_charges=()):
    from types import SimpleNamespace

    def configs(charge_lists):
        return [SimpleNamespace(properties={"charges": np.asarray(q)}) for q in charge_lists]

    return SimpleNamespace(train=configs(train_charges), valid=configs(valid_charges), tests=[])


def test_loading_warns_about_charges_outside_the_range(caplog):
    from mace_scf.utils.load_data import check_formal_charges_in_oxidation_state_range

    collections = _collections(train_charges=[[1.0, -2.0], [7.0, -1.0]], valid_charges=[[4.0, -4.0]])
    with caplog.at_level("WARNING"):
        check_formal_charges_in_oxidation_state_range(collections, (-4.0, 4.0))
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1  # valid lies on the boundary, which is in range
    assert "1/4 atoms" in warnings[0] and "split=train" in warnings[0]
    assert "observed range=(-2, 7)" in warnings[0]


def test_loading_is_silent_when_charges_are_in_range(caplog):
    from mace_scf.utils.load_data import check_formal_charges_in_oxidation_state_range

    collections = _collections(train_charges=[[7.0, -8.0]])
    with caplog.at_level("WARNING"):
        check_formal_charges_in_oxidation_state_range(collections, RANGE)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
