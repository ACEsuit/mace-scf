"""Energies, forces and dipoles must be continuous when a pair crosses r_max.

RealAgnosticResidualNonLinearInteractionBlock (and the Att block) feed element embeddings into the edge MLP, so their
edge weights are not zero at r_max unless the block applies the cutoff itself, as MACE does with apply_cutoff=False.
mace-scf used to drop that cutoff, and a trained LSC model jumped by meV and 1e-2 polarization quanta whenever an atom
pair crossed r_max.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

import mace.modules
from mace_scf.calculators.localsources import MACELocalCharges, MACELocalSplitCharges
from mace_scf.electrostatics import LocalCharges, LocalSplitCharges
from mace_scf.electrostatics.compiled_localsources import build_compiled_local_source_evaluator

R_MAX = 4.0
NONLINEAR = "RealAgnosticResidualNonLinearInteractionBlock"
EPS = 1e-9  # small enough that the smooth change across the crossing is ~1e-9


def _model(model_cls, interaction_name, apply_cutoff):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    interaction = mace.modules.interaction_classes[interaction_name]
    kwargs = dict(
        r_max=R_MAX,
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
        gate=torch.nn.functional.silu,
        atomic_multipoles_max_l=1,
        apply_cutoff=apply_cutoff,
    )
    if model_cls is LocalSplitCharges:
        kwargs.update(
            formal_charges_from_data=True,
            static_bond_transfer_block="OxidationDependentSymmetricPredictionSourceBlock",
            oxidation_state_mixer="SumOxidationStateMixer",
            oxidation_state_range=(-2.0, 2.0),
        )
    return model_cls(**kwargs)


def _atoms(distance):
    """O-H plus a second H at `distance` from the first H, out of range of the O."""
    atoms = Atoms(
        "OHH",
        positions=[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [6.0 + distance, 5.0, 5.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    atoms.arrays["charges"] = np.zeros(3)
    atoms.info["total_charge"] = 0.0
    return atoms


def _evaluate(model_path, calc_cls, atoms, compiled):
    calc = calc_cls(model_path=str(model_path), device="cpu", pbc_handling="pbc")
    data = next(iter(calc._build_data_loader(atoms))).to(calc.device).to_dict()
    if compiled:
        evaluator = build_compiled_local_source_evaluator(calc.model, pbc_handling="pbc", enabled=False)
        out = evaluator.evaluate(data)
    else:
        out = calc.model(data, compute_force=True)
    return {k: out[k].detach().cpu().numpy() for k in ("energy", "forces", "dipole")}


def _across_r_max(tmp_path, model, calc_cls, compiled):
    path = tmp_path / "model.model"
    torch.save(model, path)
    inside = _evaluate(path, calc_cls, _atoms(R_MAX - EPS), compiled)
    outside = _evaluate(path, calc_cls, _atoms(R_MAX + EPS), compiled)
    return {k: np.abs(inside[k] - outside[k]).max() for k in inside}


MODELS = [(LocalSplitCharges, MACELocalSplitCharges), (LocalCharges, MACELocalCharges)]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "interaction,apply_cutoff",
    [("RealAgnosticResidualInteractionBlock", True), ("RealAgnosticResidualInteractionBlock", False), (NONLINEAR, False)],
)
@pytest.mark.parametrize("model_cls,calc_cls", MODELS)
def test_continuous_at_r_max(tmp_path, model_cls, calc_cls, interaction, apply_cutoff, compiled):
    jumps = _across_r_max(tmp_path, _model(model_cls, interaction, apply_cutoff), calc_cls, compiled)
    for key, jump in jumps.items():
        assert jump < 1e-7, (key, jump)


@pytest.mark.parametrize("model_cls,calc_cls", MODELS)
def test_nonlinear_block_with_folded_cutoff_jumps(tmp_path, model_cls, calc_cls):
    """As in MACE, the nonlinear block needs apply_cutoff=False; with the cutoff folded into the features it jumps."""
    jumps = _across_r_max(tmp_path, _model(model_cls, NONLINEAR, apply_cutoff=True), calc_cls, compiled=False)
    assert jumps["energy"] > 1e-4, jumps
