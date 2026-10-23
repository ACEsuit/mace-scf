from types import SimpleNamespace

from ase import Atoms
from ase.io import write
import numpy as np
import pytest
import torch

from mace.data import KeySpecification
from mace.data.utils import config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric

from mace_scf.data.new_atomic_data import ExtAtomicData
from mace_scf.electrostatics.loss import weighted_mean_squared_stress
from mace_scf.utils.eval_metrics import MaceSCFLoss
from mace_scf.utils.load_data import (
    check_explicit_component_weights,
    check_explicit_component_weights_for_paths,
)


STRESS = np.arange(1.0, 10.0).reshape(3, 3)


def _atoms(*, pbc=(True, True, True), include_stress=True, stress_weight=None):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]],
        cell=np.diag([4.0, 4.0, 12.0]),
        pbc=pbc,
    )
    atoms.info["REF_energy"] = 0.0
    if include_stress:
        atoms.info["REF_stress"] = STRESS.copy()
    if stress_weight is not None:
        atoms.info["config_stress_weight"] = np.asarray(stress_weight, dtype=float)
    return atoms


def _write_atoms(path, **kwargs):
    write(path, _atoms(**kwargs), format="extxyz")
    return path


def _keyspec(stress_key="REF_stress"):
    return SimpleNamespace(info_keys={"stress": stress_key})


def test_explicit_stress_weight_required_when_stress_is_present(tmp_path):
    xyz_path = _write_atoms(tmp_path / "missing_weight.xyz")

    with pytest.raises(ValueError, match="without explicit config_stress_weight"):
        check_explicit_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_stress_weight_accepts_three_component_mask(tmp_path):
    xyz_path = _write_atoms(tmp_path / "weighted.xyz", stress_weight=[1.0, 1.0, 1.0])

    check_explicit_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_stress_weight_rejects_scalar_weight(tmp_path):
    xyz_path = _write_atoms(tmp_path / "scalar_weight.xyz", stress_weight=1.0)

    with pytest.raises(ValueError, match="must be a 3-vector"):
        check_explicit_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_stress_weight_check_ignores_missing_stress(tmp_path):
    xyz_path = _write_atoms(tmp_path / "no_stress.xyz", include_stress=False)

    check_explicit_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_stress_weight_check_ignores_disabled_stress_key(tmp_path):
    xyz_path = _write_atoms(tmp_path / "disabled_stress.xyz")

    check_explicit_component_weights(xyz_path, _keyspec("none"), "train")
    check_explicit_component_weights(
        xyz_path, SimpleNamespace(info_keys={}), "train"
    )


def test_explicit_stress_weight_check_covers_train_valid_and_test_paths(tmp_path):
    train_path = _write_atoms(tmp_path / "train.xyz", stress_weight=[1.0, 1.0, 1.0])
    valid_path = _write_atoms(tmp_path / "valid.xyz", stress_weight=[1.0, 1.0, 1.0])
    test_path = _write_atoms(tmp_path / "test.xyz")
    args = SimpleNamespace(
        train_file=train_path,
        valid_file=valid_path,
        test_file=[test_path],
        key_specification=_keyspec(),
    )

    with pytest.raises(ValueError, match="split=test"):
        check_explicit_component_weights_for_paths(args)


def _atomic_data(**kwargs):
    config = config_from_atoms(
        _atoms(**kwargs),
        key_specification=KeySpecification(
            info_keys={"energy": "REF_energy", "stress": "REF_stress"}, arrays_keys={}
        ),
    )
    return ExtAtomicData.from_config(config, z_table=AtomicNumberTable([1]), cutoff=3.0)


def _batch(*data):
    return torch_geometric.batch.Batch.from_data_list(list(data))


def test_from_config_builds_outer_product_component_weight():
    data = _atomic_data(pbc=(True, True, False), stress_weight=[1.0, 1.0, 0.0])

    expected = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(data.stress_component_weight[0], expected.to(data.stress.dtype))
    assert data.stress_weight.item() == 1.0


def test_from_config_without_stress_label_has_zero_weight():
    data = _atomic_data(include_stress=False)

    assert data.stress_weight.item() == 0.0
    torch.testing.assert_close(
        data.stress_component_weight, torch.ones_like(data.stress_component_weight)
    )


def test_stress_loss_ignores_masked_components():
    batch = _batch(
        _atomic_data(pbc=(True, True, False), stress_weight=[1.0, 1.0, 0.0])
    )
    ref_stress = batch.stress.clone()

    out_of_plane = ref_stress.clone()
    out_of_plane[:, 2, :] += 5.0
    out_of_plane[:, :, 2] += 5.0
    assert weighted_mean_squared_stress(batch, {"stress": out_of_plane}).item() == 0.0

    in_plane = ref_stress.clone()
    in_plane[:, 0, 0] += 1.0
    assert weighted_mean_squared_stress(batch, {"stress": in_plane}).item() > 0.0


def test_stress_metrics_ignore_masked_components_and_unlabelled_configs():
    slab = _atomic_data(pbc=(True, True, False), stress_weight=[1.0, 1.0, 0.0])
    unlabelled = _atomic_data(include_stress=False)
    batch = _batch(slab, unlabelled)

    pred = batch.stress.clone()
    pred[0, 0, 0] += 1.0  # counted: in-plane component of the slab
    pred[0, 2, 2] += 100.0  # masked: out-of-plane component of the slab
    pred[1] += 100.0  # masked: config without a stress label

    metric = MaceSCFLoss(loss_fn=lambda pred, ref: torch.tensor(0.0))
    metric.update(batch, {"stress": pred})
    _, aux = metric.compute()

    # 4 in-plane components of the slab are included, one with error 1.
    np.testing.assert_allclose(aux["mae_stress"], 0.25)
    np.testing.assert_allclose(aux["rmse_stress"], 0.5)
