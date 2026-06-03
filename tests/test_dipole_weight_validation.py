from types import SimpleNamespace

from ase import Atoms
from ase.io import write
import numpy as np
import pytest

from mace_scf.utils.load_data import (
    check_explicit_dipole_component_weights,
    check_explicit_dipole_component_weights_for_paths,
)


def _write_atoms(path, *, include_dipole=True, dipole_weight=None):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["AIMS_energy"] = 0.0
    if include_dipole:
        atoms.info["AIMS_dipole"] = np.array([1.0, 2.0, 3.0])
    if dipole_weight is not None:
        atoms.info["config_dipole_weight"] = np.asarray(dipole_weight, dtype=float)
    write(path, atoms, format="extxyz")
    return path


def _keyspec(dipole_key="AIMS_dipole"):
    return SimpleNamespace(info_keys={"dipole": dipole_key})


def test_explicit_dipole_weight_required_when_dipole_is_present(tmp_path):
    xyz_path = _write_atoms(tmp_path / "missing_weight.xyz")

    with pytest.raises(ValueError, match="without explicit config_dipole_weight"):
        check_explicit_dipole_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_dipole_weight_accepts_three_component_mask(tmp_path):
    xyz_path = _write_atoms(tmp_path / "weighted.xyz", dipole_weight=[1.0, 1.0, 1.0])

    check_explicit_dipole_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_dipole_weight_accepts_non_binary_three_component_values(tmp_path):
    xyz_path = _write_atoms(tmp_path / "fractional_weight.xyz", dipole_weight=[0.5, 0.0, 2.0])

    check_explicit_dipole_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_dipole_weight_rejects_scalar_weight(tmp_path):
    xyz_path = _write_atoms(tmp_path / "scalar_weight.xyz", dipole_weight=1.0)

    with pytest.raises(ValueError, match="must be a 3-vector"):
        check_explicit_dipole_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_dipole_weight_check_ignores_missing_dipoles(tmp_path):
    xyz_path = _write_atoms(tmp_path / "no_dipole.xyz", include_dipole=False)

    check_explicit_dipole_component_weights(xyz_path, _keyspec(), "train")


def test_explicit_dipole_weight_check_ignores_disabled_dipole_key(tmp_path):
    xyz_path = _write_atoms(tmp_path / "disabled_dipole.xyz")

    check_explicit_dipole_component_weights(xyz_path, _keyspec("none"), "train")


def test_explicit_dipole_weight_check_covers_train_valid_and_test_paths(tmp_path):
    train_path = _write_atoms(
        tmp_path / "train.xyz", dipole_weight=[1.0, 1.0, 1.0]
    )
    valid_path = _write_atoms(
        tmp_path / "valid.xyz", dipole_weight=[1.0, 1.0, 1.0]
    )
    test_path = _write_atoms(tmp_path / "test.xyz")
    args = SimpleNamespace(
        train_file=train_path,
        valid_file=valid_path,
        test_file=[test_path],
        key_specification=_keyspec(),
    )

    with pytest.raises(ValueError, match="split=test"):
        check_explicit_dipole_component_weights_for_paths(args)
