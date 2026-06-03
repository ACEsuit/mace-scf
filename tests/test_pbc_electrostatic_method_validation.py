from types import SimpleNamespace

from ase import Atoms
from ase.io import write
import pytest

from mace_scf.utils.load_data import (
    check_pbc_consistent_with_electrostatic_method,
    check_pbc_consistent_with_electrostatic_method_for_paths,
)


def _write_atoms(path, pbcs):
    """Write one Atoms per entry in ``pbcs`` (each a 3-tuple of bools)."""
    images = []
    for pbc in pbcs:
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        atoms.cell = [10.0, 10.0, 10.0]
        atoms.pbc = pbc
        images.append(atoms)
    write(path, images, format="extxyz")
    return path


@pytest.mark.parametrize(
    "method,pbc",
    [
        ("realspace", (False, False, False)),
        ("pbc", (True, True, True)),
        ("slab", (True, True, False)),
        ("molecule_in_box", (False, False, False)),
        ("mixed_periodic", (True, True, True)),
        ("mixed_periodic", (True, True, False)),
        ("mixed_periodic", (False, False, False)),
    ],
)
def test_accepts_compatible_pbc(tmp_path, method, pbc):
    xyz_path = _write_atoms(tmp_path / "ok.xyz", [pbc])
    check_pbc_consistent_with_electrostatic_method(xyz_path, method, "train")


@pytest.mark.parametrize(
    "method,pbc",
    [
        ("realspace", (True, True, True)),
        ("realspace", (True, True, False)),
        ("pbc", (False, False, False)),
        ("pbc", (True, True, False)),
        ("slab", (True, True, True)),
        ("slab", (True, False, True)),
        ("slab", (False, False, False)),
        ("molecule_in_box", (True, True, True)),
        ("molecule_in_box", (True, False, False)),
        ("mixed_periodic", (True, False, False)),
        ("mixed_periodic", (False, True, True)),
        ("mixed_periodic", (True, False, True)),
    ],
)
def test_rejects_incompatible_pbc(tmp_path, method, pbc):
    xyz_path = _write_atoms(tmp_path / "bad.xyz", [pbc])
    with pytest.raises(ValueError, match="incompatible with"):
        check_pbc_consistent_with_electrostatic_method(xyz_path, method, "train")


def test_rejects_in_mixed_file(tmp_path):
    xyz_path = _write_atoms(
        tmp_path / "mixed.xyz",
        [(False, False, False), (False, False, False), (True, True, True)],
    )
    with pytest.raises(ValueError, match="incompatible with"):
        check_pbc_consistent_with_electrostatic_method(xyz_path, "realspace", "train")


def test_for_paths_covers_train_valid_and_test(tmp_path):
    train_path = _write_atoms(tmp_path / "train.xyz", [(False, False, False)])
    valid_path = _write_atoms(tmp_path / "valid.xyz", [(False, False, False)])
    test_path = _write_atoms(tmp_path / "test.xyz", [(True, True, True)])
    args = SimpleNamespace(
        train_file=str(train_path),
        valid_file=str(valid_path),
        test_file=str(test_path),
        electrostatic_pbc_method="realspace",
    )
    with pytest.raises(ValueError, match="test"):
        check_pbc_consistent_with_electrostatic_method_for_paths(args)


def test_for_paths_skipped_when_method_unset(tmp_path):
    xyz_path = _write_atoms(tmp_path / "anything.xyz", [(True, True, True)])
    args = SimpleNamespace(
        train_file=str(xyz_path),
        valid_file=None,
        test_file=None,
    )
    check_pbc_consistent_with_electrostatic_method_for_paths(args)


def test_for_paths_skipped_when_override_set(tmp_path):
    xyz_path = _write_atoms(tmp_path / "bad.xyz", [(True, True, True)])
    args = SimpleNamespace(
        train_file=str(xyz_path),
        valid_file=None,
        test_file=None,
        electrostatic_pbc_method="realspace",
        override_pbc_checks=True,
    )
    check_pbc_consistent_with_electrostatic_method_for_paths(args)
