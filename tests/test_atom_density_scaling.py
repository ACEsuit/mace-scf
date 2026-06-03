from types import SimpleNamespace

from ase import Atoms
import mace.data
import mace.tools
import mace_scf.data
import numpy as np
import torch

from mace_scf.utils.run_train_utils import get_atom_density_scaling


class FakeZTable:
    zs = [1, 8]

    def __len__(self):
        return len(self.zs)


class FakeBatch:
    def __init__(self, node_attrs, density_coefficients, batch=None, density_weight=None):
        self.node_attrs = node_attrs
        self.density_coefficients = density_coefficients
        self.batch = (
            torch.zeros(node_attrs.shape[0], dtype=torch.long)
            if batch is None
            else batch
        )
        self.density_coefficients_weight = density_weight

    def to(self, device):
        self.node_attrs = self.node_attrs.to(device)
        self.density_coefficients = self.density_coefficients.to(device)
        self.batch = self.batch.to(device)
        if self.density_coefficients_weight is not None:
            self.density_coefficients_weight = self.density_coefficients_weight.to(device)
        return self

    def to_dict(self):
        return {
            "node_attrs": self.node_attrs,
            "density_coefficients": self.density_coefficients,
            "batch": self.batch,
            "density_coefficients_weight": self.density_coefficients_weight,
        }


def _args(atom_density_scaling):
    return SimpleNamespace(
        atom_density_scaling=atom_density_scaling,
        atomic_multipoles_max_l=1,
    )


def test_average_atom_density_scaling_uses_element_rms_over_all_components():
    node_attrs = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    density_coefficients = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0, 2.0, 2.0],
            [4.0, 0.0, 0.0, 0.0],
        ]
    )
    dataloader = [FakeBatch(node_attrs, density_coefficients)]

    scaling = get_atom_density_scaling(dataloader, _args("average"), "cpu", FakeZTable())

    expected_h = np.sqrt(46.0 / 8.0)
    expected_o = np.sqrt(16.0 / 4.0)
    assert np.allclose(scaling, np.array([expected_h, expected_o]))


def test_average_atom_density_scaling_truncates_to_requested_multipoles():
    node_attrs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    density_coefficients = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0, 100.0],
            [0.0, 0.0, 0.0, 0.0, 100.0],
        ]
    )
    dataloader = [FakeBatch(node_attrs, density_coefficients)]

    scaling = get_atom_density_scaling(dataloader, _args("average"), "cpu", FakeZTable())

    assert np.allclose(scaling, np.array([2.0, 1.0e-8]))


def test_average_atom_density_scaling_ignores_zero_weight_configs():
    atoms_with_multipoles = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    atoms_with_multipoles.arrays["multipoles"] = np.array(
        [
            [2.0, 2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0, 4.0],
        ]
    )
    atoms_without_multipoles = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )

    keyspec = mace.data.KeySpecification()
    keyspec = mace_scf.data.update_keyspec_from_kwargs(
        keyspec,
        {"atomic_multipoles_key": "multipoles"},
    )
    configs = mace.data.config_from_atoms_list(
        [atoms_with_multipoles, atoms_without_multipoles],
        key_specification=keyspec,
    )
    assert [config.property_weights["atomic_multipoles"] for config in configs] == [
        1.0,
        0.0,
    ]

    z_table = mace.tools.get_atomic_number_table_from_zs([1, 8])
    dataset = [
        mace_scf.data.ExtAtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=3.0,
            atomic_multipoles_max_l=1,
        )
        for config in configs
    ]
    dataloader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    scaling = get_atom_density_scaling(dataloader, _args("average"), "cpu", z_table)

    assert np.allclose(scaling, np.array([2.0, 4.0]))


def test_manual_atom_density_scaling_is_unchanged():
    scaling = get_atom_density_scaling(
        [],
        _args("{1: 0.5, 8: 2.0}"),
        "cpu",
        FakeZTable(),
    )

    assert np.allclose(scaling, np.array([0.5, 2.0]))
