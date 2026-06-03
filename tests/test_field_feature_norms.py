from types import SimpleNamespace

from ase import Atoms
import mace.data
import mace.tools
import mace_scf.data
import numpy as np

from mace_scf.utils.run_train_utils import (
    compute_average_feature_norms,
    get_fermi_level_offset,
    get_field_feature_norms,
)


def _args(field_feature_widths="[1.0]", field_feature_max_l=1, fermi_level_offset=None):
    return SimpleNamespace(
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.5,
        electrostatic_pbc_method="mixed_periodic",
        field_feature_max_l=field_feature_max_l,
        field_feature_norms="average",
        field_feature_widths=field_feature_widths,
        include_field_si=False,
        kspace_cutoff_factor=1.0,
        model="FixedPoint",
        quadrupole_feature_corrections=False,
        fermi_level_offset=fermi_level_offset,
    )


def _atoms(*, include_multipoles=True, include_fermi_level=True, fermi_level=1.25):
    atoms = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=[40.0, 40.0, 40.0],
        pbc=False,
    )
    if include_multipoles:
        atoms.arrays["multipoles"] = np.array(
            [
                [2.0, 2.0, 2.0, 2.0],
                [4.0, 4.0, 4.0, 4.0],
            ]
        )
    if include_fermi_level:
        atoms.info["the_fermi_level"] = fermi_level
    atoms.info["external_field"] = np.array([0.2, -0.1, 0.3])
    return atoms


def _loader_from_atoms(atoms_list, *, field_feature_max_l=1):
    keyspec = mace.data.KeySpecification()
    keyspec = mace_scf.data.update_keyspec_from_kwargs(
        keyspec,
        {
            "atomic_multipoles_key": "multipoles",
            "fermi_level_key": "the_fermi_level",
            "external_field_key": "external_field",
        },
    )
    configs = mace.data.config_from_atoms_list(
        atoms_list,
        key_specification=keyspec,
    )
    z_table = mace.tools.get_atomic_number_table_from_zs([1, 8])
    dataset = [
        mace_scf.data.ExtAtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=3.0,
            atomic_multipoles_max_l=field_feature_max_l,
        )
        for config in configs
    ]
    return mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False,
    ), configs


def test_average_feature_norms_ignore_missing_multipoles_and_fermi_level_configs():
    valid_atoms = _atoms()
    reference_loader, _ = _loader_from_atoms([valid_atoms])
    mixed_loader, configs = _loader_from_atoms(
        [
            valid_atoms,
            _atoms(include_multipoles=False),
            _atoms(include_fermi_level=False),
        ]
    )
    assert [config.property_weights["atomic_multipoles"] for config in configs] == [
        1.0,
        0.0,
        1.0,
    ]
    assert [config.property_weights["fermi_level"] for config in configs] == [
        1.0,
        1.0,
        0.0,
    ]

    expected = compute_average_feature_norms(reference_loader, _args(), "cpu")
    actual = compute_average_feature_norms(mixed_loader, _args(), "cpu")

    assert np.allclose(actual, expected)


def test_average_feature_norms_use_centered_fermi_level():
    offset = 2.0
    absolute_loader, _ = _loader_from_atoms([_atoms(fermi_level=3.25)])
    shifted_loader, _ = _loader_from_atoms([_atoms(fermi_level=1.25)])

    absolute_norms = compute_average_feature_norms(
        absolute_loader, _args(), "cpu", fermi_level_offset=offset
    )
    shifted_norms = compute_average_feature_norms(
        shifted_loader, _args(), "cpu", fermi_level_offset=0.0
    )

    assert np.allclose(absolute_norms, shifted_norms)


def test_fermi_level_offset_uses_present_fermi_levels_and_supports_override():
    loader, _ = _loader_from_atoms(
        [
            _atoms(fermi_level=1.0),
            _atoms(fermi_level=3.0),
            _atoms(include_fermi_level=False),
        ]
    )

    assert get_fermi_level_offset(loader, _args(), "cpu") == 2.0
    assert get_fermi_level_offset(loader, _args(fermi_level_offset=-4.0), "cpu") == -4.0


def test_fermi_level_offset_does_not_move_dataset_items(monkeypatch):
    loader, _ = _loader_from_atoms([_atoms(fermi_level=2.0)])
    data_type = type(loader.dataset[0])

    def fail_if_called(self, *args, **kwargs):
        raise AssertionError("get_fermi_level_offset should not mutate dataset devices")

    monkeypatch.setattr(data_type, "to", fail_if_called)

    assert get_fermi_level_offset(loader, _args(), "cuda") == 2.0


def test_average_feature_norms_return_one_norm_per_width_and_order():
    loader, _ = _loader_from_atoms([_atoms()])

    norms = compute_average_feature_norms(
        loader,
        _args(field_feature_widths="[1.0, 2.0]", field_feature_max_l=1),
        "cpu",
    )

    assert norms.shape == (4,)
    assert np.all(norms > 0.0)


def test_non_fixedpoint_models_ignore_field_feature_norms():
    args = _args()
    args.model = "MACE"
    args.field_feature_norms = "not a valid fixedpoint norm setting"

    assert get_field_feature_norms([], args, "cpu") is None
