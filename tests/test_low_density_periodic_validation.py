from types import SimpleNamespace

import numpy as np
import pytest

from mace_scf.utils.load_data import check_low_density_periodic_configs


def _config(*, pbc, cell_size, num_atoms=2, config_type="test_config"):
    return SimpleNamespace(
        pbc=np.asarray(pbc, dtype=bool),
        cell=np.eye(3) * cell_size,
        atomic_numbers=[1] * num_atoms,
        properties={"config_type": config_type},
    )


def _collections(train=(), valid=(), tests=()):
    return SimpleNamespace(
        train=list(train),
        valid=list(valid),
        tests=list(tests),
    )


def test_low_density_periodic_check_ignores_nonperiodic_large_cell():
    collections = _collections(
        train=[_config(pbc=[False, False, False], cell_size=100.0)]
    )

    check_low_density_periodic_configs(
        collections,
        max_volume_per_atom=1000.0,
        allow_low_density_pbc=False,
    )


def test_low_density_periodic_check_allows_dense_periodic_config():
    collections = _collections(
        train=[_config(pbc=[True, True, True], cell_size=4.0, num_atoms=8)]
    )

    check_low_density_periodic_configs(
        collections,
        max_volume_per_atom=1000.0,
        allow_low_density_pbc=False,
    )


def test_low_density_periodic_check_rejects_large_periodic_cluster():
    collections = _collections(
        train=[_config(pbc=[True, True, True], cell_size=40.0, config_type="cluster")]
    )

    with pytest.raises(ValueError, match="Suspicious low-density fully periodic config"):
        check_low_density_periodic_configs(
            collections,
            max_volume_per_atom=1000.0,
            allow_low_density_pbc=False,
        )


def test_low_density_periodic_check_error_mentions_context():
    collections = _collections(
        tests=[
            (
                "heldout",
                [_config(pbc=[True, True, True], cell_size=40.0, config_type="cluster")],
            )
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        check_low_density_periodic_configs(
            collections,
            max_volume_per_atom=1000.0,
            allow_low_density_pbc=False,
        )

    message = str(exc_info.value)
    assert "split=test:heldout" in message
    assert "index=0" in message
    assert "config_type=cluster" in message
    assert "--allow_low_density_pbc" in message


def test_low_density_periodic_check_override_allows_large_periodic_cluster():
    collections = _collections(
        train=[_config(pbc=[True, True, True], cell_size=40.0)]
    )

    check_low_density_periodic_configs(
        collections,
        max_volume_per_atom=1000.0,
        allow_low_density_pbc=True,
    )
