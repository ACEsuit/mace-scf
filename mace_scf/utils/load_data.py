
from typing import Dict, Iterable, Tuple

import ase.io
from mace_scf.data.new_atomic_data import ExtAtomicData
import mace_scf
import mace
from mace.tools.scripts_utils import get_dataset_from_xyz, get_atomic_energies
import argparse
import ast
import logging
import os
import numpy as np
from mace import tools


def _config_type(config) -> str:
    properties = getattr(config, "properties", {}) or {}
    config_type = properties.get("config_type", None)
    if config_type is None:
        config_type = getattr(config, "config_type", None)
    return "unknown" if config_type is None else str(config_type)


def _is_fully_periodic(pbc) -> bool:
    if pbc is None:
        return False
    pbc_array = np.asarray(pbc, dtype=bool).reshape(-1)
    return pbc_array.shape == (3,) and bool(np.all(pbc_array))


def _volume_from_cell(cell) -> float:
    if cell is None:
        raise ValueError("fully periodic config has no cell")
    cell_array = np.asarray(cell, dtype=float)
    if cell_array.shape != (3, 3):
        raise ValueError(f"fully periodic config has invalid cell shape {cell_array.shape}")
    return float(abs(np.linalg.det(cell_array)))


def _named_config_splits(collections) -> Iterable[Tuple[str, Iterable]]:
    yield "train", collections.train
    yield "valid", collections.valid
    for test_name, test_configs in collections.tests:
        yield f"test:{test_name}", test_configs


def _as_path_list(path_or_paths):
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, (str, os.PathLike)):
        return [path_or_paths]
    return list(path_or_paths)


def check_explicit_dipole_component_weights(file_path, key_specification, split_name):
    dipole_key = key_specification.info_keys.get("dipole")
    if dipole_key is None or dipole_key == "none":
        return

    for config_index, atoms in enumerate(ase.io.iread(file_path, index=":")):
        if dipole_key not in atoms.info:
            continue
        if "config_dipole_weight" not in atoms.info:
            raise ValueError(
                "Dipole data found without explicit config_dipole_weight. "
                f"split={split_name}, file={file_path}, index={config_index}, "
                f"dipole_key={dipole_key!r}. "
                "Set config_dipole_weight to a 3-vector such as [1, 1, 1], "
                "[0, 0, 1], or [0, 0, 0]."
            )

        dipole_weight = np.asarray(
            atoms.info["config_dipole_weight"], dtype=float
        ).reshape(-1)
        if dipole_weight.shape != (3,):
            raise ValueError(
                "config_dipole_weight must be a 3-vector whenever dipole data is "
                "present. "
                f"split={split_name}, file={file_path}, index={config_index}, "
                f"dipole_key={dipole_key!r}, shape={dipole_weight.shape}."
            )


_ALLOWED_PBC_BY_METHOD = {
    "realspace": {(False, False, False)},
    "pbc": {(True, True, True)},
    "slab": {(True, True, False)},
    "molecule_in_box": {(False, False, False)},
    "mixed_periodic": {
        (True, True, True),
        (True, True, False),
        (False, False, False),
    },
}


def _pbc_tuple(atoms) -> Tuple[bool, bool, bool]:
    if not hasattr(atoms, "pbc"):
        raise ValueError("config has no pbc attribute")
    pbc = np.asarray(atoms.pbc, dtype=bool).reshape(-1)
    return tuple(bool(x) for x in pbc)


def check_pbc_consistent_with_electrostatic_method(
    file_path, electrostatic_pbc_method, split_name
):
    allowed = _ALLOWED_PBC_BY_METHOD.get(electrostatic_pbc_method)
    if allowed is None:
        return
    for atoms in ase.io.iread(file_path, index=":"):
        pbc = _pbc_tuple(atoms)
        if pbc not in allowed:
            allowed_str = ", ".join(
                "".join("T" if x else "F" for x in p) for p in sorted(allowed)
            )
            pbc_str = "".join("T" if x else "F" for x in pbc)
            raise ValueError(
                f"Found pbc={pbc_str} in {split_name} file {file_path}, "
                f"which is incompatible with "
                f"--electrostatic_pbc_method={electrostatic_pbc_method} "
                f"(allowed: {allowed_str})."
            )


def check_pbc_consistent_with_electrostatic_method_for_paths(args):
    if getattr(args, "override_pbc_checks", False):
        return
    method = getattr(args, "electrostatic_pbc_method", None)
    if method is None:
        return
    for path in _as_path_list(args.train_file):
        check_pbc_consistent_with_electrostatic_method(path, method, "train")
    for path in _as_path_list(args.valid_file):
        check_pbc_consistent_with_electrostatic_method(path, method, "valid")
    for path in _as_path_list(args.test_file):
        check_pbc_consistent_with_electrostatic_method(path, method, "test")


def check_explicit_dipole_component_weights_for_paths(args):
    for path in _as_path_list(args.train_file):
        check_explicit_dipole_component_weights(
            path, args.key_specification, "train"
        )
    for path in _as_path_list(args.valid_file):
        check_explicit_dipole_component_weights(
            path, args.key_specification, "valid"
        )
    for path in _as_path_list(args.test_file):
        check_explicit_dipole_component_weights(
            path, args.key_specification, "test"
        )


def check_low_density_periodic_configs(
    collections,
    *,
    max_volume_per_atom: float,
    allow_low_density_pbc: bool,
) -> None:
    if allow_low_density_pbc:
        logging.info("Skipping low-density pbc=TTT config check.")
        return
    if max_volume_per_atom <= 0.0:
        raise ValueError("low_density_pbc_max_volume_per_atom must be positive")

    for split_name, configs in _named_config_splits(collections):
        for config_index, config in enumerate(configs):
            if not _is_fully_periodic(getattr(config, "pbc", None)):
                continue
            volume = _volume_from_cell(getattr(config, "cell", None))
            num_atoms = len(config.atomic_numbers)
            volume_per_atom = volume / num_atoms
            if volume_per_atom <= max_volume_per_atom:
                continue
            raise ValueError(
                "Suspicious low-density fully periodic config detected. "
                f"split={split_name}, index={config_index}, "
                f"config_type={_config_type(config)}, pbc={getattr(config, 'pbc', None)}, "
                f"volume={volume:.6g}, num_atoms={num_atoms}, "
                f"volume_per_atom={volume_per_atom:.6g}, "
                f"max_volume_per_atom={max_volume_per_atom:.6g}. "
                "If this is intentional, rerun with --allow_low_density_pbc."
            )


def load_train_valid_sets_from_xyz(args: argparse.Namespace,  config_type_weights: Dict):
    # data
    check_explicit_dipole_component_weights_for_paths(args)
    check_pbc_consistent_with_electrostatic_method_for_paths(args)
    collections, atomic_energies_dict = get_dataset_from_xyz(
        work_dir=args.work_dir,
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.valid_set_seed,
        key_specification=args.key_specification,
    )
    logging.info(
        f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
        f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    )
    check_low_density_periodic_configs(
        collections,
        max_volume_per_atom=args.low_density_pbc_max_volume_per_atom,
        allow_low_density_pbc=args.allow_low_density_pbc,
    )

    # Atomic number table
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    logging.info(z_table)

    # energies
    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        # if args.train_file.endswith(".xyz"):
        atomic_energies_dict = get_atomic_energies(
            args.E0s, collections.train, z_table
        )
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )

    train_set = [
        mace_scf.data.ExtAtomicData.from_config(
            config, 
            z_table=z_table, 
            cutoff=args.r_max, 
            atomic_multipoles_max_l=args.atomic_multipoles_max_l
        )
        for config in collections.train
    ]
   
    valid_set = [
        mace_scf.data.ExtAtomicData.from_config(
            config, 
            z_table=z_table, 
            cutoff=args.r_max, 
            atomic_multipoles_max_l=args.atomic_multipoles_max_l
        )
        for config in collections.valid
    ]

    return train_set, valid_set, z_table, atomic_energies, collections.tests


def load_train_valid_sets_from_h5(args: argparse.Namespace):

    # has to come from command line
    zs_list = ast.literal_eval(args.atomic_numbers)
    z_table = tools.get_atomic_number_table_from_zs(zs_list)
    atomic_energies_dict = get_atomic_energies(args.E0s, None, z_table)
    train_set = mace.data.HDF5Dataset(args.train_file, r_max=args.r_max, z_table=z_table, atomic_dataclass=ExtAtomicData, atomic_multipoles_max_l=args.atomic_multipoles_max_l)
    valid_set = mace.data.HDF5Dataset(args.valid_file, r_max=args.r_max, z_table=z_table, atomic_dataclass=ExtAtomicData, atomic_multipoles_max_l=args.atomic_multipoles_max_l)
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    return train_set, valid_set, z_table, atomic_energies

def load_train_valid_sets_from_sharded_h5(args: argparse.Namespace):

    # has to come from command line
    zs_list = ast.literal_eval(args.atomic_numbers)
    z_table = tools.get_atomic_number_table_from_zs(zs_list)
    atomic_energies_dict = get_atomic_energies(args.E0s, None, z_table)
    train_set = mace.data.dataset_from_sharded_hdf5(
        args.train_file, r_max=args.r_max, z_table=z_table, atomic_dataclass=ExtAtomicData, atomic_multipoles_max_l=args.atomic_multipoles_max_l
    )
    valid_set = mace.data.dataset_from_sharded_hdf5(
        args.valid_file, r_max=args.r_max, z_table=z_table, atomic_dataclass=ExtAtomicData, atomic_multipoles_max_l=args.atomic_multipoles_max_l
    )
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    
    return train_set, valid_set, z_table, atomic_energies
