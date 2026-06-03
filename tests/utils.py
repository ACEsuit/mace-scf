from ase.atoms import Atoms
from copy import deepcopy
import numpy as np
import torch
from contextlib import contextmanager
from e3nn import o3, get_optimization_defaults, set_optimization_defaults

import mace_scf.electrostatics
import mace_scf.data
import mace.tools
import mace.modules



@contextmanager
def disable_e3nn_codegen():
    """Context manager that disables the legacy PyTorch code generation used in e3nn."""
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    yield
    set_optimization_defaults(jit_script_fx=init_val)


def water_configs():
    return [
        Atoms(
            symbols='O2H3', pbc=True, cell=[10.0, 10.0, 10.0], 
            positions=[
                [7.138587, 7.588621, 6.437299],
                [5.281413, 4.285047, 5.411264],
                [7.107233, 6.847326, 7.008736],
                [6.405095, 8.134953, 6.671676],
                [5.420569, 4.75065 , 6.229309]
        ]),
        Atoms(
            symbols='O2H3', pbc=True, cell=[10.0, 10.0, 10.0], 
            positions=[
                [7.438815, 7.253235, 6.172224],
                [5.422642, 5.456331, 5.305246],
                [6.863324, 6.525475, 5.943047],
                [7.697893, 7.106493, 7.114754],
                [4.722107, 5.166765, 5.934254]
        ]),
        Atoms(
            symbols='O2H3', pbc=True, cell=[10.0, 10.0, 10.0], 
            positions=[
                [6.345852, 7.465707, 4.642786],
                [6.088641, 4.954293, 7.078166],
                [5.534056, 7.426354, 5.137674],
                [6.885944, 6.725388, 4.922928],
                [6.728624, 5.063142, 7.777214]
        ]),
        Atoms(
            symbols='O2H3', pbc=True, cell=[10.0, 10.0, 10.0], 
            positions=[
                [6.315726, 5.057041, 5.674563],
                [6.933788, 7.362959, 7.704516],
                [5.486212, 5.116388, 6.174892],
                [5.940488, 5.143974, 4.715484],
                [6.860906, 7.215707, 6.771715]
        ]),
    ]


def seed_torch(i):
    np.random.seed(i)
    torch.manual_seed(i)


def split_to_graphs(tensor, batch_ptr):
    return np.split(
        tensor, 
        indices_or_sections=batch_ptr[1:],
        axis=0)[:-1]


def wrap_loader(dataset, device='cpu', batch_size=1, shuffle=False):
    train_loader = mace.tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    for batch in train_loader:
        yield batch.to(device).to_dict()


def dataset_from_atoms(atoms, cutoff=3.0, **kwargs):
    keyspec = mace.data.KeySpecification()
    keyspec = mace_scf.data.update_keyspec_from_kwargs(keyspec, kwargs)
    configs = mace.data.config_from_atoms_list(atoms, key_specification=keyspec)
    z_table = mace.tools.get_atomic_number_table_from_zs(list(set(atoms[0].get_atomic_numbers())))
    dataset = [
        mace_scf.data.ExtAtomicData.from_config(
            config, 
            z_table=z_table, 
            cutoff=cutoff,
            atomic_multipoles_max_l=kwargs.get('atomic_multipoles_max_l', 1),
        ) for config in configs
    ]
    return dataset 


def dataset_from_position_scan(atoms_obj, delta=5e-3, num=20, **kwargs):
    all_ats = [deepcopy(atoms_obj) for _ in range(num)]
    for i, at in enumerate(all_ats):
        at.positions[0][0] += i * delta
    return dataset_from_atoms(all_ats, **kwargs)


def make_polarizable_model_random(seed, atoms_obj, cutoff=3.0, irreps="4x0e+4x1o", increase_weights=True):
    seed_torch(seed)
    z_table = mace.tools.get_atomic_number_table_from_zs(list(set(atoms_obj.get_atomic_numbers())))
    atomic_energies = np.array([1.0]*len(z_table))
    model_config = dict(
        r_max=cutoff,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=mace.modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=10.0,
        atomic_numbers=z_table.zs,
        correlation=3,
        gate=mace.modules.gate_dict["silu"],
        MLP_irreps=o3.Irreps("16x0e"),
        radial_MLP=[64, 64, 64],
        radial_type="bessel",
    )
    with disable_e3nn_codegen():
        the_model = mace_scf.electrostatics.Polarizable(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            kspace_cutoff_factor=0.75,
            atomic_multipoles_max_l=1,
            atomic_multipoles_smearing_width=1.5,
            field_feature_widths=[1.5],
            include_electrostatic_self_interaction=True,
            add_local_electron_energy=True,
            field_dependence_type='local_linear',
            final_field_readout_type='StrictQuadraticFieldEnergyReadout',
            quadrupole_feature_corrections=False,
            return_electrostatic_potentials=True,
        )
    if increase_weights:
        the_weight = the_model.field_dependent_charges_map.tp.weight
        zzs = 10.0 * torch.ones_like(the_weight.clone().detach())
        the_weight.requires_grad_(False)
        the_weight.copy_(zzs + the_weight)
        the_weight.requires_grad_(True)
    return the_model