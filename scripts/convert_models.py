"""
If you have trained models using the older macetools repo, this script can local and convert them.
"""

import pickle
import argparse
import torch
from mace.tools import torch_tools
torch.set_default_dtype(torch.float64)

from mace_scf.electrostatics.fixed_point_core import FixedPointCore
from mace_scf.electrostatics.fixed_point import FixedPoint
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures

try:
    from graph_longrange.gto_electrostatics import GTOChargeDensityFourierSeriesBlock
except ImportError:
    GTOChargeDensityFourierSeriesBlock = None

DEVICE=torch_tools.init_device('cpu')

CLASS_RENAMES = {
    (
        "macetools.electrostatics.localsources", "LocalSymmetricCharges",
    ): (
        "mace_scf.electrostatics.localsources", "LocalSplitCharges",
    ), (
        "macetools.electrostatics.localsources", "NonPolarizable",
    ): (
        "mace_scf.electrostatics.localsources", "LocalCharges",
    ), (
        "macetools.electrostatics.localsources", "FixedPointCore",
    ): (
        "mace_scf.electrostatics.localsources", "FixedPointCore",
    ), (
        "macetools.electrostatics.localsources", "MACEQEq",
    ): (
        "mace_scf.electrostatics.localsources", "MACEQEq",
    ),
}

OLD_PREFIX = "macetools"
NEW_PREFIX = "mace_scf"


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in CLASS_RENAMES:
            module, name = CLASS_RENAMES[(module, name)]
        # Otherwise do the broad package rename
        elif module == OLD_PREFIX or module.startswith(OLD_PREFIX + "."):
            module = NEW_PREFIX + module[len(OLD_PREFIX):]
        return super().find_class(module, name)


class PickleModule:
    Unpickler = RenameUnpickler
    load = pickle.load
    loads = pickle.loads
    dump = pickle.dump
    dumps = pickle.dumps




def get_specs_energy(model):
    return {
        "kspace_cutoff": model.coulomb_energy.kspace_cutoff,
        "density_max_l": model.coulomb_energy.realspace_energy.density_max_l,
        "density_smearing_width": model.coulomb_energy.realspace_energy.density_smearing_width,
        "include_self_interaction": model.coulomb_energy.realspace_energy.include_self_interaction,
        "pbc_handling": "mixed_periodic",
    }


def get_specs_features(model):
    realspace_features = model.electric_potential_descriptor.realspace_features
    return {
        "density_max_l": realspace_features.density_max_l,
        "density_smearing_width": realspace_features.density_smearing_width,
        "feature_max_l": realspace_features.projection_max_l,
        "feature_smearing_widths": realspace_features.projection_smearing_widths,
        "kspace_cutoff": model.coulomb_energy.kspace_cutoff,
        "include_self_interaction": realspace_features.include_self_interaction,
    }


def add_kspace_cutoff(model):
    model.register_buffer(
        "kspace_cutoff", 
        model.coulomb_energy.kspace_cutoff.clone().detach()
    )


def swap_energy_block(model):
    new_energy_fun = GTOElectrostaticEnergy(
        **get_specs_energy(model)
    ).to(DEVICE)
    model.coulomb_energy = new_energy_fun


def add_default_heads_dict(model):
    model.heads = ["Default"]


def swap_field_block_pol(model):
    new_features_fun = GTOElectrostaticFeatures(
        **get_specs_features(model)
    ).to(DEVICE)
    model.electric_potential_descriptor = new_features_fun


def convert_fixedpoint_to_core(model):
    """
    Convert a legacy FixedPoint model to FixedPointCore by changing its
    class and patching any missing attributes.
    """
    model.__class__ = FixedPointCore

    if not hasattr(model, "from_ell_max_field_update"):
        model.from_ell_max_field_update = 9

    if not hasattr(model, "return_electrostatic_potentials"):
        model.return_electrostatic_potentials = False

    if not hasattr(model, "fermi_level_offset"):
        model.register_buffer(
            "fermi_level_offset",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model to convert")
    parser.add_argument("--output_path", type=str, help="Path to save the converted model", default="converted_model.model")
    args = parser.parse_args()

    model = torch.load(
        f=args.model_path, 
        map_location=DEVICE, 
        weights_only=False,
        pickle_module=PickleModule,
    ).to(DEVICE)
    
    if not hasattr(model, "kspace_cutoff"):
        add_kspace_cutoff(model)
    
    if hasattr(model, "coulomb_energy"):# and (not isinstance(model.coulomb_energy, GTOElectrostaticEnergy)): # "PBCAgnosticDirectElectrostaticEnergyBlock"
        print('swapping energy block')
        swap_energy_block(model)

    if hasattr(model, "electric_potential_descriptor"):# and (not isinstance(model.electric_potential_descriptor, GTOElectrostaticFeatures)): # "PBCAgnosticElectrostaticFeatureBlock"
        print('swapping feature block')
        swap_field_block_pol(model)

    if not hasattr(model, "heads"):
        add_default_heads_dict(model)

    if isinstance(model, FixedPoint) or isinstance(model, FixedPointCore):
        model = convert_fixedpoint_to_core(model)
    
    torch.save(model, args.output_path)
    print(f"Converted model saved to {args.output_path}")
    print(f"Model class: {type(model).__name__}")
