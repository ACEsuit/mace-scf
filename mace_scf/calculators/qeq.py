import torch
from ase.calculators.calculator import Calculator, all_changes

import mace.data
import mace_scf.data

from mace.tools import torch_tools, utils, torch_geometric


class MACEQEqCalculator(Calculator):
    implemented_properties = [
        "energy",
        "free_energy",
        "qeq_energy",
        "forces",
        "charges",
        "partial_charges",
        "density_coefficients",
        "dipole",
        "enegs",
        "hardness",
    ]

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        external_field_key: str = "external_field",
        enegs_key: str = "enegs",
        hardness_key: str = "hardness",
        total_charge_key: str = "total_charge",
        atomic_multipoles_key: str = "initial_density_coefficients",
        default_dtype="float64",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.device = torch_tools.init_device(device)
        torch_tools.set_default_dtype(default_dtype)

        self.model = torch.load(f=model_path, map_location=self.device).to(self.device)
        self.r_max = self.model.r_max.cpu().item()
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.keyspec_singlepoint = mace.data.KeySpecification(
            info_keys={
                "external_field": external_field_key,
                "total_charge": total_charge_key,
            },
            arrays_keys={
                "atomic_multipoles": atomic_multipoles_key,
                "enegs": enegs_key,
                "hardness": hardness_key,
            },
        )
        self.head = self.model.heads[0] if hasattr(self.model, "heads") else "Default"

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, system_changes=system_changes)
        keyspec = self.keyspec_singlepoint
        config = mace.data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=self.head
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                mace_scf.data.ExtAtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.r_max,
                    atomic_multipoles_max_l=0,
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        out = self.model(batch.to_dict(), compute_stress=False)
        energy = out["energy"].detach().cpu().item()
        qeq_energy = out["qeq_energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()
        charges = out["density_coefficients"].detach().cpu().numpy()
        enegs = out["enegs"].detach().cpu().numpy()
        hardness = out["hardness"].detach().cpu().numpy()
        dipole = out["dipole"].squeeze().detach().cpu().numpy()

        E = energy * self.energy_units_to_eV
        qeqE = qeq_energy * self.energy_units_to_eV
        self.results = {
            "energy": E,
            "free_energy": E,
            "qeq_energy": qeqE,
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
            "enegs": enegs,
            "hardness": hardness,
            "charges": charges,
            "partial_charges": charges[:, 0],
            "density_coefficients": charges,
            "dipole": dipole,
        }
