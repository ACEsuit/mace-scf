import numpy as np
import torch
from typing import Optional
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.stress import full_3x3_to_voigt_6_stress

import mace.data
import mace_scf.data
from mace_scf.electrostatics.compiled_localsources import (
    build_compiled_local_source_evaluator,
)
from mace.tools import torch_geometric, torch_tools, utils


class _MACELocalSourceCalculator(Calculator):
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "partial_charges",
        "partial_dipoles",
        "dipole",
    ]
    uses_formal_charges = False
    includes_polarizability = False

    @staticmethod
    def _validate_compensating_jellium_options(
        compensating_jellium: bool,
        pbc_handling: str,
        jellium_slab_bounds,
    ) -> None:
        if not compensating_jellium:
            return
        if pbc_handling != "slab":
            raise ValueError(
                "compensating_jellium=True requires pbc_handling='slab'."
            )
        if jellium_slab_bounds is None:
            raise ValueError(
                "compensating_jellium=True requires "
                "jellium_slab_bounds=(z_lower, z_upper)."
            )
        if len(jellium_slab_bounds) != 2:
            raise ValueError(
                "jellium_slab_bounds must be a 2-tuple of (z_lower, z_upper)."
            )

    @staticmethod
    def _build_jellium_energy_specs(model, slab_bounds):
        energy = model.coulomb_energy
        return {
            "density_max_l": energy.density_max_l,
            "density_smearing_width": energy.density_smearing_width,
            "kspace_cutoff": energy.kspace_cutoff,
            "slab_bounds": slab_bounds,
            "include_self_interaction": energy.include_self_interaction,
        }

    @classmethod
    def _apply_compensating_jellium(cls, model, slab_bounds, device):
        from graph_longrange.jellium_energy import JelliumSlabSolvatedEnergy

        model.coulomb_energy = JelliumSlabSolvatedEnergy(
            **cls._build_jellium_energy_specs(model, slab_bounds)
        ).to(device)

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype: str = "float64",
        formal_charges_key: str = "charges",
        external_field_key: str = "external_field",
        fermi_level_key: str = "e_fermi",
        pbc_handling: str = "mixed_periodic",
        use_compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "reduce-overhead",
        compile_dynamic: bool = False,
        compile_fullgraph: bool = False,
        compile_warmup: bool = True,
        compensating_jellium: bool = False,
        jellium_slab_bounds: Optional[tuple[float, float]] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.device = torch_tools.init_device(device)
        torch_tools.set_default_dtype(default_dtype)

        self.model = torch.load(f=model_path, map_location=self.device).to(self.device)
        self._validate_compensating_jellium_options(
            compensating_jellium=compensating_jellium,
            pbc_handling=pbc_handling,
            jellium_slab_bounds=jellium_slab_bounds,
        )
        if compensating_jellium:
            self._apply_compensating_jellium(
                self.model,
                slab_bounds=jellium_slab_bounds,
                device=self.device,
            )
        self.model.coulomb_energy.set_pbc_handling(pbc_handling)
        self.r_max = float(self.model.r_max)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.use_compile = use_compile
        self.compile_warmup = compile_warmup
        self.compensating_jellium = compensating_jellium
        self.jellium_slab_bounds = jellium_slab_bounds
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.max_l = int(self.model.coulomb_energy.density_max_l)
        self.head = self.model.heads[0] if hasattr(self.model, "heads") else "Default"
        self.keyspec = self._build_keyspec(
            formal_charges_key=formal_charges_key,
            external_field_key=external_field_key,
            fermi_level_key=fermi_level_key,
        )
        self.compiled_evaluator = None
        if self.use_compile:
            self.compiled_evaluator = build_compiled_local_source_evaluator(
                model=self.model,
                pbc_handling=pbc_handling,
                backend=compile_backend,
                mode=compile_mode,
                dynamic=compile_dynamic,
                fullgraph=compile_fullgraph,
                enabled=True,
            )

    def _build_keyspec(
        self,
        formal_charges_key: str,
        external_field_key: str,
        fermi_level_key: str,
    ) -> mace.data.KeySpecification:
        arrays_keys = {}
        if self.uses_formal_charges:
            arrays_keys["charges"] = formal_charges_key
        return mace.data.KeySpecification(
            info_keys={
                "external_field": external_field_key,
                "fermi_level": fermi_level_key,
            },
            arrays_keys=arrays_keys,
        )

    def _build_data_loader(self, atoms):
        config = mace.data.config_from_atoms(
            atoms, key_specification=self.keyspec, head_name=self.head
        )
        return torch_geometric.dataloader.DataLoader(
            dataset=[
                mace_scf.data.ExtAtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.r_max,
                    atomic_multipoles_max_l=self.max_l,
                    heads=getattr(self.model, "heads", None),
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

    def _extract_partial_dipoles(self, density_coefficients: np.ndarray, num_atoms: int):
        if density_coefficients.shape[1] > 1:
            return density_coefficients[:, [3, 1, 2]]
        return np.zeros((num_atoms, 3))

    def _extract_polarizability(self, output):
        polarizability_out = output.get("polarizability")
        if polarizability_out is None:
            return np.zeros((3, 3))
        polarizability = polarizability_out.detach().cpu().numpy()
        assert polarizability.shape == (1, 3, 3), polarizability.shape
        return polarizability[0]

    def _build_results(self, atoms, output):
        energy = output["energy"].detach().cpu().item()
        forces = output["forces"].detach().cpu().numpy()
        stress = full_3x3_to_voigt_6_stress(
            output["stress"].detach().cpu().numpy()[0]
        )
        density_coefficients = output["density_coefficients"].detach().cpu().numpy()
        partial_charges = density_coefficients[:, 0]
        partial_dipoles = self._extract_partial_dipoles(
            density_coefficients=density_coefficients,
            num_atoms=len(atoms),
        )
        dipole = output["dipole"].detach().cpu().numpy()
        external_field = output["external_field"].detach().cpu().numpy()

        assert dipole.shape == (1, 3), dipole.shape
        assert external_field.shape == (1, 3), external_field.shape

        energy_eV = energy * self.energy_units_to_eV
        results = {
            "energy": energy_eV,
            "free_energy": energy_eV,
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
            "stress": stress * (self.energy_units_to_eV / self.length_units_to_A**3),
            "partial_charges": partial_charges,
            "partial_dipoles": partial_dipoles,
            "density_coefficients": density_coefficients,
            "fermi_level": 0.0,
            "external_field": external_field[0],
            "dipole": dipole[0],
        }
        if self.includes_polarizability:
            results["polarizability"] = self._extract_polarizability(output)
        return results

    def _build_compiled_results(self, atoms, output):
        energy = output["energy"].detach().cpu().item()
        forces = output["forces"].detach().cpu().numpy()
        density_coefficients = output["density_coefficients"].detach().cpu().numpy()
        partial_charges = density_coefficients[:, 0]
        partial_dipoles = self._extract_partial_dipoles(
            density_coefficients=density_coefficients,
            num_atoms=len(atoms),
        )
        dipole = output["dipole"].detach().cpu().numpy()
        external_field = output["external_field"].detach().cpu().numpy()

        assert dipole.shape == (1, 3), dipole.shape
        assert external_field.shape == (1, 3), external_field.shape

        energy_eV = energy * self.energy_units_to_eV
        return {
            "energy": energy_eV,
            "free_energy": energy_eV,
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
            "partial_charges": partial_charges,
            "partial_dipoles": partial_dipoles,
            "density_coefficients": density_coefficients,
            "fermi_level": 0.0,
            "external_field": external_field[0],
            "dipole": dipole[0],
        }

    def _validate_compiled_properties(self, properties):
        if not self.use_compile or properties is None:
            return
        unsupported = {"stress", "polarizability"}.intersection(properties)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise PropertyNotImplementedError(
                f"Compiled local-source calculator does not support: {names}"
            )

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        self._validate_compiled_properties(properties)
        Calculator.calculate(self, atoms, system_changes=system_changes)

        data_loader = self._build_data_loader(atoms)
        batch = next(iter(data_loader)).to(self.device)
        if self.compiled_evaluator is None:
            output = self.model(
                batch.to_dict(),
                compute_force=True,
                compute_stress=True,
            )
            self.results = self._build_results(atoms, output)
        else:
            output = self.compiled_evaluator.evaluate(batch.to_dict())
            self.results = self._build_compiled_results(atoms, output)


class MACELocalSplitCharges(_MACELocalSourceCalculator):
    implemented_properties = _MACELocalSourceCalculator.implemented_properties + [
        "polarizability"
    ]
    uses_formal_charges = True
    includes_polarizability = True


class MACELocalCharges(_MACELocalSourceCalculator):
    implemented_properties = _MACELocalSourceCalculator.implemented_properties
    uses_formal_charges = False
    includes_polarizability = False


class MACEFixedChargeBaselined(_MACELocalSourceCalculator):
    implemented_properties = _MACELocalSourceCalculator.implemented_properties
    uses_formal_charges = True
    includes_polarizability = False
