import warnings
from typing import Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

import mace.data
import mace.tools
import mace_scf.data
from mace.tools import torch_geometric, torch_tools, utils
from mace_scf.electrostatics.fixed_point_options import (
    validate_fixed_point_scf_options,
)
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.electrostatics.fixed_point_core import FixedPointCore
from mace_scf.electrostatics.fixed_point import FixedPoint
from mace_scf.electrostatics.compiled_fixed_point import (
    build_compiled_fixed_point_evaluator,
)


CALCULATOR_SCF_DEFAULTS = {
    "num_scf_steps": 100,
    "scf_tolerance": 1e-6,
    "mixing_parameter": 0.25,
    "constant_charge": True,
    "use_autograd_forces": True,
    "initial_density": "local_guess",
    "initial_fermi_level": "from_data",
}


class MACEFixedPointSCF(Calculator):
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "density_coefficients",
        "electrostatic_potentials",
        "dipole",
        "fermi_level",
        "electrostatic_energy",
        "electron_energy",
        "convergence_history",
    ]

    @staticmethod
    def _validate_glr_compile_ready_model(model) -> None:
        missing_paths = []
        checks = (
            (
                "coulomb_energy.self_interaction_terms.features_dim",
                getattr(getattr(model, "coulomb_energy", None), "self_interaction_terms", None),
            ),
            (
                "electric_potential_descriptor.self_interaction_terms.features_dim",
                getattr(
                    getattr(model, "electric_potential_descriptor", None),
                    "self_interaction_terms",
                    None,
                ),
            ),
        )
        for path, module in checks:
            if module is None or not hasattr(module, "features_dim"):
                missing_paths.append(path)
        if missing_paths:
            formatted = ", ".join(missing_paths)
            raise ValueError(
                "Model is not compatible with the current graph_longrange runtime. "
                f"Missing required attribute(s): {formatted}. "
                "Please rebuild/convert the model with scripts/convert_models.py."
            )

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
                "compensating_jellium=True requires jellium_slab_bounds=(z_lower, z_upper)."
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

    @staticmethod
    def _build_jellium_feature_specs(model, slab_bounds):
        features = model.electric_potential_descriptor
        realspace_features = features.realspace_features
        return {
            "density_max_l": realspace_features.density_max_l,
            "density_smearing_width": realspace_features.density_smearing_width,
            "feature_max_l": realspace_features.projection_max_l,
            "feature_smearing_widths": realspace_features.projection_smearing_widths,
            "kspace_cutoff": model.coulomb_energy.kspace_cutoff,
            "slab_bounds": slab_bounds,
            "include_self_interaction": realspace_features.include_self_interaction,
            "integral_normalization": features.feature_basis.normalize,
        }

    @classmethod
    def _apply_compensating_jellium(cls, model, slab_bounds, device):
        from graph_longrange.jellium_energy import JelliumSlabSolvatedEnergy
        from graph_longrange.jellium_features import JelliumSlabSolvatedFeatures

        model.coulomb_energy = JelliumSlabSolvatedEnergy(
            **cls._build_jellium_energy_specs(model, slab_bounds)
        ).to(device)
        model.electric_potential_descriptor = JelliumSlabSolvatedFeatures(
            **cls._build_jellium_feature_specs(model, slab_bounds)
        ).to(device)

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype: str = "float64",
        scf_options: dict = None,
        atomic_multipoles_key: str = "initial_density_coefficients",
        fermi_level_key: str = "initial_fermi_level",
        external_field_key: str = "external_field",
        total_charge_key: str = "total_charge",
        ignore_nonconverged: bool = False,
        save_full_scf_history: bool = False,
        scf_restart: bool = False,
        restart_density: bool = None,
        restart_fermi_level: bool = None,
        use_compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
        compile_dynamic: bool = True,
        compile_fullgraph: bool = False,
        compile_enabled: bool = True,
        compile_scope: str = "scf_chunk",
        compile_chunk_size: int = 1,
        verbose: bool = False,
        pbc_handling: str = "mixed_periodic",
        compensating_jellium: bool = False,
        jellium_slab_bounds: Optional[tuple[float, float]] = None,
        adjust_kspace_cutoff: float = 1.0,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        if verbose:
            mace.tools.setup_logger(level="DEBUG")

        self.results = {}
        self.device = torch_tools.init_device(device)
        torch_tools.set_default_dtype(default_dtype)

        self.scf_options = validate_fixed_point_scf_options(
            scf_options or {},
            CALCULATOR_SCF_DEFAULTS,
        )

        self.model = torch.load(
            f=model_path, map_location=self.device, weights_only=False
        ).to(self.device)
        self.model.kspace_cutoff *= adjust_kspace_cutoff

        if isinstance(self.model, FixedPoint) and not isinstance(
            self.model, FixedPointCore
        ):
            import logging
            logging.getLogger(__name__).info(
                "Auto-converting legacy FixedPoint model to FixedPointCore"
            )
            from scripts.convert_models import convert_fixedpoint_to_core
            self.model = convert_fixedpoint_to_core(self.model)

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
        self.model.electric_potential_descriptor.set_pbc_handling(pbc_handling)
        self._validate_glr_compile_ready_model(self.model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.runner = FixedPointSCFRunner(self.scf_options)
        self.use_compile = use_compile
        self.pbc_handling = pbc_handling
        self.compensating_jellium = compensating_jellium
        self.jellium_slab_bounds = jellium_slab_bounds
        self.compiled_evaluator = None

        self.r_max = float(self.model.r_max)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.head = self.model.heads[0] if hasattr(self.model, "heads") else "Default"
        self.max_l = int(self.model.coulomb_energy.density_max_l)
        self.total_charge_key = total_charge_key
        self.keyspec = self._build_keyspec(
            atomic_multipoles_key=atomic_multipoles_key,
            fermi_level_key=fermi_level_key,
            external_field_key=external_field_key,
            total_charge_key=total_charge_key,
        )
        self.ignore_nonconverged = ignore_nonconverged
        self.save_full_scf_history = save_full_scf_history
        self.scf_restart = scf_restart
        self.restart_density = scf_restart if restart_density is None else restart_density
        self.restart_fermi_level = (
            scf_restart if restart_fermi_level is None else restart_fermi_level
        )
        self.most_recent_density = None
        self.most_recent_fermi_level = None

        if self.use_compile:
            self._validate_compiled_mode()
            self.compiled_evaluator = build_compiled_fixed_point_evaluator(
                model=self.model,
                pbc_handling=pbc_handling,
                num_scf_steps=self.scf_options.num_scf_steps,
                mixing_parameter=self.scf_options.mixing_parameter,
                constant_charge=self.scf_options.constant_charge,
                backend=compile_backend,
                mode=compile_mode,
                dynamic=compile_dynamic,
                fullgraph=compile_fullgraph,
                enabled=compile_enabled,
                scope=compile_scope,
                chunk_size=compile_chunk_size,
            )

    def _validate_compiled_mode(self):
        if self.pbc_handling not in ("pbc", "slab"):
            raise ValueError(
                "Compiled MACEFixedPointSCF supports only "
                f"pbc_handling='pbc' or 'slab', got {self.pbc_handling!r}."
            )
        if self.scf_options.num_scf_steps <= 1:
            raise ValueError(
                "Compiled MACEFixedPointSCF requires num_scf_steps > 1."
            )
        if self.scf_options.initial_density != "local_guess":
            raise ValueError(
                "Compiled MACEFixedPointSCF currently requires "
                "initial_density='local_guess'."
            )
        if self.scf_options.initial_fermi_level != "from_data":
            raise ValueError(
                "Compiled MACEFixedPointSCF currently requires "
                "initial_fermi_level='from_data'."
            )
        if self.restart_density:
            raise ValueError(
                "Compiled MACEFixedPointSCF does not support SCF restart density."
            )
        if self.restart_fermi_level:
            raise ValueError(
                "Compiled MACEFixedPointSCF does not support SCF restart Fermi level."
            )
        if self.save_full_scf_history:
            raise ValueError(
                "Compiled MACEFixedPointSCF does not support save_full_scf_history."
            )

    def _build_keyspec(
        self,
        atomic_multipoles_key: str,
        fermi_level_key: str,
        external_field_key: str,
        total_charge_key: str,
    ) -> mace.data.KeySpecification:
        return mace.data.KeySpecification(
            info_keys={
                "external_field": external_field_key,
                "fermi_level": fermi_level_key,
                "total_charge": total_charge_key,
            },
            arrays_keys={
                "atomic_multipoles": atomic_multipoles_key,
            },
        )

    def _build_batch(self, atoms):
        config = mace.data.config_from_atoms(
            atoms,
            key_specification=self.keyspec,
            head_name=self.head,
        )
        data_loader = torch_geometric.dataloader.DataLoader(
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
        return next(iter(data_loader)).to(self.device)

    def warmup_compiled_evaluator(
        self,
        atoms,
        num_scf_steps: int = 2,
        num_calls: int = 1,
    ) -> None:
        if self.compiled_evaluator is None:
            raise ValueError("warmup_compiled_evaluator requires use_compile=True.")
        if self.compiled_evaluator.compile_scope != "scf_chunk":
            raise ValueError(
                "warmup_compiled_evaluator is supported only for "
                "compile_scope='scf_chunk'."
            )
        if num_calls <= 0:
            return

        intended_num_scf_steps = self.compiled_evaluator.num_scf_steps
        batch = self._build_batch(atoms)
        batch_dict = batch.to_dict()

        try:
            self.compiled_evaluator.set_num_scf_steps(num_scf_steps)
            for _ in range(num_calls):
                self.compiled_evaluator.evaluate(batch_dict)
        finally:
            self.compiled_evaluator.set_num_scf_steps(intended_num_scf_steps)

    def _resolve_restart_state(self, atoms, batch):
        restart_state = {
            "initial_charge_density": None,
            "fermi_level": None,
        }
        has_restart_density = self.restart_density and self.most_recent_density is not None
        has_restart_fermi_level = (
            self.restart_fermi_level and self.most_recent_fermi_level is not None
        )

        if has_restart_density:
            if self.most_recent_density.shape[0] != len(atoms):
                raise ValueError(
                    "previous density coefficients (calc.results) do not match "
                    "the number of atoms in the system"
                )
            restart_state["initial_charge_density"] = torch.as_tensor(
                self.most_recent_density,
                dtype=torch.get_default_dtype(),
                device=self.device,
            )
        if has_restart_fermi_level and self.scf_options.constant_charge:
            restart_state["fermi_level"] = torch.as_tensor(
                self.most_recent_fermi_level,
                dtype=torch.get_default_dtype(),
                device=self.device,
            ).reshape(1)
        return restart_state

    def _run_model(self, batch, restart_state):
        batch_dict = batch.to_dict()
        if self.compiled_evaluator is not None:
            return self.compiled_evaluator.evaluate(batch_dict)

        for p in self.model.parameters():
            p.requires_grad = False
        local_state = self.model.local_part(batch_dict, compute_force=True)

        initial_density = restart_state["initial_charge_density"]
        if initial_density is None:
            initial_density = self.runner.get_initial_density(local_state, batch_dict)

        initial_fermi_level = restart_state["fermi_level"]
        if initial_fermi_level is None:
            initial_fermi_level = self.runner.get_initial_fermi(
                self.model, local_state, batch_dict
            )

        scf_result = self.runner.converge(
            self.model, batch_dict, local_state,
            initial_density, initial_fermi_level, compute_force=True,
        )
        output = self.model.build_observables(
            data=batch_dict,
            local_state=local_state,
            density=scf_result.density,
            fermi_level=scf_result.fermi_level,
            field_feats=scf_result.field_feats,
            training=False,
            compute_force=True,
        )
        output["charges_history"] = scf_result.charges_history
        return output

    def _extract_electrostatic_potentials(self, output):
        if output["esps"] is None:
            return None
        return output["esps"].detach().cpu().numpy()

    def _extract_scf_diagnostics(self, output, atoms):
        charge_history = output["charges_history"].detach().cpu().numpy()
        abs_difference = [
            np.average(np.abs(charge_history[:, :, i + 1] - charge_history[:, :, i]))
            for i in range(charge_history.shape[-1] - 1)
        ]
        final_abs_difference = abs_difference[-1] if abs_difference else 0.0
        total_charge_target = float(atoms.info.get(self.total_charge_key, 0.0))
        partial_charges = output["density_coefficients"].detach().cpu().numpy()[:, 0]
        charge_error = float(np.sum(partial_charges) - total_charge_target)
        total_charge_error = abs(charge_error) > 0.1
        return {
            "charge_history": charge_history,
            "convergence_history": abs_difference,
            "final_abs_difference": final_abs_difference,
            "charge_error": charge_error,
            "total_charge_error": total_charge_error,
            "num_scf_steps": charge_history.shape[-1],
        }

    def _handle_scf_status(self, diagnostics):
        final_abs_difference = diagnostics["final_abs_difference"]
        if final_abs_difference > 1.0 or (
            self.scf_options.constant_charge and diagnostics["total_charge_error"]
        ):
            raise ValueError(
                "SCF diverged or total charge does not match the expected value.\n "
                f"abs_diff={final_abs_difference}, charge_err={diagnostics['charge_error']}"
            )

        if (
            not self.ignore_nonconverged
            and final_abs_difference > self.scf_options.scf_tolerance
        ):
            tol = self.scf_options.scf_tolerance
            steps = self.scf_options.num_scf_steps
            mix = self.scf_options.mixing_parameter
            if final_abs_difference > 10.0:
                logstring = (
                    "\nThe scf cycle diverged, reducing the mixing parameter should help. "
                    "Starting from a better guess for the fermi level can also help."
                )
                logstring += (
                    "\nInspect the convergence_history in calc.results to see what happened"
                )
                warnings.warn(logstring)
            else:
                logstring = (
                    "scf did not converge. mean absolute change at last step="
                    f"{final_abs_difference}, requested tolerance is {tol}"
                )
                logstring += f"\nnum_scf_steps is {steps}, increasing this could help."
                logstring += (
                    f"\nmixing_parameter is {mix}, increasing it could help if the scf is stable. "
                    "If the scf is oscillating or diverging, decreasing it will help."
                )
                logstring += (
                    "\nInspect the convergence_history in calc.results to see what happened"
                )
                warnings.warn(logstring)

    def _build_results(self, output, diagnostics):
        energy = output["energy"].detach().cpu().item()
        forces = output["forces"].detach().cpu().numpy()
        density_coefficients = output["density_coefficients"].detach().cpu().numpy()
        dipole = output["dipole"].detach().cpu().numpy()
        fermi_level = output["fermi_level"].detach().cpu().item()
        external_field = output["external_field"].detach().cpu().numpy()
        electrostatic_energy = output["electrostatic_energy"].detach().cpu().item()
        electron_energy = output["electron_energy"].detach().cpu().item()
        electrostatic_features = output["electrostatic_features"].detach().cpu().numpy()
        electrostatic_potentials = self._extract_electrostatic_potentials(output)

        assert dipole.shape == (1, 3), dipole.shape
        assert external_field.shape == (1, 3), external_field.shape

        energy_eV = energy * self.energy_units_to_eV
        results = {
            "energy": energy_eV,
            "free_energy": energy_eV,
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
            "density_coefficients": density_coefficients,
            "dipole": dipole[0],
            "electrostatic_potentials": electrostatic_potentials,
            "fermi_level": fermi_level,
            "external_field": external_field[0],
            "electrostatic_energy": electrostatic_energy,
            "electron_energy": electron_energy,
            "convergence_history": diagnostics["convergence_history"],
            "num_scf_steps": diagnostics["num_scf_steps"],
            "electrostatic_features": electrostatic_features,
        }
        if self.save_full_scf_history:
            results["full_scf_history"] = diagnostics["charge_history"]
        return results

    def _update_restart_cache(self, output):
        self.most_recent_density = (
            output["density_coefficients"].detach().cpu().numpy()
        )
        self.most_recent_fermi_level = (
            output["fermi_level"].detach().cpu().item()
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, system_changes=system_changes)

        batch = self._build_batch(atoms)
        restart_state = self._resolve_restart_state(atoms, batch)
        output = self._run_model(batch, restart_state)
        diagnostics = self._extract_scf_diagnostics(output, atoms)
        self._handle_scf_status(diagnostics)
        self.results = self._build_results(output, diagnostics)
        self._update_restart_cache(output)


MACEPolarizable = MACEFixedPointSCF
