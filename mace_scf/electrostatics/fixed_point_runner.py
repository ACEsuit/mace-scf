"""
FixedPointSCFRunner: high-level driver that binds fixed-point SCF policy
decisions at construction time so callers never branch on options.
"""
from typing import Dict, Optional
import torch

from .fixed_point_state import LocalState, SCFState, FixedPointSCFOptions
from .fixed_point_scf import converge_constant_fermi, converge_constant_charge


class FixedPointSCFRunner:
    """
    Drives a FixedPointCore model through local_part -> converge -> build_observables.

    Policy decisions (which convergence loop, how to initialise density and
    Fermi level) are resolved once in __init__ and exposed as bound public
    methods so downstream code never needs to inspect FixedPointSCFOptions.
    """

    def __init__(self, scf_options: FixedPointSCFOptions):
        self.scf_options = scf_options

        # Bind convergence function
        if scf_options.constant_charge:
            self._converge_fn = converge_constant_charge
        else:
            self._converge_fn = converge_constant_fermi

        # Bind initial-density provider
        if scf_options.initial_density == "from_data":
            self._get_initial_density = self._density_from_data
        else:
            self._get_initial_density = self._density_from_local_guess

        # Bind initial Fermi-level provider
        if scf_options.initial_fermi_level == "from_data":
            self._get_initial_fermi = self._fermi_from_data
        else:
            self._get_initial_fermi = self._fermi_zero

    # ------ public bound methods ------ #

    def get_initial_density(
        self, local_state: LocalState, data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._get_initial_density(local_state, data)

    def get_initial_fermi(
        self, model, local_state: LocalState, data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._get_initial_fermi(model, local_state, data)

    def converge(
        self,
        model,
        data: Dict[str, torch.Tensor],
        local_state: LocalState,
        initial_density: torch.Tensor,
        initial_fermi_level: torch.Tensor,
        compute_force: bool = False,
        num_scf_steps: Optional[int] = None,
    ) -> SCFState:
        scf_options = self.scf_options
        if num_scf_steps is not None:
            scf_options = FixedPointSCFOptions(
                num_scf_steps=int(num_scf_steps),
                scf_tolerance=self.scf_options.scf_tolerance,
                mixing_parameter=self.scf_options.mixing_parameter,
                constant_charge=self.scf_options.constant_charge,
                use_autograd_forces=self.scf_options.use_autograd_forces,
                initial_density=self.scf_options.initial_density,
                initial_fermi_level=self.scf_options.initial_fermi_level,
            )
        return self._converge_fn(
            model=model,
            data=data,
            local_state=local_state,
            initial_density=initial_density,
            initial_fermi_level=initial_fermi_level,
            scf_options=scf_options,
            compute_force=compute_force,
        )

    # ------ main entry point ------ #

    def eval(
        self,
        model,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        num_scf_steps: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Full evaluation: backbone -> SCF -> observables.

        Uses the bound providers for initial density and Fermi level.
        For custom initial values, call local_part/converge/build_observables
        directly instead.
        """
        if training:
            for p in model.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = False

        local_state = model.local_part(
            data,
            compute_force=compute_force,
        )

        initial_density = self.get_initial_density(local_state, data)
        initial_fermi_level = self.get_initial_fermi(model, local_state, data)

        scf_result = self.converge(
            model=model,
            data=data,
            local_state=local_state,
            initial_density=initial_density,
            initial_fermi_level=initial_fermi_level,
            compute_force=compute_force,
            num_scf_steps=num_scf_steps,
        )
        
        output = model.build_observables(
            data=data,
            local_state=local_state,
            density=scf_result.density,
            fermi_level=scf_result.fermi_level,
            field_feats=scf_result.field_feats,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )
        output["charges_history"] = scf_result.charges_history
        return output

    # ------ private density / Fermi providers ------ #

    @staticmethod
    def _density_from_data(
        local_state: LocalState, data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return data["density_coefficients"].clone()

    @staticmethod
    def _density_from_local_guess(
        local_state: LocalState, data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return local_state.field_independent_charge_density.clone()

    @staticmethod
    def _fermi_from_data(
        model, local_state: LocalState, data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return data["fermi_level"].clone()

    @staticmethod
    def _fermi_zero(
        model, local_state: LocalState, data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return model.fermi_level_offset.expand_as(data["fermi_level"]).clone()
