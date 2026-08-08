import torch
from typing import Dict, Optional
from contextlib import nullcontext
from torch_ema import ExponentialMovingAverage
from mace.tools.scatter import scatter_sum

from mace_scf.electrostatics.fixed_point_state import (
    FixedPointSCFOptions,
    FixedPointTrainingOptions,
)
from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
from mace_scf.utils.implicit import make_implicit_scf_module
from mace_scf.utils.linearize_solve import linearize_and_solve_density

import logging


SCF_FALLBACK_ABS_CHANGE_THRESHOLD = 5e-3
SCF_FALLBACK_UNROLLED_STEPS = 10
SCF_FALLBACK_DIVERGED_MIN_STEPS = 2
LINEARIZE_FALLBACK_UNROLLED_STEPS = 15

LINEARIZE_POST_SOLVE_DENSITY_WARNING_THRESHOLD = 1e-7
LINEARIZE_POST_SOLVE_DENSITY_FALLBACK_THRESHOLD = 1e-5
LINEARIZE_POST_SOLVE_CHARGE_WARNING_THRESHOLD = 1e-6
LINEARIZE_POST_SOLVE_CHARGE_FALLBACK_THRESHOLD = 1e-4


def make_model_wrapper(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_args: Dict[str, bool],
    fixed_point_training_options: FixedPointTrainingOptions = None,
) -> "BaseModelWrapper":
    model_class = model.__class__.__name__


    if model_class in ["MACE", "ScaleShiftMACE"]:
        return DefaultModelWrapper(
            model=model,
            optimizer=optimizer,
            output_args=output_args,
        )
    elif model_class in ["FixedPoint", "FixedPointCore"]:
        if fixed_point_training_options is None:
            raise ValueError(
                "fixed_point_training_options must be provided for FixedPoint models"
            )
        return FixedPointWrapper(
            model=model,
            optimizer=optimizer,
            output_args=output_args,
            training_options=fixed_point_training_options,
        )
    elif model_class in [
        "LocalSplitCharges",
        "LocalCharges",
        "FixedChargeBaselinedMACE",
    ]:
        return LocalSourcesModelWrapper(
            model=model,
            optimizer=optimizer,
            output_args=output_args,
        )
    elif model_class == "MACEQEq":
        return QEqModelWrapper(
            model=model,
            optimizer=optimizer,
            output_args=output_args,
        )
    else:
        raise ValueError(f"Model class {model_class} does not have a wrapper class")


class BaseModelWrapper(torch.nn.Module):
    """Abstract base for model wrappers whose forward() computes observables.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        output_args: Dict[str, bool],
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.output_args = output_args
    def forward(self):
        raise NotImplementedError("BaseModelWrapper is an abstract class and cannot be used directly.")


class FixedPointWrapper(BaseModelWrapper):
    """Training wrapper for FixedPointCore models.

    Supports fixed-point training modes:
      - direct: single fixed-point update using reference density from data
      - unroll_scf: full SCF convergence with gradients through the loop
      - implicit: converge SCF then use implicit differentiation for gradients
      - linearize_solve: converge SCF then differentiate a dense linearization
    """

    MODES = ("direct", "unroll_scf", "implicit", "linearize_solve")

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        output_args: Dict[str, bool],
        training_options: FixedPointTrainingOptions,
    ):
        super().__init__(model, optimizer, output_args)
        if not isinstance(training_options, FixedPointTrainingOptions):
            raise TypeError(
                "training_options must be a FixedPointTrainingOptions instance"
            )
        if training_options.mode not in self.MODES:
            raise ValueError(
                f"mode must be one of {self.MODES}, got {training_options.mode}"
            )

        self.training_options = training_options
        self.mode = training_options.mode
        self.scf_options = training_options.scf

        logging.info(
            f"FixedPointWrapper: mode={self.mode}, scf_options={self.scf_options}, "
            f"linear_solve={training_options.linear_solve}, "
            f"fixedpoint_scf_stability={training_options.fixedpoint_scf_stability}"
        )

        if self.mode in ("unroll_scf", "implicit", "linearize_solve"):
            if not isinstance(self.scf_options, FixedPointSCFOptions):
                raise ValueError(f"mode={self.mode} requires FixedPointSCFOptions")
            self._runner = FixedPointSCFRunner(self.scf_options)
        if self.mode in ("implicit", "linearize_solve"):
            self._linear_solve = training_options.linear_solve

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        training: bool = False,
        ema: Optional[ExponentialMovingAverage] = None,
    ):
        param_context = (
            ema.average_parameters()
            if (ema is not None and not training)
            else nullcontext()
        )
        with param_context:
            if self.mode == "direct":
                return self._forward_direct(batch_dict, training)
            elif self.mode == "unroll_scf":
                return self._forward_unroll_scf(batch_dict, training)
            elif self.mode == "implicit":
                return self._forward_implicit(batch_dict, training)
            elif self.mode == "linearize_solve":
                return self._forward_linearize_solve(batch_dict, training)

    def _forward_direct(self, batch_dict, training):
        if training:
            for p in self.model.parameters():
                p.requires_grad = True

        local_state = self.model.local_part(
            batch_dict,
            compute_force=self.output_args["forces"],
        )

        fermi_level_features = self.model.features_from_fermi_level(
            batch_dict["batch"], local_state.positions, batch_dict["fermi_level"]
        )

        field_dep, field_feats = self.model.scf_step(
            batch_dict,
            local_state,
            charge_density_in=batch_dict["density_coefficients"],
            total_charges=batch_dict["density_coefficients"],
            fermi_level_features=fermi_level_features,
        )
        density = local_state.field_independent_charge_density + field_dep

        output = self.model.build_observables(
            data=batch_dict,
            local_state=local_state,
            density=density,
            fermi_level=batch_dict["fermi_level"],
            field_feats=field_feats,
            training=training,
            compute_force=self.output_args["forces"],
            compute_virials=self.output_args["virials"],
            compute_stress=self.output_args["stress"],
        )
        output["charges_history"] = torch.stack([density.detach()], dim=-1)
        return output

    def _forward_unroll_scf(
        self, batch_dict, training, num_scf_steps: Optional[int] = None
    ):
        return self._runner.eval(
            model=self.model,
            data=batch_dict,
            training=training,
            compute_force=self.output_args["forces"],
            compute_virials=self.output_args["virials"],
            compute_stress=self.output_args["stress"],
            num_scf_steps=num_scf_steps,
        )

    def _linearize_post_solve_residuals(
        self, model, batch_dict, local_state, density, fermi_level
    ):
        with torch.no_grad():
            density_check = density.detach()
            fermi_check = fermi_level.detach()
            fermi_features = model.features_from_fermi_level(
                batch_dict["batch"], local_state.positions, fermi_check
            )
            field_dep, _ = model.scf_step(
                data=batch_dict,
                local_state=local_state,
                charge_density_in=density_check,
                total_charges=density_check,
                fermi_level_features=fermi_features,
            )
            density_out = local_state.field_independent_charge_density + field_dep
            density_residual = torch.max(torch.abs(density_check - density_out))

            if self._runner.scf_options.constant_charge:
                num_graphs = batch_dict["ptr"].numel() - 1
                total_charge = scatter_sum(
                    src=density_check[:, 0],
                    index=batch_dict["batch"],
                    dim=-1,
                    dim_size=num_graphs,
                )
                charge_residual = torch.max(
                    torch.abs(total_charge - batch_dict["total_charge"])
                )
            else:
                charge_residual = density_residual.new_tensor(0.0)

        return density_residual, charge_residual

    @staticmethod
    def _max_final_avg_abs_change(scf_result):
        return torch.max(scf_result.final_avg_abs_change.detach()).item()

    def _linearize_fallback_unroll(self, batch_dict, training, reason):
        logging.warning("linearize_solve fallback to unroll_scf: %s", reason)
        return self._forward_unroll_scf(
            batch_dict,
            training,
            num_scf_steps=LINEARIZE_FALLBACK_UNROLLED_STEPS,
        )

    def _forward_implicit(self, batch_dict, training):
        for p in self.model.parameters():
            p.requires_grad = True

        local_state = self.model.local_part(
            batch_dict, compute_force=self.output_args["forces"]
        )

        initial_density = self._runner.get_initial_density(local_state, batch_dict)
        initial_fermi = self._runner.get_initial_fermi(self.model, local_state, batch_dict)
        scf_result = self._runner.converge(
            self.model, batch_dict, local_state, initial_density, initial_fermi,
            compute_force=False,
        )

        if scf_result.status == "diverged":
            fallback_steps = max(
                SCF_FALLBACK_DIVERGED_MIN_STEPS,
                min(SCF_FALLBACK_UNROLLED_STEPS, scf_result.terminated_step + 1),
            )
            return self._forward_unroll_scf(
                batch_dict,
                training,
                num_scf_steps=fallback_steps,
            )

        if not torch.all(
            scf_result.final_avg_abs_change <= SCF_FALLBACK_ABS_CHANGE_THRESHOLD
        ):
            return self._forward_unroll_scf(
                batch_dict,
                training,
                num_scf_steps=SCF_FALLBACK_UNROLLED_STEPS,
            )

        num_graphs = batch_dict["ptr"].numel() - 1
        implicit = make_implicit_scf_module(
            model=self.model,
            solved_density=scf_result.density,
            solved_fermi_level=scf_result.fermi_level,
            positions=local_state.positions,
            constant_charge=self._runner.scf_options.constant_charge,
            num_graphs=num_graphs,
            linear_solve=self._linear_solve,
        )

        implicit.solve(batch_dict)

        density, fermi = implicit.extract()
        if fermi is None:
            fermi = batch_dict["fermi_level"]

        fermi_features = self.model.features_from_fermi_level(
            batch_dict["batch"], local_state.positions, fermi
        )

        field_dep, field_feats = self.model.scf_step(
            batch_dict, local_state, density, density, fermi_features
        )
        density_out = local_state.field_independent_charge_density + field_dep

        output = self.model.build_observables(
            data=batch_dict,
            local_state=local_state,
            density=density_out,
            fermi_level=fermi,
            field_feats=field_feats,
            training=training,
            compute_force=self.output_args["forces"],
            compute_virials=self.output_args["virials"],
            compute_stress=self.output_args["stress"],
        )
        output["charges_history"] = scf_result.charges_history
        return output

    def _forward_linearize_solve(self, batch_dict, training):
        for p in self.model.parameters():
            p.requires_grad = True

        local_state = self.model.local_part(
            batch_dict,
            compute_force=self.output_args["forces"],
        )

        initial_density = self._runner.get_initial_density(local_state, batch_dict)
        initial_fermi = self._runner.get_initial_fermi(self.model, local_state, batch_dict)
        scf_result = self._runner.converge(
            self.model,
            batch_dict,
            local_state,
            initial_density,
            initial_fermi,
            compute_force=False,
        )

        if scf_result.status == "diverged":
            fallback_steps = max(
                SCF_FALLBACK_DIVERGED_MIN_STEPS,
                min(LINEARIZE_FALLBACK_UNROLLED_STEPS, scf_result.terminated_step + 1),
            )
            return self._forward_unroll_scf(
                batch_dict,
                training,
                num_scf_steps=fallback_steps,
            )

        try:
            density, fermi_level, field_feats = linearize_and_solve_density(
                model=self.model,
                data=batch_dict,
                local_state=local_state,
                solved_density=scf_result.density,
                solved_fermi_level=scf_result.fermi_level,
                constant_charge=self._runner.scf_options.constant_charge,
                linear_solve=self._linear_solve,
            )
        except Exception as exc:
            return self._linearize_fallback_unroll(
                batch_dict,
                training,
                (
                    f"linear solve failed with {exc!r}; "
                    f"scf_status={scf_result.status}, "
                    f"terminated_step={scf_result.terminated_step}, "
                    f"max_final_avg_abs_change="
                    f"{self._max_final_avg_abs_change(scf_result):.6e}"
                ),
            )

        density_residual, charge_residual = self._linearize_post_solve_residuals(
            self.model, batch_dict, local_state, density, fermi_level
        )
        density_residual_value = density_residual.item()
        charge_residual_value = charge_residual.item()
        residuals_finite = bool(
            torch.isfinite(density_residual).item()
            and torch.isfinite(charge_residual).item()
        )

        if (
            not residuals_finite
            or density_residual_value > LINEARIZE_POST_SOLVE_DENSITY_FALLBACK_THRESHOLD
            or charge_residual_value > LINEARIZE_POST_SOLVE_CHARGE_FALLBACK_THRESHOLD
        ):
            return self._linearize_fallback_unroll(
                batch_dict,
                training,
                (
                    f"post-solve residual check failed; "
                    f"density_residual={density_residual_value:.6e}, "
                    f"charge_residual={charge_residual_value:.6e}, "
                    f"scf_status={scf_result.status}, "
                    f"terminated_step={scf_result.terminated_step}, "
                    f"max_final_avg_abs_change="
                    f"{self._max_final_avg_abs_change(scf_result):.6e}"
                ),
            )

        if (
            density_residual_value > LINEARIZE_POST_SOLVE_DENSITY_WARNING_THRESHOLD
            or charge_residual_value > LINEARIZE_POST_SOLVE_CHARGE_WARNING_THRESHOLD
        ):
            logging.warning(
                "linearize_solve accepted with elevated post-solve residuals: "
                "density_residual=%.6e, charge_residual=%.6e, "
                "scf_status=%s, terminated_step=%s, "
                "max_final_avg_abs_change=%.6e",
                density_residual_value,
                charge_residual_value,
                scf_result.status,
                scf_result.terminated_step,
                self._max_final_avg_abs_change(scf_result),
            )

        output = self.model.build_observables(
            data=batch_dict,
            local_state=local_state,
            density=density,
            fermi_level=fermi_level,
            field_feats=field_feats,
            training=training,
            compute_force=self.output_args["forces"],
            compute_virials=self.output_args["virials"],
            compute_stress=self.output_args["stress"],
        )
        output["charges_history"] = scf_result.charges_history
        return output


class DefaultModelWrapper(BaseModelWrapper):
    """Training wrapper for standard MACE models (no electrostatics)."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        output_args: Dict[str, bool],
    ):
        super().__init__(model, optimizer, output_args)

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        training: bool = False,
        ema: Optional[ExponentialMovingAverage] = None,
    ):
        param_context = (
            ema.average_parameters()
            if (ema is not None and not training)
            else nullcontext()
        )
        with param_context:
            return self.model(
                batch_dict,
                training=training,
                compute_force=self.output_args["forces"],
                compute_virials=self.output_args["virials"],
                compute_stress=self.output_args["stress"],
            )


class LocalSourcesModelWrapper(BaseModelWrapper):
    """Training wrapper for non-polarizable local-sources models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        output_args: Dict[str, bool],
    ):
        super().__init__(model, optimizer, output_args)

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        training: bool = False,
        ema: Optional[ExponentialMovingAverage] = None,
    ):
        param_context = (
            ema.average_parameters()
            if (ema is not None and not training)
            else nullcontext()
        )
        with param_context:
            return self.model(
                batch_dict,
                training=training,
                compute_force=self.output_args["forces"],
                compute_virials=self.output_args["virials"],
                compute_stress=self.output_args["stress"],
            )


class QEqModelWrapper(BaseModelWrapper):
    """Training wrapper for MaceQEq models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        output_args: Dict[str, bool],
    ):  
        super().__init__(model, optimizer, output_args)

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        training: bool = False,
        ema: Optional[ExponentialMovingAverage] = None,
    ):
        param_context = (
            ema.average_parameters()
            if (ema is not None and not training)
            else nullcontext()
        )
        with param_context:
            return self.model(
                batch_dict,
                training=training,
                compute_force=self.output_args["forces"],
                compute_virials=self.output_args["virials"],
                compute_stress=self.output_args["stress"],
            )