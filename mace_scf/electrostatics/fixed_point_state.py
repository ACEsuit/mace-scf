from dataclasses import dataclass
from typing import Any, NamedTuple, Optional
import torch


class LocalState(NamedTuple):
    node_feats: torch.Tensor
    all_layer_feats: torch.Tensor
    edge_attrs: torch.Tensor
    edge_feats: torch.Tensor
    field_independent_charge_density: torch.Tensor
    positions: torch.Tensor
    energies: torch.Tensor
    electrostatics_cache: Any
    external_field_features: torch.Tensor
    k_vectors: torch.Tensor
    k_vectors_norms_squared: torch.Tensor
    k_vectors_batch: torch.Tensor
    k0_mask: torch.Tensor


class SCFState(NamedTuple):
    density: torch.Tensor
    fermi_level: torch.Tensor
    field_feats: torch.Tensor
    charges_history: torch.Tensor
    terminated_step: int
    status: str
    final_avg_abs_change: torch.Tensor


@dataclass(frozen=True)
class FixedPointSCFOptions:
    num_scf_steps: int = 100
    scf_tolerance: float = 1e-6
    mixing_parameter: float = 0.25
    constant_charge: bool = True
    use_autograd_forces: bool = True
    initial_density: str = "local_guess"
    initial_fermi_level: str = "from_data"


@dataclass(frozen=True)
class FixedPointTrainingOptions:
    mode: str
    scf: Optional[FixedPointSCFOptions] = None
    linear_solve: str = "inverse"
    fixedpoint_scf_stability: bool = False
