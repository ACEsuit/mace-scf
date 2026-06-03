from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Type

import torch
import numpy as np
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules import (
    EquivariantProductBasisBlock,
    RealAgnosticDensityResidualInteractionBlock,
    InteractionBlock,
)
from mace.modules.irreps_tools import reshape_irreps, tp_out_irreps_with_instructions
from mace.modules.radial import RadialMLP
from mace.tools.scatter import scatter_sum

from .utils import undo_reshape


class MultiLayerFeatureMixer(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        num_interactions: int,
    ):
        super().__init__()
        self.linears = torch.nn.ModuleList()
        for _ in range(num_interactions):
            self.linears.append(o3.Linear(node_feats_irreps, node_feats_irreps))

    def forward(
        self,
        all_node_feats: torch.Tensor, # [num_interactions, n_node, hidden_irreps]
    ) -> torch.Tensor:
        node_feats_out = torch.zeros_like(all_node_feats[0])
        for i, linear in enumerate(self.linears):
            node_feats_out += linear(all_node_feats[i])
        return node_feats_out


@compile_mode("script")
class FeatureMixerBlock(torch.nn.Module):
    def __init__(
        self, 
        feat_irreps_1: o3.Irreps, 
        feat_irreps_2: o3.Irreps
    ):
        super().__init__()
        self.feat_irreps_1 = feat_irreps_1
        self.feat_irreps_2 = feat_irreps_2
        self._setup()  # Call the subclass-specific setup method

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def forward(
        self, 
        feats1: torch.Tensor, 
        feats2: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


@compile_mode("script")
class UncoupledProductMixer(FeatureMixerBlock):
    def _setup(self):
        self.linear_up = o3.Linear(
            self.feat_irreps_2,
            self.feat_irreps_1,
            biases=True
        )
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.feat_irreps_1, self.feat_irreps_1, self.feat_irreps_1
        )
        self.conv_tp = o3.TensorProduct(
            self.feat_irreps_1,
            self.feat_irreps_1,
            irreps_mid,
            instructions=instructions,
            internal_weights=True,
        )
        self.linear_out = o3.Linear(
            irreps_mid.simplify(), self.feat_irreps_1
        )

    def forward(self, node_feats, field_feats):
        field_ = self.linear_up(field_feats)
        product = self.conv_tp(node_feats, field_, None)
        return self.linear_out(product)


@compile_mode("script")
class AdditiveFieldMixer(FeatureMixerBlock):
    def _setup(self):
        self.linear = o3.Linear(
            self.feat_irreps_2,
            self.feat_irreps_1,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

    def forward(self, node_feats, field_feats):
        return node_feats + self.linear(field_feats)


@compile_mode("script")
class NoMixer(FeatureMixerBlock):
    def _setup(self):
        pass

    def forward(self, feats1, feats2):
        del feats2
        return feats1


class PotentialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        potential_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        node_attrs_irreps: o3.Irreps,
        **kwargs,
    ):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.potential_irreps = potential_irreps
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class LinearPotentialEmbedding(PotentialEmbeddingBlock):
    def _setup(self) -> None:
        self.prod = o3.FullyConnectedTensorProduct(
            irreps_in1=self.potential_irreps,
            irreps_in2=self.node_attrs_irreps,
            irreps_out=self.node_feats_irreps,
        )

    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        return self.prod(potential_feats, node_attrs)


class BiasedLinearPotentialEmbedding(PotentialEmbeddingBlock):
    def _setup(self) -> None:
        self.prod = o3.FullyConnectedTensorProduct(
            irreps_in1=self.potential_irreps,
            irreps_in2=self.node_attrs_irreps,
            irreps_out=self.node_feats_irreps,
        )
        self.node_feats_linear = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        return self.prod(potential_feats, node_attrs) + self.node_feats_linear(node_feats)


class TanhPotentialEmbedding(PotentialEmbeddingBlock):
    def _setup(self) -> None:
        width = 10.0
        N = 5
        self.register_buffer(
            "widths",
            torch.logspace(0, np.log10(width), N, dtype=torch.get_default_dtype()),
        )
        self.linear = o3.Linear(
            irreps_in=self.potential_irreps * N,
            irreps_out=self.node_feats_irreps,
            biases=True,
        )
        self.node_feats_linear = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        expanded_feats = potential_feats.unsqueeze(-1) / self.widths
        expanded_feats = expanded_feats.swapaxes(1, 2).flatten(start_dim=-2)
        nonlinear_feats = torch.tanh(expanded_feats)
        return self.linear(nonlinear_feats) + self.node_feats_linear(node_feats)



class GeneralNonLinearBiasReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        irreps_out: Optional[o3.Irreps] = None,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.irreps_out = irrep_out
        if irreps_out is not None:
            self.irreps_out = irreps_out
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        activation_fn = torch.nn.functional.silu
        act_gates_fn = torch.nn.functional.sigmoid
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[activation_fn for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[act_gates_fn] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_nonlin)
        self.linear_mid = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.irreps_nonlin
        )
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.irreps_out
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.linear_1(node_feats)
        x = self.equivariant_nonlin(x)
        x = self.linear_mid(x)
        x = self.equivariant_nonlin(x)
        return self.linear_2(x)  # [n_nodes, 1]



class NoNonLinearity(torch.nn.Module):
    def __init__(
        self,
        invar_irreps: o3.Irreps,
    ):
        super().__init__()
        self.invar_irreps = invar_irreps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ShallowNonLinearity(torch.nn.Module):
    def __init__(
        self,
        invar_irreps: o3.Irreps,
    ):
        super().__init__()
        channels = invar_irreps.count(o3.Irrep(0, 1))
        self.mlp = nn.FullyConnectedNet(
            [channels, channels],
            mysilu
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SkipMLPLinearity(torch.nn.Module):
    def __init__(
        self,
        invar_irreps: o3.Irreps,
    ):
        super().__init__()
        channels = invar_irreps.count(o3.Irrep(0, 1))
        """ self.mlp = nn.FullyConnectedNet(
            [channels, 64, 64, channels],
            torch.nn.functional.silu
        ) """
        self.mlp = RadialMLP([channels, 64, 64, channels])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class MLPNonLinearity(torch.nn.Module):
    def __init__(
        self,
        invar_irreps: o3.Irreps,
    ):
        super().__init__()
        channels = invar_irreps.count(o3.Irrep(0, 1))
        self.mlp = nn.FullyConnectedNet(
            [channels, 64, 64, channels],
            mysilu
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


@compile_mode("script")
class EnvironmentDependentSourceBlock(torch.nn.Module):
    def __init__(
        self, 
        irreps_in: o3.Irreps, 
        max_l: int,
        zero_charges: bool = False
    ):
        super().__init__()
        irreps_out = o3.Irreps.spherical_harmonics(max_l)
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        self.zero_charges = zero_charges

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor # [n_node, hidden_irreps.dim]
    ) -> torch.Tensor:
        mpoles = self.linear(node_feats) # [n_node, (max_l+1)**2]
        if self.zero_charges:
            zeroed_mpoles = torch.zeros_like(mpoles)
            zeroed_mpoles[:,1:] = mpoles[:,1:]
        else:
            zeroed_mpoles = mpoles
        return zeroed_mpoles.unsqueeze(-2) # [n_node, 1, (max_l+1)**2]


@compile_mode("script")
class ScaledEnvironmentDependentSourceBlock(torch.nn.Module):
    def __init__(
        self, 
        irreps_in: o3.Irreps, 
        max_l: int,
        atom_density_scaling: torch.Tensor,
        zero_charges: bool = False
    ):
        super().__init__()
        irreps_out = o3.Irreps.spherical_harmonics(max_l)
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        self.zero_charges = zero_charges
        self.register_buffer(
            "atom_density_scale", torch.tensor(atom_density_scaling)
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor # [n_node, hidden_irreps.dim]
    ) -> torch.Tensor:
        mpoles = self.linear(node_feats) # [n_node, (max_l+1)**2]
        if self.zero_charges:
            zeroed_mpoles = torch.zeros_like(mpoles)
            zeroed_mpoles[:,1:] = mpoles[:,1:]
        else:
            zeroed_mpoles = mpoles
        scaling = torch.matmul(node_attrs, torch.atleast_2d(self.atom_density_scale).T)
        return (scaling * zeroed_mpoles).unsqueeze(-2) # [n_node, 1, (max_l+1)**2]



@compile_mode("script")
class LinearPolarizabilityReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
    ):
        super().__init__()
        self.irreps_out = o3.Irreps("1x0e+1x2e")
        self.linear1 = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_in)
        self.tp = o3.ElementwiseTensorProduct(irreps_in1=irreps_in, irreps_in2=irreps_in)
        self.linear2 = o3.Linear(irreps_in=self.tp.irreps_out, irreps_out=self.irreps_out)
        self.bypass_linear = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        bit = self.linear1(x)
        bit2 = self.tp(bit, bit)
        return self.linear2(bit2) + self.bypass_linear(x)


class FieldUpdateBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        field_norm_factor: float,
        radial_MLP: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self.potential_irreps = potential_irreps
        self.charges_irreps = charges_irreps
        self.register_buffer(
            "field_norm_factor", torch.tensor(field_norm_factor)
        )

        self._setup(**kwargs)

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
        total_charges: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError



def instructions_for_sparse_tp(feat_in1, feat_in2, feat_out):
    channels1 = feat_in1.count(o3.Irrep(0, 1))
    channels2 = feat_in2.count(o3.Irrep(0, 1))
    channels3 = feat_out.count(o3.Irrep(0, 1))
    assert channels1 == channels2 and channels1 == channels3
    _, instructions = tp_out_irreps_with_instructions(
        feat_in1, feat_in2, feat_out
    )
    new_instructions = []
    for instr in instructions:
        i, j, k, mode, trainable = instr
        new_instructions.append((i, j, 0, mode, trainable))
    return new_instructions





def mysilu(x):
    return x * torch.sigmoid(x)



@compile_mode("script")
class OneBodyVariableUpdate(FieldUpdateBlock):
    def _setup(
        self,
        potential_embedding_cls: Type[PotentialEmbeddingBlock] = BiasedLinearPotentialEmbedding,
        nonlinearity_cls: Type[torch.nn.Module] = NoNonLinearity,
        num_elements: Optional[int] = None,
        **kwargs
    ) -> None:
        # product irreps is node_feats_irreps but only l=0
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
        )
        
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps,
            self.node_feats_irreps,
            invar_irreps
        )
        self.dot_products = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )
        self.nonlinearity = nonlinearity_cls(
            invar_irreps=invar_irreps,
        )
        _, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )
        self.element_select_out = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.charges_irreps
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
        total_charges: torch.Tensor,
    ) -> torch.Tensor:
        mixed_feats = self.potential_embedding(
            potential_features,
            node_feats,
            node_attrs
        )
        invariant_descriptors = self.dot_products(node_feats, mixed_feats)
        nonlin_feats = self.nonlinearity(invariant_descriptors)
        new_feats = self.tp_out(node_feats, nonlin_feats)
        multipoles = self.element_select_out(new_feats, node_attrs)
        return multipoles


@compile_mode("script")
class OneBodyScaledNonLinearUpdate(FieldUpdateBlock):
    def _setup(
        self,
        atom_density_scaling: torch.Tensor,
        potential_embedding_cls: Type[PotentialEmbeddingBlock] = BiasedLinearPotentialEmbedding,
        nonlinearity_cls: Type[torch.nn.Module] = NoNonLinearity,
        num_elements: Optional[int] = None,
        **kwargs
    ) -> None:
        # product irreps is node_feats_irreps but only l=0
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
        )
        self.nonlinearity = nonlinearity_cls(
            invar_irreps=invar_irreps,
        )
        _, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )
        instructions = [
            (0,0,0,'uvu',True),
            (1,1,0,'uvu',True),
            (0,1,1,'uvu',True),
            (1,0,1,'uvu',True),
        ]
        self.tp_skip = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )
        self.element_select_out = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.charges_irreps
        )
        self.num_channels = invar_irreps.count(o3.Irrep(0, 1))
        self.register_buffer(
            "atom_density_scale", torch.tensor(atom_density_scaling)
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
        total_charges: torch.Tensor,
    ) -> torch.Tensor:
        mixed_feats = self.potential_embedding(
            potential_features,
            node_feats,
            node_attrs
        )
        skip_piece = self.tp_skip(node_feats, mixed_feats)
        nonlin_feats = self.nonlinearity(mixed_feats[:,:self.num_channels])
        new_feats = self.tp_out(node_feats, nonlin_feats) + skip_piece
        multipoles = self.element_select_out(new_feats, node_attrs)

        scaling = torch.matmul(node_attrs, torch.atleast_2d(self.atom_density_scale).T)
        return scaling * multipoles


@compile_mode("script")
class OneBodyScaledDPNonLinearUpdate(FieldUpdateBlock):
    def _setup(
        self,
        atom_density_scaling: torch.Tensor,
        potential_embedding_cls: Type[PotentialEmbeddingBlock] = BiasedLinearPotentialEmbedding,
        nonlinearity_cls: Type[torch.nn.Module] = NoNonLinearity,
        num_elements: Optional[int] = None,
        **kwargs
    ) -> None:
        # product irreps is node_feats_irreps but only l=0
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
        )
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps,
            self.node_feats_irreps,
            invar_irreps
        )
        self.dot_products = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )
        self.nonlinearity = nonlinearity_cls(
            invar_irreps=invar_irreps,
        )
        _, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )
        instructions = [
            (0,0,0,'uvu',True),
            (1,1,0,'uvu',True),
            (0,1,1,'uvu',True),
            (1,0,1,'uvu',True),
        ]
        self.tp_skip = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )
        self.element_select_out = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.charges_irreps
        )
        self.num_channels = invar_irreps.count(o3.Irrep(0, 1))
        self.register_buffer(
            "atom_density_scale", torch.tensor(atom_density_scaling)
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
        total_charges: torch.Tensor,
    ) -> torch.Tensor:
        mixed_feats = self.potential_embedding(
            potential_features,
            node_feats,
            node_attrs
        )
        skip_piece = self.tp_skip(node_feats, mixed_feats)
        invariant_descriptors = self.dot_products(node_feats, mixed_feats)
        nonlin_feats = self.nonlinearity(invariant_descriptors)
        new_feats = self.tp_out(node_feats, nonlin_feats) + skip_piece
        multipoles = self.element_select_out(new_feats, node_attrs)

        scaling = torch.matmul(node_attrs, torch.atleast_2d(self.atom_density_scale).T)
        return scaling * multipoles




@compile_mode("script")
class AugRealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear_2 = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        self.reshape = reshape_irreps(self.target_irreps)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        central_feats = self.linear_2(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender] + central_feats[receiver], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        density = scatter_sum(src=edge_density, index=receiver, dim=0, dim_size=num_nodes)
        message = self.linear(message) / (density + 1)
        return (
            self.reshape(message),
            sc,
        )



@compile_mode("script")
class ManyBodyUpdate(FieldUpdateBlock):
    def _setup(
        self,
        potential_embedding_cls: Type[PotentialEmbeddingBlock] = LinearPotentialEmbedding,
        central_atom_mixer_cls: Type[FeatureMixerBlock] = AdditiveFieldMixer,
        central_atom_feats_mixer_cls: Type[FeatureMixerBlock] = NoMixer,
        interaction_cls: Type[InteractionBlock] = RealAgnosticDensityResidualInteractionBlock,
        nonlinearity_cls: Type[torch.nn.Module] = NoNonLinearity,
        correlation: int = 1,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        **kwargs
    ) -> None:
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
        )
        self.central_atom_mixer = central_atom_mixer_cls(
            self.target_irreps, self.node_feats_irreps
        )
        self.central_atom_node_feats_mixer = central_atom_feats_mixer_cls(
            self.target_irreps, self.node_feats_irreps
        )
        self.interaction = interaction_cls(
            node_attrs_irreps=self.node_attrs_irreps,
            node_feats_irreps=self.node_feats_irreps,
            edge_attrs_irreps=self.edge_attrs_irreps,
            edge_feats_irreps=self.edge_feats_irreps,
            target_irreps=self.target_irreps,
            hidden_irreps=self.hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_MLP=self.radial_MLP,
        )
        self.reshape = reshape_irreps(self.target_irreps)
        self.unreshape = undo_reshape(self.target_irreps)
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=self.target_irreps,
            target_irreps=self.hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc,
        )
        self.activation = nonlinearity_cls(invar_irreps)
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps,
            self.node_feats_irreps,
            invar_irreps
        )
        self.dot_products = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )
        _, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )
        self.element_select_out = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.charges_irreps
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor, # local_self_interaction_terms
        total_charges: torch.Tensor,
    ) -> torch.Tensor:
        mixed_feats = self.potential_embedding(
            potential_features,
            node_feats,
            node_attrs
        )
        new_feats_, sc = self.interaction(
            node_attrs=node_attrs,
            node_feats=mixed_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )
        #new_feats_ = self.reshape(self.central_atom_mixer(self.unreshape(new_feats_), mixed_feats))
        new_feats = self.unreshape(new_feats_)
        new_feats = self.central_atom_mixer(new_feats, mixed_feats)
        if hasattr(self, "central_atom_node_feats_mixer"):
            new_feats = self.central_atom_node_feats_mixer(new_feats, node_feats)
        product = self.product(
            node_feats=self.reshape(new_feats), 
            sc=sc,
            node_attrs=node_attrs,
        )
        invariant_descriptors = self.dot_products(node_feats, product)
        invariant_descriptors = self.activation(invariant_descriptors)
        new_feats = self.tp_out(node_feats, invariant_descriptors)
        multipoles = self.element_select_out(new_feats, node_attrs)
        return multipoles



class PostScfReadout(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        radial_MLP: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self.potential_irreps = potential_irreps
        self.charges_irreps = charges_irreps
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError



class OneBodyMLPFieldReadout(PostScfReadout):
    def _setup(self, **kwargs):
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.linear_up_q = o3.Linear(self.charges_irreps, self.node_feats_irreps, biases=True)
        self.linear_up_v = o3.Linear(self.potential_irreps, self.node_feats_irreps, biases=True)
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps,
            self.node_feats_irreps,
            invar_irreps
        )
        self.dot_products_q = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps,
            self.node_feats_irreps,
            invar_irreps
        )
        self.dot_products_v = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )

        self.mlp = nn.FullyConnectedNet(
            [invar_irreps.count(o3.Irrep(0, 1)), 128, 128, 128, 1],
            torch.nn.functional.silu
        ) 

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ):
        q_up = self.linear_up_q(charges_induced+charges_0)
        v_up = self.linear_up_v(field_feats)
        invar_feats = self.dot_products_q(node_feats, q_up) + self.dot_products_v(node_feats, v_up)
        return self.mlp(invar_feats).squeeze(-1)



class NullFieldReadout(PostScfReadout):
    def _setup(self, **kwargs):
        pass

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ):
        return torch.zeros((node_attrs.shape[0],),dtype=node_attrs.dtype, device=node_attrs.device)


class StrictQuadraticFieldEnergyReadout(PostScfReadout):
    def _setup(self, **kwargs):
        irreps_mid = o3.Irreps("32x0e + 32x1o")

        # mixing of the fixed and induced components
        self.linear_qa = o3.Linear(self.charges_irreps, irreps_mid, biases=False)
        self.linear_va = o3.Linear(self.potential_irreps, irreps_mid, biases=False)

        # tensor products between charges and fields
        self.qv_tp = o3.FullyConnectedTensorProduct(
            irreps_mid, irreps_mid, o3.Irreps("32x0e")
        )

        # tp down
        self.tp_down = o3.FullyConnectedTensorProduct(
            o3.Irreps("32x0e"), self.node_feats_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ):
        fields = 0.01 * field_feats
        charges = self.linear_qa(charges_induced - charges_0)
        fields = self.linear_va(fields)

        # tensor product between charges and fields
        qv = self.qv_tp(charges, fields)

        # final contraction
        energy = self.tp_down(qv, node_feats)

        return energy.squeeze(-1)

    
class QuadraticChargesEnergyReadout(PostScfReadout):
    def _setup(self, **kwargs):
        irreps_mid = o3.Irreps("32x0e + 32x1o")

        # mixing of the fixed and induced components
        self.linear_qa = o3.Linear(self.charges_irreps, irreps_mid, biases=True)
        self.linear_qb = o3.Linear(self.charges_irreps, irreps_mid, biases=True)

        # tensor products between charges and fields
        self.qq_tp = o3.FullyConnectedTensorProduct(
            irreps_mid, irreps_mid, irreps_mid
        )

        # tp down
        self.tp_down = o3.FullyConnectedTensorProduct(
            irreps_mid, self.node_feats_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ):
        charges = self.linear_qa(charges_induced) + self.linear_qb(charges_0)

        # tensor product between charges and fields
        qq = self.qq_tp(charges, charges)

        # final contraction
        energy = self.tp_down(qq, node_feats)
    
        return energy.squeeze(-1)


class LinearChargesEnergyReadout(PostScfReadout):
    def _setup(self, **kwargs):
        self.tp = o3.FullyConnectedTensorProduct(
            self.charges_irreps, self.node_feats_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ):
        energy = self.tp(charges_induced, node_feats)
    
        return energy.squeeze(-1)

@compile_mode("script")
class ManyBodyChargesReadout(PostScfReadout):
    def _setup(
        self,
        num_elements,
        **kwargs
    ) -> None:
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        charges_embedding_cls = BiasedLinearPotentialEmbedding
        nonlinearity_cls = MLPNonLinearity
        central_atom_mixer_cls = AdditiveFieldMixer
        interaction_cls = RealAgnosticDensityResidualInteractionBlock

        self.charges_embedding = charges_embedding_cls(
            potential_irreps=self.charges_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
        )
        self.central_atom_mixer = central_atom_mixer_cls(
            self.target_irreps, self.node_feats_irreps
        )
        self.interaction = interaction_cls(
            node_attrs_irreps=self.node_attrs_irreps,
            node_feats_irreps=self.node_feats_irreps,
            edge_attrs_irreps=self.edge_attrs_irreps,
            edge_feats_irreps=self.edge_feats_irreps,
            target_irreps=self.target_irreps,
            hidden_irreps=self.hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_MLP=self.radial_MLP,
        )
        self.reshape = reshape_irreps(self.target_irreps)
        self.unreshape = undo_reshape(self.target_irreps)
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=self.target_irreps,
            target_irreps=self.hidden_irreps,
            correlation=2,
            num_elements=num_elements,
            use_sc=True,
        )
        self.activation = nonlinearity_cls(invar_irreps)
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps,
            self.node_feats_irreps,
            invar_irreps
        )
        self.dot_products = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )
        _, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )
        self.element_select_out = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ) -> torch.Tensor:
        # create pot feats
        mixed_feats = self.charges_embedding(
            charges_induced + charges_0,
            node_feats,
            node_attrs
        )
        new_feats_, sc = self.interaction(
            node_attrs=node_attrs,
            node_feats=mixed_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )
        new_feats = self.unreshape(new_feats_)
        new_feats = self.central_atom_mixer(new_feats, mixed_feats)
        product = self.product(
            node_feats=self.reshape(new_feats), 
            sc=sc,
            node_attrs=node_attrs,
        )
        invariant_descriptors = self.dot_products(node_feats, product)
        invariant_descriptors = self.activation(invariant_descriptors)
        new_feats = self.tp_out(node_feats, invariant_descriptors)
        energy = self.element_select_out(new_feats, node_attrs)
        return energy.squeeze(-1)


@compile_mode("script")
class ManyBodyChargesFieldReadout(PostScfReadout):
    def _setup(
        self,
        num_elements,
        **kwargs
    ) -> None:
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        charges_embedding_cls = BiasedLinearPotentialEmbedding
        nonlinearity_cls = MLPNonLinearity
        central_atom_mixer_cls = AdditiveFieldMixer
        interaction_cls = RealAgnosticDensityResidualInteractionBlock

        self.charges_embedding = charges_embedding_cls(
            potential_irreps=self.charges_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
        )
        self.field_embedding = charges_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
        )
        self.central_atom_mixer = central_atom_mixer_cls(
            self.target_irreps, self.node_feats_irreps
        )
        self.interaction = interaction_cls(
            node_attrs_irreps=self.node_attrs_irreps,
            node_feats_irreps=self.node_feats_irreps,
            edge_attrs_irreps=self.edge_attrs_irreps,
            edge_feats_irreps=self.edge_feats_irreps,
            target_irreps=self.target_irreps,
            hidden_irreps=self.hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_MLP=self.radial_MLP,
        )
        self.reshape = reshape_irreps(self.target_irreps)
        self.unreshape = undo_reshape(self.target_irreps)
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=self.target_irreps,
            target_irreps=self.hidden_irreps,
            correlation=2,
            num_elements=num_elements,
            use_sc=True,
        )
        self.activation = nonlinearity_cls(invar_irreps)
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps,
            self.node_feats_irreps,
            invar_irreps
        )
        self.dot_products = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )
        _, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )
        self.element_select_out = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ) -> torch.Tensor:
        # create pot feats
        mixed_feats = self.charges_embedding(
            charges_induced + charges_0,
            node_feats,
            node_attrs
        ) + self.field_embedding(
            field_feats,
            node_feats,
            node_attrs
        )
        new_feats_, sc = self.interaction(
            node_attrs=node_attrs,
            node_feats=mixed_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )
        new_feats = self.unreshape(new_feats_)
        new_feats = self.central_atom_mixer(new_feats, mixed_feats)
        product = self.product(
            node_feats=self.reshape(new_feats), 
            sc=sc,
            node_attrs=node_attrs,
        )
        invariant_descriptors = self.dot_products(node_feats, product)
        invariant_descriptors = self.activation(invariant_descriptors)
        new_feats = self.tp_out(node_feats, invariant_descriptors)
        energy = self.element_select_out(new_feats, node_attrs)
        return energy.squeeze(-1)
