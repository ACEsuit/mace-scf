from typing import Callable, Optional, Tuple, Union, List
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode
import numpy as np

from mace.tools.scatter import scatter_sum
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from mace.modules.radial import ChebychevBasis
from .field_blocks import EnvironmentDependentSourceBlock


@compile_mode("script")
class PerAtomFormalChargesBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_attr: torch.Tensor, node_charges: torch.Tensor):
        return node_charges


@compile_mode("script")
class PerSpeciesFormalChargesBlock(torch.nn.Module):
    formal_charges: torch.Tensor

    def __init__(
        self,
        formal_charges: Union[np.ndarray, torch.Tensor],
    ):
        super().__init__()
        assert len(formal_charges.shape) == 1
        self.register_buffer(
            "formal_charges",
            torch.tensor(formal_charges, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self, node_attr: torch.Tensor, node_charges: torch.Tensor
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(node_attr, self.formal_charges)

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.formal_charges])
        return f"{self.__class__.__name__}(charges=[{formatted_energies}])"



class OxidationStateEmbeddingBlock(torch.nn.Module):
    def __init__(
        self, 
        node_feats_irreps: o3.Irreps, 
        oxidation_state_range: Tuple[float, float] = (-4.,4.),
        num_basis_oxidation=8
    ):
        super().__init__()

        self.register_buffer(
            "oxidation_state_range", torch.tensor(oxidation_state_range, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "factor", torch.tensor(2.0 / (oxidation_state_range[1] - oxidation_state_range[0]))
        ) 
        self.oxidation_basis = ChebychevBasis(
            r_max=oxidation_state_range[1] - oxidation_state_range[0],
            num_basis=num_basis_oxidation,
        )
        invariant_node_feats = o3.Irreps(f"{node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.oxidation_state_linear = o3.Linear(
            o3.Irreps(f"{num_basis_oxidation}x0e"), invariant_node_feats, biases=True
        )
    
    def forward(
        self,
        node_formal_charges: torch.Tensor
    ) -> torch.Tensor:
        oxidation_feats = self.oxidation_state_linear(self.oxidation_basis(
            (node_formal_charges.unsqueeze(-1) - self.oxidation_state_range[0]) * self.factor - 1.
        ))
        return oxidation_feats


class NoOxidationStateMixer(torch.nn.Module):
    def __init__(self, node_feats_irreps: o3.Irreps, oxidation_state_range: Tuple):
        super().__init__()
    def forward(
        self, 
        node_attrs: torch.Tensor, 
        node_feats: torch.Tensor, 
        node_formal_charges: torch.Tensor
    ) -> torch.Tensor:
        return node_feats


class ProductOxidationStateMixer(torch.nn.Module):
    def __init__(self, node_feats_irreps: o3.Irreps, oxidation_state_range: Tuple):
        super().__init__()

        self.ox_embedding = OxidationStateEmbeddingBlock(node_feats_irreps=node_feats_irreps, oxidation_state_range=oxidation_state_range)

        invariant_node_feats = o3.Irreps(f"{node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.oxidation_state_tp = o3.FullyConnectedTensorProduct(
            node_feats_irreps,
            invariant_node_feats,
            node_feats_irreps,
        )

    def forward(
        self, 
        node_attrs: torch.Tensor, 
        node_feats: torch.Tensor, 
        node_formal_charges: torch.Tensor
    ) -> torch.Tensor:
        oxidation_feats = self.ox_embedding(node_formal_charges)
        node_feats = self.oxidation_state_tp(node_feats, oxidation_feats)
        return node_feats


class SumOxidationStateMixer(torch.nn.Module):
    def __init__(self, node_feats_irreps: o3.Irreps, oxidation_state_range: Tuple):
        super().__init__()
        self.ox_embedding = OxidationStateEmbeddingBlock(node_feats_irreps=node_feats_irreps, oxidation_state_range=oxidation_state_range)

    def forward(
        self, 
        node_attrs: torch.Tensor, 
        node_feats: torch.Tensor, 
        node_formal_charges: torch.Tensor
    ) -> torch.Tensor:
        ox_feats = self.ox_embedding(node_formal_charges)
        return node_feats + ox_feats


@compile_mode("script")
class OxidationDependentSymmetricPredictionSourceBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        max_l: int,
        num_elements: int,
        oxidation_state_range: Tuple[float, float] = (-4.,4.),
        num_basis_oxidation: int = 8,
    ):
        super().__init__()
        self.register_buffer(
            "oxidation_state_range", torch.tensor(oxidation_state_range, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "factor", torch.tensor(2.0 / (oxidation_state_range[1] - oxidation_state_range[0]))
        ) 

        # higher multipoles
        self.multipole_block = EnvironmentDependentSourceBlock(
            irreps_in=node_feats_irreps,
            max_l=max_l,
            zero_charges=True
        )

        # oxidation embedding
        self.oxidation_basis = ChebychevBasis(
            r_max=oxidation_state_range[1] - oxidation_state_range[0],
            num_basis=num_basis_oxidation,
        )
        invariant_node_feat_irreps = o3.Irreps(f"{node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.oxidation_state_linear = o3.Linear(
            o3.Irreps(f"{num_basis_oxidation}x0e"), invariant_node_feat_irreps, biases=True
        )

        _, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps, invariant_node_feat_irreps, node_feats_irreps
        )
        self.oxidation_state_tp = o3.TensorProduct(
            node_feats_irreps,
            invariant_node_feat_irreps,
            node_feats_irreps,
            instructions=instructions,
            internal_weights=True,
        )

        self.linear1 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear2 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        _, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps, edge_attrs_irreps, invariant_node_feat_irreps
        )
        # compress all the scalar outputs into the same multiplicties as the input
        new_instructions = []
        for instr in instructions:
            i, j, k, mode, trainable = instr
            new_instructions.append((i, j, 0, mode, trainable))
        
        self.conv_tp = o3.TensorProduct(
            node_feats_irreps,
            edge_attrs_irreps,
            invariant_node_feat_irreps,
            instructions=new_instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [32] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # Linear
        irreps_out = o3.Irreps("1x0e")  # map straight to a scalar
        self.linear = o3.Linear(
            invariant_node_feat_irreps, irreps_out, internal_weights=True, shared_weights=True
        )

    def forward(
        self,
        node_attrs: torch.Tensor,  # [n_node, num_el]
        node_formal_charges: torch.Tensor,  # [n_node,]
        node_feats: torch.Tensor,  # [n_node, hidden_irreps.dim]
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        # embed oxidation state, and product into the node feats.
        oxidation_feats = self.oxidation_state_linear(self.oxidation_basis(
            (node_formal_charges.unsqueeze(-1) - self.oxidation_state_range[0]) * self.factor - 1.
        ))
        node_feats_ = self.oxidation_state_tp(node_feats, oxidation_feats)

        tp_weights = self.conv_tp_weights(edge_feats)
        bond_feats = (
            self.linear1(node_feats_)[sender] + self.linear2(node_feats_)[receiver]
        )
        mji = self.conv_tp(bond_feats, edge_attrs, tp_weights)  # [n_edges, irreps]

        p_ji = self.linear(mji).squeeze(-1) / 40.0

        # sum over edges
        charges = scatter_sum(
            src=p_ji, index=receiver, dim=0, dim_size=num_nodes
        ) - scatter_sum(src=p_ji, index=sender, dim=0, dim_size=num_nodes)

        multipoles = self.multipole_block(node_feats=node_feats_, node_attrs=node_attrs)
        multipoles[:,0,0] = charges

        return multipoles.squeeze(-2), p_ji.unsqueeze(-1) # [n_node, 1, (max_l+1)**2]



@compile_mode("script")
class NoFieldSymmetricPredictionSourceBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        max_l: int,
        num_elements: int,
    ):
        super().__init__()

        self.multipole_block = EnvironmentDependentSourceBlock(
            irreps_in=node_feats_irreps,
            max_l=max_l,
            zero_charges=True
        )

        self.linear1 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear2 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            node_feats_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # Linear
        irreps_mid = irreps_mid.simplify()
        irreps_out = o3.Irreps("1x0e")  # map straight to a scalar
        self.linear = o3.Linear(
            irreps_mid, irreps_out, internal_weights=True, shared_weights=True
        )

    def forward(
        self,
        node_attrs: torch.Tensor,  # [n_node, num_el]
        node_formal_charges: torch.Tensor,  # [n_node,]
        node_feats: torch.Tensor,  # [n_node, hidden_irreps.dim]
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        tp_weights = self.conv_tp_weights(edge_feats)
        bond_feats = (
            self.linear1(node_feats)[sender] + self.linear2(node_feats)[receiver]
        )
        mji = self.conv_tp(bond_feats, edge_attrs, tp_weights)  # [n_edges, irreps]

        p_ji = self.linear(mji).squeeze(-1) / 40.0

        # sum over edges
        charges = scatter_sum(
            src=p_ji, index=receiver, dim=0, dim_size=num_nodes
        ) - scatter_sum(src=p_ji, index=sender, dim=0, dim_size=num_nodes)

        multipoles = self.multipole_block(node_feats=node_feats, node_attrs=node_attrs)
        multipoles[:,0,0] = charges

        return multipoles.squeeze(-2), p_ji.unsqueeze(-1) # [n_node, 1, (max_l+1)**2]


static_bond_transfer_blocks = {
    "NoFieldSymmetricPredictionSourceBlock": NoFieldSymmetricPredictionSourceBlock,
    "OxidationDependentSymmetricPredictionSourceBlock": OxidationDependentSymmetricPredictionSourceBlock
}

oxidation_state_mixers = {
    "NoOxidationStateMixer": NoOxidationStateMixer,
    "SumOxidationStateMixer": SumOxidationStateMixer,
    "ProductOxidationStateMixer": ProductOxidationStateMixer
}

