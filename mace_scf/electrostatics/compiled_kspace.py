from typing import Dict, NamedTuple, Tuple

import torch

from graph_longrange.kspace import compute_k_vectors_flat


class KSpacePlan(NamedTuple):
    k_vectors: torch.Tensor
    k_norm2: torch.Tensor
    k_vector_batch: torch.Tensor
    k0_mask: torch.Tensor


class KSpacePlanner:
    def __init__(self, kspace_cutoff: torch.Tensor):
        self.kspace_cutoff = kspace_cutoff
        self._plans: Dict[Tuple, KSpacePlan] = {}

    @staticmethod
    def _tensor_key(tensor: torch.Tensor) -> Tuple:
        stable = tensor.detach().cpu().contiguous()
        return (
            tuple(stable.shape),
            str(stable.dtype),
            stable.numpy().tobytes(),
        )

    def _key(
        self,
        cell: torch.Tensor,
        rcell: torch.Tensor,
        pbc_handling: str,
    ) -> Tuple:
        return (
            pbc_handling,
            str(cell.device),
            str(cell.dtype),
            self._tensor_key(self.kspace_cutoff.reshape(())),
            self._tensor_key(cell),
            self._tensor_key(rcell),
        )

    def get_plan(
        self,
        cell: torch.Tensor,
        rcell: torch.Tensor,
        pbc_handling: str,
    ) -> KSpacePlan:
        cell = cell.view(-1, 3, 3)
        rcell = rcell.view(-1, 3, 3)
        key = self._key(cell=cell, rcell=rcell, pbc_handling=pbc_handling)
        if key not in self._plans:
            k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
                self.kspace_cutoff.to(device=cell.device, dtype=cell.dtype),
                cell,
                rcell,
            )
            self._plans[key] = KSpacePlan(
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
            )
        return self._plans[key]
