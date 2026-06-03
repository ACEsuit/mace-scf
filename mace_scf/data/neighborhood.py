from typing import Optional, Tuple
import warnings
import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)
    # we now accept slabs, but the out of plane box length is treated as physical
    
    # no cell -> zero cell 
    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    identity = np.identity(3, dtype=float)
    ranges = np.max(positions, axis=0) - np.min(positions, axis=0)

    # check for problem: if not pbc, check for any overrun
    # for mixed systems, no checks are made
    if not(any(pbc)): # if [F, F, F]
        # min box size should always be 30
        for dim in range(3):
            if cell[dim, dim] < 30.:
                cell[dim, dim] = 30.
                warnings.warn(f"Warning: min cell dimension for pbc=False is 30A")
        # and otherwise, it should be 2.5* the ranges
        if any((
            cell[0,0] < ranges[0]*2.0,
            cell[1,1] < ranges[1]*2.0,
            cell[2,2] < ranges[2]*2.0,
        )): # if overlap
            cell = np.eye(3) * (np.max(ranges) * 2.0)
            # in this repo, the cell actually matters for non-periodic systems
            warnings.warn(f"Warning: cell was too small, setting to {cell[0,0]} cube")

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell
