import torch 
from typing import List, Tuple, Optional, Dict


def group_configurations_by_max(batch: List[int], M: int) -> List[List[int]]:
    """
    Given a list of configuration labels (one per sample) and a maximum group size M,
    form groups by combining whole configurations (without splitting any configuration)
    in the order they first appear. Each group will have at most M atoms.
    """
    # Build mapping: configuration label -> list of sample indices
    config_to_indices = {}
    config_order = []
    for idx, cfg in enumerate(batch):
        if cfg not in config_to_indices:
            config_to_indices[cfg] = []
            config_order.append(cfg)
        config_to_indices[cfg].append(idx)
    
    groups = []
    current_group = []
    current_total = 0
    
    for cfg in config_order:
        indices = config_to_indices[cfg]
        count = len(indices)
        if current_total + count <= M:
            current_group.append(indices)
            current_total += count
        else:
            if current_group:
                # Finalize current group (flatten the list of lists)
                groups.append([i for sublist in current_group for i in sublist])
                current_group = []
                current_total = 0
            if count > M:
                # If a single configuration exceeds M, form its own group
                groups.append(indices)
            else:
                current_group.append(indices)
                current_total = count
                
    if current_group:
        groups.append([i for sublist in current_group for i in sublist])
    
    return groups


def process_group_dict(atom_dict: dict, group_indices: List[int]) -> dict:
    """
    Process an atomic dictionary into a group, handling all tensor types appropriately.
    """
    group_dict = {}
    
    # Get batch info and device
    if isinstance(atom_dict["batch"], (list, tuple)):
        batch = torch.tensor(atom_dict["batch"], device=atom_dict["edge_index"].device)
    else:
        batch = atom_dict["batch"]
    device = batch.device
    
    old_batch = batch[group_indices]
    _, new_batch = torch.unique(old_batch, return_inverse=True)
    config_indices = torch.unique(old_batch).tolist()
    group_dict["batch"] = new_batch
    
    # 1. Graph-Level Tensors
    graph_level_keys = ["energy", "total_charge", "volume", "fermi_level"]
    for key in graph_level_keys:
        if key in atom_dict and atom_dict[key] is not None:
            if len(atom_dict[key].shape) == 0:  # Scalar
                group_dict[key] = atom_dict[key]
            else:
                group_dict[key] = atom_dict[key][config_indices]
    
    # Handle cell tensors specially
    for key in ["cell", "rcell"]:
        if key in atom_dict and atom_dict[key] is not None:
            cell_data = atom_dict[key].view(-1, 3, 3)
            group_dict[key] = cell_data[config_indices].reshape(-1, 3)
            
    # Handle stress/virial tensors
    for key in ["stress", "virials"]:
        if key in atom_dict and atom_dict[key] is not None:
            if atom_dict[key].shape[0] == 1:  # Global tensor
                group_dict[key] = atom_dict[key]
            else:  # Per-configuration tensor
                group_dict[key] = atom_dict[key][config_indices]
    
    for key in ["pbc", "external_field", "dipole"]:
        if key in atom_dict and atom_dict[key] is not None:
            pbc_data = atom_dict[key].view(-1, 3)
            group_dict[key] = pbc_data[config_indices].reshape(-1, 3)

    # 2. Node-Level Tensors
    node_level_keys = [
        "positions", "forces", "node_attrs", "charges",
        "weight_forces", "weight_q_forces", "q_forces", "density_coefficients"
    ]
    for key in node_level_keys:
        if key in atom_dict and atom_dict[key] is not None:
            group_dict[key] = atom_dict[key][group_indices]
    
    # 3. Edge-Level Tensors
    # Create edge mask
    group_set = set(group_indices)
    edge_index = atom_dict["edge_index"]
    """ edge_mask = torch.tensor([
        edge_index[0, i].item() in group_set and edge_index[1, i].item() in group_set
        for i in range(edge_index.shape[1])
    ], device=device) """
    edge_mask = torch.isin(edge_index[0], torch.tensor(list(group_set), device=device)) & torch.isin(edge_index[1], torch.tensor(list(group_set), device=device))
    
    edge_level_keys = ["shifts", "unit_shifts", "edge_attr"]
    for key in edge_level_keys:
        if key in atom_dict and atom_dict[key] is not None:
            group_dict[key] = atom_dict[key][edge_mask]
    
    # Handle edge_index specially
    if "edge_index" in atom_dict:
        # Create a tensor for old_to_new mapping
        old_to_new = torch.full((max(group_indices) + 1,), -1, device=device)  # Initialize with -1 (invalid)
        old_to_new[group_indices] = torch.arange(len(group_indices), device=device)  # Map old indices to new indices

        # Filter and remap edge_index
        edge_indices_filtered = edge_index[:, edge_mask]
        new_edge_index = old_to_new[edge_indices_filtered]  # Vectorized remapping
        group_dict["edge_index"] = new_edge_index
    
    # Handle ptr specially
    if "ptr" in atom_dict:
        atom_counts = []
        for cfg_idx in config_indices:
            mask = old_batch == cfg_idx
            atom_counts.append(mask.sum().item())
        group_dict["ptr"] = torch.cat([
            torch.tensor([0], device=device),
            torch.cumsum(torch.tensor(atom_counts, device=device), dim=0)
        ])
    
    return group_dict


def split_batch(atom_dict: dict, max_atoms: int) -> List[Tuple[dict, torch.Tensor]]:
    """
    Process atomic data into groups based on maximum number of atoms.
    
    Args:
        atom_dict (dict): Dictionary containing atomic data with various tensor types
        max_atoms (int): Maximum number of atoms per group
        
    Returns:
        List[Tuple[dict, torch.Tensor]]: List of tuples containing (group_dict) 
    """
    # Get batch assignments
    if isinstance(atom_dict["batch"], (list, tuple)):
        batch_ids = atom_dict["batch"]
    else:
        batch_ids = atom_dict["batch"].tolist()
    
    # Get groups based on max atoms
    groups = group_configurations_by_max(batch_ids, max_atoms)
    
    # Process each group
    group_results = []
    for group_indices in groups:
        # Process atomic data
        group_dict = process_group_dict(atom_dict, group_indices)
        group_results.append(group_dict)
    
    return groups, group_results


def recombine_output_dict(group_results):
    num_meta = len(group_results[0])
    full_results = [torch.cat([group[meta_idx] for group in group_results], dim=0) for meta_idx in range(num_meta)]


def recombine_output_manual(group_results):
    output_dict = {}
    omit = ["charges_history", "electrostatic_potentials"]
    isnone = []
    for key, tensr in group_results[0].items():
        if key in omit:
            continue
        if tensr is None:
            isnone.append(key)
            continue
        output_dict[key] = [tensr]

    for group in group_results[1:]:
        for key in output_dict.keys():
            output_dict[key].append(group[key])

    for key in output_dict.keys():
        output_dict[key] = torch.cat(output_dict[key])
    for key in isnone:
        output_dict[key] = None
    
    return output_dict