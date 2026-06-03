import torch 
from contextlib import contextmanager
from e3nn import get_optimization_defaults, set_optimization_defaults
import mace.modules
from e3nn import o3
import ast 
import logging
from mace_scf import electrostatics
import numpy as np
from graph_longrange.gto_utils import gto_basis_kspace_cutoff, DisplacedGTOExternalFieldBlock
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.features import GTOElectrostaticFeatures
from mace.tools import torch_tools


# needed for torchopt
@contextmanager
def disable_e3nn_codegen():
    """Context manager that disables the legacy PyTorch code generation used in e3nn."""
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    yield
    set_optimization_defaults(jit_script_fx=init_val)



def build_model(
    args,
    z_table,
    atomic_energies,
    atomic_charges,
    train_loader,
):
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=mace.modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=args.correlation,
        gate=mace.modules.gate_dict[args.gate],
        MLP_irreps=o3.Irreps(args.MLP_irreps),
        radial_MLP=ast.literal_eval(args.radial_MLP),
        radial_type=args.radial_type,
    )

    model: torch.nn.Module

    if args.model == "MACE":
        """ if args.scaling == "no_scaling":
            std = 1.0
            logging.info("No scaling selected")
        else:
            mean, std = mace.modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
            )
        model = mace.modules.ScaleShiftMACE(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
        ) """
        model = mace.modules.MACE(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
        )
    elif args.model == "ScaleShiftMACE":
        mean, std = mace.modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
        model = mace.modules.ScaleShiftMACE(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif args.model == "FixedChargeBaselinedMACE":
        formal_charges = ast.literal_eval(args.atomic_formal_charges)
        assert args.formal_charges_from_data or len(formal_charges) == len(z_table)
        model = electrostatics.FixedChargeBaselinedMACE(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            atomic_formal_charges=atomic_charges,
            formal_charges_from_data=args.formal_charges_from_data,
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
            use_linear_final_readout=args.use_linear_final_readout,
            pbc_handling=args.electrostatic_pbc_method,
        )
    elif args.model == "LocalSplitCharges":
        formal_charges = ast.literal_eval(args.atomic_formal_charges)
        assert args.formal_charges_from_data or len(formal_charges) == len(z_table)
        model = electrostatics.LocalSplitCharges(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            atomic_formal_charges=atomic_charges,
            formal_charges_from_data=args.formal_charges_from_data,
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
            static_bond_transfer_block=args.static_bond_transfer_block,
            oxidation_state_mixer=args.oxidation_state_mixer,
            oxidation_state_range=ast.literal_eval(args.oxidation_state_range),
            compute_polarizability=args.compute_polarizability,
            use_linear_final_readout=args.use_linear_final_readout,
            pbc_handling=args.electrostatic_pbc_method,
        )
    elif args.model == "LocalCharges":
        model = electrostatics.LocalCharges(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
            pbc_handling=args.electrostatic_pbc_method,
        )
    elif args.model == "FixedPoint":
        with disable_e3nn_codegen():
            model = electrostatics.FixedPointCore(
                **model_config,
                interaction_cls_first=mace.modules.interaction_classes[
                    args.interaction_first
                ],
                kspace_cutoff_factor=args.kspace_cutoff_factor,
                atomic_multipoles_max_l=args.atomic_multipoles_max_l,
                atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
                field_feature_widths=ast.literal_eval(args.field_feature_widths),
                field_feature_max_l=args.field_feature_max_l,
                include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
                add_local_electron_energy=args.include_local_electron_energy,
                quadrupole_feature_corrections=args.quadrupole_feature_corrections,
                return_electrostatic_potentials=args.return_electrostatic_potentials,
                fixedpoint_update_config=args.fixedpoint_update_config,
                field_readout_config=args.field_readout_config,
                field_feature_norms=args.field_feature_norms,
                field_norm_factor=args.field_norm_factor,
                field_si=args.include_field_si,
                use_linear_local_charges=args.use_linear_local_charges,
                atom_density_scaling=args.atom_density_scaling,
                pbc_handling=args.electrostatic_pbc_method,
                fermi_level_offset=args.fermi_level_offset,
            )
    elif args.model == "MACEQEq":
        with disable_e3nn_codegen():
            model = electrostatics.MACEQEq(
                **model_config,
                interaction_cls_first=mace.modules.interaction_classes[
                    args.interaction_first
                ],
                kspace_cutoff_factor=args.kspace_cutoff_factor,
                atomic_multipoles_max_l=args.atomic_multipoles_max_l,
                atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
                field_feature_widths=ast.literal_eval(args.field_feature_widths),
                include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
                qeq_charges=args.qeq_charges,
                pbc_handling=args.electrostatic_pbc_method,
                train_hardness=args.train_hardness,
                read_enegs=args.read_enegs,
                read_hardness=args.read_hardness,
                default_hardness=args.default_hardness,
            )
    else:
        raise RuntimeError(f"Unknown model: '{args.model}'")

    return model


def get_param_options(model, args):
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
        amsgrad=args.amsgrad,
        betas=(args.beta, args.beta_two),
    )

    if args.model == "LocalSplitCharges":
        param_options["params"].append(
            {
                "name": "lr_source_maps",
                "params": model.lr_source_maps.parameters(),
                "weight_decay": 0.0,
                "lr": 0.01,
            }
        )
        if hasattr(model, "polarizability_readouts"):
            param_options["params"].append(
                {
                    "name": "polarizability_readouts",
                    "params": model.polarizability_readouts.parameters(),
                    "weight_decay": 0.0,
                    "lr": 0.01,
                }
            )
    if args.model == "LocalCharges":
        param_options["params"].append(
            {
                "name": "lr_source_maps",
                "params": model.lr_source_maps.parameters(),
                "weight_decay": 0.0,
            }
        )
    if args.model in ("FixedPoint", "FixedPointCore"):
        param_options["params"].append(
            {
                "name": "lr_source_maps",
                "params": model.lr_source_maps.parameters(),
                "weight_decay": args.local_charges_weight_decay,
            }
        )
        param_options["params"].append(
            {
                "name": "field_dependent_charges_map",
                "params": model.field_dependent_charges_map.parameters(),
                "weight_decay": args.field_block_weight_decay,
            }
        )
        param_options["params"].append(
            {
                "name": "local_electron_energy",
                "params": model.local_electron_energy.parameters(),
                "weight_decay": args.weight_decay,
            }
        )
    if args.model == "FixedPoint":
        param_options["params"].append(
            {
                "name": "layer_feature_mixer",
                "params": model.layer_feature_mixer.parameters(),
                "weight_decay": args.weight_decay,
            }
        )
    if args.model == "MACEQEq":
        param_options["params"].append(
            {
                "name": "enegs_readouts",
                "params": model.enegs_readouts.parameters(),
                "weight_decay": args.weight_decay,
            }
        )
        param_options["params"].append(
            {
                "name": "hardness_readouts",
                "params": model.hardness_readouts.parameters(),
                "weight_decay": args.weight_decay,
            }
        )
    return param_options


def get_formal_charges(model_name, formal_charges_from_data, args_atomic_formal_charges, z_table):
    if model_name in ["FixedChargeBaselinedMACE", "LocalSplitCharges"]:
        if formal_charges_from_data:
            logging.info("charged models: taking formal charges per config from data")
            atomic_charges = None
        elif args_atomic_formal_charges is not None:
            logging.info("charged models: taking formal charges from command line")
            atomic_charges_dict = ast.literal_eval(args_atomic_formal_charges)
            assert isinstance(atomic_charges_dict, dict)
            missing_atomic_numbers = [
                atomic_number
                for atomic_number in z_table.zs
                if atomic_number not in atomic_charges_dict
            ]
            if missing_atomic_numbers:
                raise ValueError(
                    "Missing formal charges for atomic numbers: "
                    f"{missing_atomic_numbers}"
                )
            atomic_charges = np.array([atomic_charges_dict[z] for z in z_table.zs])
        else:
            raise ValueError("--formal_charges_from_data is False, but no formal charges were provided")
        return atomic_charges
    else:
        return None


def get_field_feature_norms(dataloader, args, device, fermi_level_offset=0.0):
    if args.field_feature_norms == "None":
        return None
    if args.model != "FixedPoint":
        logging.info(
            "Ignoring field_feature_norms=%r for model %s; field features are "
            "only used by FixedPoint.",
            args.field_feature_norms,
            args.model,
        )
        return None

    field_feature_widths = ast.literal_eval(args.field_feature_widths)
    expected_len = len(field_feature_widths) * (args.field_feature_max_l + 1)
    if args.field_feature_norms != "average":
        try:
            field_feat_norms = ast.literal_eval(args.field_feature_norms)
            assert type(field_feat_norms) == list
            assert len(field_feat_norms) == expected_len
        except (AssertionError, SyntaxError, ValueError) as e:
            print(
                "field feature norms specified incorrectly: "
                f"{args.field_feature_norms}; expected a list of length {expected_len}"
            )
            raise e
        return np.array(field_feat_norms)
    else:
        return compute_average_feature_norms(
            dataloader, args, device, fermi_level_offset=fermi_level_offset
        )


def get_fermi_level_offset(dataloader, args, device):
    if args.fermi_level_offset is not None:
        logging.info(
            "using manually specified Fermi level offset %s",
            args.fermi_level_offset,
        )
        return float(args.fermi_level_offset)

    fermi_sum = 0.0
    num_values = 0
    for data in dataloader.dataset:
        fermi_level = getattr(data, "fermi_level", None)
        if fermi_level is None:
            continue
        fermi_level_weight = getattr(data, "fermi_level_weight", None)
        if fermi_level_weight is None:
            continue
        if not bool(torch.atleast_1d(fermi_level_weight.detach().cpu() > 0.0).all()):
            continue
        fermi_sum += float(fermi_level.detach().cpu())
        num_values += 1

    if num_values == 0:
        raise ValueError("No Fermi level data found, can't compute an average fermi level")

    fermi_level_offset = fermi_sum / num_values
    logging.info("computed Fermi level offset %s", fermi_level_offset)
    return fermi_level_offset


def get_atom_density_scaling(dataloader, args, device, z_table):
    if args.atom_density_scaling == "average":
        atom_density_scaling = compute_average_atom_density_scaling(
            dataloader=dataloader,
            args=args,
            device=device,
            z_table=z_table,
        )
    elif args.atom_density_scaling == "None":
        atom_density_scaling = {atomic_number: 1.0 for atomic_number in z_table.zs}
    else:
        try:
            atom_density_scaling = ast.literal_eval(args.atom_density_scaling)
            assert type(atom_density_scaling) == dict
            assert len(atom_density_scaling) == len(z_table)
        except (AssertionError, SyntaxError, ValueError) as e:
            print(f"atom density scaling specified incorrectly: {args.atom_density_scaling}")
            raise e
    return np.array([atom_density_scaling[z] for z in z_table.zs])


def compute_average_atom_density_scaling(
    dataloader,
    args,
    device,
    z_table,
    minimum_scale: float = 1.0e-8,
):
    num_components = (args.atomic_multipoles_max_l + 1) ** 2
    sum_squares = {atomic_number: 0.0 for atomic_number in z_table.zs}
    num_values = {atomic_number: 0 for atomic_number in z_table.zs}
    z_tensor = torch.tensor(z_table.zs, device=device)

    for batch in dataloader:
        batch = batch.to(device)
        batch = batch.to_dict()
        density_coefficients = batch.get("density_coefficients")
        node_attrs = batch.get("node_attrs")
        if density_coefficients is None or node_attrs is None:
            continue

        density_coefficients = density_coefficients[:, :num_components]
        density_weight = batch.get("density_coefficients_weight")
        if density_weight is not None and "batch" in batch:
            density_weight = torch.atleast_1d(density_weight)
            node_has_density = density_weight[batch["batch"]] > 0.0
            density_coefficients = density_coefficients[node_has_density]
            node_attrs = node_attrs[node_has_density]
            if density_coefficients.numel() == 0:
                continue

        atomic_numbers = z_tensor[torch.argmax(node_attrs, dim=-1)]

        for atomic_number in z_table.zs:
            mask = atomic_numbers == atomic_number
            if not torch.any(mask):
                continue
            element_density = density_coefficients[mask]
            sum_squares[atomic_number] += float(
                torch.sum(element_density.detach() ** 2).cpu()
            )
            num_values[atomic_number] += element_density.numel()

    atom_density_scaling = {}
    for atomic_number in z_table.zs:
        if num_values[atomic_number] == 0:
            logging.warning(
                "No atomic multipole data found for atomic number %s; using atom "
                "density scaling factor 1.0",
                atomic_number,
            )
            atom_density_scaling[atomic_number] = 1.0
            continue

        rms = np.sqrt(sum_squares[atomic_number] / num_values[atomic_number])
        if rms < minimum_scale:
            logging.warning(
                "Average atomic multipole RMS for atomic number %s is %.6g; using "
                "minimum atom density scaling factor %.6g",
                atomic_number,
                rms,
                minimum_scale,
            )
            rms = minimum_scale
        atom_density_scaling[atomic_number] = float(rms)

    logging.info(
        "computed average atom density scaling factors %s",
        atom_density_scaling,
    )
    return atom_density_scaling


def compute_average_feature_norms(
    dataloader,
    args,
    device,
    fermi_level_offset: float = 0.0,
    minimum_norm: float = 1.0e-8,
):
    field_feature_widths = ast.literal_eval(args.field_feature_widths)
    min_sigma = min(field_feature_widths + [args.atomic_multipoles_smearing_width])
    kspace_max_l = max(args.atomic_multipoles_max_l, args.field_feature_max_l)
    kspace_cutoff = args.kspace_cutoff_factor * gto_basis_kspace_cutoff(
        [min_sigma], kspace_max_l
    )
    electric_potential_descriptor = GTOElectrostaticFeatures(
        density_max_l=args.atomic_multipoles_max_l, 
        density_smearing_width=args.atomic_multipoles_smearing_width, 
        feature_max_l=args.field_feature_max_l,
        feature_smearing_widths=field_feature_widths,
        kspace_cutoff=kspace_cutoff,
        include_self_interaction=args.include_field_si,
        quadrupole_feature_corrections=args.quadrupole_feature_corrections,
        integral_normalization="receiver",
        pbc_handling=args.electrostatic_pbc_method,
    )
    electric_potential_descriptor.to(device)
    external_field_contribution = DisplacedGTOExternalFieldBlock(
        args.field_feature_max_l,
        field_feature_widths,
        "receiver"
    )
    external_field_contribution.to(device)

    field_features_collection = []
    for batch in dataloader:
        batch = batch.to(device)
        batch = batch.to_dict()
        num_graphs = batch["ptr"].numel() - 1

        k_vectors, kv_norms_squared, k_vectors_batch, k0_mask = compute_k_vectors_flat(
            kspace_cutoff, batch["cell"].view(-1,3,3), batch["rcell"].view(-1,3,3)
        )
        electrostatics_cache = electric_potential_descriptor.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k0_mask,
            node_positions=batch["positions"],
            batch=batch["batch"],
            volume=batch["volume"],
            pbc=batch["pbc"].view(-1,3),
        )
        field_feats = electric_potential_descriptor.forward_dynamic(
            cache=electrostatics_cache,
            source_feats=batch["density_coefficients"],
        )
        centered_fermi_level = batch["fermi_level"] - fermi_level_offset
        external_potential = torch.hstack(
            (centered_fermi_level.unsqueeze(-1), batch["external_field"])
        )
        field_feats += external_field_contribution(
            batch["batch"], batch["positions"], external_potential
        )

        graph_has_inputs = torch.ones(
            num_graphs, dtype=torch.bool, device=field_feats.device
        )
        density_weight = batch.get("density_coefficients_weight")
        if density_weight is not None:
            graph_has_inputs &= torch.atleast_1d(density_weight) > 0.0
        fermi_level_weight = batch.get("fermi_level_weight")
        if fermi_level_weight is not None:
            graph_has_inputs &= torch.atleast_1d(fermi_level_weight) > 0.0
        node_has_inputs = graph_has_inputs[batch["batch"]]
        if not torch.any(node_has_inputs):
            continue

        field_features_all = torch_tools.to_numpy(field_feats[node_has_inputs])
        field_features_collection.append(field_features_all)

    if len(field_features_collection) == 0:
        raise ValueError(
            "Cannot compute average field feature norms: no training configs had "
            "both atomic multipoles and fermi_level data."
        )

    all_feats = np.concatenate(field_features_collection, axis=0)
    norms = []

    num_channels = len(field_feature_widths)
    for order in range(args.field_feature_max_l + 1):
        feats = all_feats[:, num_channels*(order**2):num_channels*((order+1)**2)]
        for j in range(num_channels):
            norm = np.sqrt(np.average(feats[:, j*(2*order+1):(j+1)*(2*order+1)]**2))
            norms.append(max(float(norm), minimum_norm))

    norms = np.asarray(norms)
    logging.info("computed average field feature norms %s", norms)
    return norms


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    print(f"\nWarning: {message}")
    print(f"Category: {category.__name__}")
    print(f"File: {filename}, Line: {lineno}")
    print("Traceback:")
    traceback.print_stack()
