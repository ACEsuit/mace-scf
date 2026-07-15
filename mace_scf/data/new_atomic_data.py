from typing import Optional, Sequence

import torch.utils.data

from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
)

from mace.data.utils import Configuration
from mace.data import AtomicData
from scipy.constants import pi

from .neighborhood import get_neighborhood
from mace.data import KeySpecification
import numpy as np


def update_keyspec_from_kwargs(keyspec, keydict) -> KeySpecification:
    # convert command line style property_key arguments into a keyspec
    infos = [
        "energy_key",
        "stress_key",
        "virials_key",
        "dipole_key",
        "head_key",
        "fermi_level_key",
        "external_field_key",
        "polarizability_key",
        "rho_fftn_key",
    ]
    arrays = [
        "forces_key",
        "charges_key",
        "enegs_key",
        "hardness_key",
        "atomic_multipoles_key",
    ]
    info_keys = {}
    arrays_keys = {}
    for key in infos:
        if key in keydict:
            info_keys[key[:-4]] = keydict[key]
    for key in arrays:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]
    keyspec.update(info_keys=info_keys, arrays_keys=arrays_keys)
    return keyspec


class ExtAtomicData(AtomicData):
    density_coefficients: torch.Tensor
    density_coefficients_weight: torch.Tensor
    rcell: torch.Tensor
    volume: torch.Tensor
    pbc: torch.Tensor
    total_charge: torch.Tensor
    external_field: torch.Tensor
    fermi_level: torch.Tensor
    fermi_level_weight: torch.Tensor
    enegs: torch.Tensor
    hardness: torch.Tensor

    def __init__(
        self,
        **kwargs,
    ):
        # new bits
        density_coefficients = kwargs.pop("density_coefficients", None)
        density_coefficients_weight = kwargs.pop(
            "density_coefficients_weight", None
        )  # [,]
        electrostatic_potentials = kwargs.pop("electrostatic_potentials", None)
        electrostatic_potentials_weight = kwargs.pop(
            "electrostatic_potentials_weight", None
        )  # [,]
        rcell = kwargs.pop("rcell", None)  # [3,3]
        volume = kwargs.pop("volume", None)  # [1]
        pbc = kwargs.pop("pbc", None)  # [3,1]
        total_charge = kwargs.pop("total_charge", None)  # [1]
        external_field = kwargs.pop("external_field", None)  # [3]
        fermi_level = kwargs.pop("fermi_level", None)  # [1]
        fermi_level_weight = kwargs.pop("fermi_level_weight", None)  # [,]
        # polarizability = kwargs.pop("polarizability", None) # [3,3]
        # polarizability_weight = kwargs.pop("polarizability_weight", None) #[1]
        cluster_batch = kwargs.pop("cluster_batch", None)
        cluster_loss_weight = kwargs.pop("cluster_loss_weight", None)
        enegs = kwargs.pop("enegs", None)
        hardness = kwargs.pop("hardness", None)
        rho_fftn = kwargs.pop("rho_fftn", None)
        rho_fftn_shape = kwargs.pop("rho_fftn_shape", None)
        super().__init__(**kwargs)
        assert (
            density_coefficients_weight is None
            or len(density_coefficients_weight.shape) == 0
        )
        assert (
            electrostatic_potentials_weight is None
            or len(electrostatic_potentials_weight.shape) == 0
        )
        assert (
            electrostatic_potentials is None or electrostatic_potentials.shape[-1] == 1
        )
        assert total_charge is None or len(total_charge.shape) == 0
        assert external_field is None or external_field.shape == torch.Size([1, 3])
        assert fermi_level is None or len(fermi_level.shape) == 0
        assert fermi_level_weight is None or len(fermi_level_weight.shape) == 0
        # assert polarizability is None or polarizability.shape == torch.Size([1,3,3])
        # assert polarizability_weight is None or len(polarizability_weight.shape) == 0
        assert cluster_loss_weight is None or len(cluster_loss_weight.shape) == 0

        # Aggregate data
        data = {
            "density_coefficients": density_coefficients,
            "density_coefficients_weight": density_coefficients_weight,
            "electrostatic_potentials": electrostatic_potentials,
            "electrostatic_potentials_weight": electrostatic_potentials_weight,
            "volume": volume,
            "rcell": rcell,
            "pbc": pbc,
            "total_charge": total_charge,
            "external_field": external_field,
            "fermi_level": fermi_level,
            "fermi_level_weight": fermi_level_weight,
            # "polarizability": polarizability,
            # "polarizability_weight": polarizability_weight,
            "cluster_batch": cluster_batch,
            "cluster_loss_weight": cluster_loss_weight,
            "enegs": enegs,
            "hardness": hardness,
            "rho_fftn": rho_fftn,
            "rho_fftn_shape": rho_fftn_shape,
        }
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        z_table: AtomicNumberTable,
        cutoff: float,
        heads: Optional[list] = None,
        atomic_multipoles_max_l: int = 0,
    ) -> "ExtAtomicData":
        atomic_data = super().from_config(config, z_table, cutoff, heads=heads)
        num_atoms = len(config.atomic_numbers)

        # redo cell
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
        )
        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )
        atomic_data.cell = cell

        density_coefficients = (
            torch.tensor(
                np.atleast_2d(config.properties.get("atomic_multipoles").T).T[
                    ..., : (atomic_multipoles_max_l + 1) ** 2
                ],
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("atomic_multipoles") is not None
            else torch.zeros((num_atoms, (atomic_multipoles_max_l + 1) ** 2))
        )
        density_coefficients_weight = (
            torch.tensor(
                config.property_weights.get("atomic_multipoles"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("atomic_multipoles") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )
        electrostatic_potentials = (
            torch.tensor(config.properties.get("electrostatic_potentials")).unsqueeze(
                -1
            )
            if config.properties.get("electrostatic_potentials") is not None
            else torch.zeros((num_atoms, 1))
        )
        electrostatic_potentials_weight = (
            torch.tensor(
                config.property_weights.get("electrostatic_potentials"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("electrostatic_potentials") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )
        volume = (
            torch.linalg.det(atomic_data.cell) if atomic_data.cell is not None else None
        )
        rcell = (
            2 * pi * torch.linalg.inv(atomic_data.cell.mT)
            if volume > 0
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )
        pbc = (
            torch.as_tensor(np.asarray(config.pbc, dtype=bool), dtype=torch.bool)
            if config.pbc is not None
            else torch.tensor([False, False, False], dtype=torch.bool)
        )
        total_charge = (
            torch.tensor(
                config.properties.get("total_charge"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("total_charge") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        external_field = (
            torch.tensor(
                config.properties.get("external_field"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("external_field") is not None
            else torch.tensor([3 * [0.0]], dtype=torch.get_default_dtype())
        )
        external_field = torch.atleast_2d(external_field)
        fermi_level = (
            torch.tensor(
                config.properties.get("fermi_level"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("fermi_level") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        fermi_level_weight = (
            torch.tensor(
                config.property_weights.get("fermi_level"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("fermi_level") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )
        """ polarizability = (
            voigt_to_matrix(
                torch.tensor(config.properties.get("polarizability"), dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.properties.get("polarizability") is not None
            else torch.zeros((1,3,3), dtype=torch.get_default_dtype())
        ) """
        polarizability_weight = (
            torch.tensor(
                config.property_weights.get("polarizability"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("polarizability") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        cluster_batch = (
            torch.tensor(config.properties.get("molID"), dtype=torch.long)
            if config.properties.get("molID") is not None
            else torch.zeros((num_atoms,), dtype=torch.long)
        )
        cluster_loss_weight = (
            torch.tensor(
                config.property_weights.get("molID"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("molID") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        enegs = (
            torch.tensor(
                config.properties.get("enegs"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("enegs") is not None
            else torch.zeros((num_atoms))
        )
        hardness = (
            torch.tensor(
                config.properties.get("hardness"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("hardness") is not None
            else torch.zeros((num_atoms))
        )
        # rho_fftn: raw DFT reference charge density fftn. Stored on the config
        # as a real+imag pair with shape [2, N1, N2, N3]. We fold it into a
        # complex tensor, remember the grid shape (as a [1, 3] tensor so torch
        # _geometric stacks it into [n_graphs, 3] under batching), and flatten
        # the values (concatenated across the batch under the default dim-0
        # cat). The model forward reads the flat tensor and slices per graph
        # using rho_fftn_shape.
        if config.properties.get("rho_fftn") is not None:
            rho_fftn_np = np.asarray(config.properties.get("rho_fftn"))
            rho_fftn_complex = torch.as_tensor(
                rho_fftn_np[0] + 1j * rho_fftn_np[1],
                dtype=torch.complex64
                if torch.get_default_dtype() == torch.float32
                else torch.complex128,
            )
            rho_fftn_shape = torch.tensor(
                [list(rho_fftn_complex.shape)], dtype=torch.long
            )  # [1, 3]
            rho_fftn = rho_fftn_complex.flatten()
        else:
            rho_fftn = torch.zeros((1,), dtype=torch.complex64
                if torch.get_default_dtype() == torch.float32
                else torch.complex128)
            rho_fftn_shape = torch.tensor([[1, 1, 1]], dtype=torch.long)
        return cls(
            edge_index=atomic_data.edge_index,
            positions=atomic_data.positions,
            shifts=atomic_data.shifts,
            unit_shifts=atomic_data.unit_shifts,
            cell=cell,  # use cell from new neighbourhood fn
            node_attrs=atomic_data.node_attrs,
            weight=atomic_data.weight,
            head=atomic_data.head,
            energy_weight=atomic_data.energy_weight,
            forces_weight=atomic_data.forces_weight,
            stress_weight=atomic_data.stress_weight,
            virials_weight=atomic_data.virials_weight,
            dipole_weight=atomic_data.dipole_weight,
            charges_weight=atomic_data.charges_weight,
            forces=atomic_data.forces,
            energy=atomic_data.energy,
            stress=atomic_data.stress,
            virials=atomic_data.virials,
            dipole=atomic_data.dipole,
            charges=atomic_data.charges,
            polarizability=atomic_data.polarizability,
            polarizability_weight=polarizability_weight,
            elec_temp=atomic_data.elec_temp,  # new things below
            density_coefficients=density_coefficients,
            density_coefficients_weight=density_coefficients_weight,
            electrostatic_potentials=electrostatic_potentials,
            electrostatic_potentials_weight=electrostatic_potentials_weight,
            volume=volume,
            rcell=rcell,
            pbc=pbc,
            total_charge=total_charge,
            external_field=external_field,
            fermi_level=fermi_level,
            fermi_level_weight=fermi_level_weight,
            cluster_batch=cluster_batch,
            cluster_loss_weight=cluster_loss_weight,
            enegs=enegs,
            hardness=hardness,
            rho_fftn=rho_fftn,
            rho_fftn_shape=rho_fftn_shape,
        )
