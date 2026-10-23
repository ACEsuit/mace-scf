from typing import List, Union

import numpy as np
import torch
from torchmetrics import Metric

from mace.tools.scatter import scatter_sum
from mace.tools.torch_tools import to_numpy
from mace.tools.utils import (
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)


class MaceSCFLoss(Metric):
    """DDP-aware accumulator for mace-scf evaluation metrics.

    Mirrors ``mace.tools.train.MACELoss``: per-batch ``update()`` only
    accumulates local state, and the single ``compute()`` call after the
    data loader is exhausted triggers torchmetrics' implicit all_reduce
    (for "sum" states) / all_gather+cat (for "cat" states) across DDP
    ranks, so every rank ends up with metrics over the full dataset.
    """

    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_batches", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")

        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")

        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state("delta_stress_per_atom", default=[], dist_reduce_fx="cat")

        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")

        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")

        self.add_state(
            "dmas_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("dmas", default=[], dist_reduce_fx="cat")
        self.add_state("delta_dmas", default=[], dist_reduce_fx="cat")

        self.add_state(
            "esps_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("esps", default=[], dist_reduce_fx="cat")
        self.add_state("delta_esps", default=[], dist_reduce_fx="cat")

        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

        self.add_state(
            "total_charge_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_total_charge", default=[], dist_reduce_fx="cat")

        self.add_state(
            "fermi_level_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_fermi_level", default=[], dist_reduce_fx="cat")

    def update(self, batch, output) -> None:  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_batches += 1

        for key in output:
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].detach()

        if output.get("energy") is not None and batch.energy is not None:
            self.E_computed += 1
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.Fs_computed += 1
            self.delta_fs.append(batch.forces - output["forces"])
            self.fs.append(batch.forces)
        if output.get("stress") is not None and batch.stress is not None:
            self.stress_computed += 1
            self.delta_stress.append(batch.stress - output["stress"])
            self.delta_stress_per_atom.append(
                (batch.stress - output["stress"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.virials_computed += 1
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("density_coefficients") is not None and batch.total_charge is not None:
            self.total_charge_computed += 1
            total_charge = scatter_sum(
                src=output["density_coefficients"][:, 0], index=batch.batch, dim=-1
            )
            self.delta_total_charge.append(batch.total_charge - total_charge)
        if output.get("fermi_level") is not None and batch.fermi_level is not None:
            self.fermi_level_computed += 1
            self.delta_fermi_level.append(batch.fermi_level - output["fermi_level"])
        if output.get("dipole") is not None and batch.dipole is not None:
            dipole_components_to_include = batch.dipole_weight.view(-1, 3) > 0.0
            if torch.any(dipole_components_to_include):
                self.Mus_computed += 1
                dipole_differences = batch.dipole - output["dipole"]
                num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1)
                num_atoms = num_atoms.repeat(1, 3)

                self.delta_mus.append(
                    dipole_differences[dipole_components_to_include]
                )
                self.delta_mus_per_atom.append(
                    dipole_differences[dipole_components_to_include]
                    / num_atoms[dipole_components_to_include]
                )
                self.mus.append(batch.dipole[dipole_components_to_include])
        if (
            output.get("density_coefficients") is not None
            and batch.density_coefficients is not None
        ):
            self.dmas_computed += 1
            self.delta_dmas.append(
                batch.density_coefficients - output["density_coefficients"]
            )
            self.dmas.append(batch.density_coefficients)
        if (
            output.get("electrostatic_potentials") is not None
            and batch.electrostatic_potentials is not None
        ):
            self.esps_computed += 1
            self.delta_esps.append(
                batch.electrostatic_potentials - output["electrostatic_potentials"]
            )
            self.esps.append(batch.electrostatic_potentials)
        if output.get("polarizability") is not None and batch.polarizability is not None:
            polars_to_include = batch.polarizability_weight > 0.0
            if torch.any(polars_to_include):
                self.polarizability_computed += 1
                self.delta_polarizability.append(
                    batch.polarizability[polars_to_include]
                    - output["polarizability"][polars_to_include]
                )
                self.delta_polarizability_per_atom.append(
                    (batch.polarizability - output["polarizability"])[
                        polars_to_include
                    ]
                    / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)[
                        polars_to_include
                    ]
                )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def _add_error_stats(self, aux, suffix, delta, ref=None):
        aux[f"mae_{suffix}"] = compute_mae(delta)
        if ref is not None:
            aux[f"rel_mae_{suffix}"] = compute_rel_mae(delta, ref)
        aux[f"rmse_{suffix}"] = compute_rmse(delta)
        if ref is not None:
            aux[f"rel_rmse_{suffix}"] = compute_rel_rmse(delta, ref)
        aux[f"q95_{suffix}"] = compute_q95(delta)

    def compute(self):
        aux = {"loss": to_numpy(self.total_loss / self.num_batches).item()}

        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            self._add_error_stats(aux, "e", delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            offset = np.mean(delta_es_per_atom)
            reduced = delta_es_per_atom - offset
            aux["offset_e_per_atom"] = offset
            aux["rmse_spread_e_per_atom"] = compute_rmse(reduced)
            aux["mae_spread_e_per_atom"] = compute_mae(reduced)
        if self.Fs_computed:
            delta_fs = self.convert(self.delta_fs)
            fs = self.convert(self.fs)
            self._add_error_stats(aux, "f", delta_fs, ref=fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            delta_stress_per_atom = self.convert(self.delta_stress_per_atom)
            self._add_error_stats(aux, "stress", delta_stress)
            aux["rmse_stress_per_atom"] = compute_rmse(delta_stress_per_atom)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            self._add_error_stats(aux, "virials", delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
        if self.Mus_computed:
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            mus = self.convert(self.mus)
            self._add_error_stats(aux, "mu", delta_mus, ref=mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
        if self.dmas_computed:
            delta_dmas = self.convert(self.delta_dmas)
            dmas = self.convert(self.dmas)
            self._add_error_stats(aux, "dma", delta_dmas, ref=dmas)
            if delta_dmas.shape[0] > 0:
                aux["rmse_charges"] = compute_rmse(delta_dmas[:, 0:1])
            if delta_dmas.shape[1] > 1:
                aux["rmse_local_dipoles"] = compute_rmse(delta_dmas[:, 1:4])
        if self.esps_computed:
            delta_esps = self.convert(self.delta_esps)
            esps = self.convert(self.esps)
            self._add_error_stats(aux, "esp", delta_esps, ref=esps)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            self._add_error_stats(aux, "polarizability", delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
        if self.total_charge_computed:
            delta_total_charge = self.convert(self.delta_total_charge)
            self._add_error_stats(aux, "total_charge", delta_total_charge)
        if self.fermi_level_computed:
            delta_fermi_level = self.convert(self.delta_fermi_level)
            self._add_error_stats(aux, "fermi_level", delta_fermi_level)

        return aux["loss"], aux
