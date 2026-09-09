"""Eager vs compiled-adaptive SCF parity on the bulk water fixtures, cold and
warm-restarted from 0.99 x the cold density (Fermi level unscaled).

Needs CUDA and a local FixedPoint model at
``tests/fixtures/models/fixedpoint_scf_gate.model`` (any model covering H and
O; see tests/fixtures/models/README.md); skips otherwise. Real torch.compile
(inductor) on GPU, one run per chunk size. Prints the per-structure SCF step
table and writes report.json / report.csv under tmp_path.
"""

import csv
import json

import numpy as np
import pytest
import torch
from ase.io import read

from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
from tests.paths import FIXTURES_DIR, reference_model, require_file

ENERGY_TOL_PER_ATOM = 1e-5
FORCE_TOL = 1e-4
FORCE_TOL_WARM = 1e-3  # warm-restart forces are step-count sensitive (see docs)
MULTIPOLE_TOL = 1e-4  # max-abs on density_coefficients; 10x the per-step SCF stop criterion
DIPOLE_TOL_PER_ATOM = 1e-5  # |d dipole| / natoms: the dipole is extensive
WARM_SEED_SCALE = 0.99

MODEL_NAME = "fixedpoint_scf_gate"
CONFIGS = FIXTURES_DIR / "configs" / "bulk_test_configs.xyz"
SCF_OPTIONS = {
    "num_scf_steps": 201,  # (201 - 1) is divisible by every chunk size below
    "scf_tolerance": 1e-8,
    "mixing_parameter": 0.25,
    "constant_charge": True,
    "initial_density": "local_guess",
    "initial_fermi_level": "from_data",
}

STEP_COLUMNS = ["steps_eager_cold", "steps_compiled_cold", "steps_eager_warm", "steps_compiled_warm"]
DEV_COLUMNS = ["dE_atom_cold", "maxdF_cold", "maxdMult_cold", "dDip_atom_cold",
               "dE_atom_warm", "maxdF_warm", "maxdMult_warm", "dDip_atom_warm",
               "maxdF_warm_vs_cold_eager", "maxdF_warm_vs_cold_compiled"]  # last two: report only


def _make_calc(model_path, use_compile, scf_restart, chunk_size):
    kwargs = dict(
        model_path=model_path, device="cuda", default_dtype="float64", pbc_handling="pbc",
        scf_options=SCF_OPTIONS, use_compile=use_compile, scf_restart=scf_restart,
        atomic_multipoles_key="some_multipoles", fermi_level_key="the_VBM",
        external_field_key="an_applied_field", total_charge_key="total_charge",
    )
    if use_compile:
        kwargs.update(compile_enabled=True, compile_scope="scf_chunk",
                      compile_chunk_size=chunk_size, compile_adaptive_scf=True)
    return MACEFixedPointSCF(**kwargs)


def _run_pass(calc, structures, seeds=None):
    rows = []
    for i, atoms in enumerate(structures):
        if seeds is not None:
            calc.most_recent_density = WARM_SEED_SCALE * seeds[i]["density_coefficients"]
            calc.most_recent_fermi_level = seeds[i]["fermi_level"]
        calc.calculate(atoms)
        r = calc.results
        rows.append({
            "energy": r["energy"],
            "forces": np.asarray(r["forces"]),
            "density_coefficients": np.asarray(r["density_coefficients"]),
            "dipole": np.asarray(r["dipole"]),
            "fermi_level": r["fermi_level"],
            "steps": len(r["convergence_history"]),
        })
    return rows


def _deviations(eager_row, compiled_row, natoms):
    return (
        abs(compiled_row["energy"] - eager_row["energy"]) / natoms,
        np.max(np.abs(compiled_row["forces"] - eager_row["forces"])),
        np.max(np.abs(compiled_row["density_coefficients"] - eager_row["density_coefficients"])),
        np.max(np.abs(compiled_row["dipole"] - eager_row["dipole"])) / natoms,
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("chunk_size", [4, 8])
def test_eager_vs_compiled_adaptive_scf(tmp_path, chunk_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model_path = str(require_file(reference_model(MODEL_NAME), "Gate model"))
    structures = read(require_file(CONFIGS, "Bulk configs"), index=":")

    cold_eager = _run_pass(_make_calc(model_path, False, False, chunk_size), structures)
    cold_compiled = _run_pass(_make_calc(model_path, True, False, chunk_size), structures)
    warm_eager = _run_pass(_make_calc(model_path, False, True, chunk_size), structures, seeds=cold_eager)
    warm_compiled = _run_pass(_make_calc(model_path, True, True, chunk_size), structures, seeds=cold_eager)

    violations, rows = [], []
    for i, atoms in enumerate(structures):
        natoms = len(atoms)
        dev_c = _deviations(cold_eager[i], cold_compiled[i], natoms)
        dev_w = _deviations(warm_eager[i], warm_compiled[i], natoms)
        rows.append({
            "idx": i, "natoms": natoms,
            "steps_eager_cold": cold_eager[i]["steps"], "steps_compiled_cold": cold_compiled[i]["steps"],
            "steps_eager_warm": warm_eager[i]["steps"], "steps_compiled_warm": warm_compiled[i]["steps"],
            **dict(zip(DEV_COLUMNS[:4], dev_c)), **dict(zip(DEV_COLUMNS[4:8], dev_w)),
            "maxdF_warm_vs_cold_eager": np.max(np.abs(warm_eager[i]["forces"] - cold_eager[i]["forces"])),
            "maxdF_warm_vs_cold_compiled": np.max(np.abs(warm_compiled[i]["forces"] - cold_compiled[i]["forces"])),
        })
        for label, (de, maxdf, maxdmult, ddip), force_tol in (("cold", dev_c, FORCE_TOL), ("warm", dev_w, FORCE_TOL_WARM)):
            for name, value, tol in (("dE/atom", de, ENERGY_TOL_PER_ATOM), ("maxdF", maxdf, force_tol),
                                     ("maxdMult", maxdmult, MULTIPOLE_TOL), ("dDip/atom", ddip, DIPOLE_TOL_PER_ATOM)):
                if value > tol:
                    violations.append(f"structure {i} {label}: {name}={value:.3e} > {tol}")
        for mode, cold_row, warm_row in (("eager", cold_eager[i], warm_eager[i]),
                                          ("compiled", cold_compiled[i], warm_compiled[i])):
            if warm_row["steps"] > cold_row["steps"]:
                violations.append(f"structure {i} {mode}: steps_warm > steps_cold")
            if abs(warm_row["energy"] - cold_row["energy"]) / natoms > ENERGY_TOL_PER_ATOM:
                violations.append(f"structure {i} {mode}: warm-vs-cold energy mismatch")

    totals = {c: sum(r[c] for r in rows) for c in STEP_COLUMNS}
    for mode in ("eager", "compiled"):
        if totals[f"steps_{mode}_warm"] >= totals[f"steps_{mode}_cold"]:
            violations.append(f"{mode}: total warm steps not < total cold steps")

    columns = ["idx", "natoms", *STEP_COLUMNS, *DEV_COLUMNS]
    lines = ["\t".join(columns)] + ["\t".join(str(r[c]) for c in columns) for r in rows]
    lines.append("\t".join(["TOTAL", str(sum(r["natoms"] for r in rows))]
                           + [str(totals[c]) for c in STEP_COLUMNS] + ["-"] * len(DEV_COLUMNS)))
    table = "\n".join(lines)
    print(f"chunk_size={chunk_size}\n{table}")

    (tmp_path / "report.json").write_text(json.dumps(rows, indent=2))
    with open(tmp_path / "report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"report written to {tmp_path}")

    assert not violations, "\n".join(violations) + "\n\n" + table
