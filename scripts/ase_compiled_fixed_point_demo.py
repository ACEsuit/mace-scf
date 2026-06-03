#!/usr/bin/env python
import argparse
import time

import ase.io
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pbc_handling", choices=["pbc", "slab"], default="pbc")
    parser.add_argument("--atomic_multipoles_key", default="DMA_coeficients")
    parser.add_argument("--external_field_key", default="external_field")
    parser.add_argument("--fermi_level_key", default="fermi_level")
    parser.add_argument("--total_charge_key", default="total_charge")
    parser.add_argument("--num_scf_steps", type=int, default=20)
    parser.add_argument("--mixing_parameter", type=float, default=0.2)
    parser.add_argument("--scf_tolerance", type=float, default=1.0e-5)
    parser.add_argument("--constant_charge", action="store_true")
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--compile_scope",
        choices=("scf", "scf_observables", "scf_chunk", "full"),
        default="scf",
    )
    parser.add_argument("--compile_chunk_size", type=int, default=1)
    parser.add_argument("--compile_warmup_steps", type=int, default=0)
    parser.add_argument("--compile_warmup_scf_steps", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timestep_fs", type=float, default=0.5)
    parser.add_argument("--temperature_K", type=float, default=300.0)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--log", default="log.log")
    parser.add_argument("--cell_z", type=float, default=80.0)
    return parser.parse_args()


def main():
    args = parse_args()
    atoms = ase.io.read(args.input)
    thecell = atoms.get_cell()
    thecell[2, 2] = args.cell_z
    atoms.set_cell(thecell)

    atoms.calc = MACEFixedPointSCF(
        model_path=args.model,
        device=args.device,
        atomic_multipoles_key=args.atomic_multipoles_key,
        external_field_key=args.external_field_key,
        fermi_level_key=args.fermi_level_key,
        total_charge_key=args.total_charge_key,
        pbc_handling=args.pbc_handling,
        scf_options={
            "constant_charge": args.constant_charge,
            "num_scf_steps": args.num_scf_steps,
            "mixing_parameter": args.mixing_parameter,
            "scf_tolerance": args.scf_tolerance,
            "initial_density": "local_guess",
            "initial_fermi_level": "from_data",
        },
        use_compile=args.compile,
        compile_backend=args.backend,
        compile_mode=args.mode,
        compile_dynamic=args.dynamic,
        compile_fullgraph=args.fullgraph,
        compile_scope=args.compile_scope,
        compile_chunk_size=args.compile_chunk_size,
    )

    if args.compile_warmup_steps > 0:
        compile_warmup_start = time.perf_counter()
        atoms.calc.warmup_compiled_evaluator(
            atoms,
            num_scf_steps=args.compile_warmup_scf_steps,
            num_calls=args.compile_warmup_steps,
        )
        compile_warmup_time = time.perf_counter() - compile_warmup_start
        print(f"compile_warmup_time_s={compile_warmup_time:.6f}")

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K)
    dyn = VelocityVerlet(
        atoms,
        timestep=args.timestep_fs * units.fs,
        trajectory=args.trajectory,
        logfile=args.log,
    )

    warmup_start = time.perf_counter()
    dyn.run(args.warmup_steps)
    warmup_time = time.perf_counter() - warmup_start

    md_start = time.perf_counter()
    dyn.run(args.steps)
    md_time = time.perf_counter() - md_start

    print(f"warmup_time_s={warmup_time:.6f}")
    print(f"md_time_s={md_time:.6f}")
    if args.steps > 0:
        print(f"mean_step_time_s={md_time / args.steps:.6f}")


if __name__ == "__main__":
    main()
