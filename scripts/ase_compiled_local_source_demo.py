#!/usr/bin/env python
import argparse
import time

import ase.io
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from mace_scf.calculators.localsources import MACELocalSplitCharges, MACELocalCharges


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pbc_handling", choices=["pbc", "slab"], default="pbc")
    parser.add_argument("--formal_charges_key", default="formal_oxidation_states")
    parser.add_argument("--external_field_key", default="external_field")
    parser.add_argument("--fermi_level_key", default="fermi_level")
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--mode", default="reduce-overhead")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timestep_fs", type=float, default=0.5)
    parser.add_argument("--temperature_K", type=float, default=300.0)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--log", default="log.log")
    return parser.parse_args()


def main():
    args = parse_args()
    atoms = ase.io.read(args.input)
    thecell = atoms.get_cell()
    thecell[2, 2] = 80.0
    atoms.set_cell(thecell)

    atoms.calc = MACELocalCharges(
        model_path=args.model,
        device=args.device,
        formal_charges_key=args.formal_charges_key,
        external_field_key=args.external_field_key,
        fermi_level_key=args.fermi_level_key,
        pbc_handling=args.pbc_handling,
        use_compile=args.compile,
        compile_backend=args.backend,
        compile_mode=args.mode,
        compile_dynamic=args.dynamic,
        compile_fullgraph=args.fullgraph,
    )
    
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
