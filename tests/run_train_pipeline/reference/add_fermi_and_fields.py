import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def fermi_from_charge(total_charge: float) -> float:
    if total_charge < 0.0:
        return 0.0
    if total_charge > 0.0:
        return -10.0
    return -5.0


def main():
    parser = argparse.ArgumentParser(
        description="Add simple fermi_level and external_field info keys to an extxyz file."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--field-std", type=float, default=0.01)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    atoms_list = read(args.input, index=":")

    for atoms in atoms_list:
        total_charge = float(atoms.info.get("total_charge", 0.0))
        del atoms.info["fermi_level"]
        del atoms.info["external_field"]
        atoms.info["the_fermi_level"] = fermi_from_charge(total_charge)
        atoms.info["the_external_field"] = rng.normal(
            loc=0.0,
            scale=args.field_std,
            size=3,
        )

    write(args.output, atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
