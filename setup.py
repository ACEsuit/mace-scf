from pathlib import Path

from setuptools import find_packages, setup


version_ns = {}
exec(
    (Path(__file__).resolve().parent / "mace_scf" / "__version__.py").read_text(
        encoding="utf-8"
    ),
    version_ns,
)

setup(
    name="mace_scf",
    version=version_ns["__version__"],
    packages=find_packages(include=["mace_scf", "mace_scf.*"]),
)
