"""Validate the Python environment needed by the spray-cooling project."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def show_distribution(name: str) -> None:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise SystemExit(f"ERROR: required distribution is missing: {name}")

    print(f"{name:22s} {version}")


print("Python")
print("------")
print(sys.version)
print()

print("Required distributions")
print("----------------------")

for distribution in (
    "numpy",
    "jax",
    "jaxlib",
    "pyvista",
    "vtk",
    "matplotlib",
    "trimesh",
    "ikpy",
    "robot-descriptions",
    "yourdfpy",
):
    show_distribution(distribution)

print()
print("Project imports")
print("---------------")

import jax
import numpy
import pyvista
import vtk
import matplotlib
import trimesh
import ikpy
import robot_descriptions
import yourdfpy

from relevant_PULSE_files.jax_kernels import Pose, deposit
from relevant_PULSE_files.jax_pulse import Pulse
from spray_cooling.config import load_config
from spray_cooling.geometry.surface_mesh import load_surface_mesh

print("PASS: project imports succeeded.")
print()

print("JAX")
print("---")
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
print()

config_path = PROJECT_ROOT / "configs" / "flat_plate.toml"
config = load_config(config_path, project_root=PROJECT_ROOT)

assert config.geometry.mesh_path.is_file()
assert config.simulation.steps == 6000
assert config.path.points_per_row * config.path.number_of_rows == 400

print("Project assets")
print("--------------")
print("Configuration:", config_path)
print("Mesh:", config.geometry.mesh_path)
print("Simulation steps:", config.simulation.steps)
print(
    "Waypoints:",
    config.path.points_per_row * config.path.number_of_rows,
)
print()
print("PASS: environment and project assets are valid.")
