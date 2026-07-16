# Robot Spray Cooling Research

Robotic spray-cooling simulation using:

- JAX for thermal time integration
- JAX-PULSE for mesh-based spray deposition
- PyVista for interactive 3D visualization
- ikpy and robot_descriptions for UR5e inverse kinematics

## Current baseline

The first modularization target is the working flat-plate demonstration.

The numerical and visual behavior must remain unchanged while the original
monolithic implementation is separated into reusable modules.

## Project areas

- `configs/`: geometry and simulation configurations
- `data/meshes/`: canonical input geometry
- `demos/`: executable research demonstrations
- `src/spray_cooling/`: project-owned reusable code
- `src/relevant_PULSE_files/`: vendored JAX-PULSE dependency
- `tests/`: automated unit and regression tests
- `legacy/`: frozen pre-refactor baseline
- `artifacts/generated/`: runtime-generated URDF and temporary geometry
- `results/`: simulation outputs

## Development rule

The flat plate and half-sphere use the same thermal, spray, robotics,
simulation, and visualization systems. Geometry-specific behavior belongs in
geometry configuration and path-planning modules.
