"""
Typed project configuration for the robotic spray-cooling simulation.

Configuration files contain experiment-specific values such as:

- material properties
- simulation duration and time step
- spray parameters
- geometry placement
- path density
- robot settings

Reusable solver and visualization modules should receive these values through
AppConfig rather than embedding experiment-specific constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


class ConfigurationError(ValueError):
    """Raised when a project configuration is missing or physically invalid."""


@dataclass(frozen=True)
class SimulationConfig:
    dt_s: float
    end_time_s: float

    @property
    def steps(self) -> int:
        return int(self.end_time_s / self.dt_s)


@dataclass(frozen=True)
class ThermalConfig:
    initial_temperature_c: float
    ambient_temperature_c: float
    thermal_conductivity_w_mk: float
    density_kg_m3: float
    specific_heat_j_kgk: float
    ambient_h_w_m2k: float
    spray_h_scale_w_m2k: float
    thickness_m: float

    @property
    def volumetric_heat_capacity_j_m3k(self) -> float:
        return self.density_kg_m3 * self.specific_heat_j_kgk

    @property
    def diffusivity_m2_s(self) -> float:
        return (
            self.thermal_conductivity_w_mk
            / self.volumetric_heat_capacity_j_m3k
        )


@dataclass(frozen=True)
class SprayConfig:
    sigma: float
    a: float
    reference_distance_m: float
    resolution: int
    field_of_view_deg: float
    volumetric_flow_rate_m3_s: float


@dataclass(frozen=True)
class GeometryConfig:
    mesh_path: Path
    plate_size_m: float
    plate_center_m: tuple[float, float, float]


@dataclass(frozen=True)
class PathConfig:
    points_per_row: int
    number_of_rows: int
    edge_margin_m: float
    nozzle_world_z_m: float


@dataclass(frozen=True)
class RobotConfig:
    description: str
    base_position_m: tuple[float, float, float]
    visual_nozzle_length_m: float


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig
    thermal: ThermalConfig
    spray: SprayConfig
    geometry: GeometryConfig
    path: PathConfig
    robot: RobotConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)

    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Configuration section [{name}] is missing or invalid."
        )

    return value


def _positive(value: float, name: str) -> float:
    value = float(value)

    if value <= 0.0:
        raise ConfigurationError(f"{name} must be greater than zero.")

    return value


def _positive_int(value: int, name: str) -> int:
    value = int(value)

    if value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer.")

    return value


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigurationError(
            f"{name} must contain exactly three numeric values."
        )

    return float(value[0]), float(value[1]), float(value[2])


def load_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> AppConfig:
    """
    Load and validate one spray-cooling TOML configuration.

    Relative mesh paths are resolved from project_root. When project_root is
    omitted, the parent of the configs directory is treated as the project root.
    """
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    if project_root is None:
        root = config_path.parent.parent
    else:
        root = Path(project_root).expanduser().resolve()

    with config_path.open("rb") as file:
        data = tomllib.load(file)

    simulation_data = _section(data, "simulation")
    thermal_data = _section(data, "thermal")
    spray_data = _section(data, "spray")
    geometry_data = _section(data, "geometry")
    path_data = _section(data, "path")
    robot_data = _section(data, "robot")

    simulation = SimulationConfig(
        dt_s=_positive(simulation_data["dt_s"], "simulation.dt_s"),
        end_time_s=_positive(
            simulation_data["end_time_s"],
            "simulation.end_time_s",
        ),
    )

    if simulation.steps < 1:
        raise ConfigurationError(
            "simulation.end_time_s must contain at least one time step."
        )

    thermal = ThermalConfig(
        initial_temperature_c=float(
            thermal_data["initial_temperature_c"]
        ),
        ambient_temperature_c=float(
            thermal_data["ambient_temperature_c"]
        ),
        thermal_conductivity_w_mk=_positive(
            thermal_data["thermal_conductivity_w_mk"],
            "thermal.thermal_conductivity_w_mk",
        ),
        density_kg_m3=_positive(
            thermal_data["density_kg_m3"],
            "thermal.density_kg_m3",
        ),
        specific_heat_j_kgk=_positive(
            thermal_data["specific_heat_j_kgk"],
            "thermal.specific_heat_j_kgk",
        ),
        ambient_h_w_m2k=_positive(
            thermal_data["ambient_h_w_m2k"],
            "thermal.ambient_h_w_m2k",
        ),
        spray_h_scale_w_m2k=_positive(
            thermal_data["spray_h_scale_w_m2k"],
            "thermal.spray_h_scale_w_m2k",
        ),
        thickness_m=_positive(
            thermal_data["thickness_m"],
            "thermal.thickness_m",
        ),
    )

    if thermal.initial_temperature_c < thermal.ambient_temperature_c:
        raise ConfigurationError(
            "Initial temperature must not be below ambient temperature."
        )

    spray = SprayConfig(
        sigma=_positive(spray_data["sigma"], "spray.sigma"),
        a=_positive(spray_data["a"], "spray.a"),
        reference_distance_m=_positive(
            spray_data["reference_distance_m"],
            "spray.reference_distance_m",
        ),
        resolution=_positive_int(
            spray_data["resolution"],
            "spray.resolution",
        ),
        field_of_view_deg=_positive(
            spray_data["field_of_view_deg"],
            "spray.field_of_view_deg",
        ),
        volumetric_flow_rate_m3_s=_positive(
            spray_data["volumetric_flow_rate_m3_s"],
            "spray.volumetric_flow_rate_m3_s",
        ),
    )

    mesh_path = Path(geometry_data["mesh_path"]).expanduser()

    if not mesh_path.is_absolute():
        mesh_path = root / mesh_path

    mesh_path = mesh_path.resolve()

    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Configured geometry mesh does not exist: {mesh_path}"
        )

    geometry = GeometryConfig(
        mesh_path=mesh_path,
        plate_size_m=_positive(
            geometry_data["plate_size_m"],
            "geometry.plate_size_m",
        ),
        plate_center_m=_vector3(
            geometry_data["plate_center_m"],
            "geometry.plate_center_m",
        ),
    )

    path = PathConfig(
        points_per_row=_positive_int(
            path_data["points_per_row"],
            "path.points_per_row",
        ),
        number_of_rows=_positive_int(
            path_data["number_of_rows"],
            "path.number_of_rows",
        ),
        edge_margin_m=float(path_data["edge_margin_m"]),
        nozzle_world_z_m=float(path_data["nozzle_world_z_m"]),
    )

    if path.edge_margin_m < 0.0:
        raise ConfigurationError(
            "path.edge_margin_m must not be negative."
        )

    if 2.0 * path.edge_margin_m >= geometry.plate_size_m:
        raise ConfigurationError(
            "The path edge margin consumes the entire plate."
        )

    robot = RobotConfig(
        description=str(robot_data["description"]),
        base_position_m=_vector3(
            robot_data["base_position_m"],
            "robot.base_position_m",
        ),
        visual_nozzle_length_m=_positive(
            robot_data["visual_nozzle_length_m"],
            "robot.visual_nozzle_length_m",
        ),
    )

    return AppConfig(
        simulation=simulation,
        thermal=thermal,
        spray=spray,
        geometry=geometry,
        path=path,
        robot=robot,
    )
