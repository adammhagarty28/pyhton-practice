"""
Typed project configuration for the robotic spray-cooling simulation.
Extended for volume-mesh POC 3: T-dependent properties, boiling curve,
radiation, adaptive hysteresis, S-curve valve dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple
import tomllib


class ConfigurationError(ValueError):
    pass


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
    thermal_conductivity_w_mk: float          # legacy constant fallback
    density_kg_m3: float
    specific_heat_j_kgk: float                # legacy constant fallback
    ambient_h_w_m2k: float
    spray_h_scale_w_m2k: float                # legacy scale (unused with boiling curve)
    thickness_m: float
    # T-dependent property tables
    temp_table_k: Tuple[float, ...]           # [K]
    k_table_w_mk: Tuple[float, ...]           # [W/(m K)]
    cp_table_j_kgk: Tuple[float, ...]         # [J/(kg K)]
    emissivity_table: Tuple[float, ...]       # [-]
    # Volume mesh
    layer_thicknesses_m: Tuple[float, ...]    # [m]

    @property
    def volumetric_heat_capacity_j_m3k(self) -> float:
        return self.density_kg_m3 * self.specific_heat_j_kgk

    @property
    def diffusivity_m2_s(self) -> float:
        return self.thermal_conductivity_w_mk / self.volumetric_heat_capacity_j_m3k

    @property
    def n_layers(self) -> int:
        return len(self.layer_thicknesses_m)


@dataclass(frozen=True)
class RadiationConfig:
    stefan_boltzmann_w_m2k4: float            # [W/(m^2 K^4)]


@dataclass(frozen=True)
class SprayConfig:
    sigma: float
    a: float
    reference_distance_m: float
    resolution: int
    field_of_view_deg: float
    volumetric_flow_rate_m3_s: float
    boiling_temp_table_c: Tuple[float, ...]   # [C]
    boiling_h_table_w_m2k: Tuple[float, ...]  # [W/(m^2 K)]


@dataclass(frozen=True)
class ControlConfig:
    delta_off_fraction: float                 # [-]
    delta_on_fraction: float                  # [-]
    spray_ramp_tau_s: float                   # [s]


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
    radiation: RadiationConfig
    spray: SprayConfig
    control: ControlConfig
    geometry: GeometryConfig
    path: PathConfig
    robot: RobotConfig


def _section(data, name):
    v = data.get(name)
    if not isinstance(v, dict):
        raise ConfigurationError(f"Configuration section [{name}] is missing or invalid.")
    return v


def _positive(value, name):
    v = float(value)
    if v <= 0.0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return v


def _positive_int(value, name):
    v = int(value)
    if v <= 0:
        raise ConfigurationError(f"{name} must be a positive integer.")
    return v


def _vector3(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigurationError(f"{name} must contain exactly three numeric values.")
    return float(value[0]), float(value[1]), float(value[2])


def _float_tuple(value, name):
    if not isinstance(value, list) or len(value) < 1:
        raise ConfigurationError(f"{name} must be a non-empty list of numbers.")
    return tuple(float(v) for v in value)


def load_config(config_path, *, project_root=None):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    if project_root is None:
        root = config_path.parent.parent
    else:
        root = Path(project_root).expanduser().resolve()

    with config_path.open("rb") as file:
        data = tomllib.load(file)

    simulation_data = _section(data, "simulation")
    thermal_data    = _section(data, "thermal")
    radiation_data  = _section(data, "radiation")
    spray_data      = _section(data, "spray")
    control_data    = _section(data, "control")
    geometry_data   = _section(data, "geometry")
    path_data       = _section(data, "path")
    robot_data      = _section(data, "robot")

    simulation = SimulationConfig(
        dt_s=_positive(simulation_data["dt_s"], "simulation.dt_s"),
        end_time_s=_positive(simulation_data["end_time_s"], "simulation.end_time_s"),
    )
    if simulation.steps < 1:
        raise ConfigurationError("simulation.end_time_s must contain at least one time step.")

    layer_thicks = _float_tuple(thermal_data["layer_thicknesses_m"], "thermal.layer_thicknesses_m")
    thickness = _positive(thermal_data["thickness_m"], "thermal.thickness_m")
    if abs(sum(layer_thicks) - thickness) > 1e-9:
        raise ConfigurationError(
            f"layer_thicknesses_m sums to {sum(layer_thicks)} but thickness_m is {thickness}"
        )

    thermal = ThermalConfig(
        initial_temperature_c=float(thermal_data["initial_temperature_c"]),
        ambient_temperature_c=float(thermal_data["ambient_temperature_c"]),
        thermal_conductivity_w_mk=_positive(thermal_data["thermal_conductivity_w_mk"], "thermal.thermal_conductivity_w_mk"),
        density_kg_m3=_positive(thermal_data["density_kg_m3"], "thermal.density_kg_m3"),
        specific_heat_j_kgk=_positive(thermal_data["specific_heat_j_kgk"], "thermal.specific_heat_j_kgk"),
        ambient_h_w_m2k=_positive(thermal_data["ambient_h_w_m2k"], "thermal.ambient_h_w_m2k"),
        spray_h_scale_w_m2k=_positive(thermal_data["spray_h_scale_w_m2k"], "thermal.spray_h_scale_w_m2k"),
        thickness_m=thickness,
        temp_table_k=_float_tuple(thermal_data["temp_table_k"], "thermal.temp_table_k"),
        k_table_w_mk=_float_tuple(thermal_data["k_table_w_mk"], "thermal.k_table_w_mk"),
        cp_table_j_kgk=_float_tuple(thermal_data["cp_table_j_kgk"], "thermal.cp_table_j_kgk"),
        emissivity_table=_float_tuple(thermal_data["emissivity_table"], "thermal.emissivity_table"),
        layer_thicknesses_m=layer_thicks,
    )
    if thermal.initial_temperature_c < thermal.ambient_temperature_c:
        raise ConfigurationError("Initial temperature must not be below ambient temperature.")

    radiation = RadiationConfig(
        stefan_boltzmann_w_m2k4=_positive(radiation_data["stefan_boltzmann_w_m2k4"], "radiation.stefan_boltzmann_w_m2k4"),
    )

    spray = SprayConfig(
        sigma=_positive(spray_data["sigma"], "spray.sigma"),
        a=_positive(spray_data["a"], "spray.a"),
        reference_distance_m=_positive(spray_data["reference_distance_m"], "spray.reference_distance_m"),
        resolution=_positive_int(spray_data["resolution"], "spray.resolution"),
        field_of_view_deg=_positive(spray_data["field_of_view_deg"], "spray.field_of_view_deg"),
        volumetric_flow_rate_m3_s=_positive(spray_data["volumetric_flow_rate_m3_s"], "spray.volumetric_flow_rate_m3_s"),
        boiling_temp_table_c=_float_tuple(spray_data["boiling_temp_table_c"], "spray.boiling_temp_table_c"),
        boiling_h_table_w_m2k=_float_tuple(spray_data["boiling_h_table_w_m2k"], "spray.boiling_h_table_w_m2k"),
    )

    control = ControlConfig(
        delta_off_fraction=_positive(control_data["delta_off_fraction"], "control.delta_off_fraction"),
        delta_on_fraction=_positive(control_data["delta_on_fraction"], "control.delta_on_fraction"),
        spray_ramp_tau_s=_positive(control_data["spray_ramp_tau_s"], "control.spray_ramp_tau_s"),
    )

    mesh_path = Path(geometry_data["mesh_path"]).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = root / mesh_path
    mesh_path = mesh_path.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Configured geometry mesh does not exist: {mesh_path}")

    geometry = GeometryConfig(
        mesh_path=mesh_path,
        plate_size_m=_positive(geometry_data["plate_size_m"], "geometry.plate_size_m"),
        plate_center_m=_vector3(geometry_data["plate_center_m"], "geometry.plate_center_m"),
    )

    path_cfg = PathConfig(
        points_per_row=_positive_int(path_data["points_per_row"], "path.points_per_row"),
        number_of_rows=_positive_int(path_data["number_of_rows"], "path.number_of_rows"),
        edge_margin_m=float(path_data["edge_margin_m"]),
        nozzle_world_z_m=float(path_data["nozzle_world_z_m"]),
    )
    if path_cfg.edge_margin_m < 0.0:
        raise ConfigurationError("path.edge_margin_m must not be negative.")
    if 2.0 * path_cfg.edge_margin_m >= geometry.plate_size_m:
        raise ConfigurationError("The path edge margin consumes the entire plate.")

    robot = RobotConfig(
        description=str(robot_data["description"]),
        base_position_m=_vector3(robot_data["base_position_m"], "robot.base_position_m"),
        visual_nozzle_length_m=_positive(robot_data["visual_nozzle_length_m"], "robot.visual_nozzle_length_m"),
    )

    return AppConfig(
        simulation=simulation,
        thermal=thermal,
        radiation=radiation,
        spray=spray,
        control=control,
        geometry=geometry,
        path=path_cfg,
        robot=robot,
    )
