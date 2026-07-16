from pathlib import Path

from spray_cooling.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_flat_plate_configuration_matches_baseline() -> None:
    cfg = load_config(
        PROJECT_ROOT / "configs" / "flat_plate.toml",
        project_root=PROJECT_ROOT,
    )

    assert cfg.simulation.dt_s == 0.1
    assert cfg.simulation.end_time_s == 600.0
    assert cfg.simulation.steps == 6000

    assert cfg.thermal.initial_temperature_c == 900.0
    assert cfg.thermal.ambient_temperature_c == 25.0
    assert cfg.thermal.thermal_conductivity_w_mk == 50.0
    assert cfg.thermal.density_kg_m3 == 7800.0
    assert cfg.thermal.specific_heat_j_kgk == 500.0
    assert cfg.thermal.thickness_m == 0.010

    assert cfg.geometry.plate_size_m == 0.4
    assert cfg.geometry.plate_center_m == (0.6, 0.0, -0.15)
    assert cfg.geometry.mesh_path.is_file()

    assert cfg.path.points_per_row == 20
    assert cfg.path.number_of_rows == 20

    assert cfg.robot.description == "ur5e_description"
    assert cfg.robot.base_position_m == (0.0, 0.0, 0.0)
