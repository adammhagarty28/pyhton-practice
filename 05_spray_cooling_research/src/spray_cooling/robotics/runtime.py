"""Shared functionality extracted from the geometry demos."""

from __future__ import annotations

import numpy as np


def world_to_base(pos_world):
    return np.array(pos_world)


def set_initial_joint(_last_joint_state, chain, name, value):
    """Set initial joint value by joint name, not by fragile hardcoded index."""
    for idx, link in enumerate(chain.links):
        if link.name == name:
            _last_joint_state[idx] = value
            return
    print(f'WARNING: joint {name} not found in ikpy chain')


def q_to_cfg(UR5E_JOINT_NAMES, chain, q):
    """
    Convert ikpy q vector into a yourdfpy config dictionary.

    Do this by matching joint names, not by assuming q[i + 1].
    This prevents ikpy's OriginLink or fixed links from shifting the UR5e joints.
    """
    cfg = {}
    for idx, link in enumerate(chain.links):
        if link.name in UR5E_JOINT_NAMES:
            cfg[link.name] = float(q[idx])
    return cfg


def get_tool0_transform_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q):
    """Return full tool0 transform in world coordinates."""
    cfg = q_to_cfg(UR5E_JOINT_NAMES, chain, q)
    urdf.update_cfg(cfg)
    T_tool0 = np.array(urdf.get_transform('tool0'))
    return robot_base_T @ T_tool0


def get_tool0_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q):
    """Return tool0 position in world coordinates."""
    return get_tool0_transform_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q)[:3, 3]


def get_nozzle_tip_world(NOZZLE_LENGTH, UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q):
    """
    Return physical nozzle exit position.

    The visual nozzle points along local -Z from tool0, so the tip is
    NOZZLE_LENGTH below tool0 in the tool0 local frame.
    """
    T_tool0_world = get_tool0_transform_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q)
    tip_local = np.array([0.0, 0.0, -NOZZLE_LENGTH, 1.0])
    return (T_tool0_world @ tip_local)[:3]


def get_nozzle_direction_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q):
    """
    Return spray direction in world coordinates.

    The nozzle sprays along local -Z of tool0.
    """
    T_tool0_world = get_tool0_transform_world(UR5E_JOINT_NAMES, chain, robot_base_T, urdf, q)
    local_minus_z = np.array([0.0, 0.0, -1.0])
    d = T_tool0_world[:3, :3] @ local_minus_z
    return d / (np.linalg.norm(d) + 1e-08)
