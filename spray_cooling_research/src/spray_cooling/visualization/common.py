"""Shared functionality extracted from the geometry demos."""

from __future__ import annotations

import numpy as np
import vtk


def set_actor_matrix(actor, M):
    """Apply a 4x4 numpy matrix to a vtk actor."""
    vtk_m = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_m.SetElement(i, j, M[i, j])
    actor.SetUserMatrix(vtk_m)


def set_spray_head_position(spray_head_actor, pos_world):
    M = np.eye(4)
    M[:3, 3] = np.array(pos_world)
    set_actor_matrix(spray_head_actor, M)
