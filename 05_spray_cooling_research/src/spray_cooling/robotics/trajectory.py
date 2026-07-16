"""Shared robot trajectory utilities.

These functions are geometry-independent and are shared by
the flat-plate and half-sphere demonstrations.
"""

from __future__ import annotations

import numpy as np


def unwrap_to_reference(q_candidate, q_reference):
    """
    Keep IK continuous by choosing equivalent joint angles closest to previous pose.
    This prevents +/-2pi branch jumps that look like vibration.
    """
    q_candidate = np.array(q_candidate, dtype=float)
    q_reference = np.array(q_reference, dtype=float)
    delta = q_candidate - q_reference
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return q_reference + delta


def smooth_joint_path(chain, q_path, passes=3):
    """
    Post-process cached IK trajectory to suppress tiny rapid vibrations.

    This is a visual trajectory smoother, similar in spirit to robot motion planning:
    don't execute raw noisy inverse-kinematics frame outputs directly.
    """
    q_path = np.array(q_path, dtype=float)
    max_steps = np.full(q_path.shape[1], 0.04)
    for idx, link in enumerate(chain.links):
        if link.name in ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint'):
            max_steps[idx] = 0.022
        elif link.name in ('wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'):
            max_steps[idx] = 0.026
    q = q_path.copy()
    for _ in range(passes):
        for i in range(1, len(q)):
            dq = q[i] - q[i - 1]
            dq = np.clip(dq, -max_steps, max_steps)
            q[i] = q[i - 1] + dq
        for i in range(len(q) - 2, -1, -1):
            dq = q[i] - q[i + 1]
            dq = np.clip(dq, -max_steps, max_steps)
            q[i] = q[i + 1] + dq
        q2 = q.copy()
        for i in range(1, len(q) - 1):
            q2[i] = 0.25 * q[i - 1] + 0.5 * q[i] + 0.25 * q[i + 1]
        q = q2
    return q
