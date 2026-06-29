#!/usr/bin/env python3
"""
predictor.py — graceful missing-frame handling.
===============================================

When a frame is late or dropped, the control loop must not stall or jerk.
This module extrapolates the target's image features forward using the most
recent observation plus the perception velocity estimate (constant-velocity
model). Pure math; no I/O.

Two prediction paths:
  * If a camera-frame velocity (vel) and depth are available, project the 3D
    constant-velocity motion into pixel motion via the interaction matrix
    relationship (du/dt ~ depends on vx,vy,vz). This is the principled path.
  * Otherwise, hold the last observation (zero-order hold) — still safe, just
    less anticipatory.

The loop decides WHEN to predict (based on frame age); this module decides
WHAT the predicted features are.
"""
from __future__ import annotations
from typing import Optional, List
import numpy as np

from schema import TargetState


class FeaturePredictor:
    """Extrapolates pixel corners forward from the last good TargetState."""

    def __init__(self, K: np.ndarray):
        self.K = np.asarray(K, dtype=float)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def predict_corners(self, last: TargetState, dt: float) -> Optional[np.ndarray]:
        """
        Return predicted Nx2 pixel corners dt seconds after `last`.

        Uses the per-point image Jacobian to turn the camera-frame linear
        velocity (vx,vy,vz) into pixel velocity:
            u_dot = fx * ( -vx/Z + x*vz/Z )      (x = (u-cx)/fx)
            v_dot = fy * ( -vy/Z + y*vz/Z )      (y = (v-cy)/fy)
        Falls back to zero-order hold (return last corners) if vel/depth absent.
        """
        if last.corners is None:
            return None
        corners = np.asarray(last.corners, dtype=float)        # Nx2 pixels

        if last.vel is None or not last.has_depth():
            return corners                                     # zero-order hold

        vx, vy, vz = last.vel
        Z = float(last.xyz[2])
        if Z <= 1e-6:
            return corners

        pred = np.empty_like(corners)
        for i, (u, v) in enumerate(corners):
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
            u_dot = self.fx * (-vx / Z + x * vz / Z)
            v_dot = self.fy * (-vy / Z + y * vz / Z)
            pred[i, 0] = u + u_dot * dt
            pred[i, 1] = v + v_dot * dt
        return pred

    def predict_depth(self, last: TargetState, dt: float, fallback: float) -> float:
        """Extrapolate depth Z forward with vz (constant-velocity)."""
        if not last.has_depth():
            return float(fallback)
        Z = float(last.xyz[2])
        if last.vel is not None:
            Z = Z + float(last.vel[2]) * dt
        return max(Z, 1e-3)
