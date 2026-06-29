#!/usr/bin/env python3
"""
controller.py — pure visual-servoing controller (NO robot, NO I/O).
===================================================================

This is the CONTROLLER layer. It is a pure function of image features:

        features (normalized x,y + depth Z)  ->  camera-frame 6-twist

It wraps the patched IBVS_Controller (Chaumette mean-interaction IBVS law,
    v = -lambda * ( 1/2 (L_s + L_s*) )^+ * e ,   e = s - s* )
and exposes a single clean output: a full 6-vector camera-frame twist
[vx, vy, vz, wx, wy, wz]. Inactive DOFs (per the chosen control mode) are
zero-filled so the SEAM to the robot layer is always a full 6-twist,
regardless of control mode.

It imports numpy and the IBVS class ONLY. No zmq, no rtde, no mujoco.
That purity is the whole point: you can unit-test this with synthetic
points and reason about the control law in isolation.

Camera frame (matches perception.py / OpenCV): +x right, +y down, +z into image.
"""

import numpy as np
from ibvs_controller_new import IBVS_Controller


class VisualServoController:
    """Features -> camera-frame 6-twist. Knows nothing about any robot."""

    # map each control-mode axis to its slot in the full 6-twist
    _AXIS_SLOT = {"vx": 0, "vy": 1, "vz": 2, "wx": 3, "wy": 4, "wz": 5}

    def __init__(self,
                 linear_velocities: str = "xyz",
                 angular_velocities: str = "y",
                 interaction_mode: str = "mean",
                 num_pts: int = 4,
                 lam: float = 0.5):
        """
        :param linear_velocities:  subset of 'xyz' (translational DOFs).
        :param angular_velocities: subset of 'xyz' (rotational DOFs).
                                   Default 'xyz'+'y' == the '4xyzy' mode.
        :param interaction_mode:   'curr' | 'desired' | 'mean'. 'mean' realizes
                                   the slide's 1/2(L_s+L_s*) law (needs Z and Z*).
        :param num_pts:            number of feature points (4 ArUco corners).
        :param lam:                proportional gain lambda (exponential decay).
        """
        self.ibvs = IBVS_Controller(linear_velocities, angular_velocities,
                                    interaction_mode, num_pts)
        self.ibvs.set_lambda_matrix([float(lam)] * self.ibvs.num_degs)
        self.num_pts = num_pts
        self.mode = interaction_mode

        # active full-6-twist slots, in the controller's output order
        dof = self.ibvs.degrees_of_freedom            # [vx,vy,vz,wx,wy,wz]
        self._active_slots = [i for i in range(6) if dof[i]]

        self._goal_set = False

    # --- teach step -----------------------------------------------------------
    def set_goal(self, star_pts):
        """
        Set the desired feature configuration s*.

        :param star_pts: list of (x, y, Z*) tuples — normalized desired corner
                         positions AND the reference depth Z*. Z* matters for
                         'desired' and 'mean' modes (the slide needs Z*); in
                         'curr' mode it is ignored and may be any positive value.
        """
        assert len(star_pts) == self.num_pts, \
            f"need {self.num_pts} desired points, got {len(star_pts)}"
        self.ibvs.set_desired_points([(float(x), float(y), float(z))
                                      for (x, y, z) in star_pts])
        self._goal_set = True

    # --- per-frame control ----------------------------------------------------
    def compute(self, curr_pts) -> np.ndarray:
        """
        Compute the camera-frame 6-twist for the current features.

        :param curr_pts: list of (x, y, Z) tuples — normalized current corner
                         positions and current depth Z.
        :return: np.ndarray shape (6,) = [vx, vy, vz, wx, wy, wz] in camera frame.
                 Inactive DOFs are 0.
        """
        assert self._goal_set, "call set_goal(s*) before compute()"
        assert len(curr_pts) == self.num_pts, \
            f"need {self.num_pts} current points, got {len(curr_pts)}"

        self.ibvs.set_current_points([(float(x), float(y), float(z))
                                      for (x, y, z) in curr_pts])
        self.ibvs.calculate_interaction_matrix()
        v_subset = np.asarray(self.ibvs.calculate_velocities(),
                              dtype=float).flatten()

        v_cam = np.zeros(6)
        for slot, val in zip(self._active_slots, v_subset):
            v_cam[slot] = val
        return v_cam                                  # <-- the SEAM

    # --- convergence ----------------------------------------------------------
    @property
    def error_norm(self) -> float:
        """L2 norm of the current feature error e = s - s* (pixels-normalized)."""
        if self.ibvs.errs is None:
            return float("inf")
        return float(np.linalg.norm(self.ibvs.errs))


# --------------------------------------------------------------------------- #
# Self-test: pure, no robot, no perception. Run: python controller.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # 4 corners, target deeper than goal -> expect forward motion (+vz)
    star = [(-0.5, -0.5, 1.0), (0.5, -0.5, 1.0),
            (-0.5,  0.5, 1.0), (0.5,  0.5, 1.0)]
    far  = [(-0.2, -0.2, 5.0), (0.2, -0.2, 5.0),
            (-0.2,  0.2, 5.0), (0.2,  0.2, 5.0)]

    c = VisualServoController("xyz", "y", "mean", num_pts=4, lam=0.5)
    c.set_goal(star)
    v = c.compute(far)
    print("camera twist [vx vy vz wx wy wz]:", np.round(v, 4))
    print("error norm:", round(c.error_norm, 4))
    assert v.shape == (6,)
    assert v[2] > 0, "deeper target should command +vz (move into image)"
    assert abs(v[3]) < 1e-9 and abs(v[5]) < 1e-9, "wx,wz inactive in 4xyzy"
    print("controller self-test OK")
