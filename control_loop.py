#!/usr/bin/env python3
"""
control_loop.py — deterministic fixed-rate IBVS loop.
=====================================================

Ties the three layers together at a FIXED frequency (50-100 Hz) and stays
stable under jitter and dropped perception frames.

State machine driven by the age of the latest perception frame:

    TRACKING    fresh frame (age <= fresh_dt)
                -> use observed features directly.
    PREDICTING  frame is stale but recent (fresh_dt < age <= lost_dt)
                -> extrapolate features with the constant-velocity predictor;
                   keep commanding so the arm doesn't stall on a dropped frame.
    LOST        no frame for too long (age > lost_dt)
                -> stop the robot, hold, wait for re-acquisition.

Determinism / jitter:
  * The loop targets a fixed period and sleeps only the remaining time each
    tick (busy-margin compensation), so control rate stays near-constant even
    when compute time varies.
  * Perception runs asynchronously; the loop never blocks on it (CONFLATE SUB).
  * dt passed to the predictor is the measured age, so prediction is correct
    regardless of how stale the frame is.

This module owns orchestration ONLY. It computes no control law (controller)
and talks to no socket/robot directly (interface). It just schedules and routes.
"""
from __future__ import annotations
from enum import Enum
import time
import numpy as np


class LoopState(Enum):
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    LOST = "LOST"


class ControlLoop:
    def __init__(self, subscriber, controller, robot, predictor, K,
                 rate_hz=100.0,
                 fresh_dt=0.05,      # <=50 ms old -> trust observation
                 lost_dt=0.40,       # >400 ms old -> declare LOST
                 conv_tol=0.02,      # ||e|| convergence threshold
                 z_fallback=0.35):
        self.sub = subscriber
        self.ctrl = controller
        self.robot = robot
        self.pred = predictor
        self.K = np.asarray(K, float)
        self.Kinv = np.linalg.inv(self.K)

        self.period = 1.0 / rate_hz
        self.fresh_dt = fresh_dt
        self.lost_dt = lost_dt
        self.conv_tol = conv_tol
        self.z_fallback = z_fallback

        self.state = LoopState.LOST
        self._last_good = None       # last TargetState with found=True

        # diagnostics
        self.stats = {"ticks": 0, "tracking": 0, "predicting": 0,
                      "lost": 0, "commands": 0, "max_jitter_ms": 0.0}

    # --- pixel -> normalized ------------------------------------------------
    def _normalize(self, corners_px):
        c = np.asarray(corners_px, float)
        uv1 = np.hstack([c, np.ones((c.shape[0], 1))])
        return (self.Kinv @ uv1.T).T[:, :2]

    # --- teach goal ---------------------------------------------------------
    def teach_goal(self, timeout_s=30.0):
        """Capture s* from the first fresh, usable frame."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            st, age = self.sub.latest()
            if st and st.found and st.corners is not None and age <= self.fresh_dt:
                Zstar = st.depth(self.z_fallback)
                star_n = self._normalize(st.corners)
                self.ctrl.set_goal([(x, y, Zstar) for x, y in star_n])
                return True, Zstar
            time.sleep(0.02)
        return False, None

    # --- one tick -----------------------------------------------------------
    def _classify(self, st, age):
        if st is None or self._last_good is None and not (st and st.found):
            return LoopState.LOST
        if st.found and age <= self.fresh_dt:
            return LoopState.TRACKING
        if self._last_good is not None and age <= self.lost_dt:
            return LoopState.PREDICTING
        return LoopState.LOST

    def _features_for(self, state, st, age):
        """Return (curr_pts, ok). curr_pts is list of (x,y,Z)."""
        if state == LoopState.TRACKING:
            self._last_good = st
            Z = st.depth(self.z_fallback)
            return [(x, y, Z) for x, y in self._normalize(st.corners)], True
        if state == LoopState.PREDICTING:
            pred_px = self.pred.predict_corners(self._last_good, age)
            if pred_px is None:
                return None, False
            Z = self.pred.predict_depth(self._last_good, age, self.z_fallback)
            return [(x, y, Z) for x, y in self._normalize(pred_px)], True
        return None, False

    def tick(self):
        """Run exactly one control step. Returns the current LoopState."""
        st, age = self.sub.latest()
        state = self._classify(st, age)
        self.state = state

        if state == LoopState.LOST:
            self.robot.stop()
            self.stats["lost"] += 1
            return state

        curr_pts, ok = self._features_for(state, st, age)
        if not ok:
            self.robot.stop()
            self.stats["lost"] += 1
            self.state = LoopState.LOST
            return self.state

        v_cam = self.ctrl.compute(curr_pts)

        if self.ctrl.error_norm < self.conv_tol:
            self.robot.stop()
        else:
            self.robot.send_camera_twist(v_cam)
            self.stats["commands"] += 1

        self.stats["tracking" if state == LoopState.TRACKING else "predicting"] += 1
        return state

    # --- fixed-rate driver --------------------------------------------------
    def run(self, max_ticks=None, stop_on_converge=True):
        """Block, running tick() at the fixed rate until converged/interrupted."""
        next_t = time.monotonic()
        try:
            while True:
                tick_start = time.monotonic()
                jitter = (tick_start - next_t) * 1000.0
                self.stats["max_jitter_ms"] = max(self.stats["max_jitter_ms"],
                                                  abs(jitter))

                state = self.tick()
                self.stats["ticks"] += 1

                if (stop_on_converge and state != LoopState.LOST
                        and self.ctrl.error_norm < self.conv_tol):
                    return "converged"
                if max_ticks and self.stats["ticks"] >= max_ticks:
                    return "max_ticks"

                # fixed-rate sleep with jitter compensation
                next_t += self.period
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.monotonic()      # we fell behind; resync
        except KeyboardInterrupt:
            return "interrupted"
        finally:
            self.robot.stop()
