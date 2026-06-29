#!/usr/bin/env python3
"""
robot_interface.py — controller → UR5e execution layer (RTDE preferred).
========================================================================

ONLY executes commands. Owns:
  * cVn  : eye-in-hand twist transform (camera-frame twist -> EE-frame twist)
  * saturation : safe speed clamps
  * comms : ur-rtde speedL / speedJ
  * continuous feedback : a background thread reading RTDEReceiveInterface so
    robot state (q, qd, TCP pose) is always fresh for logging / safety / a
    future joint-space law.

The controller hands this layer a CAMERA-FRAME 6-twist and nothing else.

Backends:
  RTDEInterface : real UR5e / URSim. speedL (EE twist) by default; speedJ path
                  included for joint-velocity control if you compute qd upstream.
  MockInterface : logs commands + fakes feedback; for offline loop validation.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import threading
import time
import numpy as np


# --------------------------------------------------------------------------- #
# eye-in-hand twist transform
# --------------------------------------------------------------------------- #
def _skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=float)


def twist_transform(R, t):
    """cVn: v_cam = cVn @ v_ee. R = EE->camera rot, t = camera origin in EE frame."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    V = np.zeros((6, 6))
    V[:3, :3] = R
    V[:3, 3:] = _skew(t) @ R
    V[3:, 3:] = R
    return V


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class RobotInterface(ABC):
    def __init__(self, R_cam_ee, t_cam_ee, v_max=0.25, w_max=1.0):
        self.cVn = twist_transform(R_cam_ee, t_cam_ee)
        self.cVn_inv = np.linalg.inv(self.cVn)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self._feedback = {"q": None, "qd": None, "tcp": None, "t": 0.0}
        self._fb_lock = threading.Lock()

    def _saturate(self, twist):
        twist = np.asarray(twist, dtype=float).copy()
        v, w = twist[:3], twist[3:]
        nv, nw = np.linalg.norm(v), np.linalg.norm(w)
        if nv > self.v_max and nv > 0:
            v *= self.v_max / nv
        if nw > self.w_max and nw > 0:
            w *= self.w_max / nw
        return np.hstack([v, w])

    def send_camera_twist(self, v_cam):
        """camera-frame twist -> EE-frame twist -> saturate -> execute."""
        v_ee = self.cVn_inv @ np.asarray(v_cam, dtype=float)
        self.send_ee_twist(self._saturate(v_ee))

    def feedback(self) -> dict:
        with self._fb_lock:
            return dict(self._feedback)

    @abstractmethod
    def send_ee_twist(self, v_ee): ...
    @abstractmethod
    def send_joint_velocity(self, qd): ...
    @abstractmethod
    def stop(self): ...

    def close(self):
        try:
            self.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# real UR5e / URSim
# --------------------------------------------------------------------------- #
class RTDEInterface(RobotInterface):
    def __init__(self, ip, R_cam_ee, t_cam_ee,
                 v_max=0.25, w_max=1.0, acceleration=0.25, dt=1.0/125.0,
                 feedback_hz=125.0):
        super().__init__(R_cam_ee, t_cam_ee, v_max, w_max)
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        self.rtde_c = RTDEControlInterface(ip)
        self.rtde_r = RTDEReceiveInterface(ip)
        self.acc = float(acceleration)
        self.dt = float(dt)

        # continuous feedback reader
        self._fb_stop = False
        self._fb_period = 1.0 / feedback_hz
        self._fb_thread = threading.Thread(target=self._fb_spin, daemon=True)
        self._fb_thread.start()

    def _fb_spin(self):
        while not self._fb_stop:
            try:
                q = self.rtde_r.getActualQ()
                qd = self.rtde_r.getActualQd()
                tcp = self.rtde_r.getActualTCPPose()
                with self._fb_lock:
                    self._feedback = {"q": q, "qd": qd, "tcp": tcp,
                                      "t": time.monotonic()}
            except Exception:
                pass
            time.sleep(self._fb_period)

    def send_ee_twist(self, v_ee):
        # speedL(xd, acceleration, time): re-sent every tick; short time = responsive
        self.rtde_c.speedL(list(map(float, v_ee)), self.acc, self.dt)

    def send_joint_velocity(self, qd):
        self.rtde_c.speedJ(list(map(float, qd)), self.acc, self.dt)

    def stop(self):
        try:
            self.rtde_c.speedStop()
        except Exception:
            pass

    def close(self):
        self._fb_stop = True
        super().close()
        for x in ("rtde_c", "rtde_r"):
            try:
                getattr(self, x).disconnect()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# mock for offline validation
# --------------------------------------------------------------------------- #
class MockInterface(RobotInterface):
    def __init__(self, R_cam_ee=None, t_cam_ee=None, v_max=0.25, w_max=1.0):
        super().__init__(np.eye(3) if R_cam_ee is None else R_cam_ee,
                         np.zeros(3) if t_cam_ee is None else t_cam_ee,
                         v_max, w_max)
        self.log = []
        with self._fb_lock:
            self._feedback = {"q": [0]*6, "qd": [0]*6, "tcp": [0]*6,
                              "t": time.monotonic()}

    def send_ee_twist(self, v_ee):
        self.log.append(("ee", np.asarray(v_ee, float)))

    def send_joint_velocity(self, qd):
        self.log.append(("j", np.asarray(qd, float)))

    def stop(self):
        self.log.append(("stop", np.zeros(6)))


if __name__ == "__main__":
    m = MockInterface(v_max=0.25)
    m.send_camera_twist([10, 0, 0, 0, 0, 0])
    kind, cmd = m.log[-1]
    print("clamped:", kind, np.round(cmd, 4))
    assert abs(np.linalg.norm(cmd[:3]) - 0.25) < 1e-9
    print("feedback:", m.feedback()["q"])
    print("robot_interface self-test OK")
