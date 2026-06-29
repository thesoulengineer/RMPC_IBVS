#!/usr/bin/env python3
"""
main.py — IBVS pipeline orchestrator.
=====================================

    [perception.py]  --ZMQ PUB-->  [StreamSubscriber]
                                          |
                                   [ControlLoop @ fixed Hz]
                                    /            \\
                      [VisualServoController]   [FeaturePredictor]
                                          |
                                   v_cam (6-twist)
                                          |
                                  [RobotInterface] --RTDE speedL--> [UR5e]

Each layer has ONE job:
  * StreamSubscriber       : receive target state (no control, no robot)
  * VisualServoController  : compute v = -lambda L^+ e (no I/O, no robot)
  * FeaturePredictor       : extrapolate features on dropped frames (pure math)
  * ControlLoop            : deterministic fixed-rate scheduler + state machine
  * RobotInterface         : execute commands, read feedback (no control law)

Run perception first (separate process, headless, ZMQ):

    python perception.py --source 0 --marker-size 0.05 \\
        --intrinsics cam.json --marker-id 0 --kalman \\
        --publish zmq --no-show

Then:

    python main.py                 # MockInterface (no robot)
    python main.py --robot rtde --ip 192.168.1.10   # real UR5e / URSim

cam.json intrinsics MUST equal K below, or normalized coords / depth disagree.
"""
from __future__ import annotations
import argparse
import numpy as np

from stream_subscriber import StreamSubscriber, FileSubscriber
from controller import VisualServoController
from predictor import FeaturePredictor
from control_loop import ControlLoop
from robot_interface import RTDEInterface, MockInterface


# --- configuration ---------------------------------------------------------
# Camera intrinsics — MUST match perception.py --intrinsics cam.json.
K = np.array([[600.0,   0.0, 320.0],
              [  0.0, 600.0, 240.0],
              [  0.0,   0.0,   1.0]])

# Eye-in-hand calibration (EE -> camera). Replace with YOUR calibration.
R_CAM_EE = np.eye(3)
T_CAM_EE = np.zeros(3)


def build_robot(kind, ip):
    if kind == "mock":
        return MockInterface(R_CAM_EE, T_CAM_EE, v_max=0.25, w_max=1.0)

    # --- real UR5e / URSim via RTDE ---------------------------------------
    from robot_interface import RTDEInterface
    return RTDEInterface(
        ip=ip, R_cam_ee=R_CAM_EE, t_cam_ee=T_CAM_EE,
        v_max=0.1, w_max=1.0,
        acceleration=0.1, dt=1.0 / 100.0, feedback_hz=100.0,
    )
    raise SystemExit("RTDE backend is commented out; uncomment build_robot() "
                     "and install ur-rtde to drive a real UR5e.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["zmq", "stdin"], default="zmq")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    ap.add_argument("--topic", default="perception")
    ap.add_argument("--robot", choices=["mock", "rtde"], default="mock")
    ap.add_argument("--ip", default="192.168.5.11")
    ap.add_argument("--rate", type=float, default=100.0, help="control Hz")
    ap.add_argument("--fresh-ms", type=float, default=50.0)
    ap.add_argument("--lost-ms", type=float, default=400.0)
    ap.add_argument("--conv-tol", type=float, default=0.02)
    ap.add_argument("--max-ticks", type=int, default=None)
    args = ap.parse_args()

    # transport (perception -> controller)
    if args.transport == "zmq":
        sub = StreamSubscriber(args.endpoint, args.topic.encode())
    else:
        sub = FileSubscriber()

    controller = VisualServoController(
        linear_velocities="xyz", angular_velocities="y",
        interaction_mode="mean", num_pts=4, lam=0.5)
    predictor = FeaturePredictor(K)
    robot = build_robot("mock" if args.robot == "mock" else "rtde", args.ip)

    loop = ControlLoop(
        sub, controller, robot, predictor, K,
        rate_hz=args.rate,
        fresh_dt=args.fresh_ms / 1000.0,
        lost_dt=args.lost_ms / 1000.0,
        conv_tol=args.conv_tol)

    try:
        print("[teach] hold the marker at the desired pose...")
        ok, Zstar = loop.teach_goal(timeout_s=30.0)
        if not ok:
            print("[teach] no valid target frame; is perception publishing?")
            return
        print(f"[teach] goal captured (Z*={Zstar:.3f} m)")

        print(f"[servo] running at {args.rate:.0f} Hz. Ctrl-C to stop.")
        
        # input("[teach] goal set. MOVE the marker, then press Enter to start servoing...")
        result = loop.run(max_ticks=args.max_ticks)
        print(f"[servo] stopped: {result}")
        print(f"[stats] {loop.stats}")
    finally:
        robot.close()
        sub.close()
        if isinstance(robot, MockInterface) and robot.log:
            kind, cmd = robot.log[-1]
            print(f"[mock] {len(robot.log)} commands; last={kind} {np.round(cmd,4)}")


if __name__ == "__main__":
    main()