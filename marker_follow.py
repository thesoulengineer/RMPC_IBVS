#!/usr/bin/env python3
"""
marker_follow.py — 2-DOF poll-and-follow ArUco tracker (discrete moveL).
========================================================================

A simple, safe alternative to continuous IBVS:

  1. On startup, move the UR5e once to a fixed HOME pose (test.py-style moveL).
  2. Every PERIOD_S seconds, read the marker's metric position from perception.
  3. If the marker moved more than DEADBAND_M since the last check, make ONE
     2-DOF moveL in base X/Y to re-center on it. Otherwise, don't move.
  4. Run until Ctrl-C.

This is a *position* tracker, not a velocity servo: it does NOT use the IBVS
controller / ControlLoop. It reuses the RTDE moveL idiom from test.py and the
existing perception transport (StreamSubscriber + the RealSense perception node).

Run perception first (separate terminal), then this:
    python perception.py --source realsense --marker-size 0.05 --marker-id 0 \
        --publish zmq --no-show
    python marker_follow.py --ip 192.168.5.5
    python marker_follow.py --dry-run          # no robot; validate signs/deadband

cam->base axis signs are NOT calibrated. Validate with --dry-run, then tune
CAM_TO_BASE on hardware at low speed (see the README / plan verification steps).
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np

from stream_subscriber import StreamSubscriber

# --- tunable configuration -------------------------------------------------- #
HOME_XYZ = (0.40, 0.20, 0.50)     # base-frame metres (test.py center + height)
PERIOD_S = 1.0                    # seconds between marker checks
DEADBAND_M = 0.01                 # ignore marker moves smaller than this (~1 cm)
MOVE_V = 0.1                     # moveL tool speed [m/s]
MOVE_A = 0.30                     # moveL tool acceleration [m/s^2]
FRESH_S = 0.20                    # only trust a perception frame younger than this

# Camera image-plane (x=right, y=down) -> base (X, Y). Signs/axes are NOT
# hand-eye calibrated; tune these after --dry-run. Identity = camera-x maps to
# base-X, camera-y maps to base-Y, both same sign.
CAM_TO_BASE = np.array([[-1.0, 0.0],
                        [0.0, 1.0]])


def read_marker_xy(sub, samples=5, gap=0.03):
    """Median camera-frame (mx, my) in metres over a few fresh frames, or None.

    Requires a found marker WITH metric depth (perception run with intrinsics).
    """
    xs, ys = [], []
    for _ in range(samples):
        st, age = sub.latest()
        if st and st.found and st.has_depth() and age < FRESH_S:
            xs.append(float(st.xyz[0]))
            ys.append(float(st.xyz[1]))
        time.sleep(gap)
    if not xs:
        return None
    return (float(np.median(xs)), float(np.median(ys)))


def main():
    ap = argparse.ArgumentParser(description="2-DOF poll-and-follow marker tracker.")
    ap.add_argument("--ip", default="192.168.5.5", help="UR robot IP.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5556",
                    help="perception ZMQ endpoint.")
    ap.add_argument("--topic", default="perception", help="perception ZMQ topic.")
    ap.add_argument("--period", type=float, default=PERIOD_S)
    ap.add_argument("--deadband", type=float, default=DEADBAND_M)
    ap.add_argument("--vel", type=float, default=MOVE_V)
    ap.add_argument("--acc", type=float, default=MOVE_A)
    ap.add_argument("--home-x", type=float, default=HOME_XYZ[0])
    ap.add_argument("--home-y", type=float, default=HOME_XYZ[1])
    ap.add_argument("--home-z", type=float, default=HOME_XYZ[2])
    ap.add_argument("--dry-run", action="store_true",
                    help="no robot: just read the marker and print intended moves.")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="stop after N check cycles (default: run until Ctrl-C).")
    args = ap.parse_args()

    sub = StreamSubscriber(args.endpoint, args.topic.encode())
    print(f"[follow] subscribed to {args.endpoint} topic={args.topic!r}")

    rtde_c = rtde_r = None
    if not args.dry_run:
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError:
            print("ERROR: ur-rtde not installed. Use --dry-run, or install ur_rtde.",
                  file=sys.stderr)
            sub.close()
            sys.exit(1)
        rtde_c = RTDEControlInterface(args.ip)
        rtde_r = RTDEReceiveInterface(args.ip)

        # capture current orientation so the tool keeps pointing the same way
        cur = rtde_r.getActualTCPPose()
        rx, ry, rz = cur[3], cur[4], cur[5]
        home = [args.home_x, args.home_y, args.home_z, rx, ry, rz]
        print(f"[home] moving to {np.round(home, 4).tolist()} ...")
        rtde_c.moveL(home, args.vel, args.acc)
        print("[home] at home pose.")
    else:
        print("[dry-run] no robot connection; printing intended moves only.")

    prev = None
    cycles = 0
    try:
        while args.max_cycles is None or cycles < args.max_cycles:
            cycles += 1
            time.sleep(args.period)
            m = read_marker_xy(sub)
            if m is None:
                print("[check] marker not visible / no depth -> hold.")
                continue
            mx, my = m
            if prev is None:
                prev = m
                print(f"[check] baseline marker cam-xy = ({mx:+.3f}, {my:+.3f}) m")
                continue

            moved = float(np.hypot(mx - prev[0], my - prev[1]))
            dX, dY = (CAM_TO_BASE @ np.array([mx, my])).tolist()
            if moved > args.deadband:
                print(f"[move] marker cam-xy=({mx:+.3f},{my:+.3f}) moved {moved*100:.1f}cm "
                      f"-> base dX={dX:+.3f} dY={dY:+.3f} m")
                if not args.dry_run:
                    pose = rtde_r.getActualTCPPose()
                    pose[0] += dX
                    pose[1] += dY                      # 2-DOF: only base X,Y change
                    rtde_c.moveL(pose, args.vel, args.acc)
                    prev = read_marker_xy(sub) or m    # refresh baseline (now ~centered)
                else:
                    prev = m
            else:
                print(f"[hold] marker cam-xy=({mx:+.3f},{my:+.3f}) moved {moved*100:.1f}cm "
                      f"< deadband -> no move.")
                prev = m
    except KeyboardInterrupt:
        print("\n[stop] interrupted.")
    finally:
        if rtde_c is not None:
            try:
                rtde_c.speedStop()
            except Exception:
                pass
            try:
                rtde_c.disconnect()
            except Exception:
                pass
        if rtde_r is not None:
            try:
                rtde_r.disconnect()
            except Exception:
                pass
        sub.close()
        print("[stop] disconnected.")


if __name__ == "__main__":
    main()
