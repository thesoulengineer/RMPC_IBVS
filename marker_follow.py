#!/usr/bin/env python3
"""
marker_follow.py — 2-DOF ArUco tracker (discrete moveL + optional continuous speedL).
=====================================================================================

A simple, safe 2-DOF "center the marker" tracker:

  1. On startup, move the UR5e once to a fixed HOME pose (test.py-style moveL).
  2. Every PERIOD_S seconds, read the marker's metric offset from the image center.
  3. While that offset exceeds CENTER_TOL_M, make a 2-DOF moveL in base X/Y that
     steps GAIN x the offset toward centering it (discrete, IBVS-like proportional
     control); inside the tolerance it holds, and re-chases when the marker moves.
  4. Run until Ctrl-C.

Two modes: the discrete moveL stepper above is the default. With --continuous it
instead streams speedL TCP velocity (v = VEL_GAIN x offset, capped at MOVE_V) at
RATE_HZ for smooth, continuous 2-DOF velocity servoing — the continuous IBVS
centering law, standalone.

It reuses the RTDE idiom from test.py / robot_interface.py and the existing perception
transport (StreamSubscriber + the RealSense perception node); it does NOT use the IBVS
ControlLoop/controller.

Run perception first (separate terminal), then this:
    python perception.py --source realsense --marker-size 0.05 --marker-id 0 \
        --publish zmq --no-show
    python marker_follow.py --ip 192.168.5.5
    python marker_follow.py --dry-run          # no robot; validate signs/gain/tol

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
PERIOD_S = 0.001                    # seconds between marker checks
CENTER_TOL_M = 0.01               # "centered enough" stop band (~1 cm); no move inside it
GAIN = 0.5                        # lambda: fraction of the offset moved per step (<1 = IBVS-like)
RATE_HZ = 125.0                   # continuous-mode loop frequency [Hz]
VEL_GAIN = 1.5                    # continuous-mode velocity gain [1/s]: v = VEL_GAIN * offset
MOVE_V = 0.1                     # moveL tool speed [m/s]; also the continuous max-speed cap
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


def run_discrete(sub, rtde_c, rtde_r, args):
    """Default mode: every --period s, step GAIN x offset toward center via moveL."""
    cycles = 0
    while args.max_cycles is None or cycles < args.max_cycles:
        cycles += 1
        time.sleep(args.period)
        m = read_marker_xy(sub)
        if m is None:
            print("[check] marker not visible / no depth -> hold.")
            continue
        mx, my = m
        offset = float(np.hypot(mx, my))           # distance from image center = the error
        dX, dY = (CAM_TO_BASE @ np.array([args.gain * mx, args.gain * my])).tolist()
        if offset > args.tol:                      # off-center -> step toward center
            print(f"[move] off-center {offset*100:.1f}cm "
                  f"(cam-xy={mx:+.3f},{my:+.3f}) -> step base dX={dX:+.3f} dY={dY:+.3f} m")
            if not args.dry_run:
                pose = rtde_r.getActualTCPPose()
                pose[0] += dX
                pose[1] += dY                      # 2-DOF: only base X,Y change
                rtde_c.moveL(pose, args.vel, args.acc)
        else:
            print(f"[hold] centered ({offset*100:.1f}cm < {args.tol*100:.1f}cm) -> no move.")


def run_continuous(sub, rtde_c, rtde_r, args):
    """--continuous mode: stream speedL velocity (v = vel_gain x offset) at --rate Hz."""
    dt = 1.0 / args.rate
    print_every = max(1, int(args.rate / 5.0))     # throttle prints to ~5 Hz
    moving = False
    cycles = 0
    while args.max_cycles is None or cycles < args.max_cycles:
        cycles += 1
        t0 = time.monotonic()
        m = read_marker_xy(sub, samples=1, gap=0.0)   # freshest frame, no median sleeps
        if m is None:
            if moving and not args.dry_run:
                rtde_c.speedStop()
            moving = False
            if cycles % print_every == 0:
                print("[lost] marker not visible -> stop.")
        else:
            mx, my = m
            offset = float(np.hypot(mx, my))
            if offset > args.tol:                  # off-center -> velocity toward center
                vx, vy = (CAM_TO_BASE @ np.array([args.vel_gain * mx,
                                                  args.vel_gain * my])).tolist()
                speed = float(np.hypot(vx, vy))
                if speed > args.vel and speed > 0:        # cap to max speed
                    vx *= args.vel / speed
                    vy *= args.vel / speed
                if not args.dry_run:
                    rtde_c.speedL([vx, vy, 0.0, 0.0, 0.0, 0.0], args.acc, dt)
                moving = True
                if cycles % print_every == 0:
                    print(f"[servo] off-center {offset*100:.1f}cm -> "
                          f"v=({vx:+.3f},{vy:+.3f}) m/s")
            else:                                  # centered -> hold
                if moving and not args.dry_run:
                    rtde_c.speedStop()
                moving = False
                if cycles % print_every == 0:
                    print(f"[hold] centered ({offset*100:.1f}cm) -> stop.")
        rem = dt - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)


def main():
    ap = argparse.ArgumentParser(description="2-DOF poll-and-follow marker tracker.")
    ap.add_argument("--ip", default="192.168.5.5", help="UR robot IP.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5556",
                    help="perception ZMQ endpoint.")
    ap.add_argument("--topic", default="perception", help="perception ZMQ topic.")
    ap.add_argument("--period", type=float, default=PERIOD_S)
    ap.add_argument("--tol", type=float, default=CENTER_TOL_M,
                    help="centered-enough stop band [m]; no move when offset < tol.")
    ap.add_argument("--gain", type=float, default=GAIN,
                    help="fraction of the offset moved per step (the lambda).")
    ap.add_argument("--vel", type=float, default=MOVE_V,
                    help="moveL speed [discrete] / max speed cap [continuous] [m/s].")
    ap.add_argument("--acc", type=float, default=MOVE_A)
    ap.add_argument("--continuous", action="store_true",
                    help="continuous speedL velocity servo instead of discrete moveL steps.")
    ap.add_argument("--rate", type=float, default=RATE_HZ,
                    help="continuous-mode control loop frequency [Hz].")
    ap.add_argument("--vel-gain", type=float, default=VEL_GAIN,
                    help="continuous-mode velocity gain [1/s]: v = vel_gain * offset.")
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

    try:
        if args.continuous:
            print(f"[mode] CONTINUOUS speedL servo @ {args.rate:.0f} Hz "
                  f"(vel_gain={args.vel_gain}, v_max={args.vel} m/s).")
            run_continuous(sub, rtde_c, rtde_r, args)
        else:
            print(f"[mode] DISCRETE moveL stepper (period={args.period}s, gain={args.gain}).")
            run_discrete(sub, rtde_c, rtde_r, args)
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
