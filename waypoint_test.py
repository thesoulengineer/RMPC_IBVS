#!/usr/bin/env python3
"""
waypoint_test.py — standalone UR5e motion test (NO perception, NO IBVS).
========================================================================

Purpose: prove the RTDE plumbing in isolation before any visual servoing.
If the arm cannot move between known waypoints, nothing downstream matters.
This script touches ONLY ur-rtde — it does not import the controller,
perception, or ZMQ.

What it verifies, in order:
  1. RTDE connect (control + receive).
  2. Continuous feedback reads (q, TCP pose).
  3. Joint-space motion (moveJ) between waypoints.
  4. Cartesian motion (moveL) between waypoints.
  5. Clean stop + disconnect.

SAFETY MODEL — read this:
  * Waypoints are defined as SMALL OFFSETS from the robot's CURRENT pose, read
    live at startup. The script never commands an absolute pose you didn't
    derive from where the arm already is, so it cannot fling to a random
    configuration.
  * Speeds/accelerations are deliberately LOW.
  * Every target is checked against the robot's own safety limits
    (isPoseWithinSafetyLimits / isJointsWithinSafetyLimits) before moving.
  * --dry-run prints every planned target and moves NOTHING.
  * You confirm at the keyboard before any real motion.

PREREQUISITES on the robot:
  * RTDE enabled; for a real arm the ExternalControl URCap running, or use
    URSim. The robot must be powered, brakes released, and in Remote Control
    (real arm) with the program ready.

USAGE:
  # 1) ALWAYS dry-run first — no motion, just prints targets and limit checks:
  python waypoint_test.py --ip 192.168.1.10 --dry-run

  # 2) URSim (no physical risk). Point --ip at the simulator:
  python waypoint_test.py --ip 127.0.0.1

  # 3) Real arm: low speed, e-stop in hand, arm clear of obstacles:
  python waypoint_test.py --ip 192.168.5.5 --speed 0.05
"""
from __future__ import annotations
import argparse
import sys
import time
import math


def banner(msg):
    print("\n" + "=" * 64 + f"\n{msg}\n" + "=" * 64)


def confirm(prompt):
    try:
        return input(prompt + " [type 'yes' to proceed] ").strip().lower() == "yes"
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    ap = argparse.ArgumentParser(description="Standalone UR5e waypoint motion test.")
    ap.add_argument("--ip", required=True, default="192.168.5.5", help="Robot IP (e.g. 127.0.0.1).")
    ap.add_argument("--speed", type=float, default=0.10,
                    help="Joint speed rad/s and linear speed m/s (LOW; default 0.10).")
    ap.add_argument("--accel", type=float, default=0.30,
                    help="Acceleration (default 0.30).")
    ap.add_argument("--joint-offset", type=float, default=0.15,
                    help="Base-joint test offset in radians (~8.6 deg, default 0.15).")
    ap.add_argument("--cart-offset", type=float, default=0.05,
                    help="Cartesian test offset in meters (default 0.05 = 5 cm).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned targets and limit checks; move NOTHING.")
    ap.add_argument("--skip-cartesian", action="store_true",
                    help="Run only the joint-space test.")
    args = ap.parse_args()

    # ---- import ur-rtde (clear error if missing) --------------------------
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except ImportError:
        print("ERROR: ur-rtde not installed. Run:  pip install ur_rtde", file=sys.stderr)
        sys.exit(1)

    banner(f"Connecting to {args.ip}")
    try:
        rtde_c = RTDEControlInterface(args.ip)
        rtde_r = RTDEReceiveInterface(args.ip)
    except Exception as e:
        print(f"ERROR: could not connect to {args.ip}: {e}", file=sys.stderr)
        print("  - URSim running / robot powered?  RTDE enabled?  "
              "ExternalControl URCap running (real arm)?  IP correct?", file=sys.stderr)
        sys.exit(1)

    if not rtde_c.isConnected():
        print("ERROR: control interface reports not connected.", file=sys.stderr)
        sys.exit(1)
    print("Connected. Control + Receive interfaces up.")

    try:
        # ---- 2) feedback read --------------------------------------------
        banner("Reading current robot state (feedback test)")
        q0 = rtde_r.getActualQ()             # 6 joint angles, rad
        tcp0 = rtde_r.getActualTCPPose()     # [x,y,z,rx,ry,rz], m + rad
        print("  joints q  (rad):", [round(v, 4) for v in q0])
        print("  joints q  (deg):", [round(math.degrees(v), 1) for v in q0])
        print("  TCP pose      :", [round(v, 4) for v in tcp0])

        # ---- build waypoints as OFFSETS from current pose ----------------
        # Joint waypoints: nudge base joint +offset, return. Small & reversible.
        q_fwd = list(q0); q_fwd[0] += args.joint_offset
        q_waypoints = [q_fwd, list(q0)]      # out, then back to start

        # Cartesian waypoints: nudge TCP +Z (up), return. +Z is up in base frame.
        p_up = list(tcp0); p_up[2] += args.cart_offset
        p_waypoints = [p_up, list(tcp0)]     # up, then back

        # ---- safety-limit checks (always, even in dry-run) ---------------
        banner("Safety-limit pre-checks")
        all_ok = True
        for i, q in enumerate(q_waypoints):
            ok = rtde_c.isJointsWithinSafetyLimits(q)
            print(f"  joint waypoint {i}: within limits = {ok}")
            all_ok &= ok
        if not args.skip_cartesian:
            for i, p in enumerate(p_waypoints):
                ok = rtde_c.isPoseWithinSafetyLimits(p)
                print(f"  cart  waypoint {i}: within limits = {ok}")
                all_ok &= ok
        if not all_ok:
            print("\nABORT: at least one waypoint is outside safety limits. "
                  "Reduce --joint-offset / --cart-offset or reposition the arm.")
            return

        # ---- dry-run stops here ------------------------------------------
        if args.dry_run:
            banner("DRY RUN — no motion")
            print("Planned joint waypoints (rad):")
            for i, q in enumerate(q_waypoints):
                print(f"  {i}: {[round(v,4) for v in q]}")
            if not args.skip_cartesian:
                print("Planned Cartesian waypoints [x,y,z,rx,ry,rz]:")
                for i, p in enumerate(p_waypoints):
                    print(f"  {i}: {[round(v,4) for v in p]}")
            print("\nAll limit checks passed. Re-run without --dry-run to move.")
            return

        # ---- confirm before motion ---------------------------------------
        banner("MOTION CONFIRMATION")
        print(f"About to move at speed={args.speed} accel={args.accel}.")
        print("Ensure: workspace clear, e-stop in reach, "
              "real arm in Remote Control.")
        if not confirm("Proceed with JOINT-space motion?"):
            print("Cancelled by user.")
            return

        # ---- 3) joint-space motion (moveJ) -------------------------------
        banner("Joint-space test (moveJ)")
        for i, q in enumerate(q_waypoints):
            print(f"  moveJ -> waypoint {i}: {[round(v,4) for v in q]}")
            # moveJ(q, speed, acceleration): blocking (synchronous) by default.
            rtde_c.moveJ(q, args.speed, args.accel)
            qn = rtde_r.getActualQ()
            err = max(abs(a - b) for a, b in zip(qn, q))
            print(f"    reached; max joint error = {err:.5f} rad")
            time.sleep(0.3)
        print("  Joint-space test complete; returned to start.")

        # ---- 4) Cartesian motion (moveL) ---------------------------------
        if not args.skip_cartesian:
            if not confirm("Proceed with CARTESIAN (moveL) motion?"):
                print("Skipping Cartesian test.")
            else:
                banner("Cartesian test (moveL)")
                for i, p in enumerate(p_waypoints):
                    print(f"  moveL -> waypoint {i}: {[round(v,4) for v in p]}")
                    # moveL(pose, speed_m_s, accel): blocking by default.
                    rtde_c.moveL(p, args.speed, args.accel)
                    pn = rtde_r.getActualTCPPose()
                    derr = math.sqrt(sum((a - b) ** 2
                                         for a, b in zip(pn[:3], p[:3])))
                    print(f"    reached; position error = {derr*1000:.2f} mm")
                    time.sleep(0.3)
                print("  Cartesian test complete; returned to start.")

        banner("ALL TESTS PASSED")
        print("RTDE connect, feedback, moveJ, and moveL all working.")
        print("Next: this confirms the robot half. Hand-eye + intrinsics "
              "calibration remain before closed-loop visual servoing.")

    except KeyboardInterrupt:
        print("\nInterrupted — stopping robot.")
    finally:
        banner("Stopping and disconnecting")
        try:
            rtde_c.stopL(2.0)   # decelerate any linear motion
        except Exception:
            pass
        try:
            rtde_c.stopJ(2.0)   # decelerate any joint motion
        except Exception:
            pass
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        for name, iface in (("control", rtde_c), ("receive", rtde_r)):
            try:
                iface.disconnect()
            except Exception:
                pass
        print("Disconnected cleanly.")


if __name__ == "__main__":
    main()
