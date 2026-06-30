#!/usr/bin/env python3
"""
pose_tap.py — observe what perception actually sends to the controller.
=======================================================================

Subscribes to the SAME ZMQ stream that main.py / ControlLoop consume
(via StreamSubscriber) and prints, per received frame:

    found, frame_id, corners?, 3D pose (xyz)?, calibrated?, velocity?, arrival age

Then prints a summary that tells you whether poses are flowing or the
controller is falling back to a guessed depth.

Usage (perception in one terminal, this in another):

    # perception WITH intrinsics -> real metric pose/depth:
    python perception.py --source 0 --marker-size 0.05 \
        --intrinsics cam.json --marker-id 0 --kalman --publish zmq --no-show

    python pose_tap.py
    python pose_tap.py --endpoint tcp://127.0.0.1:5556 --topic perception --seconds 15
"""
import argparse
import time

from stream_subscriber import StreamSubscriber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    ap.add_argument("--topic", default="perception")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    sub = StreamSubscriber(args.endpoint, args.topic.encode())
    print(f"[tap] subscribed to {args.endpoint} topic={args.topic!r}; "
          f"watching for {args.seconds:.0f}s (Ctrl-C to stop early)...")

    seen = with_corners = with_pose = with_cal = 0
    last_id = None
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            st, age = sub.latest()
            if st is not None and st.frame_id != last_id:
                last_id = st.frame_id
                seen += 1
                has_c = st.corners is not None
                has_p = st.xyz is not None
                with_corners += int(has_c)
                with_pose += int(has_p)
                with_cal += int(st.calibrated)

                xyz = ("[%+.3f %+.3f %+.3f]" % tuple(st.xyz)) if has_p else "None"
                depth = f"{st.depth(float('nan')):.3f}" if st.has_depth() else "FALLBACK"
                print(f"  frame={st.frame_id:<6} found={str(st.found):<5} "
                      f"corners={'Y' if has_c else 'n'} "
                      f"xyz={xyz:<24} cal={str(st.calibrated):<5} "
                      f"depth={depth:<9} age={age*1000:5.0f}ms")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()

    print("\n[tap] summary")
    print(f"  unique frames received : {seen}")
    print(f"  with 2D corners        : {with_corners}")
    print(f"  with 3D pose (xyz)     : {with_pose}")
    print(f"  flagged calibrated     : {with_cal}")
    if seen == 0:
        print("  >> NOTHING received. Perception is not publishing to this "
              "endpoint/topic.\n     Check: is perception running with "
              "--publish zmq, same --zmq-endpoint and --zmq-topic?")
    elif with_pose == 0 or with_cal == 0:
        print("  >> Frames arrive but carry NO usable 3D pose -> the controller "
              "uses the FALLBACK depth (0.35 m).\n     Fix: run perception WITH "
              "--intrinsics cam.json AND --marker-size <meters>.")
    else:
        print("  >> Poses are flowing. The controller is receiving metric pose/depth.")


if __name__ == "__main__":
    main()
