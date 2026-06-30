#!/usr/bin/env python3
"""
realsense_setup.py — RealSense -> cam.json (factory intrinsics, no calibration).
================================================================================

perception.py computes the marker's 3D pose by PnP from the marker size + camera
intrinsics. A RealSense already knows its own intrinsics (factory-calibrated), so
instead of running a checkerboard calibration you can just dump them here.

What it does:
  1. opens the RealSense COLOR stream at the requested resolution (via pyrealsense2),
  2. reads that stream's factory intrinsics (fx, fy, cx, cy, distortion),
  3. writes them to cam.json in the format Intrinsics.load() expects.

IMPORTANT: intrinsics are resolution-specific. Whatever --width/--height you dump
here, you MUST capture at the SAME size in perception.py (--width/--height), or the
pose/depth will be wrong.

Usage:
    python realsense_setup.py --list                     # show connected devices
    python realsense_setup.py                             # 640x480 -> cam.json
    python realsense_setup.py --width 1280 --height 720 --out cam.json
    python realsense_setup.py --probe-cv2                 # find the cv2 source index

Then run perception with a MATCHING resolution, e.g.:
    python perception.py --source 1 --width 640 --height 480 \
        --marker-size 0.05 --intrinsics cam.json --marker-id 0 --kalman \
        --publish zmq --no-show
"""
import argparse
import json
import sys


def list_devices():
    import pyrealsense2 as rs
    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
        print("No RealSense devices found. Is it plugged in (USB3) and detected?")
        return
    for i, d in enumerate(devs):
        name = d.get_info(rs.camera_info.name)
        serial = d.get_info(rs.camera_info.serial_number)
        fw = d.get_info(rs.camera_info.firmware_version)
        print(f"  [{i}] {name}  serial={serial}  fw={fw}")


def probe_cv2(max_index=6):
    """Open cv2 indices to help identify which one is the RealSense RGB."""
    import cv2
    print("Probing cv2.VideoCapture indices (use the one with a color image):")
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                ch = 1 if frame.ndim == 2 else frame.shape[2]
                print(f"  index {idx}: OPEN  {w}x{h}  channels={ch}")
            else:
                print(f"  index {idx}: opened but no frame")
            cap.release()
        else:
            print(f"  index {idx}: --")


def dump_intrinsics(width, height, fps, out_path):
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        print(f"ERROR starting color stream at {width}x{height}@{fps}: {e}", file=sys.stderr)
        print("Try a supported color mode, e.g. 640x480, 1280x720, or 1920x1080.",
              file=sys.stderr)
        sys.exit(1)
    try:
        vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vsp.get_intrinsics()
    finally:
        pipe.stop()

    cam = {
        "fx": intr.fx, "fy": intr.fy,
        "cx": intr.ppx, "cy": intr.ppy,
        "dist": [float(c) for c in intr.coeffs],   # [k1,k2,p1,p2,k3]
        # --- reference only (Intrinsics.load ignores these) ---
        "width": intr.width, "height": intr.height,
        "distortion_model": str(intr.model),
        "source": "realsense_factory",
    }
    with open(out_path, "w") as f:
        json.dump(cam, f, indent=2)

    print(f"Wrote {out_path} from RealSense factory intrinsics:")
    print(f"  resolution : {intr.width}x{intr.height}")
    print(f"  fx,fy      : {intr.fx:.2f}, {intr.fy:.2f}")
    print(f"  cx,cy      : {intr.ppx:.2f}, {intr.ppy:.2f}")
    print(f"  dist model : {intr.model}  coeffs={[round(c,5) for c in intr.coeffs]}")
    print()
    print("Now run perception at the SAME resolution, e.g.:")
    print(f"  python perception.py --source <cv2_index> --width {intr.width} "
          f"--height {intr.height} \\")
    print(f"      --marker-size 0.05 --intrinsics {out_path} --marker-id 0 "
          f"--kalman --publish zmq --no-show")


def main():
    ap = argparse.ArgumentParser(description="Dump RealSense factory intrinsics to cam.json.")
    ap.add_argument("--list", action="store_true", help="List connected RealSense devices and exit.")
    ap.add_argument("--probe-cv2", action="store_true",
                    help="Probe cv2 indices to find the RealSense RGB source, then exit.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="cam.json")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return
    if args.probe_cv2:
        probe_cv2()
        return
    dump_intrinsics(args.width, args.height, args.fps, args.out)


if __name__ == "__main__":
    main()
