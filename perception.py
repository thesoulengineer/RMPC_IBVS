#!/usr/bin/env python3
"""
Eye-in-Hand ArUco Perception for UR5e Tracking of a Moving Box
==============================================================

The target box carries an ArUco marker. The marker gives us, per frame:
    - a STABLE id (used directly as track_id -> no separate tracker needed)
    - the marker center in pixels
    - full 6-DoF pose (X, Y, Z + orientation) in the camera frame, IF the
      camera is calibrated (solvePnP with known marker size). This is the
      metric depth a plain webcam otherwise cannot provide.

SCOPE (perception only): detect, locate, (optionally) estimate pose, and
publish. No pixel-error / control commands; the control team consumes output.

WHY NO TRACKER: ArUco re-detects every frame and is fast + robust. The marker
id is constant, so a moving box is tracked by identity automatically. An
OPTIONAL constant-velocity Kalman filter (--kalman) smooths the position and
adds a velocity estimate, which helps control anticipate a moving target.

INSTALL (cv2.aruco requires the contrib build):
    pip uninstall opencv-python -y          # avoid conflicting installs
    pip install opencv-contrib-python numpy
    # optional socket output:
    pip install pyzmq

RUN:
    # 2D only (no calibration): center + stable id, still tracks a moving box
    python perception.py --source 0 --marker-size 0.05 --show

    # full 3D pose (recommended): provide intrinsics from calibration
    python perception.py --source 0 --marker-size 0.05 \
        --intrinsics cam.json --kalman --show

OUTPUT CONTRACT (one JSON object per frame):
    {
      "timestamp": float, "frame_id": int,
      "target_found": bool,
      "marker_id": int | null,          # also the track_id (stable)
      "center": [u, v] | null,          # pixels
      "corners": [[x,y]*4] | null,      # pixels
      "pose": {"x":..,"y":..,"z":..} | null,     # meters, camera frame
      "rvec": [rx,ry,rz] | null,        # Rodrigues rotation, camera frame
      "velocity": {"vx":..,"vy":..,"vz":..} | null,  # only with --kalman
      "image_size": [W, H],
      "intrinsics_calibrated": bool,
      "dict": str
    }
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    stream=sys.stderr)
log = logging.getLogger("perception_aruco")


# --------------------------------------------------------------------------- #
# Intrinsics (+ distortion, needed for accurate pose)
# --------------------------------------------------------------------------- #
@dataclass
class Intrinsics:
    K: Optional[np.ndarray]          # 3x3 or None
    dist: np.ndarray                 # distortion coeffs (zeros if unknown)
    cx: float
    cy: float
    calibrated: bool

    @staticmethod
    def default(width, height):
        return Intrinsics(K=None, dist=np.zeros(5), cx=width / 2.0,
                          cy=height / 2.0, calibrated=False)

    @staticmethod
    def load(path, width, height):
        with open(path) as f:
            d = json.load(f)
        if "K" in d:
            K = np.array(d["K"], dtype=float)
        else:
            K = np.array([[d["fx"], 0, d["cx"]],
                          [0, d["fy"], d["cy"]],
                          [0, 0, 1]], dtype=float)
        dist = np.array(d.get("dist", [0, 0, 0, 0, 0]), dtype=float)
        return Intrinsics(K=K, dist=dist, cx=float(K[0, 2]), cy=float(K[1, 2]),
                          calibrated=True)


# --------------------------------------------------------------------------- #
# Per-frame result
# --------------------------------------------------------------------------- #
@dataclass
class PerceptionResult:
    timestamp: float
    frame_id: int
    target_found: bool
    image_size: List[int]
    intrinsics_calibrated: bool
    dict: str
    marker_id: Optional[int] = None
    center: Optional[List[float]] = None
    corners: Optional[List[List[float]]] = None
    pose: Optional[dict] = None
    rvec: Optional[List[float]] = None
    velocity: Optional[dict] = None

    def to_json(self):
        return json.dumps(asdict(self))


# --------------------------------------------------------------------------- #
# Publishers
# --------------------------------------------------------------------------- #
class StdoutPublisher:
    def publish(self, r): sys.stdout.write(r.to_json() + "\n"); sys.stdout.flush()
    def close(self): pass


class ZmqPublisher:
    def __init__(self, endpoint="tcp://127.0.0.1:5556", topic="perception"):
        import zmq
        self.topic = topic.encode()
        self.ctx = zmq.Context(); self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(endpoint); log.info("ZMQ PUB bound at %s", endpoint)
        time.sleep(0.2)
    def publish(self, r): self.sock.send_multipart([self.topic, r.to_json().encode()])
    def close(self): self.sock.close(linger=0); self.ctx.term()


# --------------------------------------------------------------------------- #
# ArUco detector (handles new >=4.7 and old <4.7 OpenCV APIs)
# --------------------------------------------------------------------------- #
DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "7X7_50": cv2.aruco.DICT_7X7_50,
    "APRILTAG_36h11": getattr(cv2.aruco, "DICT_APRILTAG_36h11", cv2.aruco.DICT_6X6_250),
}


class ArucoDetector:
    def __init__(self, dict_name="4X4_50"):
        if dict_name not in DICT_MAP:
            raise ValueError(f"Unknown dict '{dict_name}'. Options: {list(DICT_MAP)}")
        self.dict_name = dict_name
        dict_id = DICT_MAP[dict_name]

        if hasattr(cv2.aruco, "ArucoDetector"):           # OpenCV >= 4.7
            self._new = True
            self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
        else:                                              # OpenCV < 4.7
            self._new = False
            self._dictionary = cv2.aruco.Dictionary_get(dict_id)
            self._params = cv2.aruco.DetectorParameters_create()

    def detect(self, gray):
        if self._new:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._params)
        return corners, ids


# --------------------------------------------------------------------------- #
# Pose via solvePnP (works across versions; estimatePoseSingleMarkers is gone
# in some builds)
# --------------------------------------------------------------------------- #
def marker_object_points(size):
    h = size / 2.0
    # Order matches ArUco corner order: TL, TR, BR, BL
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                    dtype=np.float32)


def estimate_pose(corners, size, K, dist):
    objp = marker_object_points(size)
    imgp = corners.reshape(-1, 2).astype(np.float32)
    # IPPE_SQUARE is purpose-built for square markers but can occasionally
    # return non-finite values; validate and fall back to the iterative solver.
    for flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=flag)
        if ok:
            r, t = rvec.reshape(-1), tvec.reshape(-1)
            if np.all(np.isfinite(r)) and np.all(np.isfinite(t)):
                return r, t
    return None, None


# --------------------------------------------------------------------------- #
# Optional constant-velocity Kalman filter (helps with a MOVING target)
# --------------------------------------------------------------------------- #
class CVKalman:
    """Tracks a 3-D point with constant-velocity model. State: [x,y,z,vx,vy,vz]."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        self.initialized = False
        self.last_t = None

    def _set_dt(self, dt):
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.kf.transitionMatrix = F

    def update(self, meas, t):
        meas = np.array(meas, dtype=np.float32).reshape(3, 1)
        # Reject non-finite measurements: coast on prediction instead of
        # letting a single nan poison the filter state permanently.
        if not np.all(np.isfinite(meas)):
            if self.initialized:
                dt = max(1e-3, t - self.last_t); self.last_t = t
                self._set_dt(dt)
                p = self.kf.predict()
                return p[:3].reshape(-1), p[3:].reshape(-1)
            return None, None
        if not self.initialized:
            self.kf.statePost = np.vstack([meas, np.zeros((3, 1), np.float32)])
            self.initialized = True
            self.last_t = t
            return meas.reshape(-1), np.zeros(3)
        dt = max(1e-3, t - self.last_t); self.last_t = t
        self._set_dt(dt)
        self.kf.predict()
        est = self.kf.correct(meas)
        return est[:3].reshape(-1), est[3:].reshape(-1)


# --------------------------------------------------------------------------- #
# RealSense capture (duck-typed cv2.VideoCapture) for depth-only modules
# --------------------------------------------------------------------------- #
class RealSenseCapture:
    """Feeds the RealSense LEFT-INFRARED stream through a cv2.VideoCapture-like
    interface (read / isOpened / set / release).

    For depth modules without an RGB sensor (e.g. D430): ArUco detects fine in
    grayscale IR. The IR projector is disabled so the marker image is clean, and
    the stream's factory intrinsics are exposed as .K / .dist for PnP pose.
    """
    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self._rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        try:
            self.profile = self.pipe.start(cfg)
        except RuntimeError as e:
            raise RuntimeError(
                f"RealSense IR start failed at {width}x{height}@{fps}: {e}. "
                f"Pick a supported IR mode (e.g. 640x480, 848x480, 1280x720).")
        # kill the IR dot projector -> clean grayscale image for marker detection
        try:
            ds = self.profile.get_device().first_depth_sensor()
            if ds.supports(rs.option.emitter_enabled):
                ds.set_option(rs.option.emitter_enabled, 0)
        except Exception:
            pass
        vsp = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        intr = vsp.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], dtype=float)
        self.dist = np.array(intr.coeffs, dtype=float)
        self.width, self.height = intr.width, intr.height
        self._opened = True

    def isOpened(self):
        return self._opened

    def set(self, *args, **kwargs):
        return False                      # resolution is fixed at stream start

    def read(self):
        try:
            frames = self.pipe.wait_for_frames(2000)
            ir = frames.get_infrared_frame(1)
            if not ir:
                return False, None
            img = np.asanyarray(ir.get_data())             # HxW uint8 grayscale
            return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except Exception:
            return False, None

    def release(self):
        self._opened = False
        try:
            self.pipe.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Main node
# --------------------------------------------------------------------------- #
class PerceptionNode:
    def __init__(self, args):
        self.args = args
        self.marker_size = args.marker_size
        self.target_id = args.marker_id   # None = any / closest to center

        src = args.source
        self._is_realsense = isinstance(src, str) and src.lower() == "realsense"
        if self._is_realsense:
            self.cap = RealSenseCapture(args.width or 640, args.height or 480, args.fps)
            log.info("RealSense left-IR stream (depth module / no RGB).")
        else:
            try:
                src = int(src)
            except ValueError:
                pass
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source '{args.source}'.")
        if not self._is_realsense:
            if args.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            if args.height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Source opened but returned no frame.")
        self.height, self.width = frame.shape[:2]
        self._first_frame = frame
        log.info("Source open: %dx%d", self.width, self.height)
        if not self._is_realsense and ((args.width and args.width != self.width) or
                                       (args.height and args.height != self.height)):
            log.warning("Requested %sx%s but camera gave %dx%d -- intrinsics MUST "
                        "match the ACTUAL size above, or pose/depth will be wrong.",
                        args.width, args.height, self.width, self.height)

        if args.intrinsics:
            self.intr = Intrinsics.load(args.intrinsics, self.width, self.height)
            log.info("Loaded intrinsics from %s (3D pose enabled)", args.intrinsics)
        elif self._is_realsense:
            self.intr = Intrinsics(K=self.cap.K, dist=self.cap.dist,
                                   cx=float(self.cap.K[0, 2]),
                                   cy=float(self.cap.K[1, 2]), calibrated=True)
            log.info("RealSense factory IR intrinsics (3D pose enabled): "
                     "fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                     self.cap.K[0, 0], self.cap.K[1, 1],
                     self.cap.K[0, 2], self.cap.K[1, 2])
        else:
            self.intr = Intrinsics.default(self.width, self.height)
            log.warning("No --intrinsics: 2D center + marker id only, NO 3D pose. "
                        "Calibrate to enable metric pose/depth.")

        self.detector = ArucoDetector(args.dict)
        log.info("ArUco dict: %s | marker size: %.3f m", args.dict, self.marker_size)

        self.kalman = CVKalman() if args.kalman else None
        if args.kalman and not self.intr.calibrated:
            log.warning("--kalman with no intrinsics: filtering pixel center "
                        "(2D) instead of 3D pose.")

        if args.publish == "zmq":
            self.publisher = ZmqPublisher(args.zmq_endpoint, args.zmq_topic)
        else:
            self.publisher = StdoutPublisher()

    def _select(self, corners, ids):
        """Pick target marker. Prefer --marker-id; else the one nearest center."""
        ids = ids.flatten()
        idx_list = list(range(len(ids)))
        if self.target_id is not None:
            matches = [i for i in idx_list if int(ids[i]) == self.target_id]
            if not matches:
                return None
            idx_list = matches
        cx, cy = self.width / 2.0, self.height / 2.0
        best, best_d = None, 1e18
        for i in idx_list:
            c = corners[i].reshape(-1, 2)
            u, v = c[:, 0].mean(), c[:, 1].mean()
            d = (u - cx) ** 2 + (v - cy) ** 2
            if d < best_d:
                best_d, best = d, i
        return best

    def run(self):
        frame_id = 0
        t_fps, fps = time.time(), 0.0
        show = self.args.show
        win = "aruco perception (q to quit)"
        if show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        try:
            while True:
                frame = self._first_frame if frame_id == 0 else None
                if frame is None:
                    ok, frame = self.cap.read()
                    if not ok:
                        log.info("End of stream."); break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids = self.detector.detect(gray)

                found = ids is not None and len(ids) > 0
                marker_id = center = corners_out = pose = rvec_out = vel = None
                draw_rvec = draw_tvec = None   # pose of selected marker for axis overlay

                if found:
                    sel = self._select(corners, ids)
                    if sel is None:
                        found = False
                    else:
                        c = corners[sel].reshape(-1, 2)
                        marker_id = int(ids.flatten()[sel])
                        center = [float(c[:, 0].mean()), float(c[:, 1].mean())]
                        corners_out = c.astype(float).tolist()

                        meas3d = None
                        if self.intr.calibrated:
                            rvec, tvec = estimate_pose(
                                corners[sel], self.marker_size,
                                self.intr.K, self.intr.dist)
                            if tvec is not None:
                                draw_rvec, draw_tvec = rvec, tvec
                                pose = {"x": float(tvec[0]), "y": float(tvec[1]),
                                        "z": float(tvec[2])}
                                rvec_out = [float(v) for v in rvec]
                                meas3d = tvec

                        if self.kalman is not None:
                            now = time.time()
                            if self.intr.calibrated:
                                # 3-D filter; coast (nan) on frames with no pose
                                m = meas3d if meas3d is not None else \
                                    [np.nan, np.nan, np.nan]
                                est, v = self.kalman.update(m, now)
                                if est is not None and np.all(np.isfinite(est)):
                                    pose = {"x": float(est[0]), "y": float(est[1]),
                                            "z": float(est[2])}
                                    vel = {"vx": float(v[0]), "vy": float(v[1]),
                                           "vz": float(v[2])}
                            else:
                                # 2-D filter on the pixel center
                                est, v = self.kalman.update(
                                    [center[0], center[1], 0.0], now)
                                if est is not None and np.all(np.isfinite(est)):
                                    center = [float(est[0]), float(est[1])]
                                    vel = {"vx": float(v[0]), "vy": float(v[1]),
                                           "vz": 0.0}

                result = PerceptionResult(
                    timestamp=time.time(), frame_id=frame_id,
                    target_found=bool(found),
                    image_size=[self.width, self.height],
                    intrinsics_calibrated=self.intr.calibrated,
                    dict=self.args.dict, marker_id=marker_id, center=center,
                    corners=corners_out, pose=pose, rvec=rvec_out, velocity=vel,
                )
                self.publisher.publish(result)

                now = time.time()
                inst = 1.0 / (now - t_fps) if now > t_fps else 0.0
                fps = 0.9 * fps + 0.1 * inst if fps else inst
                t_fps = now

                if show:
                    self._draw(frame, found, center, pose, marker_id, fps,
                               corners, ids, draw_rvec, draw_tvec)
                    cv2.imshow(win, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_id += 1
        finally:
            self.cap.release()
            if show:
                cv2.destroyAllWindows()
            self.publisher.close()

    def _draw(self, frame, found, center, pose, marker_id, fps,
              corners=None, ids=None, rvec=None, tvec=None):
        # 1) Outline every detected marker (green polygon + id) so you can see
        #    what the detector sees, even markers that were not selected.
        if corners is not None and ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # 2) Image-center crosshair (where the camera is "looking").
        cx, cy = int(self.intr.cx), int(self.intr.cy)
        cv2.line(frame, (cx - 18, cy), (cx + 18, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy - 18), (cx, cy + 18), (0, 255, 255), 1)

        # 3) Highlight the SELECTED target: center dot + line from image center,
        #    plus a 3D pose axis when calibration is available.
        if found and center is not None:
            u, v = int(center[0]), int(center[1])
            cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)
            cv2.line(frame, (cx, cy), (u, v), (255, 0, 0), 2)
            txt = f"id{marker_id}"
            if pose is not None:
                txt += f"  X={pose['x']:+.2f} Y={pose['y']:+.2f} Z={pose['z']:.2f}m"
            cv2.putText(frame, txt, (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if self.intr.calibrated and rvec is not None and tvec is not None:
                cv2.drawFrameAxes(frame, self.intr.K, self.intr.dist,
                                  rvec, tvec, self.marker_size * 0.75, 2)

        # 4) Status banner: FPS, calibration mode, and lock state.
        cal = "calib" if self.intr.calibrated else "NO-CALIB(2D)"
        status = f"id{marker_id} LOCKED" if found else "searching..."
        cv2.putText(frame, f"{fps:4.1f} FPS [{cal}]  {status}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "press 'q' to quit", (8, self.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def parse_args():
    p = argparse.ArgumentParser(description="Eye-in-hand ArUco perception for UR5e.")
    p.add_argument("--source", default="0",
                   help="Webcam index, video/image path, or 'realsense' for the "
                        "Intel RealSense left-IR stream (depth modules w/o RGB).")
    p.add_argument("--width", type=int, default=None,
                   help="Request capture width. MUST match the resolution the "
                        "--intrinsics file was generated at (see realsense_setup.py). "
                        "For --source realsense this picks the IR mode (default 640).")
    p.add_argument("--height", type=int, default=None,
                   help="Request capture height. MUST match the --intrinsics resolution. "
                        "For --source realsense this picks the IR mode (default 480).")
    p.add_argument("--fps", type=int, default=30,
                   help="Requested capture FPS (used by --source realsense).")
    p.add_argument("--dict", default="4X4_50", choices=list(DICT_MAP),
                   help="ArUco dictionary the marker belongs to.")
    p.add_argument("--marker-size", type=float, default=0.05,
                   help="Physical marker side length in METERS (needed for pose).")
    p.add_argument("--marker-id", type=int, default=None,
                   help="Lock onto a specific marker id. Omit = nearest to center.")
    p.add_argument("--intrinsics", default=None,
                   help="Calibration JSON {fx,fy,cx,cy,(dist)} or {K,(dist)}. "
                        "Required for 3D pose.")
    p.add_argument("--kalman", action="store_true",
                   help="Smooth position + estimate velocity (helps moving targets).")
    p.add_argument("--publish", default="stdout", choices=["stdout", "zmq"])
    p.add_argument("--zmq-endpoint", default="tcp://127.0.0.1:5556")
    p.add_argument("--zmq-topic", default="perception")
    p.add_argument("--show", dest="show", action="store_true", default=True,
                   help="Show the live video window with overlays (default ON).")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="Disable the video window (headless / faster).")
    return p.parse_args()


def main():
    PerceptionNode(parse_args()).run()


if __name__ == "__main__":
    main()