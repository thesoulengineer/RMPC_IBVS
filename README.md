# RMPC IBVS Pipeline

Image-Based Visual Servoing (IBVS) pipeline for a UR5e with an eye-in-hand camera.

```
[perception.py] --ZMQ PUB--> [StreamSubscriber]
                                    |
                             [ControlLoop @ fixed Hz]
                              /            \
                [VisualServoController]   [FeaturePredictor]
                                    |
                             v_cam (6-twist)
                                    |
                            [RobotInterface] --RTDE speedL--> [UR5e]
```

## Requirements

- **Python 3.11** (this is the interpreter where `ur_rtde` already builds/runs on this
  machine; 3.12+ has no `ur_rtde` Windows wheel and will not compile without MSVC).

## Setup (Windows / PowerShell)

From this folder:

```powershell
# 1) Create the venv from Python 3.11.
#    --system-site-packages lets the venv borrow the already-working ur_rtde
#    from the global 3.11 install (it cannot be pip-built on Windows).
py -3.11 -m venv .venv --system-site-packages

# 2) Activate it.
.\.venv\Scripts\Activate.ps1

# 3) Install the pip-installable deps INTO the venv (these shadow any global
#    copies, so you get opencv-contrib — required for cv2.aruco).
python -m pip install -r requirements.txt
```

Verify everything imports:

```powershell
python -c "import numpy, zmq, cv2; cv2.aruco; print('core OK'); import rtde_control; print('ur_rtde OK')"
```

## Running

Always run with the venv **activated** (so `python` = `.venv\Scripts\python.exe`).

**Simulated / no robot (mock backend):**
```powershell
# Terminal A — perception (publishes target state over ZMQ):
#   Intel RealSense depth module (D430 etc., no RGB) — uses left-IR + factory intrinsics:
python perception.py --source realsense --marker-size 0.05 --marker-id 0 --kalman --publish zmq --no-show
#   ...or a plain color webcam (needs an intrinsics file):
python perception.py --source 0 --marker-size 0.05 --intrinsics cam.json --marker-id 0 --kalman --publish zmq --no-show

# Terminal B — control loop with the mock robot (default):
python main.py
```

## Intel RealSense (depth module, e.g. D430 — no RGB)

This camera has no color sensor, so perception uses its **left-infrared** stream
(ArUco detects fine in grayscale IR) and reads the device's **factory intrinsics**
automatically — no `cam.json`, no checkerboard calibration:

```powershell
python realsense_setup.py --list        # confirm the device is detected
python perception.py --source realsense --marker-size <marker_side_m> --publish zmq --no-show
```

Notes:
- Default IR mode is 640x480; change with `--width/--height/--fps` (must be a supported
  IR mode, e.g. 640x480, 848x480, 1280x720).
- The IR dot projector is disabled automatically so the marker image is clean. If the
  room is too dark for IR, add ambient light (the left IR imager needs to see the marker).
- Print a **4X4_50** ArUco marker (the default `--dict`) and pass its real side length
  in metres via `--marker-size`.
- `realsense_setup.py` is only needed for the *webcam* path (to make a `cam.json`); the
  `--source realsense` path doesn't use it.

**Real UR5e / URSim:**
```powershell
python main.py --robot rtde --ip 192.168.1.10
```

**Standalone robot motion scripts** (need a reachable robot IP — edit the IP inside each file):
```powershell
python test.py            # trace a square with moveL
python waypoint_test.py   # waypoint sequence
```

## Real robot (ur_rtde)

`ur_rtde` is intentionally **not** in `requirements.txt`: it has no Windows wheel and
must be compiled from source (needs the Visual Studio "Desktop development with C++"
build tools). This repo's venv gets it for free via `--system-site-packages` from the
global Python 3.11 install, where it is already built.

If you ever need a fully isolated build, either:
- `conda install -c conda-forge ur_rtde` (prebuilt Windows binary), or
- install Visual Studio C++ Build Tools + CMake, then `pip install ur_rtde`.

## 2-DOF poll-and-follow tracker (`marker_follow.py`)

A simple, safe alternative to continuous IBVS: the robot homes once, then every cycle
**chases the marker until it's centered** — while the marker's offset from the image
center exceeds a tolerance, it makes a 2-DOF `moveL` in base X/Y stepping `gain × offset`
toward center (a discrete, IBVS-like proportional law). It holds once centered and
re-chases when the marker moves. Runs until `Ctrl-C`.

```powershell
# Terminal A — perception (RealSense, metric depth):
python perception.py --source realsense --marker-size 0.05 --marker-id 0 --publish zmq --no-show

# Terminal B — validate WITHOUT the robot first (prints intended moves, no motion):
python marker_follow.py --dry-run

# Terminal B — on hardware (e-stop in hand, low speed):
python marker_follow.py --ip 192.168.5.5                 # discrete moveL steps (default)
python marker_follow.py --ip 192.168.5.5 --continuous    # smooth speedL velocity servo
```

Bring-up notes:
- **Run `--dry-run` first.** Move the marker and confirm the printed base `dX,dY`
  direction matches its real-world motion. The camera→base axis signs are **not**
  hand-eye calibrated — fix them in `CAM_TO_BASE` at the top of `marker_follow.py`.
- **Two modes:** default is discrete `moveL` steps; `--continuous` streams `speedL` velocity
  (`v = --vel-gain × offset`, capped at `--vel`) at `--rate` Hz for smooth tracking. Validate
  signs with `--continuous --dry-run` first — continuous never pauses, so a wrong `CAM_TO_BASE`
  sign runs away *continuously* (the discrete mode only nudges once per cycle).
- Tunables (CLI or constants): `--period`, `--tol` (centered stop band, ~1 cm),
  `--gain` (λ: fraction of the offset moved per step; <1 = smooth IBVS-like approach,
  1.0 = one-shot), `--vel`/`--acc`, `--home-x/-y/-z`, `--max-cycles` (bounded test runs).
- Only base X and Y move; Z and tool orientation are held (that's the "2-DOF").

## Files

| File | Role |
|------|------|
| `main.py` | Pipeline orchestrator / entry point |
| `perception.py` | ArUco detection + Kalman, publishes target state over ZMQ |
| `stream_subscriber.py` | Receives target state (ZMQ or file) |
| `controller.py` | `VisualServoController` — IBVS control law |
| `ibvs_controller_new.py` | `IBVS_Controller` — interaction-matrix core |
| `predictor.py` | `FeaturePredictor` — extrapolates on dropped frames |
| `control_loop.py` | Fixed-rate scheduler + state machine |
| `robot_interface.py` | `RTDEInterface` (real) and `MockInterface` (no robot) |
| `schema.py` | `TargetState` data schema |
| `marker_follow.py` | 2-DOF poll-and-follow tracker (discrete `moveL`, standalone) |
| `realsense_setup.py` | RealSense helper: list devices, dump factory intrinsics → `cam.json` |
| `pose_tap.py` | Diagnostic: prints what perception publishes (pose/depth/calibrated) |
| `test.py`, `waypoint_test.py` | Standalone UR5e motion scripts (RTDE) |
| `cam.json` | Camera intrinsics (must match `K` in `main.py`) |
