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
python perception.py --source 0 --marker-size 0.05 --intrinsics cam.json --marker-id 0 --kalman --publish zmq --no-show

# Terminal B — control loop with the mock robot (default):
python main.py
```

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
| `test.py`, `waypoint_test.py` | Standalone UR5e motion scripts (RTDE) |
| `cam.json` | Camera intrinsics (must match `K` in `main.py`) |
