#!/usr/bin/env python3
"""
schema.py — the compact wire contract between perception and controller.
========================================================================

Single source of truth for the low-latency message. Perception fills it,
the controller reads it. NO images ever cross the wire — only this.

Fields (compact JSON, matches perception.py output):
    t           : float  capture timestamp (epoch seconds)
    frame_id    : int    monotonically increasing frame counter
    found       : bool   target present this frame
    marker_id   : int|None
    uv          : [u, v] | None        2D center, pixels
    corners     : [[u,v]*4] | None     4 corners, pixels (richer feature set)
    xyz         : [x,y,z] | None        3D pose, meters, camera frame (calibrated)
    vel         : [vx,vy,vz] | None     velocity estimate, m/s, camera frame
    calibrated  : bool   xyz/vel trustworthy only when True

`TargetState` is a thin typed view over the raw dict so the controller never
touches JSON keys directly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import json
import time


@dataclass
class TargetState:
    t: float
    frame_id: int
    found: bool
    marker_id: Optional[int] = None
    uv: Optional[List[float]] = None
    corners: Optional[List[List[float]]] = None
    xyz: Optional[List[float]] = None
    vel: Optional[List[float]] = None
    calibrated: bool = False

    # ---- (de)serialization -------------------------------------------------
    @classmethod
    def from_perception(cls, d: dict) -> "TargetState":
        """Build from a perception.py JSON frame (tolerant of its field names)."""
        pose = d.get("pose")
        vel = d.get("velocity")
        return cls(
            t=float(d.get("timestamp", time.time())),
            frame_id=int(d.get("frame_id", -1)),
            found=bool(d.get("target_found", False)),
            marker_id=d.get("marker_id"),
            uv=d.get("center"),
            corners=d.get("corners"),
            xyz=[pose["x"], pose["y"], pose["z"]] if pose else None,
            vel=[vel["vx"], vel["vy"], vel["vz"]] if vel else None,
            calibrated=bool(d.get("intrinsics_calibrated", False)),
        )

    @classmethod
    def from_json(cls, s: str) -> "TargetState":
        return cls.from_perception(json.loads(s))

    def has_depth(self) -> bool:
        return self.calibrated and self.xyz is not None

    def depth(self, fallback: float) -> float:
        return float(self.xyz[2]) if self.has_depth() else float(fallback)
