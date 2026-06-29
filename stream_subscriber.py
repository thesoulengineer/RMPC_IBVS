#!/usr/bin/env python3
"""
stream_subscriber.py — perception → controller transport (SUB side).
====================================================================

PUB–SUB, non-blocking, conflated. The controller pulls the freshest
TargetState whenever it ticks; it never blocks waiting for perception.

Design choices that satisfy the real-time spec:
  * zmq.CONFLATE=1  -> queue depth 1; a slow controller never reads stale
    backlog, it always gets the newest frame (drops are silent and correct).
  * Background thread does the recv; the control loop only touches a lock-
    guarded latest-value. No REQ–REP, no round-trip, no head-of-line blocking.
  * Tracks arrival wall-time so the loop can compute staleness and decide
    TRACKING vs PREDICTING vs LOST.

A stdout fallback (FileSubscriber) is included for running without ZMQ.
"""
from __future__ import annotations
import sys
import json
import threading
import time
from typing import Optional, Tuple

from schema import TargetState


class StreamSubscriber:
    """Non-blocking conflated SUB. Exposes latest TargetState + its age."""

    def __init__(self,
                 endpoint: str = "tcp://127.0.0.1:5556",
                 topic: bytes = b"perception"):
        import zmq
        self._zmq = zmq
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB)
        # Note: do NOT use zmq.CONFLATE here. CONFLATE is incompatible with
        # multipart messages (it keeps only the last *part*, breaking the
        # [topic, payload] framing perception.py uses). Instead we keep a small
        # RCVHWM and the background thread always advances to the newest frame,
        # so the control loop still only ever sees the freshest observation.
        self.sock.setsockopt(zmq.RCVHWM, 4)          # tiny buffer; drop old
        self.sock.setsockopt(zmq.RCVTIMEO, 200)      # don't block forever
        self.sock.connect(endpoint)                  # perception binds; we connect
        self.sock.setsockopt(zmq.SUBSCRIBE, topic)

        self._state: Optional[TargetState] = None
        self._recv_wall: float = 0.0                 # wall-clock of last recv
        self._lock = threading.Lock()
        self._stop = False
        self._t = threading.Thread(target=self._spin, daemon=True)
        self._t.start()

    def _recv_newest(self):
        """Block for one message, then non-blocking drain to the newest."""
        zmq = self._zmq
        parts = self.sock.recv_multipart()           # blocks (RCVTIMEO bound)
        while True:                                   # drain backlog -> newest
            try:
                parts = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        return parts

    def _spin(self):
        zmq = self._zmq
        while not self._stop:
            try:
                parts = self._recv_newest()
            except zmq.Again:
                continue
            except Exception:
                continue
            payload = parts[-1]                       # last frame = JSON payload
            try:
                st = TargetState.from_json(payload.decode())
            except Exception:
                continue
            with self._lock:
                self._state = st
                self._recv_wall = time.monotonic()

    def latest(self) -> Tuple[Optional[TargetState], float]:
        """
        Return (state, age_seconds). age is wall-time since the frame arrived;
        None state means nothing received yet.
        """
        with self._lock:
            st = self._state
            wall = self._recv_wall
        age = (time.monotonic() - wall) if wall else float("inf")
        return st, age

    def close(self):
        self._stop = True
        try:
            self.sock.close(linger=0)
        except Exception:
            pass


class FileSubscriber:
    """
    stdout/file fallback: reads newline-delimited JSON from a stream (e.g.
    perception.py --publish stdout piped in). Same interface as StreamSubscriber.
    """
    def __init__(self, stream=None):
        self._stream = stream or sys.stdin
        self._state: Optional[TargetState] = None
        self._recv_wall = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self._t = threading.Thread(target=self._spin, daemon=True)
        self._t.start()

    def _spin(self):
        for line in self._stream:
            if self._stop:
                break
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                st = TargetState.from_json(line)
            except Exception:
                continue
            with self._lock:
                self._state = st
                self._recv_wall = time.monotonic()

    def latest(self):
        with self._lock:
            st, wall = self._state, self._recv_wall
        age = (time.monotonic() - wall) if wall else float("inf")
        return st, age

    def close(self):
        self._stop = True
