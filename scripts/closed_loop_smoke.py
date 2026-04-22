"""Closed-loop smoke test. Runs INSIDE the ai4anim container.

Drives the live MANN+Go1 demo at trot for `--duration` seconds, records the
Go1 root height (qpos z) each frame, and reports:
  * time-to-fall (first frame with z < 0.13 m) or None if still upright
  * mean & min Go1 z
  * total forward displacement (initial MuJoCo-x vs final MuJoCo-x)

Run with CLOSED_LOOP_GAIN env controlling the fix (set before starting the
container). Run twice (GAIN=0 vs GAIN=0.3) to A/B.

Usage (inside container):
    python3 /app/scripts/closed_loop_smoke.py --duration 60

Or from host:
    docker exec ai4anim python3 /app/scripts/closed_loop_smoke.py --duration 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time
import uuid

import urllib.parse
import urllib.request

import numpy as np
import websockets


def _http_post_json(url: str, payload: dict, timeout: float = 5.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url: str, params: dict, timeout: float = 5.0) -> dict:
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

HTTP = os.environ.get("AI4ANIM_HTTP", "http://127.0.0.1:7860")
WS = os.environ.get("AI4ANIM_WS", "ws://127.0.0.1:7860")
SERVER_FPS = 30
TRAJ_SAMPLES = 16
GO1_PAYLOAD_FLOATS = 20  # active(1) + root(7) + qpos(12)


def join_and_wait(sid: str, max_wait: float = 60.0) -> None:
    j = _http_post_json(f"{HTTP}/api/queue/join", {"sid": sid})
    if j.get("promoted"):
        return
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(2.0)
        try:
            st = _http_get_json(f"{HTTP}/api/queue/status", {"sid": sid})
        except Exception:
            continue
        if st.get("promoted"):
            return
    raise RuntimeError("queue promotion timed out")


def parse_go1_tail(buf: bytes) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (active_flag, root7, qpos12) from the last 20*4 bytes."""
    tail = np.frombuffer(buf[-GO1_PAYLOAD_FLOATS * 4:], dtype=np.float32)
    return float(tail[0]), tail[1:8].copy(), tail[8:20].copy()


async def drive(duration: float, warmup: float, gait: str) -> dict:
    sid = uuid.uuid4().hex
    print(f"[q] join sid={sid[:8]}…", flush=True)
    join_and_wait(sid)
    print(f"[q] promoted", flush=True)

    url = f"{WS}/ws/quadruped?sid={sid}"
    async with websockets.connect(url, max_size=None) as ws:
        init_raw = await ws.recv()
        init = json.loads(init_raw)
        if init.get("type") != "init":
            raise RuntimeError(f"bad init: {init}")
        print(f"[ws] init go1Available={init.get('go1Available')}", flush=True)

        stop_send = asyncio.Event()

        async def send_loop():
            # Warm up without Go1 so MANN dog can start walking, then enable.
            t_start = time.time()
            while not stop_send.is_set():
                enable_go1 = (time.time() - t_start) > warmup
                payload = {
                    "type": "input",
                    "left_stick": [0.0, 1.0],
                    "right_stick": [0.0, 0.0],
                    "speed_toggle": False,
                    "canter_boost": gait == "canter",
                    "walk_modifier": gait == "walk",
                    "trot_modifier": gait == "trot",
                    "canter_modifier": gait == "canter",
                    "action_sit": False,
                    "action_stand": False,
                    "action_lie": False,
                    "guidance_index": 0,
                    "character_index": 0,
                    "go1_enabled": bool(enable_go1),
                }
                try:
                    await ws.send(json.dumps(payload))
                except Exception:
                    return
                await asyncio.sleep(1.0 / SERVER_FPS)

        send_task = asyncio.create_task(send_loop())

        go1_zs: list[float] = []
        go1_xs: list[float] = []
        yaws: list[float] = []
        active_seen_at: float | None = None
        fell_at: float | None = None
        x0: float | None = None
        t_end = time.time() + duration
        frame_count = 0

        try:
            while time.time() < t_end:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(msg, str):
                    j = json.loads(msg)
                    if j.get("type") == "timeout":
                        print(f"[ws] timeout: {j.get('message')}", flush=True)
                        break
                    continue
                active, root7, _qpos = parse_go1_tail(msg)
                frame_count += 1
                if active > 0.5:
                    z = float(root7[2])
                    x = float(root7[0])
                    w, qx, qy, qz = (float(v) for v in root7[3:7])
                    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
                    fy = 2.0 * (qx * qy + w * qz)
                    yaw = float(np.arctan2(fy, fx))
                    go1_zs.append(z)
                    go1_xs.append(x)
                    yaws.append(yaw)
                    if active_seen_at is None:
                        active_seen_at = time.time()
                        x0 = x
                    if z < 0.13 and fell_at is None:
                        fell_at = time.time() - active_seen_at
                        print(f"[!] fell at t={fell_at:.2f}s (z={z:.3f})",
                              flush=True)
                if frame_count % 60 == 0 and go1_zs:
                    t_go1 = (time.time() - active_seen_at) if active_seen_at else 0.0
                    dx = (go1_xs[-1] - (x0 or 0.0))
                    print(f"  [frame {frame_count:4d}] t_go1={t_go1:4.1f}s "
                          f"z={go1_zs[-1]:.3f} dx={dx:+.2f}m "
                          f"yaw={np.degrees(yaws[-1]):+6.1f}°", flush=True)
        finally:
            stop_send.set()
            await send_task

    return {
        "frames": frame_count,
        "go1_active_frames": len(go1_zs),
        "fell_at_s": fell_at,
        "min_z": float(np.min(go1_zs)) if go1_zs else None,
        "mean_z": float(np.mean(go1_zs)) if go1_zs else None,
        "z_last": float(go1_zs[-1]) if go1_zs else None,
        "dx_total_m": float(go1_xs[-1] - go1_xs[0]) if go1_xs else 0.0,
        "walk_time_s": (
            (fell_at if fell_at is not None else len(go1_zs) / SERVER_FPS)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="Seconds of MANN-only walk before enabling Go1")
    ap.add_argument("--gait", default="trot",
                    choices=["walk", "trot", "pace", "canter"])
    args = ap.parse_args()

    stats = asyncio.run(drive(args.duration, args.warmup, args.gait))
    stats["env"] = {
        "CLOSED_LOOP_GAIN": os.environ.get("CLOSED_LOOP_GAIN"),
        "CLOSED_LOOP_LAT_SIGN": os.environ.get("CLOSED_LOOP_LAT_SIGN"),
    }
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
