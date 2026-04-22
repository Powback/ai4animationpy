"""Headless diagnostic: connect to ai4anim.pow, drive joystick through a
scripted turn schedule, and measure **server-side** Go1 yaw vs MANN yaw.

Unlike spec-llm/mann_live_go1.py (which runs its own local MuJoCo Go1), this
reads Go1's pose back from the server's WS trailer — so it measures the
actual demo the user sees at ai4anim.pow.

Trailer layout (last 20 float32s of every binary frame):
  go1_active(1) + go1_root(7=xyz+quat) + go1_qpos(12)

Usage:
  python3 scripts/server_go1_yaw_bench.py --duration 20 \
      --turn-schedule "0:0,3:0.6,8:-0.6,13:0"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
import websockets

AI4ANIM_HTTP = os.environ.get("AI4ANIM_HTTP", "http://ai4anim.pow")
AI4ANIM_WS = os.environ.get("AI4ANIM_WS", "ws://ai4anim.pow")
SERVER_FPS = 30
TRAJ_SAMPLES = 16
GO1_TRAILER_LEN = 20  # f32s


def _quat_yaw_rad(q):
    w, x, y, z = (float(v) for v in q)
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    return float(np.arctan2(fy, fx))


def _mann_root_yaw_rad(root_4x4):
    axis_z = root_4x4[:3, 2]
    return float(np.arctan2(axis_z[0], axis_z[2]))


def _wrap_pi(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def parse_schedule(s):
    out = []
    for part in (p.strip() for p in s.split(",") if p.strip()):
        t, v = part.split(":")
        out.append((float(t), float(v)))
    out.sort()
    return out


def schedule_value(t, sched, default=0.0):
    if not sched:
        return default
    cur = sched[0][1] if t >= sched[0][0] else default
    for ts, v in sched:
        if t >= ts:
            cur = v
        else:
            break
    return cur


def parse_frame(buf, entity_count):
    a = np.frombuffer(buf, dtype=np.float32)
    o = 0
    root = a[o:o + 16].reshape(4, 4); o += 16
    o += entity_count * 16
    o += 4            # contacts
    o += TRAJ_SAMPLES * 3 * 4  # sim/ctrl traj pos+dir
    o += 1            # speed
    trailer = a[o:o + GO1_TRAILER_LEN]
    go1_active = float(trailer[0])
    go1_root = trailer[1:8]            # x y z qw qx qy qz
    go1_qpos = trailer[8:20]
    return root, go1_active, go1_root, go1_qpos


async def run(duration, turn_sched, fwd_sched, fwd_default, gait):
    r = requests.post(f"{AI4ANIM_HTTP}/api/queue/join", json={}, timeout=5)
    r.raise_for_status()
    j = r.json()
    sid = j["sid"]
    ws_url = f"{AI4ANIM_WS}/ws/quadruped?sid={sid}"
    print(f"[queue] sid={sid[:8]}… promoted={j.get('promoted')}", flush=True)

    mann_yaws, go1_yaws, go1_xys, turn_cmds, fwd_cmds = [], [], [], [], []
    go1_heights = []
    mann_xz = []  # x,z in MANN world (y-up)
    t_start = time.time()
    ever_active = False

    async with websockets.connect(ws_url, max_size=None) as ws:
        init = json.loads(await ws.recv())
        assert init.get("type") == "init", init
        entity_count = init.get("entityCount")
        print(f"[init] entities={entity_count} styles={init.get('styles')}",
              flush=True)

        async def send_inputs():
            t0 = time.time()
            while time.time() - t0 < duration + 1.0:
                t_rel = time.time() - t0
                fwd_now = schedule_value(t_rel, fwd_sched, fwd_default)
                turn_now = schedule_value(t_rel, turn_sched, 0.0)
                inp = {
                    "type": "input",
                    "left_stick": [float(turn_now), float(fwd_now)],
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
                    "go1_enabled": True,
                }
                await ws.send(json.dumps(inp))
                await asyncio.sleep(1.0 / SERVER_FPS)

        send_task = asyncio.create_task(send_inputs())
        t_end = time.time() + duration
        frame = 0
        try:
            while time.time() < t_end:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(msg, str):
                    continue
                root, active, go1_root, go1_qpos = parse_frame(
                    msg, entity_count)
                if active > 0.5:
                    ever_active = True
                    mann_yaws.append(_mann_root_yaw_rad(root))
                    go1_yaws.append(_quat_yaw_rad(go1_root[3:7]))
                    go1_xys.append((float(go1_root[0]), float(go1_root[1])))
                    go1_heights.append(float(go1_root[2]))
                    mann_xz.append((float(root[0, 3]), float(root[2, 3])))
                    t_rel = time.time() - t_start
                    turn_cmds.append(schedule_value(t_rel, turn_sched, 0.0))
                    fwd_cmds.append(
                        schedule_value(t_rel, fwd_sched, fwd_default))
                frame += 1
                if frame % 30 == 0 and ever_active:
                    t_rel = time.time() - t_start
                    print(f"  [t={t_rel:5.1f}s] turn_cmd="
                          f"{turn_cmds[-1]:+.2f} "
                          f"mann_yaw={np.degrees(mann_yaws[-1] - mann_yaws[0]):+6.1f}° "
                          f"go1_yaw={np.degrees(go1_yaws[-1] - go1_yaws[0]):+6.1f}° "
                          f"go1_z={go1_heights[-1]:.3f}m",
                          flush=True)
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

    if not mann_yaws:
        print("[ERROR] Go1 never became active — check server logs",
              flush=True)
        return {}

    m = np.asarray(mann_yaws)
    g = np.asarray(go1_yaws)
    m_rel = np.asarray([_wrap_pi(v - m[0]) for v in m])
    g_rel = np.asarray([_wrap_pi(v - g[0]) for v in g])
    err = np.asarray([_wrap_pi(a - b) for a, b in zip(m_rel, g_rel)])

    # Per-phase breakdown: split by turn_cmd segments.
    phase_stats = []
    if turn_sched:
        for i, (t_start_p, v) in enumerate(turn_sched):
            t_end_p = (turn_sched[i + 1][0] if i + 1 < len(turn_sched)
                       else duration)
            mask = [t_start_p <= (j / SERVER_FPS) < t_end_p
                    for j in range(len(err))]
            if sum(mask) == 0:
                continue
            sub_err = err[np.asarray(mask)]
            sub_mann = m_rel[np.asarray(mask)]
            sub_go1 = g_rel[np.asarray(mask)]
            phase_stats.append({
                "t": f"{t_start_p:.1f}-{t_end_p:.1f}s",
                "turn_cmd": float(v),
                "n": int(sum(mask)),
                "mann_yaw_deg": float(np.degrees(sub_mann[-1] - sub_mann[0])),
                "go1_yaw_deg": float(np.degrees(sub_go1[-1] - sub_go1[0])),
                "yaw_err_mae_deg": float(np.degrees(np.mean(np.abs(sub_err)))),
            })

    go1_fell = min(go1_heights) < 0.13 if go1_heights else False
    stats = {
        "frames_active": len(mann_yaws),
        "go1_fell": go1_fell,
        "go1_min_height_m": float(min(go1_heights)) if go1_heights else 0.0,
        "mann_yaw_total_deg": float(np.degrees(m_rel[-1])),
        "go1_yaw_total_deg": float(np.degrees(g_rel[-1])),
        "yaw_mean_abs_err_deg": float(np.degrees(np.mean(np.abs(err)))),
        "yaw_max_abs_err_deg": float(np.degrees(np.max(np.abs(err)))),
        "go1_distance_m": float(np.linalg.norm(
            np.asarray(go1_xys[-1]) - np.asarray(go1_xys[0])
        )),
        "go1_avg_speed_mps": float(np.linalg.norm(
            np.asarray(go1_xys[-1]) - np.asarray(go1_xys[0])
        ) / max(1e-6, len(mann_yaws) / SERVER_FPS)),
        "phase_stats": phase_stats,
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--turn-schedule", default="")
    ap.add_argument("--fwd-schedule", default="")
    ap.add_argument("--fwd", type=float, default=1.0)
    ap.add_argument("--gait", default="walk",
                    choices=["walk", "trot", "canter", "pace"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    turn_sched = parse_schedule(args.turn_schedule)
    fwd_sched = parse_schedule(args.fwd_schedule)

    stats = asyncio.run(run(args.duration, turn_sched, fwd_sched,
                            args.fwd, args.gait))
    print("\n[summary]", json.dumps(stats, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(stats, indent=2))
        print(f"[save] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
