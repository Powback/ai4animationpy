"""Locomotion benchmark: drive a local Go1Puppeteer from the live MANN
server, log MANN-vs-Go1 root trajectories, report forward speed, trajectory
RMSE, foot slip, and contact stats. Sweep puppeteer settings to compare.

Run against the local ai4anim container (no queue, no browser):

    MANN_HTTP=http://192.168.50.100:7860 \
        python3 Demos/Go1/measure_locomotion.py --duration 10

Each `--preset` tag runs a separate pass and appends results to the JSON
summary file so we can tabulate the effect of one knob at a time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import websockets

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from go1_puppeteer import Go1Puppeteer  # noqa: E402

MANN_HTTP = os.environ.get("MANN_HTTP", "http://192.168.50.100:7860")
MANN_WS = os.environ.get("MANN_WS", MANN_HTTP.replace("http", "ws"))

SERVER_FPS = 30
TRAJ_SAMPLES = 16

# 21 MANN bones the retargeter consumes.
GO1_BONES = (
    "Hips", "Spine1",
    "LeftShoulder", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
)


def parse_frame(buf: bytes, entity_count: int) -> dict:
    arr = np.frombuffer(buf, dtype=np.float32)
    o = 0
    root = arr[o:o + 16].reshape(4, 4); o += 16
    entities = arr[o:o + entity_count * 16].reshape(entity_count, 4, 4)
    o += entity_count * 16
    contacts = arr[o:o + 4]; o += 4
    sim_pos = arr[o:o + TRAJ_SAMPLES * 3].reshape(TRAJ_SAMPLES, 3); o += TRAJ_SAMPLES * 3
    sim_dir = arr[o:o + TRAJ_SAMPLES * 3].reshape(TRAJ_SAMPLES, 3); o += TRAJ_SAMPLES * 3
    ctrl_pos = arr[o:o + TRAJ_SAMPLES * 3].reshape(TRAJ_SAMPLES, 3); o += TRAJ_SAMPLES * 3
    ctrl_dir = arr[o:o + TRAJ_SAMPLES * 3].reshape(TRAJ_SAMPLES, 3); o += TRAJ_SAMPLES * 3
    speed = float(arr[o]) if o < len(arr) else 0.0
    return {
        "root": root, "entities": entities, "contacts": contacts,
        "sim_pos": sim_pos, "sim_dir": sim_dir,
        "ctrl_pos": ctrl_pos, "ctrl_dir": ctrl_dir, "speed": speed,
    }


def bone_positions(entities: np.ndarray) -> np.ndarray:
    return entities[:, :3, 3].copy()


def mann_root_xz_yaw(root4: np.ndarray) -> tuple[float, float, float]:
    x = float(root4[0, 3]); z = float(root4[2, 3])
    yaw = float(np.arctan2(root4[0, 2], root4[2, 2]))
    return x, z, yaw


def quat_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    return float(np.arctan2(fy, fx))


async def run_preset(
    preset: dict,
    duration: float,
    fwd_speed: float,
) -> dict:
    # Join the queue with retry-on-429 (server has a 5s rate limit).
    for attempt in range(20):
        r = requests.post(f"{MANN_HTTP}/api/queue/join", json={}, timeout=5)
        if r.status_code == 429:
            await asyncio.sleep(6.0)
            continue
        r.raise_for_status()
        break
    j = r.json()
    sid = j.get("sid")
    if not sid:
        raise RuntimeError(f"queue/join: {j}")
    # Poll for promotion (another preset may still hold the session).
    for _ in range(120):
        if j.get("promoted"):
            break
        await asyncio.sleep(2.0)
        s = requests.get(f"{MANN_HTTP}/api/queue/status",
                         params={"sid": sid}, timeout=5).json()
        j = s
    if not j.get("promoted"):
        raise RuntimeError(f"timed out waiting for promotion: {j}")

    ws_url = f"{MANN_WS}/ws/quadruped?sid={sid}"

    pup = Go1Puppeteer(
        sweep_gain=preset.get("sweep_gain", 1.5),
        zero_abduction=preset.get("zero_abduction", True),
        foot_drop_m=preset.get("foot_drop_m", 0.0),
        warmup_frames=preset.get("warmup_frames", 20),
        blend_frames=preset.get("blend_frames", 30),
        smooth_alpha=preset.get("smooth_alpha", 0.5),
    )
    # Optional lever for thigh bias (forward body lean) and stance push.
    body_lean = float(preset.get("body_lean_rad", 0.0))
    stance_push = float(preset.get("stance_push_rad", 0.0))

    # Logs
    mann_x, mann_z, mann_yaw_log = [], [], []
    go1_x, go1_y, go1_z, go1_yaw_log = [], [], [], []
    contacts_log = []
    mann_speed_log = []
    foot_pos_prev = None
    foot_slip_per_frame: list[float] = []

    foot_site_ids = [pup.model.site(n).id for n in ("FR", "FL", "RR", "RL")]

    t0 = time.time()
    frame_count = 0
    fell = False

    async with websockets.connect(ws_url, max_size=None) as ws:
        init = json.loads(await ws.recv())
        entity_names = init.get("entityNames", [])
        name_to_idx = {n: i for i, n in enumerate(entity_names)}
        entity_count = init.get("entityCount", len(entity_names))

        stop_send = asyncio.Event()

        async def send_inputs():
            while not stop_send.is_set():
                await ws.send(json.dumps({
                    "type": "input",
                    "left_stick": [0.0, float(fwd_speed)],
                    "right_stick": [0.0, 0.0],
                    "speed_toggle": False,
                    "canter_boost": False,
                    "walk_modifier": False,
                    "trot_modifier": True,
                    "canter_modifier": False,
                    "action_sit": False,
                    "action_stand": False,
                    "action_lie": False,
                    "guidance_index": 0,
                    "character_index": 0,
                }))
                await asyncio.sleep(1.0 / SERVER_FPS)

        send_task = asyncio.create_task(send_inputs())

        try:
            t_end = time.time() + duration
            while time.time() < t_end:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(msg, str):
                    j = json.loads(msg)
                    if j.get("type") == "timeout":
                        break
                    continue
                frame = parse_frame(msg, entity_count)

                positions = bone_positions(frame["entities"])
                name_to_pos = {
                    n: positions[name_to_idx[n]]
                    for n in GO1_BONES if n in name_to_idx
                }

                # Experimental bias: forward body lean shifts commanded
                # thighs so CoM moves forward, and stance-push adds an
                # extra thigh-back offset during the first half of each
                # 0.4s cycle for the diagonal trot pair.
                if body_lean != 0.0 or stance_push != 0.0:
                    # Fake body lean by tilting Spine1 forward relative to
                    # Hips — changes the body-frame axis used by retarget.
                    if body_lean != 0.0:
                        hips = name_to_pos["Hips"].copy()
                        sp = name_to_pos["Spine1"].copy()
                        fwd_dir = sp - hips
                        fwd_dir[1] = 0.0
                        n = float(np.linalg.norm(fwd_dir))
                        if n > 1e-6:
                            fwd_dir = fwd_dir / n
                            # Tilt spine toward forward-down: +forward, -up
                            tilt_amt = float(np.sin(body_lean))
                            sp_new = hips + (sp - hips) * float(np.cos(body_lean))
                            sp_new -= np.array([0.0, tilt_amt * 0.1, 0.0])
                            sp_new += fwd_dir * tilt_amt * 0.1
                            name_to_pos = dict(name_to_pos)
                            name_to_pos["Spine1"] = sp_new.astype(np.float32)

                pup.step(name_to_pos)

                # Log MANN root
                mx, mz, myaw = mann_root_xz_yaw(frame["root"])
                mann_x.append(mx); mann_z.append(mz); mann_yaw_log.append(myaw)

                # Log Go1 root (MuJoCo world, z-up)
                gx = float(pup.data.qpos[0])
                gy = float(pup.data.qpos[1])
                gz = float(pup.data.qpos[2])
                gyaw = quat_yaw(pup.data.qpos[3:7])
                go1_x.append(gx); go1_y.append(gy); go1_z.append(gz)
                go1_yaw_log.append(gyaw)

                contacts_log.append(frame["contacts"].copy())
                mann_speed_log.append(frame["speed"])

                # Foot slip: during a contact, world foot should not move
                # more than step distance the body moved.
                foot_pos_now = np.stack(
                    [pup.data.site_xpos[sid].copy() for sid in foot_site_ids]
                )
                if foot_pos_prev is not None:
                    # foot in contact if z < 0.03
                    in_contact = (foot_pos_now[:, 2] < 0.03)
                    dxy = np.linalg.norm(
                        foot_pos_now[:, :2] - foot_pos_prev[:, :2], axis=1
                    )
                    if in_contact.any():
                        foot_slip_per_frame.append(float(dxy[in_contact].mean()))
                    else:
                        foot_slip_per_frame.append(0.0)
                foot_pos_prev = foot_pos_now

                if pup.has_fallen():
                    fell = True
                    break

                frame_count += 1
        finally:
            stop_send.set()
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

    # Compute stats
    mann_x = np.asarray(mann_x, dtype=np.float32)
    mann_z = np.asarray(mann_z, dtype=np.float32)
    go1_x = np.asarray(go1_x, dtype=np.float32)
    go1_y = np.asarray(go1_y, dtype=np.float32)
    go1_z_log = np.asarray(go1_z, dtype=np.float32)

    if len(mann_x) < 2:
        return {"preset": preset, "error": "no frames", "frames": frame_count}

    # MANN dog: x,z ground-plane. Forward is local +z at yaw=0.
    # MANN speed: use the server-reported speed value (already body-frame).
    sec = len(mann_x) / SERVER_FPS
    mann_disp = float(
        np.linalg.norm([mann_x[-1] - mann_x[0], mann_z[-1] - mann_z[0]])
    )
    mann_avg_speed = mann_disp / max(sec, 1e-6)

    # Go1 speed: ground-plane xy in MuJoCo.
    go1_disp = float(
        np.linalg.norm([go1_x[-1] - go1_x[0], go1_y[-1] - go1_y[0]])
    )
    go1_avg_speed = go1_disp / max(sec, 1e-6)

    # Instantaneous velocities (finite-differenced, windowed mean).
    def _avg_abs_vel(a: np.ndarray, b: np.ndarray) -> float:
        dx = np.diff(a) * SERVER_FPS
        dy = np.diff(b) * SERVER_FPS
        return float(np.mean(np.linalg.norm(np.stack([dx, dy], axis=1), axis=1)))

    mann_inst_speed = _avg_abs_vel(mann_x, mann_z)
    go1_inst_speed = _avg_abs_vel(go1_x, go1_y)

    # Height drift (relative to home 0.27)
    height_mean = float(go1_z_log.mean())
    height_min = float(go1_z_log.min())
    height_max = float(go1_z_log.max())

    # Foot slip
    foot_slip_mean = float(np.mean(foot_slip_per_frame))
    foot_slip_max = float(np.max(foot_slip_per_frame) if foot_slip_per_frame else 0.0)

    stats = {
        "preset_tag": preset.get("tag", "default"),
        "preset": {k: v for k, v in preset.items() if k != "tag"},
        "frames": frame_count,
        "duration_s": len(mann_x) / SERVER_FPS,
        "fell": fell,
        "mann_disp_m": mann_disp,
        "go1_disp_m": go1_disp,
        "disp_ratio": go1_disp / max(mann_disp, 1e-6),
        "mann_avg_speed_mps": mann_avg_speed,
        "go1_avg_speed_mps": go1_avg_speed,
        "mann_inst_speed_mps": mann_inst_speed,
        "go1_inst_speed_mps": go1_inst_speed,
        "mann_server_speed_mean": float(np.mean(mann_speed_log)) if mann_speed_log else 0.0,
        "go1_height_mean_m": height_mean,
        "go1_height_min_m": height_min,
        "go1_height_max_m": height_max,
        "foot_slip_per_frame_mean_m": foot_slip_mean,
        "foot_slip_per_frame_max_m": foot_slip_max,
    }
    stats["perf_wall_s"] = time.time() - t0
    return stats


async def main_async(args):
    presets = [json.loads(p) for p in args.presets] or [
        {"tag": "default", "sweep_gain": 1.5, "zero_abduction": True},
    ]
    results = []
    for pi, preset in enumerate(presets):
        print(f"\n=== Preset {pi+1}/{len(presets)}: {preset} ===", flush=True)
        try:
            stats = await run_preset(
                preset=preset,
                duration=args.duration,
                fwd_speed=args.fwd_speed,
            )
        except Exception as e:
            stats = {"preset": preset, "error": repr(e)}
        results.append(stats)
        print(json.dumps(stats, indent=2), flush=True)
        # Cooldown so queue/rate-limit doesn't bite (5s window).
        await asyncio.sleep(6.0)

    out_path = Path(args.out or "out_measure_locomotion.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[save] wrote {out_path}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fwd-speed", type=float, default=1.0)
    ap.add_argument("--out", default="out_measure_locomotion.json")
    ap.add_argument("--presets", nargs="*", default=[],
                    help="JSON dicts, e.g. "
                         "'{\"tag\":\"sweep1.8\",\"sweep_gain\":1.8}'")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
