# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
import importlib.util
import json
import logging
import os
import secrets
import sys
import threading
import time
import traceback

import numpy as np

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from pathlib import Path

logger = logging.getLogger(__name__)

SERVER_FPS = 30
FRAME_DT = 1.0 / SERVER_FPS
SESSION_TIMEOUT = 3 * 60
TICKET_STALE_SECONDS = 60
MAX_QUEUE_SIZE = 20
RATE_LIMIT_JOIN_SECONDS = 5
RATE_LIMIT_STATUS_SECONDS = 1
PRELOAD_DEMOS = os.environ.get("PRELOAD_DEMOS", "").strip().lower()
MAX_CPU_THREADS = max(2, int(os.environ.get("MAX_CPU_THREADS", "4")))
# Queue identity sid is client-provided opaque token (generated server-side on first join).
RETRY_AFTER_QUEUE_FULL_SECONDS = int(os.environ.get("RETRY_AFTER_QUEUE_FULL_SECONDS", "30"))


def _log_queue_event(event: str, **fields) -> None:
    payload = {"event": event, "ts": time.time(), **fields}
    logger.info(json.dumps(payload, default=str))


def _normalize_sid(raw: str | None) -> str | None:
    if not raw:
        return None
    sid = raw.strip()
    if not sid:
        return None
    if len(sid) > 128:
        return None
    if not all(ch.isalnum() or ch in "-_." for ch in sid):
        return None
    return sid


def _new_sid() -> str:
    return secrets.token_urlsafe(16)


@asynccontextmanager
async def lifespan(app):
    process_start = time.perf_counter()
    loop = asyncio.get_event_loop()
    preload_list = []
    if PRELOAD_DEMOS in ("all", "*"):
        preload_list = list(DEMO_DIRS.keys())
    elif PRELOAD_DEMOS in DEMO_DIRS:
        preload_list = [PRELOAD_DEMOS]

    for demo_name in preload_list:
        try:
            logger.info(f"Preloading {demo_name}...")
            await loop.run_in_executor(None, _ensure_demo_model_loaded, demo_name)
            logger.info(f"{demo_name} model ready")
        except Exception as e:
            logger.error(f"{demo_name} preload failed: {e}")
            _startup_state[demo_name]["error"] = str(e)
    _startup_state["_total_startup_seconds"] = round(time.perf_counter() - process_start, 3)
    yield


app = FastAPI(lifespan=lifespan)

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
BIPED_DIR = SCRIPT_DIR / "Locomotion" / "Biped"
QUADRUPED_DIR = SCRIPT_DIR / "Locomotion" / "Quadruped"

DEMO_DIRS = {
    "biped": BIPED_DIR,
    "quadruped": QUADRUPED_DIR,
}
WEBPROGRAM_FILES = {
    "biped": BIPED_DIR / "WebProgram.py",
    "quadruped": QUADRUPED_DIR / "WebProgram.py",
}

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BIPED_DIR))
sys.path.insert(0, str(QUADRUPED_DIR))

import torch
torch.set_num_interop_threads(1)

_cached_models = {}
_model_ready = {demo: False for demo in DEMO_DIRS}
_guidance_templates = {demo: {} for demo in DEMO_DIRS}
_startup_state = {demo: {"preload_seconds": None, "error": None} for demo in DEMO_DIRS}
_model_lock = threading.Lock()
_inference_lock = threading.Lock()

# --- Lease + queue (guarded by _state_lock) ---
# Single client identity: opaque sid passed in API/WS payloads.
# FIFO _queue holds sids waiting; at most one _lease_sid holds the demo slot.
_state_lock = threading.Lock()

_lease_sid: str | None = None
_lease_expires_at: float = 0.0
_lease_heartbeat_at: float = 0.0
_active_ws: WebSocket | None = None
_session_demo: str | None = None

_queue: list[str] = []
_wait_last_seen: dict[str, float] = {}


# --- Rate limiting (guarded by _state_lock) ---
# {ip: timestamp} for join endpoint, {ip: timestamp} for status endpoint
_rate_join: dict[str, float] = {}
_rate_status: dict[str, float] = {}


def _get_client_ip(request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if hasattr(request, "client") and request.client:
        return request.client.host
    return "unknown"


def _check_rate_limit(store: dict, ip: str, interval: float) -> bool:
    """Return True if the request is allowed. Must hold _state_lock."""
    now = time.time()
    cutoff = now - 300
    stale = [k for k, v in store.items() if v < cutoff]
    for k in stale:
        del store[k]
    last = store.get(ip, 0)
    if (now - last) < interval:
        return False
    store[ip] = now
    return True


def _prune_stale_queue():
    """Remove queued sids that have not polled recently. Must hold _state_lock."""
    now = time.time()
    before = list(_queue)
    _queue[:] = [
        sid for sid in _queue
        if (now - _wait_last_seen.get(sid, 0)) < TICKET_STALE_SECONDS
    ]
    after = set(_queue)
    for sid in before:
        if sid not in after:
            _wait_last_seen.pop(sid, None)


def _clear_lease():
    """Release the demo lease (session slot). Must hold _state_lock."""
    global _lease_sid, _lease_expires_at, _lease_heartbeat_at
    _lease_sid = None
    _lease_expires_at = 0.0
    _lease_heartbeat_at = 0.0


def _detach_dead_active_ws():
    """Drop stale WebSocket refs so dead connections cannot block the queue. Must hold _state_lock."""
    global _active_ws, _session_demo
    if _active_ws is None:
        return
    try:
        from starlette.websockets import WebSocketState

        if _active_ws.application_state == WebSocketState.DISCONNECTED:
            _log_queue_event("active_ws_detached", reason="disconnected")
            _active_ws = None
            _session_demo = None
    except Exception:
        _active_ws = None
        _session_demo = None
        _log_queue_event("active_ws_detached", reason="error")


def _invalidate_lease_if_abandoned():
    """Clear lease if expired or holder left no WS and no fresh heartbeat. Must hold _state_lock."""
    _detach_dead_active_ws()
    if _lease_sid is None:
        return
    now = time.time()
    if now >= _lease_expires_at:
        _clear_lease()
        _log_queue_event("lease_expired")
        return
    if _active_ws is not None:
        return
    if (now - _lease_heartbeat_at) < TICKET_STALE_SECONDS:
        return
    _clear_lease()
    _log_queue_event("lease_abandoned")


def _try_promote():
    """Assign the lease to the next queued sid if the slot is free. Must hold _state_lock."""
    global _lease_sid, _lease_expires_at, _lease_heartbeat_at, _active_ws, _session_demo
    _invalidate_lease_if_abandoned()
    if _lease_sid is not None:
        return
    if _active_ws is not None:
        _log_queue_event("orphan_active_ws_cleared")
        _active_ws = None
        _session_demo = None
    if _active_ws is not None:
        return
    if not _queue:
        return
    sid = _queue.pop(0)
    _wait_last_seen.pop(sid, None)
    _lease_sid = sid
    _lease_expires_at = time.time() + SESSION_TIMEOUT
    _lease_heartbeat_at = time.time()
    _log_queue_event("promoted", sid_prefix=sid[:4] + "…")


def _get_lease_remaining() -> float:
    if _lease_sid is None:
        return 0.0
    return max(0.0, _lease_expires_at - time.time())


def _estimate_wait(position: int) -> float:
    """Estimate wait in seconds for the given 1-based queue position."""
    base = 0.0
    if _lease_sid is not None:
        base = max(0.0, _lease_expires_at - time.time())
    return base + (position - 1) * SESSION_TIMEOUT


def _position_in_queue(sid: str) -> int:
    return _queue.index(sid) + 1


# --- Model loading ---

def _preload_model(demo_name: str):
    import torch

    demo_dir = DEMO_DIRS[demo_name]
    local_path = os.path.join(demo_dir, "Network.pt")
    device = "cpu"

    torch.set_num_threads(MAX_CPU_THREADS)

    preload_start = time.perf_counter()
    model = torch.load(local_path, weights_only=False, map_location=device)
    model.eval()

    guidances = {}
    guidances_dir = os.path.join(demo_dir, "Guidances")
    for path in sorted(os.listdir(guidances_dir)):
        with np.load(os.path.join(guidances_dir, path), allow_pickle=True) as data:
            id = Path(path).stem
            guidances[id] = {
                "Names": data["Names"].copy(),
                "Positions": data["Positions"].copy(),
            }
    _guidance_templates[demo_name] = guidances
    _cached_models[demo_name] = model
    _model_ready[demo_name] = True
    _startup_state[demo_name]["preload_seconds"] = round(time.perf_counter() - preload_start, 3)
    _startup_state[demo_name]["error"] = None


def _ensure_demo_model_loaded(demo_name: str):
    with _model_lock:
        if _model_ready.get(demo_name, False):
            return
        _preload_model(demo_name)


def _load_web_program_class(demo_name: str):
    module_path = WEBPROGRAM_FILES[demo_name]
    module_name = f"webprogram_{demo_name}"
    demo_dir = str(DEMO_DIRS[demo_name])
    for name in ("Definitions", "LegIK", "Sequence", module_name, "WebProgram"):
        if name in sys.modules:
            del sys.modules[name]
    if demo_dir in sys.path:
        sys.path.remove(demo_dir)
    sys.path.insert(0, demo_dir)
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.WebProgram


def _create_program(demo_name: str):
    from ai4animation import AI4Animation, Time

    with _inference_lock:
        Time.TotalTime = 0.0
        Time.DeltaTime = 0.0
        Time.Timescale = 1.0

        _ensure_demo_model_loaded(demo_name)

        WebProgram = _load_web_program_class(demo_name)
        program = WebProgram()
        program._preloaded_model = _cached_models[demo_name]
        program._preloaded_guidances = _guidance_templates[demo_name]
        AI4Animation(program, mode=AI4Animation.Mode.MANUAL)
        return program, AI4Animation


# --- Frame packing & simulation ---

def _pack_frame(program) -> bytes:
    frame_data = program.get_frame_data()
    (
        root_matrix,
        entity_matrices,
        contacts,
        sim_traj_pos,
        sim_traj_dir,
        ctrl_traj_pos,
        ctrl_traj_dir,
    ) = frame_data[:7]
    current_speed = (
        float(program.GetCurrentSpeed())
        if hasattr(program, "GetCurrentSpeed")
        else 0.0
    )

    root_matrix_f32 = np.asarray(root_matrix, dtype=np.float32)
    entity_matrices_f32 = np.asarray(entity_matrices, dtype=np.float32)
    contacts_f32 = np.asarray(contacts, dtype=np.float32).reshape(-1)
    sim_traj_pos_f32 = np.asarray(sim_traj_pos, dtype=np.float32)
    sim_traj_dir_f32 = np.asarray(sim_traj_dir, dtype=np.float32)
    ctrl_traj_pos_f32 = np.asarray(ctrl_traj_pos, dtype=np.float32)
    ctrl_traj_dir_f32 = np.asarray(ctrl_traj_dir, dtype=np.float32)
    speed_f32 = np.asarray([current_speed], dtype=np.float32)

    return (
        root_matrix_f32.tobytes()
        + entity_matrices_f32.tobytes()
        + contacts_f32.tobytes()
        + sim_traj_pos_f32.tobytes()
        + sim_traj_dir_f32.tobytes()
        + ctrl_traj_pos_f32.tobytes()
        + ctrl_traj_dir_f32.tobytes()
        + speed_f32.tobytes()
    )


def _precise_sleep(target_time):
    """Sleep until target frame time without CPU spin-wait."""
    remaining = target_time - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _neutralize_input(inp: dict) -> dict:
    """Return safe neutral input while preserving optional selectors."""
    neutral = dict(inp)
    neutral["left_stick"] = [0.0, 0.0]
    neutral["right_stick"] = [0.0, 0.0]
    neutral["speed_toggle"] = False
    neutral["canter_boost"] = False
    neutral["walk_modifier"] = False
    neutral["trot_modifier"] = False
    neutral["canter_modifier"] = False
    neutral["action_sit"] = False
    neutral["action_stand"] = False
    neutral["action_lie"] = False
    return neutral


def _tick_paced(
    ai4animation,
    program,
    input_lock,
    pending_input,
    pending_input_timestamp,
    next_frame_time,
):
    """Pace + tick in executor thread. Server drives the clock."""
    _precise_sleep(next_frame_time)

    with input_lock:
        inp = pending_input.copy()
        last_input_at = pending_input_timestamp[0]

    if (time.perf_counter() - last_input_at) > 0.25:
        inp = _neutralize_input(inp)

    with _inference_lock:
        program.set_inputs(**inp)
        ai4animation.Update(FRAME_DT)
        return _pack_frame(program)


async def _run_active_session(websocket: WebSocket, demo_name: str, sid: str):
    session_sid = sid
    loop = asyncio.get_event_loop()
    try:
        program, AI4Animation = await loop.run_in_executor(
            None, _create_program, demo_name
        )
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": f"Init failed: {e}"})
        return

    if not program._ready:
        await websocket.send_json({"type": "error", "message": "Failed to initialize"})
        return

    with _state_lock:
        remaining = _get_lease_remaining()

    await websocket.send_json({
        "type": "init",
        "styles": program.GuidanceNames,
        "entityNames": program.get_entity_names(),
        "entityCount": len(program.get_entity_names()),
        "sessionLimitSeconds": SESSION_TIMEOUT,
        "remainingSeconds": remaining,
        "avgInferenceMs": program.AvgInferenceMs,
        "demo": demo_name,
    })

    disconnected = asyncio.Event()

    input_lock = threading.Lock()
    pending_input = {
        "left_stick": [0.0, 0.0],
        "right_stick": [0.0, 0.0],
        "speed_toggle": False,
        "guidance_index": 0,
        "action_sit": False,
        "action_stand": False,
        "action_lie": False,
        "character_index": 0,
    }
    pending_input_timestamp = [time.perf_counter()]

    async def receive_inputs():
        nonlocal pending_input
        try:
            while not disconnected.is_set():
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("type") == "heartbeat":
                    continue
                with input_lock:
                    pending_input = data
                    pending_input_timestamp[0] = time.perf_counter()
        except (WebSocketDisconnect, RuntimeError):
            disconnected.set()

    async def send_frames():
        global _lease_heartbeat_at
        next_frame_time = time.perf_counter()
        frame_count = 0
        try:
            while not disconnected.is_set():
                with _state_lock:
                    remaining = _get_lease_remaining()
                    still_owner = _active_ws is websocket and _lease_sid == session_sid

                if remaining <= 0 or not still_owner:
                    try:
                        await websocket.send_json({
                            "type": "timeout",
                            "message": "Your session has expired.",
                        })
                    except Exception:
                        pass
                    disconnected.set()
                    break

                next_frame_time += FRAME_DT

                frame_bytes = await loop.run_in_executor(
                    None, _tick_paced, AI4Animation, program,
                    input_lock, pending_input, pending_input_timestamp, next_frame_time,
                )

                await websocket.send_bytes(frame_bytes)
                frame_count += 1

                if frame_count % 60 == 0:
                    try:
                        with _state_lock:
                            r = _get_lease_remaining()
                            _lease_heartbeat_at = time.time()
                        await websocket.send_json({
                            "type": "time_update",
                            "remainingSeconds": r,
                        })
                    except Exception:
                        pass

                if frame_count % 15 == 0:
                    try:
                        await websocket.send_json({
                            "type": "perf_update",
                            "avgInferenceMs": round(program.AvgInferenceMs, 2) if program.AvgInferenceMs is not None else None,
                        })
                    except Exception:
                        pass

                now = time.perf_counter()
                if now > next_frame_time + FRAME_DT:
                    next_frame_time = now
        except (WebSocketDisconnect, RuntimeError):
            disconnected.set()
        except Exception:
            traceback.print_exc()
            disconnected.set()

    await asyncio.gather(receive_inputs(), send_frames())


# --- HTTP queue endpoints ---

def _promoted_payload() -> dict:
    return {
        "promoted": True,
        "remainingSeconds": round(_get_lease_remaining()),
        "queueLength": len(_queue),
    }


def _waiting_payload(sid: str) -> dict:
    pos = _position_in_queue(sid)
    return {
        "promoted": False,
        "position": pos,
        "estimatedWaitSeconds": round(_estimate_wait(pos)),
        "queueLength": len(_queue),
    }


@app.post("/api/queue/join")
async def queue_join(request: Request):
    ip = _get_client_ip(request)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    raw_sid = payload.get("sid") if isinstance(payload, dict) else None
    sid = _normalize_sid(raw_sid) or _new_sid()
    with _state_lock:
        if not _check_rate_limit(_rate_join, ip, RATE_LIMIT_JOIN_SECONDS):
            _log_queue_event("rate_limited_join", ip=ip)
            return JSONResponse(
                {
                    "error": "rate_limited",
                    "message": "Too many requests. Please wait before trying again.",
                    "retry_after_seconds": RATE_LIMIT_JOIN_SECONDS,
                },
                status_code=429,
                headers={"Retry-After": str(RATE_LIMIT_JOIN_SECONDS)},
            )

        _prune_stale_queue()

        if sid == _lease_sid and time.time() < _lease_expires_at:
            data = _promoted_payload()
            data["sid"] = sid
            return data

        if sid in _queue:
            _try_promote()
            if sid == _lease_sid and time.time() < _lease_expires_at:
                _log_queue_event("join_promoted", sid_prefix=sid[:4] + "…")
                data = _promoted_payload()
                data["sid"] = sid
                return data
            _log_queue_event("join_waiting", sid_prefix=sid[:4] + "…", position=_position_in_queue(sid))
            data = _waiting_payload(sid)
            data["sid"] = sid
            return data

        if len(_queue) >= MAX_QUEUE_SIZE:
            _log_queue_event("queue_full", queue_len=len(_queue))
            return JSONResponse(
                {
                    "error": "queue_full",
                    "message": f"The queue is full ({MAX_QUEUE_SIZE} people waiting). Please try again later.",
                    "retry_after_seconds": RETRY_AFTER_QUEUE_FULL_SECONDS,
                },
                status_code=503,
                headers={"Retry-After": str(RETRY_AFTER_QUEUE_FULL_SECONDS)},
            )

        _queue.append(sid)
        _wait_last_seen[sid] = time.time()
        _try_promote()
        if sid == _lease_sid and time.time() < _lease_expires_at:
            _log_queue_event("join_new_promoted", sid_prefix=sid[:4] + "…")
            data = _promoted_payload()
            data["sid"] = sid
            return data
        pos = _position_in_queue(sid)
        _log_queue_event("join_queued", sid_prefix=sid[:4] + "…", position=pos)
        data = _waiting_payload(sid)
        data["sid"] = sid
        return data


@app.get("/api/queue/status")
async def queue_status(request: Request):
    global _lease_heartbeat_at
    ip = _get_client_ip(request)
    sid = _normalize_sid(request.query_params.get("sid"))
    if not sid:
        return JSONResponse(
            {
                "error": "sid_required",
                "message": "Missing sid. Rejoining...",
            },
            status_code=400,
        )
    with _state_lock:
        if not _check_rate_limit(_rate_status, ip, RATE_LIMIT_STATUS_SECONDS):
            _log_queue_event("rate_limited_status", ip=ip)
            return JSONResponse(
                {
                    "error": "rate_limited",
                    "retry_after_seconds": RATE_LIMIT_STATUS_SECONDS,
                },
                status_code=429,
                headers={"Retry-After": str(RATE_LIMIT_STATUS_SECONDS)},
            )

        _prune_stale_queue()

        if sid == _lease_sid and time.time() < _lease_expires_at:
            _lease_heartbeat_at = time.time()
            data = _promoted_payload()
            data["sid"] = sid
            return data

        if sid in _queue:
            _wait_last_seen[sid] = time.time()
            _try_promote()
            if sid == _lease_sid and time.time() < _lease_expires_at:
                data = _promoted_payload()
                data["sid"] = sid
                return data
            data = _waiting_payload(sid)
            data["sid"] = sid
            return data

        return JSONResponse(
            {
                "error": "not_enqueued",
                "reason": "not_enqueued",
                "message": "Not in queue. Rejoining...",
            },
            status_code=404,
        )


# --- WebSocket demo endpoint ---

@app.websocket("/ws/{demo_name}")
async def websocket_endpoint(websocket: WebSocket, demo_name: str):
    global _active_ws, _session_demo, _lease_heartbeat_at
    await websocket.accept()

    if demo_name not in DEMO_DIRS:
        await websocket.send_json({"type": "error", "message": f"Unknown demo: {demo_name}"})
        await websocket.close()
        return

    sid = _normalize_sid(websocket.query_params.get("sid"))
    with _state_lock:
        valid = (
            sid is not None
            and sid == _lease_sid
            and time.time() < _lease_expires_at
        )

    if not valid:
        _log_queue_event("ws_reject_no_lease", demo=demo_name)
        await websocket.send_json({
            "type": "busy",
            "message": "Your session has ended. Please return to the menu to join the queue.",
        })
        await websocket.close()
        return

    with _state_lock:
        if _active_ws is not None:
            reject = True
        else:
            reject = False
            _active_ws = websocket
            _session_demo = demo_name
            _lease_heartbeat_at = time.time()

    if reject:
        _log_queue_event("ws_reject_duplicate_tab", demo=demo_name)
        await websocket.send_json({
            "type": "busy",
            "message": "A demo is already running in another tab. Please close it first.",
        })
        await websocket.close()
        return

    _log_queue_event("ws_connected", demo=demo_name, sid_prefix=sid[:4] + "…")

    try:
        await _run_active_session(websocket, demo_name, sid)
    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        with _state_lock:
            if _active_ws is websocket:
                _active_ws = None
                _session_demo = None
            if sid:
                _lease_heartbeat_at = time.time()


@app.get("/api/health")
async def health():
    with _state_lock:
        return {
            "status": "ok",
            "demos": {
                name: {"model_ready": _model_ready[name]} for name in DEMO_DIRS
            },
            "active_demo": _session_demo,
            "lease_remaining": round(_get_lease_remaining()) if _lease_sid else None,
            "queue_length": len(_queue),
        }
