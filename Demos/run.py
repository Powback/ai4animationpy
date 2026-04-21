# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import platform
import sys
import webbrowser
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
BIPED_DIR = SCRIPT_DIR / "Locomotion" / "Biped"
QUADRUPED_DIR = SCRIPT_DIR / "Locomotion" / "Quadruped"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BIPED_DIR))
sys.path.insert(0, str(QUADRUPED_DIR))

import uvicorn
from server import app
from fastapi.staticfiles import StaticFiles

CLIENT_DIR = str(SCRIPT_DIR / "client")
ASSETS_BIPED_DIR = str(SCRIPT_DIR / "_ASSETS_" / "Geno")
ASSETS_QUADRUPED_DIR = str(SCRIPT_DIR / "_ASSETS_" / "Quadruped")
app.mount("/assets/quadruped", StaticFiles(directory=ASSETS_QUADRUPED_DIR), name="assets-quadruped")
app.mount("/assets", StaticFiles(directory=ASSETS_BIPED_DIR), name="assets")
app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
WS_PING_INTERVAL = float(os.environ.get("WS_PING_INTERVAL", "20"))
WS_PING_TIMEOUT = float(os.environ.get("WS_PING_TIMEOUT", "20"))


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _log_startup_diagnostics() -> None:
    biped_model = ROOT_DIR / "Demos" / "Locomotion" / "Biped" / "Network.pt"
    biped_guidances = ROOT_DIR / "Demos" / "Locomotion" / "Biped" / "Guidances"
    quadruped_model = ROOT_DIR / "Demos" / "Locomotion" / "Quadruped" / "Network.pt"
    quadruped_guidances = ROOT_DIR / "Demos" / "Locomotion" / "Quadruped" / "Guidances"
    client_dir = SCRIPT_DIR / "client"

    print("[boot] Container startup diagnostics")
    print(f"[boot] Python: {platform.python_version()}")
    print(f"[boot] Platform: {platform.platform()}")
    print(f"[boot] CWD: {Path.cwd()}")
    print(f"[boot] SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"[boot] ROOT_DIR: {ROOT_DIR}")
    print(f"[boot] HOST={HOST} PORT={PORT}")
    print(
        "[boot] Env: "
        f"NO_BROWSER={os.environ.get('NO_BROWSER')} "
        f"DEBUG_CONTAINER={os.environ.get('DEBUG_CONTAINER')} "
        f"UVICORN_ACCESS_LOG={os.environ.get('UVICORN_ACCESS_LOG')} "
        f"PRELOAD_DEMOS={os.environ.get('PRELOAD_DEMOS')}"
    )
    print(f"[boot] Biped: model_exists={biped_model.exists()} model_size={biped_model.stat().st_size if biped_model.exists() else -1}")
    print(f"[boot] Biped: guidances_exists={biped_guidances.exists()} guidances_count={len(list(biped_guidances.glob('*.npz'))) if biped_guidances.exists() else -1}")
    print(f"[boot] Quadruped: model_exists={quadruped_model.exists()} model_size={quadruped_model.stat().st_size if quadruped_model.exists() else -1}")
    print(f"[boot] Quadruped: guidances_exists={quadruped_guidances.exists()} guidances_count={len(list(quadruped_guidances.glob('*.npz'))) if quadruped_guidances.exists() else -1}")
    print(
        f"[boot] Assets: client_exists={client_dir.exists()} "
        f"index_exists={(client_dir / 'index.html').exists()}"
    )


def main():
    if _bool_env("DEBUG_CONTAINER", True):
        _log_startup_diagnostics()

    if not os.environ.get("NO_BROWSER"):
        startup_url = f"http://localhost:{PORT}"
        webbrowser.open(startup_url)
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
        access_log=_bool_env("UVICORN_ACCESS_LOG", True),
        ws_ping_interval=WS_PING_INTERVAL,
        ws_ping_timeout=WS_PING_TIMEOUT,
    )


if __name__ == "__main__":
    main()
