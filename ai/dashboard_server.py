import asyncio
import threading
import json
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="TruePneumoniaAI Dashboard")

_clients: Set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop = None

_HTML_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/")
async def get_dashboard():
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _clients.discard(websocket)
    except Exception:
        _clients.discard(websocket)


async def _do_broadcast(data: dict):
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


def broadcast(data: dict):
    if _loop is not None and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(_do_broadcast(data), _loop)


def start_background(host: str = "0.0.0.0", port: int = 8000):
    def run():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            loop="none",
            log_level="warning",
        )
        server = uvicorn.Server(config)
        _loop.run_until_complete(server.serve())

    t = threading.Thread(target=run, daemon=True, name="dashboard-server")
    t.start()
    print(f"[Dashboard] Serveur démarré → http://{host if host != '0.0.0.0' else 'localhost'}:{port}/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
