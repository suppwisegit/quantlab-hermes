"""FastAPI-Anwendung des QuantLab-Sidecars.

Wird von der Electron-Shell auf ``127.0.0.1:<freier Port>`` gestartet (E004)
oder standalone via ``npm run backend`` (Port 8765, Browser-Fallback-Modus).
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import quantlab.api.routers.alerts as alerts  # T33: Alert-CRUD/Historie/evaluate (additiv)
import quantlab.api.routers.backtest_analyse as backtest_analyse  # B2/E059 (additiv)
import quantlab.api.routers.charts as charts  # T26: Zeichentools-Persistenz (additiv)
import quantlab.api.routers.markets as markets  # T28: Märkte-Übersicht/Mover (additiv)
import quantlab.api.routers.projects as projects  # T7: Projekt-System (additiv)
import quantlab.api.routers.screener as screener  # T32: Screener/Presets/SP-500-Import (additiv)
import quantlab.api.routers.strategies as strategies  # T30: Strategie-Editor/Sandbox (additiv)
import quantlab.api.routers.watchlists as watchlists  # T28: Multi-Watchlists (additiv)
from quantlab import APP_VERSION
from quantlab.api.routers import (  # T8 + T10  # T4: Daten-/Events-Router (additiv)
    agent,
    backtest,
    data,
    events,
    fundamentals,  # T21: Fundamentals/Freshness/Scheduler (additiv)
    health,
    planspiel,  # Welle P/E072: Planspiel-Börse (additiv)
    portfolios,  # T23: Depot-/Performance-Router (additiv)
    quant,
    quotes,  # T22: Live-Quotes (additiv)
    settings,
)
from quantlab.config import APP_NAME, ensure_directories
from quantlab.errors import register_error_handlers
from quantlab.migrations import run_migrations
from quantlab.services import scheduler  # T21: Lifespan-Scheduler (E035)

# Renderer-Origins: Vite-Dev-Server (http://localhost:<port>) und Electron
# pakettiert (file:// sendet Origin "null"). Fachliche REST-Calls gehen direkt
# vom Renderer an das Sidecar (E003) — daher CORS statt Main-Proxy.
_CORS_ORIGIN_PATTERN = r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$"

# E017: Intervall der Parent-Lebendigkeitsprüfung in Sekunden.
_PARENT_CHECK_INTERVAL_S = 2.0


def get_parent_pid() -> int | None:
    """Parent-PID aus der Env der Electron-Shell (E017) — ``None`` wenn nicht gesetzt."""
    raw = os.environ.get("QUANTLAB_PARENT_PID", "").strip()
    return int(raw) if raw.isdigit() else None


def _request_shutdown() -> None:
    """Fordert uvicorn zum sauberen Herunterfahren auf (SIGTERM an den eigenen Prozess)."""
    os.kill(os.getpid(), signal.SIGTERM)


async def watch_parent_pid(
    parent_pid: int,
    *,
    interval: float = _PARENT_CHECK_INTERVAL_S,
    on_parent_gone: Callable[[], None] = _request_shutdown,
) -> None:
    """Orphan-Watchdog (E017): prüft zyklisch, ob der Electron-Main noch lebt.

    Stirbt der Parent (SIGKILL/Crash — das Sidecar würde sonst als Waisenkind
    zu PID 1 re-parentiert und ewig weiterlaufen), wird ``on_parent_gone``
    aufgerufen, was im Betrieb den Server-Shutdown auslöst. ``os.kill(pid, 0)``
    liefert keine Signale, meldet nur die Existenz; ``PermissionError`` heißt
    „Prozess existiert, gehört aber einem anderen Benutzer" — kein Orphan.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            on_parent_gone()
            return
        except PermissionError:
            continue


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_directories()
    # Persistenz-Kern (T2): Schema beim Start idempotent auf den aktuellen Stand
    # bringen — danach existiert quantlab.db garantiert im Datenverzeichnis.
    run_migrations()
    parent_pid = get_parent_pid()
    watcher: asyncio.Task[None] | None = None
    if parent_pid is not None:
        # on_parent_gone/interval explizit (Laufzeit-Lookup), damit Tests sie
        # per Monkeypatch auf dem Modul ersetzen können.
        watcher = asyncio.create_task(
            watch_parent_pid(
                parent_pid,
                interval=_PARENT_CHECK_INTERVAL_S,
                on_parent_gone=_request_shutdown,
            )
        )
    # T21/E035: Scheduler-Jobs als Lifespan-Tasks (ein Task je aktiver Job-Art).
    # Stop im Finally mit Abwartung — Muster Watchdog (kein pending-Task-Leak).
    await scheduler.starte_scheduler()
    try:
        yield
    finally:
        await scheduler.stoppe_scheduler()
        if watcher is not None and not watcher.done():
            watcher.cancel()


def create_app() -> FastAPI:
    """Erzeugt die FastAPI-App (Router, CORS, Fehlerformat, Lifespan)."""
    application = FastAPI(title=f"{APP_NAME} Sidecar", version=APP_VERSION, lifespan=_lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_CORS_ORIGIN_PATTERN,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(application)
    application.include_router(health.router, prefix="/api")
    application.include_router(settings.router, prefix="/api")
    application.include_router(agent.router, prefix="/api")  # T12
    application.include_router(quant.router, prefix="/api")  # T8
    application.include_router(backtest.router, prefix="/api")  # T10
    # B2/E059: Monte-Carlo-Analyse (additiv)
    application.include_router(backtest_analyse.router, prefix="/api")
    application.include_router(projects.router, prefix="/api")  # T7
    application.include_router(data.router, prefix="/api")  # T4 (additiv)
    application.include_router(events.router, prefix="/api")  # T4 (additiv)
    application.include_router(fundamentals.router, prefix="/api")  # T21 (additiv)
    application.include_router(portfolios.router, prefix="/api")  # T23 (additiv)
    application.include_router(quotes.router, prefix="/api")  # T22 (additiv)
    application.include_router(charts.router, prefix="/api")  # T26: Zeichnungen (additiv)
    application.include_router(watchlists.router, prefix="/api")  # T28: Watchlists (additiv)
    application.include_router(markets.router, prefix="/api")  # T28: Märkte (additiv)
    application.include_router(strategies.router, prefix="/api")  # T30: User-Strategien (additiv)
    application.include_router(screener.router, prefix="/api")  # T32: Screener (additiv)
    application.include_router(alerts.router, prefix="/api")  # T33: Alerts (additiv)
    application.include_router(planspiel.router, prefix="/api")  # Welle P/E072 (additiv)
    return application


app = create_app()
