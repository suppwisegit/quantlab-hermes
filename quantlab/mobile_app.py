"""Mobiles FastAPI-Profil (Welle M2a) — Android/Chaquopy-tauglicher Sidecar.

``create_mobile_app()`` baut eine FastAPI-App mit denselben Grundregeln wie
die Desktop-App (:mod:`quantlab.main`): CORS-Muster für Vite-Dev-Server/
Electron-``file://``/Capacitor (``https://localhost`` matcht das Regex,
E-M11), Haus-Fehlerformat (:func:`quantlab.errors.register_error_handlers`),
Lifespan mit ``ensure_directories()`` + ``run_migrations()``.

Bewusste Abweichungen zum Desktop (E-M08 / PLAN §M2a):

- **Router-Auswahl statt Komplett-Import:** Kein Backtest/Backtest-Analyse/
  Strategien/Alerts/Portfolios/Watchlists. Quant nur mit dem ``metrics``-
  Endpunkt (numpy+pandas), Data als OHLC-light-Subset, Fundamentals ohne
  Scheduler-Steuerung, Planspiel ohne Backfill-/Altbestand-Endpunkte.
- **Kein Scheduler** im Lifespan: Hintergrundjobs (Freshness, Recorder,
  Cron-Zeitpläne) sind Desktop-Features; der Agent-Router funktioniert ohne.
- **Mobile Tool-Registry** via
  :func:`quantlab.agent.tools.mobile_registry.registriere_mobile_tools` —
  der Agent-Chat bietet mobil nur den mobilen Katalog (kein run_backtest/
  manage_alerts/manage_portfolios).

Die Router-Module selbst bleiben unangetastet: unbekannte Pfade werden beim
Include herausgefiltert (Desktop-Router unverändert, minimal-invasiv).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quantlab import APP_VERSION
from quantlab.api.routers import (
    agent,
    charts,
    data,
    events,
    fundamentals,
    health,
    markets,
    planspiel,
    projects,
    quant,
    quotes,
    screener,
    settings,
)
from quantlab.config import APP_NAME, ensure_directories
from quantlab.errors import register_error_handlers
from quantlab.migrations import run_migrations

# Dasselbe Origin-Muster wie main.py (E-M11: matcht Capacitor https://localhost).
_CORS_ORIGIN_PATTERN = r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$"

#: Planspiel mobil OHNE diese Pfade (M2a-Vorgabe „ohne Backfill/Recorder“):
#: Historien-Backfill-Steuerung + Fortschritt und den Desktop-Altbestand-
#: Import (Pfad-API ist auf Android bedeutungslos). Recorder selbst ist kein
#: REST-Endpunkt, sondern Scheduler-Job — mobil läuft kein Scheduler.
_PLANSPIEL_AUSGENOMMEN: frozenset[str] = frozenset(
    {
        "/planspiel/universum/historie",
        "/planspiel/altbestand_import",
    }
)

#: Data als OHLC-light-Subset (M2a): Symbole verwalten/lesen, Suche, OHLC-
#: Lesen, Indikatoren für die Chart-Analyse. Downloads/Gaps/Actions/
#: Provider-Status bleiben Desktop-Sachen (Freigabe/Netz-Mechanik).
_DATA_LIGHT_BEHALTEN: frozenset[str] = frozenset(
    {
        "/data/symbols",
        "/data/search",
        "/data/ohlc",
        "/data/indicators",
    }
)

#: Fundamentals mobil ohne Scheduler-Steuerung (Hintergrundjob = Desktop).
_FUNDAMENTALS_AUSGENOMMEN: frozenset[str] = frozenset({"/scheduler/run-now"})


def _include_gefiltert(
    application: FastAPI, router: APIRouter, *, behalte: Callable[[str], bool]
) -> None:
    """Include eines Routers, aber nur mit den Pfaden, die ``behalte`` erlaubt.

    Die Route-Objekte werden unverändert übernommen (gleiche Handler/Schemas);
    der Desktop-Router selbst bleibt unberührt (minimal-invasiv, M2a).
    """
    auswahl = APIRouter()
    for route in router.routes:
        if behalte(route.path):  # type: ignore[attr-defined]
            auswahl.routes.append(route)
    application.include_router(auswahl, prefix="/api")


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_directories()
    run_migrations()
    # Bewusst KEIN scheduler.starte_scheduler(): Hintergrundjobs (Freshness/
    # Recorder/Cron) sind Desktop-Features — mobil läuft nur der Sidecar.
    yield


def create_mobile_app() -> FastAPI:
    """Erzeugt die mobile FastAPI-App (Router-Subset, CORS, Tools, Lifespan)."""
    from quantlab.agent.tools.mobile_registry import registriere_mobile_tools

    registriere_mobile_tools()
    application = FastAPI(
        title=f"{APP_NAME} Mobile", version=APP_VERSION, lifespan=_lifespan
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_CORS_ORIGIN_PATTERN,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(application)
    application.include_router(health.router, prefix="/api")
    application.include_router(settings.router, prefix="/api")
    application.include_router(agent.router, prefix="/api")
    _include_gefiltert(
        application, planspiel.router, behalte=lambda p: p not in _PLANSPIEL_AUSGENOMMEN
    )
    application.include_router(projects.router, prefix="/api")
    application.include_router(events.router, prefix="/api")
    application.include_router(screener.router, prefix="/api")
    # Quant mobil: nur der compute_metrics-Pfad (numpy+pandas, E-M08) —
    # Kovarianz/PCA/VaR/Portfolio brauchen sklearn/statsmodels/scipy.
    _include_gefiltert(application, quant.router, behalte=lambda p: p == "/quant/metrics")
    _include_gefiltert(application, data.router, behalte=_DATA_LIGHT_BEHALTEN.__contains__)
    _include_gefiltert(
        application,
        fundamentals.router,
        behalte=lambda p: p not in _FUNDAMENTALS_AUSGENOMMEN,
    )
    application.include_router(quotes.router, prefix="/api")
    application.include_router(markets.router, prefix="/api")
    application.include_router(charts.router, prefix="/api")
    return application
