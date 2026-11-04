"""Entrypoint des mobilen Sidecars (Welle M2a) — ``python -m quantlab.mobile_server``.

Startet die mobile App (:func:`quantlab.mobile_app.create_mobile_app`) via
uvicorn auf ``127.0.0.1:8765`` (Fix im Vergleich zum Desktop: kein freier
Port — die Capacitor-WebView der Android-App pocht genau darauf, E-M10).
``ensure_directories()`` + ``run_migrations()`` laufen vor dem Start (und
zusätzlich idempotent im Lifespan).

Chaquopy (M3): Die Funktion :func:`main` ist auch der Einstieg aus dem
Android-``Application.onCreate``-Thread — sie blockiert (uvicorn-Run) und
soll dort in einem Hintergrundthread laufen.
"""

from __future__ import annotations

import uvicorn

from quantlab.config import ensure_directories
from quantlab.migrations import run_migrations
from quantlab.mobile_app import create_mobile_app

#: Fixer Host/Port des mobilen Sidecars (WebView-Fallback, E-M10).
HOST = "127.0.0.1"
PORT = 8765


def main() -> None:
    """Startet den mobilen Sidecar (blockierend — Desktop wie Chaquopy)."""
    ensure_directories()
    run_migrations()
    uvicorn.run(create_mobile_app(), host=HOST, port=PORT)


if __name__ == "__main__":  # pragma: no cover — manueller Start
    main()
