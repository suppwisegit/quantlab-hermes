"""Optionaler Planspiel-Recorder-Hintergrund-Tick für den Hermes-MCP-Server (E-H05/H4).

Der Hermes-MCP-Server (:mod:`quantlab.hermes.mcp_server`) läuft **ohne**
FastAPI-Lifespan — der Desktop-``planspiel_recorder``-Job des Schedulers steht
ihm deshalb nicht zur Verfügung. Dieses Modul gibt Hermes optional denselben
Recorder **ohne den ganzen Desktop-Scheduler**:

- **Env-Gate** :data:`ENV_AKTIV` (``QUANTLAB_HERMES_RECORDER``, Werte
  ``{"1", "true", "yes"}`` case-insensitive): nur dann startet
  :func:`starte_recorder_ticker` einen Task — sonst ``None`` (Default, der
  Desktop-Betrieb bleibt unberührt). Das Env ist bewusst **statt** des
  Settings-Schalters ``planspiel.recorder_aktiv`` das Gate: Auf dem Pi gibt es
  keine Desktop-UI, die den Settings-Schalter setzen würde.
- **Intervall** :data:`ENV_INTERVALL_S` (``QUANTLAB_HERMES_RECORDER_INTERVALL_S``):
  Basis-Takt in Sekunden, Default/Clamp analog dem Recorder-Fenster
  (:data:`recorder.STANDARD_INTERVALL_S` 60, 30–3600 s) — ungültige Werte
  fallen auf den Default bzw. werden ins Fenster geklemmt, nie in einen Absturz.
- **Wiederverwendung**: Der Tick ruft dieselbe Sweep-Logik wie der Scheduler
  (:func:`quantlab.services.planspiel.recorder.sweep`, Client über das
  Prozess-Singleton :func:`quantlab.services.planspiel.hole_client`) und das
  Handelszeiten-Gate 08:00–22:00 Europe/Berlin des Schedulers
  (:func:`quantlab.services.scheduler._in_handelszeiten`) — nichts dupliziert.
- **Fehler-Resilienz** analog :func:`quantlab.services.scheduler.tick_planspiel_recorder`:
  Fehlgeschlagene Sweeps (z. B. fehlende Credentials/Login-Fehler) werden
  geloggt (stderr, s. ``mcp_server._konfiguriere_logging``) und registrieren
  die Backoff-Fehler-Serie im Recorder-Modul
  (:func:`recorder.fehler_registrieren`); der Schlaf **dieser Runde** verlängert
  sich exponentiell (``basis × 2 ** min(serie, 4)``, Obergrenze 3600 s — Muster
  ``scheduler._job_schleife`` + :func:`recorder.naechstes_intervall_s`, hier mit
  Env-Basis). Der Task stürzt nie.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from quantlab.services import planspiel as planspiel_service
from quantlab.services.planspiel import fehler as planspiel_fehler
from quantlab.services.planspiel import recorder as recorder_service
from quantlab.services.scheduler import _in_handelszeiten

logger = logging.getLogger(__name__)

#: Env-Schalter des Recorder-Ticks (Werte ``1``/``true``/``yes``, case-insensitive).
ENV_AKTIV = "QUANTLAB_HERMES_RECORDER"

#: Env-Intervall (Sekunden) des Recorder-Ticks; Default/Clamp 30–3600 analog Recorder.
ENV_INTERVALL_S = "QUANTLAB_HERMES_RECORDER_INTERVALL_S"

#: Wahrheitswerte, die den Tick aktivieren.
_AKTIV_WERTE = frozenset({"1", "true", "yes"})


def _env_aktiv() -> bool:
    """Env-Gate: ``QUANTLAB_HERMES_RECORDER`` in {1,true,yes} (case-insensitive)."""
    return os.environ.get(ENV_AKTIV, "").strip().lower() in _AKTIV_WERTE


def intervall_aus_env() -> int:
    """Basis-Intervall aus :data:`ENV_INTERVALL_S`, defensiv geklemmt (30–3600 s).

    Ungültige Werte (falscher Typ/nicht numerisch/``NaN``) fallen auf den
    Default :data:`recorder_service.STANDARD_INTERVALL_S`, außerhalb des
    Fensters liegende werden geklemmt — Muster
    :func:`quantlab.services.planspiel.recorder.einstellungen`: Der Tick darf
    an einer verstümmelten Env nie sterben.
    """
    roh = os.environ.get(ENV_INTERVALL_S, "")
    try:
        wert = float(roh)
    except ValueError:
        wert = float(recorder_service.STANDARD_INTERVALL_S)
    if wert != wert:  # NaN — float("nan") parst erfolgreich
        wert = float(recorder_service.STANDARD_INTERVALL_S)
    return int(
        min(
            max(wert, float(recorder_service.MIN_INTERVALL_S)),
            float(recorder_service.MAX_INTERVALL_S),
        )
    )


def _standard_client() -> Any:
    """Client-Fabrik des Ticks (Injektionspunkt für Tests, Muster Scheduler)."""
    return planspiel_service.hole_client()


async def _schlafe(sekunden: float) -> None:
    """Injizierbarer Schlaf (Tests ersetzen ihn für deterministische Takte)."""
    await asyncio.sleep(sekunden)


async def _recorder_tick(now: Callable[[], datetime]) -> dict[str, Any]:
    """Ein Tick: Handelszeiten-Gate, dann Sweep über den Recorder-Service.

    Rückgabe-Muster :func:`quantlab.services.scheduler.tick_planspiel_recorder`
    (ohne dessen Settings-``aktiv``-Gate — das Env ist hier das Gate):
    ``ausser_handelszeiten``-Flag bei sofortiger Rückkehr, Fehler als
    ``ok=False``-Ergebnis inkl. Backoff-Stufe — nie als Exception.
    """
    jetzt = now()
    if not _in_handelszeiten(jetzt):
        return {
            "ok": True,
            "ausser_handelszeiten": True,
            "hinweis": (
                "Außerhalb der Handelszeiten (08:00–22:00 Europe/Berlin) — "
                "Hermes-Recorder-Sweep übersprungen."
            ),
        }
    try:
        details = await asyncio.to_thread(
            recorder_service.sweep, _standard_client(), jetzt=jetzt
        )
    except asyncio.CancelledError:
        raise
    except planspiel_fehler.PlanspielFehler as exc:
        serie = await asyncio.to_thread(recorder_service.fehler_registrieren)
        logger.warning(
            "Hermes-Recorder-Sweep fehlgeschlagen (%d. Serie): %s", serie, exc.message
        )
        return {"ok": False, "fehler": exc.message, "backoff_stufe": serie}
    except Exception as exc:  # noqa: BLE001 — Tick-Fehler: Ergebnis, nie Absturz
        serie = await asyncio.to_thread(recorder_service.fehler_registrieren)
        logger.exception("Hermes-Recorder-Sweep fehlgeschlagen (%d. Serie)", serie)
        return {
            "ok": False,
            "fehler": str(exc) or exc.__class__.__name__,
            "backoff_stufe": serie,
        }
    return {**details, "ok": True}


async def _ticker_schleife(
    basis_s: int,
    now: Callable[[], datetime],
) -> None:
    """Task-Leib des Ticks: sofortiger erster Sweep, danach (Backoff-)Takt.

    Muster :func:`quantlab.services.scheduler._job_schleife` (Zweig
    ``planspiel_recorder``): Nach fehlgeschlagenen Sweeps verlängert der Schlaf
    **dieser Runde** exponentiell (``basis × 2 ** min(serie, 4)``, Obergrenze
    :data:`recorder_service.MAX_INTERVALL_S`); ein erfolgreicher Sweep setzt
    die Serie zurück (:func:`recorder_service.sweep` selbst ruft dazu
    ``fehler_zuruecksetzen``).
    """
    while True:
        ergebnis = await _recorder_tick(now)
        if ergebnis.get("ok") and not ergebnis.get("ausser_handelszeiten"):
            takt_s = float(basis_s)
        else:
            serie = int(ergebnis.get("backoff_stufe") or 1)
            takt_s = min(
                float(basis_s) * 2 ** min(serie, 4),
                float(recorder_service.MAX_INTERVALL_S),
            )
        await _schlafe(takt_s)


async def starte_recorder_ticker(
    *,
    now: Callable[[], datetime] | None = None,
) -> asyncio.Task[None] | None:
    """Startet den Recorder-Tick-Task — nur wenn :data:`ENV_AKTIV` aktiv ist.

    Ohne Env-Gate ``None`` (Default — kein Task, kein Netz). Der erste Sweep
    läuft sofort, danach im Env-Intervall (bzw. Backoff). ``now`` ist
    injizierbar (Tests steuern das Handelszeiten-Gate deterministisch, Muster
    :func:`quantlab.services.scheduler.tick_planspiel_recorder`).
    """
    if not _env_aktiv():
        logger.info(
            "Hermes-Recorder-Tick nicht aktiviert (%s nicht gesetzt) — kein Task.",
            ENV_AKTIV,
        )
        return None
    basis_s = intervall_aus_env()
    uhr = now or (lambda: datetime.now(UTC))
    logger.info(
        "Hermes-Recorder-Tick aktiviert (Basis-Intervall %d s, Handelszeiten "
        "08:00–22:00 Europe/Berlin).",
        basis_s,
    )
    return asyncio.create_task(
        _ticker_schleife(basis_s, uhr), name="hermes-recorder-tick"
    )


async def stoppe_recorder_ticker(task: asyncio.Task[None] | None) -> None:
    """Bricht den Tick-Task ab und wartet sein Ende ab (sauberer Shutdown).

    Muster :func:`quantlab.services.scheduler.stoppe_scheduler`: Cancel +
    gather im selben Loop — kein „Task was destroyed but it is pending!“.
    """
    if task is None or task.done():
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


__all__ = [
    "ENV_AKTIV",
    "ENV_INTERVALL_S",
    "intervall_aus_env",
    "starte_recorder_ticker",
    "stoppe_recorder_ticker",
]
