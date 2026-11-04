"""Auto-Update-Scheduler (T21/E035): periodische Aufgaben im Sidecar-Lifespan.

Alle periodischen Jobs laufen als **asyncio-Tasks im FastAPI-Lifespan** (ein
Task je Job-Art, E035) — kein separater Thread/Prozess:

- ``data_update``: inkrementeller 1d-Download **aller Watchlist-Symbole**
  (Default-Liste 1) über den Download-Job-Manager — der Inkrement-Start
  ``last_ts − 5 Balken`` (§D.2) macht jeden Tick idempotent und netzarm.
  Intervall: ``scheduler.data_update_interval_h`` (Default 6 h).
- ``fundamentals``: Fundamentals-Refresh der Watchlist-Symbole, **seltener**
  als Kurse — pro Symbol entscheidet der Freshness-Marker über
  ``scheduler.fundamentals_refresh_days`` (Default 7 Tage) — und ``run-now``
  erzwingt einen sofortigen Refresh (Tests/Agent).
- ``quotes`` (T22/E024): Live-Quotes-Poll der Watchlist über den Quotes-Service
  (TTL-Cache/Single-Flight dort) mit ``quotes``-SSE **nur bei geänderten
  Werten**; Intervall ``live.quotes_poll_s`` (15–300 s, Default 60).
- ``alerts`` (T33): Alert-Evaluation (:mod:`quantlab.services.alerts`) über
  alle aktiven Alerts — Quotes (TTL) + Parquet-Fallback, Auslöser schreiben
  ``alert_events`` + ``alert_triggered``-SSE; Intervall ``alerts_check_s``
  (Default 60 s). Ohne relevante Alerts Netto-No-Op (kein Abruf).
- ``agent_cron`` (Welle F/E080): weckt in festem Poll-Rhythmus (60 s) den
  Agent-Loop für **fällige** Aufgaben aus der Settings-Sektion ``agent_cron``
  (Intervall je Aufgabe gegen den gespeicherten letzten Lauf; ``run-now``
  erzwingt). Fällige Aufgaben laufen als normale Agent-Runs (Chat-Historie)
  mit modus-gefilterter Tool-Registry (E081) und ``system_prompt_zusatz``;
  nach Run-Ende werden ``agent_cron``-SSE + letzter Lauf persistiert. Ohne
  fällige/aktivierte Aufgabe Netto-No-Op — der Job startet deshalb **immer**
  (neue Aufgaben greifen ohne Sidecar-Neustart).
- ``download_retry`` (E089/T-G3): weckt in festem Poll-Rhythmus (60 s) die
  Auto-Wiederholung transient fehlgeschlagener Downloads (429/502/
  ``stooq_blockiert``/``stooq_limit``) — die eigentliche Fälligkeit
  (``naechster_versuch_us`` je Plan-Zeile in ``download_retries``) liegt im
  Tick (:func:`tick_download_retry` → ``downloads.faellige_retries_starten``),
  ``run-now`` umgeht das Fälligkeits-Gate. Ohne fällige Zeilen Netto-No-Op —
  der Job startet deshalb **immer** (Wiederholungen greifen ohne Neustart).
- ``planspiel_recorder`` (Welle H/E096): fährt in adaptivem Rhythmus
  (:data:`quantlab.services.planspiel.recorder.einstellungen`, 30–3600 s)
  einen PSB-Tab-Sweep + Fokus-Quotes und schreibt 1m-Bars (quelle ``psb``)
  in den Parquet-Store. Das Handelszeiten-Gate 08:00–22:00 Europe/Berlin
  (:func:`_in_handelszeiten`, injizierbare ``now``) liefert außerhalb eine
  sofort-Rückkehr mit Ergebnis-Flag (keine Drossel-Schleife); nach
  fehlgeschlagenen Sweeps verlängert exponentieller Backoff den Takt.
  Ohne ``planspiel.recorder_aktiv`` (Default **false** — der Nutzer
  aktiviert bewusst) Netto-No-Op — der Job startet deshalb **immer**
  (Aktivierung greift ohne Sidecar-Neustart).

Jeder Tick emittiert ``scheduler_tick`` auf den globalen SSE-Bus (§C.5:
``{job, ok, durationMs, error?, nextRunAt?}`). ``POST /api/scheduler/run-now
{job}`` (Router) führt **genau einen** Tick synchron aus und liefert dasselbe
Payload-Format als Antwort; läuft bereits ein Tick derselben Job-Art, wird er
vorher abgewartet (**Single-Flight je Job-Art** — nie zwei gleichzeitige
Durchgänge, s. :func:`_job_lock`).

Lebenszyklus: :func:`starte_scheduler` (aus ``main._lifespan``) startet die
Tasks der aktivierten Jobs — der **erste Tick läuft sofort** (die Frische-Gates
machen ihn netzarm; beim App-Start sollen Daten aktuell werden), danach im
konfigurierten Intervall. :func:`stoppe_scheduler` bricht beim Shutdown ab und
wartet die Tasks ab (Muster: Watchdog in ``main.py``) — sauberes Task-Ende
ohne „Task was destroyed but it is pending!“-Warnungen.

Einstellungen werden **je Runde neu gelesen** (Settings-Sektion ``scheduler``
— Änderungen greifen ohne Neustart; T22 ergänzt ``live``-Settings analog).
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from quantlab import events
from quantlab.providers.base import ProviderError
from quantlab.services import alerts as alerts_service
from quantlab.services import downloads
from quantlab.services import fundamentals as fundamentals_service
from quantlab.services import quotes as quotes_service
from quantlab.services.planspiel import fehler as planspiel_fehler
from quantlab.services.planspiel import recorder as planspiel_recorder_service
from quantlab.storage.repos import (
    chat_repo,
    downloads_repo,
    freshness_repo,
    settings_repo,
    watchlists_repo,
)

if TYPE_CHECKING:
    from quantlab.agent.llm_client import LLMClient
    from quantlab.agent.registry import ToolRegistry
    from quantlab.services.planspiel.client import PlanspielClient

# WICHTIG (Import-Zyklus, F-A0 §5): ``agent.tools.daten`` liest
# ``scheduler.VERFUEGBARE_JOBS`` auf Modul-Ebene. Der Scheduler darf deshalb
# ``quantlab.agent.*`` NICHT auf Modul-Ebene importieren (scheduler →
# agent.loop → tools/__init__ → daten → partielles scheduler-Modul ohne die
# Job-Tupel). Alle Agent-Importe erfolgen daher funktionslokal.

logger = logging.getLogger(__name__)

#: Job-Arten des Vertrags (§C.4 ``run-now``) — seit T33 alle implementiert,
#: seit T-F1 inkl. Agent-Cron, seit E089/T-G3 inkl. Auto-Retry, seit Welle H
#: inkl. Planspiel-Recorder.
BEKANNTE_JOBS: tuple[str, ...] = (
    "data_update",
    "fundamentals",
    "quotes",
    "alerts",
    "agent_cron",
    "download_retry",
    "planspiel_recorder",
)

#: Jobs mit implementiertem Tick (startbar/lauffähig) — seit T33 inkl. Alerts,
#: seit T-F1 inkl. Agent-Cron, seit E089/T-G3 inkl. Auto-Retry, seit Welle H
#: inkl. Planspiel-Recorder.
VERFUEGBARE_JOBS: tuple[str, ...] = (
    "data_update",
    "fundamentals",
    "quotes",
    "alerts",
    "agent_cron",
    "download_retry",
    "planspiel_recorder",
)

#: SSE-Event-Name (§C.5) — Publisher-Helfer in ``events.py``, sobald weitere
#: Quellen das Event senden (T22-Muster); hier bewusst direktes ``publizieren``.
SCHEDULER_TICK = "scheduler_tick"

#: SSE-Event nach Run-Ende einer Cron-Aufgabe (T-F1, F-A0-D1): das Backend
#: emittiert nur das Event + Settings-Persistenz — die OS-Notification ist
#: Renderer-Sache (T-F2 abonniert ``agent_cron`` und ruft ``notifyAlert``).
AGENT_CRON_EVENT = "agent_cron"

#: Preset-Tasktexte der Cron-Aufgaben (Single Source; die Settings-UI, T-F2,
#: hält nur die Labels). ``custom`` nutzt ``aufgabe_prompt`` statt eines Texts.
AGENT_CRON_PRESETS: dict[str, str] = {
    "depot_check": (
        "Prüfe das Depot: Lies die aktuellen Bestände und Werte (Depot- bzw. "
        "Portfolio-Tools, Planspiel-Depot falls verbunden), stelle die "
        "Veränderung seit dem letzten Stand fest und fasse Auffälligkeiten "
        "kompakt zusammen (größte Gewinner/Verlierer, ungewöhnliche Bewegungen, "
        "offene Orders)."
    ),
    "markt_wachposten": (
        "Beobachte die Märkte: Hole die aktuellen Kurse/Quotes der relevanten "
        "Symbole (Watchlist), identifiziere markante Bewegungen (starke "
        "Tagesveränderungen, neue Hochs/Tiefs, ungewöhnliche Volumina) und "
        "melde sie kompakt mit Zahlen."
    ),
    "rebalance_vorschlag": (
        "Erstelle einen Rebalance-Vorschlag: Vergleiche die aktuelle "
        "Depotstruktur mit einer sinnvollen Zielgewichtung, benenne die "
        "Abweichungen und formuliere konkrete Orderkandidaten (Instrument, "
        "Richtung, Menge, Ordertyp) als TEXT — platziere selbst keine Orders."
    ),
}

#: Freshness-Domain der Kurse (Staleness-Marker für den data_update-Tick).
DOMAIN_DAILY_BARS = "daily_bars"

#: Fallback-Defaults, falls die Settings-Sektion fehlt (Additivität schützt).
_FALLBACK_EINSTELLUNGEN: dict[str, Any] = {
    "data_update_enabled": True,
    "data_update_interval_h": 6.0,
    "fundamentals_enabled": True,
    "fundamentals_refresh_days": 7.0,
    "alerts_check_s": 60,
}

#: Untergrenzen (Schutz vor 0/negativen Intervallen aus ungültigen Settings).
_MIN_INTERVAL_H = 0.0005  # 1,8 s — schnell genug für Tests, harmlos im Betrieb
_MIN_REFRESH_DAYS = 0.0005
_MIN_ALERTS_CHECK_S = 1

#: Timeout des data_update-Ticks auf den Download-Job (Watchlists sind klein;
#: Pacing limitiert eh — der Timeout ist ein Netz gegen Hänger).
_JOB_TIMEOUT_S = 600.0

# ---------------------------------------------------------------------------
# Agent-Cron (T-F1/E080) — Konstanten
# ---------------------------------------------------------------------------

#: Untergrenze des Aufgaben-Intervalls (Minuten; Zweit-Verteidigung neben der
#: Router-Validierung — defekte Settings dürfen den Tick nie kippen).
_MIN_INTERVALL_MIN = 5.0

#: Fester Poll-Rhythmus des Loop-Tasks (Sekunden) — die eigentliche Fälligkeit
#: (Intervall je Aufgabe gegen den gespeicherten letzten Lauf) liegt im Tick,
#: damit neue/geänderte Aufgaben ohne Sidecar-Neustart greifen.
_AGENT_CRON_POLL_S = 60.0

#: Fester Poll-Rhythmus des Auto-Retry-Jobs (Sekunden, E089/T-G3) — die
#: eigentliche Fälligkeit (``naechster_versuch_us`` je Plan-Zeile) liegt im
#: Tick; gegen die Backoff-Stufen 2/5/10/20/30 min bleibt der Drift ≤ 60 s.
_DOWNLOAD_RETRY_POLL_S = 60.0

#: Planspiel-Handelszeiten (08:00–22:00 Europe/Berlin) für ``nur_handelszeiten``.
_HANDELSZEIT_START_STUNDE = 8
_HANDELSZEIT_ENDE_STUNDE = 22
_HANDELSZEIT_ZONE = "Europe/Berlin"

#: Kappung des Kurzergebnisses (letzte-Lauf-Eintrag + SSE-Payload).
_KURZ_MAX_ZEICHEN = 280

#: Hartes Netz gegen einen hängenden Cron-Run: LLM-Requests tragen je einen
#: Client-Timeout, max. 25 Iterationen — der Tick hält derweil den Job-Lock
#: und ``run_now('agent_cron')`` wartet gewollt (F-A0 §1.1). Das Limit hier
#: ist der Notausgang gegen einen dauerhaft blockierten Scheduler-Task.
_RUN_TIMEOUT_S = 1800.0

#: Schreib-Tools, die Cron-Runs (Modus analyse/vorschlag) nie sehen — der Guard
#: baut eine GEFILTERTE :class:`ToolRegistry`-Instanz; GLOBAL_REGISTRY bleibt
#: unangetastet (F-A0 §1.3). ANNAHME (F-A0 §7): Blocklist statt Whitelist —
#: neue lesende Tools (z. B. Websuche, T-F6) stehen Cron-Runs damit automatisch
#: zur Verfügung; ``platziere_order`` kommt zusätzlich nur im Modus „trade“ mit
#: erteilter Freigabe in die Registry.
_SCHREIB_TOOLS: frozenset[str] = frozenset(
    {
        "platziere_order",  # Orders — nur Modus „trade“ mit Freigabe
        "verwalte_zeitplaene",  # H7/E099: ein Cron-Lauf konfiguriert sich nicht selbst um
        "record_transaction",  # Depot-Buchungen
        "manage_portfolios",  # Depot-Anlage/-Änderung
        "manage_watchlists",  # Watchlist-Änderung
        "manage_alerts",  # Alert-Anlage/-Änderung
        "save_strategy",  # Strategie-Artefakte
        "save_report",  # Projekt-Artefakte
        "manage_tasks",  # Projekt-Tasks
        "create_chart",  # Projekt-Artefakte
        "manage_projects",  # Welle I/E102: Projekt-Anlage/-Änderung/-Löschen
        "manage_notes",  # Welle I/E102: Notiz-Schreiben
        "manage_chart_drawings",  # Welle I/E103: Chart-Zeichnungen
        "save_analyse",  # Welle I/E103: Analyse-Artefakte
        "download_data",  # Download-Jobs (Netz- + Parquet-Schreibwirkung)
        "trigger_data_update",  # Scheduler-Fernsteuerung (Download-Jobs)
    }
)

#: Fallback-Defaults der agent_cron-Sektion (Additivität schützt).
_FALLBACK_AGENT_CRON: dict[str, Any] = {"aufgaben": [], "letzte_laeufe": {}}


def agent_cron_einstellungen() -> dict[str, Any]:
    """Settings-Sektion ``agent_cron`` (Defaults + Overrides), defensiv getypt.

    Defekte Einträge (keine Liste/kein Objekt) fallen auf die Defaults statt
    den Tick zu kippen — Hintergrundjobs dürfen an einer verstümmelten
    Einstellung nie sterben (Muster :func:`scheduler_einstellungen`).
    """
    try:
        sektion = settings_repo.get_section("agent_cron")
    except ValueError:  # pragma: no cover — Defaults sind ab T-F1 vorhanden
        sektion = dict(_FALLBACK_AGENT_CRON)
    aufgaben = sektion.get("aufgaben")
    if not isinstance(aufgaben, list):
        aufgaben = []
    letzte_laeufe = sektion.get("letzte_laeufe")
    if not isinstance(letzte_laeufe, dict):
        letzte_laeufe = {}
    return {
        "aufgaben": [a for a in aufgaben if isinstance(a, dict)],
        "letzte_laeufe": letzte_laeufe,
    }


def scheduler_einstellungen() -> dict[str, Any]:
    """Settings-Sektion ``scheduler`` (Defaults + Overrides), defensiv getypt.

    Ungültige Werte (falscher Typ/negativ) werden auf Defaults bzw. Untergrenzen
    geklemmt statt den Scheduler zu kippen — Hintergrundjobs dürfen an einer
    verstümmelten Einstellung nie sterben („Warum sind meine Daten alt?“).
    """
    try:
        sektion = settings_repo.get_section("scheduler")
    except ValueError:  # pragma: no cover — Defaults sind ab T21 vorhanden
        sektion = dict(_FALLBACK_EINSTELLUNGEN)

    def als_float(schluessel: str) -> float:
        wert = sektion.get(schluessel, _FALLBACK_EINSTELLUNGEN[schluessel])
        try:
            return float(wert)
        except (TypeError, ValueError):
            return float(_FALLBACK_EINSTELLUNGEN[schluessel])

    def als_bool(schluessel: str) -> bool:
        wert = sektion.get(schluessel, _FALLBACK_EINSTELLUNGEN[schluessel])
        return bool(wert) if isinstance(wert, bool) else bool(_FALLBACK_EINSTELLUNGEN[schluessel])

    return {
        "data_update_enabled": als_bool("data_update_enabled"),
        "data_update_interval_h": max(
            als_float("data_update_interval_h"), _MIN_INTERVAL_H
        ),
        "fundamentals_enabled": als_bool("fundamentals_enabled"),
        "fundamentals_refresh_days": max(
            als_float("fundamentals_refresh_days"), _MIN_REFRESH_DAYS
        ),
        "alerts_check_s": int(
            max(als_float("alerts_check_s"), _MIN_ALERTS_CHECK_S)
        ),
    }


def watchlist_symbole() -> list[str]:
    """Symbole der Default-Watchlist (Liste 1, Reihenfolge = Position)."""
    return [
        eintrag.symbol
        for eintrag in watchlists_repo.list_symbols(watchlists_repo.DEFAULT_WATCHLIST_ID)
    ]


# ---------------------------------------------------------------------------
# Laufzeitzustand je Event-Loop (Muster downloads._LOOPZUSTAENDE)
# ---------------------------------------------------------------------------


@dataclass
class _SchedulerZustand:
    """Tasks + Single-Flight-Locks der Jobs im aktuellen Loop (Shutdown-räumbar)."""

    tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)


_LOOPZUSTAENDE: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _SchedulerZustand] = (
    weakref.WeakKeyDictionary()
)


def _loop_zustand() -> _SchedulerZustand:
    loop = asyncio.get_running_loop()
    zustand = _LOOPZUSTAENDE.get(loop)
    if zustand is None:
        zustand = _SchedulerZustand()
        _LOOPZUSTAENDE[loop] = zustand
    return zustand


def _job_lock(job: str) -> asyncio.Lock:
    """Single-Flight-Lock einer Job-Art im aktuellen Loop (QA-Fund c(ii)).

    ``run_now`` während eines laufenden Ticks derselben Job-Art startete
    einen parallelen Doppel-Durchgang (2 gleichzeitige Refreshes). Jeder
    Tick-Runner — Loop-Tick wie ``run_now`` — erwirbt das Lock der Job-Art;
    der zweite wartet auf das Ende des ersten (**Sequenzierung** statt
    Ignorieren: ``run_now`` ist ein expliziter Auftrag, sein Ergebnis —
    Payload/Event — gehört dem Aufrufer, nicht dem zufällig laufenden Tick).
    """
    return _loop_zustand().locks.setdefault(job, asyncio.Lock())


def laufende_jobs() -> list[str]:
    """Namen der aktuell laufenden Scheduler-Tasks (Diagnose/Tests, DoD c)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return []
    zustand = _LOOPZUSTAENDE.get(loop)
    if zustand is None:
        return []
    return sorted(job for job, task in zustand.tasks.items() if not task.done())


# ---------------------------------------------------------------------------
# Ticks (idempotent + netzarm; jeder emittiert scheduler_tick)
# ---------------------------------------------------------------------------


async def tick_data_update() -> dict[str, Any]:
    """Inkrementeller 1d-Download aller Watchlist-Symbole (ein Download-Job).

    Nach erfolgreichem Job (``done`` — Teilerfolge zählen) wird der Staleness-
    Marker ``daily_bars`` je Symbol gesetzt. Keine Symbole → No-Op (ok) —
    auch der Lifespan-Starttick einer frischen Datenbank bleibt netzfrei.
    """
    symbole = await asyncio.to_thread(watchlist_symbole)
    if not symbole:
        return {"ok": True, "symbole": 0, "hinweis": "Watchlist ist leer — kein Tick nötig."}
    job = await downloads.submit_job(symbole, "1d")
    fertig = await downloads.warte_auf_abschluss(job.job_id, timeout_s=_JOB_TIMEOUT_S)
    if fertig is None:
        return {
            "ok": False,
            "symbole": len(symbole),
            "job_id": job.job_id,
            "fehler": "Download-Job wurde nicht rechtzeitig fertig (Timeout).",
        }
    ok = fertig.status == downloads_repo.STATUS_DONE
    if ok:
        for symbol in symbole:
            await asyncio.to_thread(
                freshness_repo.mark_fresh, DOMAIN_DAILY_BARS, symbol
            )
    return {
        "ok": ok,
        "symbole": len(symbole),
        "job_id": job.job_id,
        "fehler": fertig.error,
    }


async def tick_fundamentals(*, erzwingen: bool = False) -> dict[str, Any]:
    """Fundamentals-Refresh der Watchlist — nur Symbole mit altem Marker.

    ``erzwingen=True`` (``run-now``) überspringt das Frische-Gate. Fehler je
    Symbol isolieren (ein Symbol kippt nie den Tick); alle fehlgeschlagen →
    ``ok=False``.
    """
    einstellungen = await asyncio.to_thread(scheduler_einstellungen)
    alter_tage = einstellungen["fundamentals_refresh_days"]
    symbole = await asyncio.to_thread(watchlist_symbole)
    if not symbole:
        return {"ok": True, "symbole": 0, "hinweis": "Watchlist ist leer — kein Tick nötig."}

    fehler: dict[str, str] = {}
    aktualisiert: list[str] = []
    uebersprungen: list[str] = []
    for symbol in symbole:
        marker = await asyncio.to_thread(
            freshness_repo.get_freshness, fundamentals_service.DOMAIN_FUNDAMENTALS, symbol
        )
        if not erzwingen and marker is not None and _alter_tage(marker) < alter_tage:
            uebersprungen.append(symbol)
            continue
        try:
            await asyncio.to_thread(fundamentals_service.aktualisiere_symbol, symbol)
            aktualisiert.append(symbol)
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            fehler[symbol] = exc.message
        except Exception as exc:  # noqa: BLE001 — Isolation: ein Symbol kippt nie den Tick
            logger.exception("Fundamentals-Refresh für %s fehlgeschlagen", symbol)
            fehler[symbol] = str(exc) or exc.__class__.__name__

    ok = len(fehler) < len(symbole)
    return {
        "ok": ok,
        "symbole": len(symbole),
        "aktualisiert": aktualisiert,
        "uebersprungen": uebersprungen,
        "fehler": "; ".join(f"{s}: {m}" for s, m in sorted(fehler.items())) or None,
    }


# ---------------------------------------------------------------------------
# Agent-Cron (T-F1/E080): fällige Aufgaben als modus-gefilterte Agent-Runs
# ---------------------------------------------------------------------------


def _standard_now() -> datetime:
    """Injizierbare Uhr (Tests) — Standard: aktuelle UTC-Zeit."""
    return datetime.now(UTC)


def _in_handelszeiten(jetzt: datetime) -> bool:
    """Planspiel-Handelszeiten 08:00–22:00 Europe/Berlin (Stunden-Gate)."""
    lokal = jetzt.astimezone(ZoneInfo(_HANDELSZEIT_ZONE))
    return _HANDELSZEIT_START_STUNDE <= lokal.hour < _HANDELSZEIT_ENDE_STUNDE


def _intervall_min_von(aufgabe: dict[str, Any]) -> float:
    """Intervall einer Aufgabe in Minuten (defekt → Untergrenze, defensiv)."""
    roh = aufgabe.get("intervall_min")
    if isinstance(roh, bool) or not isinstance(roh, int | float) or roh <= 0:
        return _MIN_INTERVALL_MIN
    return max(float(roh), _MIN_INTERVALL_MIN)


def _letzter_lauf_zeit(letzter: Any) -> datetime | None:
    """Zeitstempel eines letzten-Lauf-Eintrags (defekt/fehlend → ``None`` = fällig)."""
    if not isinstance(letzter, dict):
        return None
    roh = letzter.get("zeit")
    if not isinstance(roh, str):
        return None
    try:
        moment = datetime.fromisoformat(roh)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment


def _agent_laeuft() -> bool:
    """Läuft gerade irgendein Agent-Run (Muster ``loop.RUN_REGISTRY``, F-A0 §1.3)?

    Der In-Memory-Registry-Stand — nicht der DB-Status ``running`` — ist die
    Quelle (Waisen werden dort erst bei Cancel/Stream aufgeräumt); fertige
    Handles bleiben fürs Replay liegen und zählen über ``task.done()`` nicht.
    """
    from quantlab.agent import loop as agent_loop

    return any(
        handle.task is not None and not handle.task.done()
        for handle in agent_loop.RUN_REGISTRY.values()
    )


def _gefilterte_registry(modus: str, *, freigabe: bool) -> ToolRegistry:
    """Tool-Whitelist je Modus (E081): neue Instanz, GLOBAL_REGISTRY bleibt intakt.

    analyse/vorschlag: alle Tools außer den Schreib-Tools — ``platziere_order``
    ist für Cron-Runs dieser Modi **nie** sichtbar. trade mit Freigabe: zusätzlich
    ``platziere_order``. Fehlt ein Tool in der GLOBAL_REGISTRY (Schema-Drift),
    wird es still übersprungen.
    """
    from quantlab.agent.registry import GLOBAL_REGISTRY, ToolRegistry

    registry = ToolRegistry()
    verboten = set(_SCHREIB_TOOLS)
    if modus == "trade" and freigabe:
        verboten.discard("platziere_order")  # einziger trade-Vorteil: Order-Tool
    else:
        verboten.add("platziere_order")  # analyse/vorschlag: nie sichtbar
    for name in GLOBAL_REGISTRY.names():
        if name in verboten:
            continue
        spec = GLOBAL_REGISTRY.get(name)
        if spec is not None:
            registry.register(spec)
    return registry


def _aufgaben_text(aufgabe: dict[str, Any]) -> str:
    """Tasktext einer Aufgabe: Preset-Text bzw. (``custom``) der gespeicherte Prompt."""
    preset = str(aufgabe.get("preset") or "")
    if preset == "custom":
        return str(aufgabe.get("aufgabe_prompt") or "").strip()
    return AGENT_CRON_PRESETS.get(preset, "")


def _kurz_text(text: str, max_zeichen: int = _KURZ_MAX_ZEICHEN) -> str:
    """Einzeilige Kurzform (Muster ``loop._kurz``) für Persistenz/SSE-Payload."""
    kompakt = " ".join(str(text).split())
    return kompakt if len(kompakt) <= max_zeichen else kompakt[: max_zeichen - 1] + "…"


def _autopilot_zusatz(
    aufgabe: dict[str, Any],
    text: str,
    jetzt: datetime,
    letzter: dict[str, Any] | None,
) -> str:
    """Systemprompt-Zusatz des Cron-Runs: Zeit, Aufgabe, Modus(-Limits), letzter Lauf."""
    modus = str(aufgabe.get("modus") or "vorschlag")
    lokal = jetzt.astimezone(ZoneInfo(_HANDELSZEIT_ZONE))
    zeilen = [
        "Autopilot-Modus (Agent-Zeitplan): Es ist kein Nutzer anwesend, Rückfragen "
        "sind nicht möglich — arbeite selbstständig und kennzeichne Annahmen.",
        f"Aktuelle Zeit: {lokal.strftime('%Y-%m-%d %H:%M')} Europe/Berlin "
        f"(UTC {jetzt.strftime('%Y-%m-%d %H:%M')}).",
        f"Aufgabe „{aufgabe.get('name')}“ (Preset: {aufgabe.get('preset')}, "
        f"Modus: {modus}):",
        text,
    ]
    if modus == "analyse":
        zeilen.append(
            "Modus-Regel: Nur analysieren und lesen — du platzierst keine Orders "
            "und nimmst keine Buchungen vor."
        )
    elif modus == "vorschlag":
        zeilen.append(
            "Modus-Regel: Analysiere und formuliere deinen Vorschlag als Text "
            "(z. B. konkrete Orderkandidaten) — du platzierst KEINE Orders."
        )
    else:  # trade — die Freigabe ist im Tick geprüft, hier nur die harten Limits.
        zeilen.append(
            "Modus-Regel: Du darfst Orders über platziere_order platzieren. Diese "
            "Limits sind hart und zwingend einzuhalten: höchstens "
            f"{int(aufgabe.get('max_orders_pro_lauf') or 0)} Orders in diesem Lauf; "
            "platziere nur Orders, solange ihr Gesamtvolumen innerhalb des "
            f"Kapital-Limits von {float(aufgabe.get('kapital_limit_eur') or 0):.2f} "
            "EUR bleibt."
        )
    if isinstance(letzter, dict) and letzter.get("kurz"):
        zeilen.append(
            f"Kurzergebnis des letzten Laufs ({letzter.get('zeit')}, "
            f"Status {letzter.get('status')}): {letzter.get('kurz')}"
        )
    zeilen.append(
        "Beende dich mit einer kompakten deutschen Kurzfassung des Ergebnisses "
        "(max. ~5 Sätze)."
    )
    return "\n".join(zeilen)


def _letzten_lauf_speichern(aufgabe_id: str, eintrag: dict[str, Any]) -> None:
    """Persistiert den letzten Lauf NUR unter dem Key ``letzte_laeufe`` (Merge).

    Die Aufgabenliste wird nicht angefasst (F-A0 §1.4) — der schlüsselweise
    Merge von ``update_section`` ersetzt ausschließlich den einen Key.
    """
    aktuell = settings_repo.get_section("agent_cron").get("letzte_laeufe")
    gemischt: dict[str, Any] = dict(aktuell) if isinstance(aktuell, dict) else {}
    gemischt[aufgabe_id] = eintrag
    settings_repo.update_section("agent_cron", {"letzte_laeufe": gemischt})


def _zaehle_platziere_order(run_id: str) -> int:
    """Post-Run-Zähler (F-A0 §7-ANNAHME): ``platziere_order``-Calls der Run-Historie.

    Die harte Vorab-Sperre existiert nicht (das Tool ist im trade-Modus erlaubt);
    die Limits gelten als verbindliche Systemprompt-Anweisung und werden hier
    nachträglich verifiziert — Überschreitungen werden markiert, nicht versteckt.
    """
    anzahl = 0
    for message in chat_repo.list_messages(run_id):
        if message.role != "assistant":
            continue
        for call in message.tool_calls() or []:
            if (
                isinstance(call, dict)
                and isinstance(call.get("function"), dict)
                and call["function"].get("name") == "platziere_order"
            ):
                anzahl += 1
    return anzahl


def _letztes_assistant_kurz(run_id: str) -> str:
    """Kurzergebnis eines Runs: letzte Assistant-Message mit Inhalt (gekappt)."""
    for message in reversed(chat_repo.list_messages(run_id)):
        if message.role == "assistant" and message.content:
            return _kurz_text(message.content)
    return ""


def _emitte_agent_cron(
    aufgabe: dict[str, Any], run_id: str | None, status: str, kurzergebnis: str
) -> None:
    """``agent_cron``-SSE nach Abschluss einer Aufgabe (F-A0-D1; OS-Notify: T-F2)."""
    events.publizieren(
        AGENT_CRON_EVENT,
        {
            "aufgabeId": aufgabe.get("id"),
            "name": aufgabe.get("name"),
            "runId": run_id,
            "status": status,
            "kurzergebnis": _kurz_text(kurzergebnis),
        },
    )


#: Run-DB-Status → letzter-Lauf-Status (alles Unbekannte gilt als ``fehler``).
_STATUS_NACH_RUN: dict[str, str] = {"done": "fertig", "cancelled": "abgebrochen"}


def _standard_client(api_key: str) -> LLMClient:
    """Client-Fabrik der Cron-Runs (Injektionspunkt für Tests; Muster routers/agent).

    Je Run ein frischer Client — der Loop übernimmt die Ownership und schließt
    ihn am Run-Ende (siehe ``loop.start_run``).
    """
    from quantlab.agent.llm_client import LLMClient  # funktionslokal, s. Import-Hinweis

    return LLMClient(api_key)


async def _fuehre_aufgabe_aus(
    aufgabe: dict[str, Any], jetzt: datetime
) -> tuple[str, dict[str, Any]]:
    """Startet den Agent-Run einer fälligen Aufgabe und wartet auf sein Ende.

    Rückgabe: ``(status, letzter_lauf_eintrag)``. Der Tick wartet bewusst AUF
    den Run (F-A0 §1.1-ANNAHME): Der Job-Lock sequenziert Cron-Ticks und
    ``run_now`` ohnehin — ein wartendes ``run_now('agent_cron')`` ist gewollt
    (UI zeigt „läuft“), ein paralleler Doppel-Run ist ausgeschlossen. Nach
    Run-Ende: ``agent_cron``-SSE; die Persistenz übernimmt der Aufrufer.
    """
    from quantlab.agent import loop as agent_loop

    aufgabe_id = str(aufgabe.get("id") or "")

    async def _ohne_run(status: str, hinweis: str) -> tuple[str, dict[str, Any]]:
        eintrag: dict[str, Any] = {
            "zeit": datetime.now(UTC).isoformat(),
            "status": status,
            "kurz": "",
            "run_id": None,
            "hinweis": hinweis,
        }
        _emitte_agent_cron(aufgabe, None, status, hinweis)
        return status, eintrag

    # Guard 1 (E081): Modus „trade“ ohne erteilte Freigabe wird abgelehnt.
    modus = str(aufgabe.get("modus") or "vorschlag")
    freigabe = bool(aufgabe.get("freigabe_ertelt"))
    if modus == "trade" and not freigabe:
        return await _ohne_run(
            "abgelehnt",
            "Modus „trade“ ohne erteilte Freigabe — Aufgabe wurde nicht ausgeführt.",
        )

    text = _aufgaben_text(aufgabe)
    if not text:  # Router-Validierung verhindert das; defensiv (Schema-Drift).
        return await _ohne_run("uebersprungen", "Aufgabe hat keinen Aufgabentext.")

    # Guard 2: fehlender API-Key → sauber überspringen, kein Crash (F-A0 §1.1).
    api_key = await asyncio.to_thread(settings_repo.api_key_read)
    if not api_key:
        return await _ohne_run(
            "uebersprungen",
            "Kein z.ai-API-Key konfiguriert — Aufgabe wurde übersprungen.",
        )

    registry = _gefilterte_registry(modus, freigabe=freigabe)
    letzter_roh = agent_cron_einstellungen()["letzte_laeufe"].get(aufgabe_id)
    letzter = letzter_roh if isinstance(letzter_roh, dict) else None
    projekt = aufgabe.get("projekt")
    projekt_id = (
        int(projekt)
        if isinstance(projekt, int) and not isinstance(projekt, bool)
        else None
    )

    handle = await agent_loop.start_run(
        message=f"⏰ Agent-Zeitplan: {aufgabe.get('name')}\n\n{text}",
        client=_standard_client(api_key),
        project_id=projekt_id,
        registry=registry,
        system_prompt_zusatz=_autopilot_zusatz(aufgabe, text, jetzt, letzter),
    )
    timeout = False
    if handle.task is not None:
        try:
            await asyncio.wait_for(handle.task, timeout=_RUN_TIMEOUT_S)
        except TimeoutError:
            timeout = True
            logger.warning(
                "Agent-Cron-Run %s (Aufgabe %s) hat das Laufzeit-Limit von %.0f s "
                "überschritten und wurde abgebrochen.",
                handle.run_id,
                aufgabe_id,
                _RUN_TIMEOUT_S,
            )

    run = await asyncio.to_thread(chat_repo.get_run, handle.run_id)
    status = "fehler" if timeout else _STATUS_NACH_RUN.get(
        run.status if run is not None else "error", "fehler"
    )
    hinweis: str | None = (
        f"Der Lauf wurde nach {_RUN_TIMEOUT_S:.0f} s abgebrochen (Laufzeit-Limit)."
        if timeout
        else None
    )
    kurz = await asyncio.to_thread(_letztes_assistant_kurz, handle.run_id)
    if modus == "trade" and freigabe:
        max_orders = int(aufgabe.get("max_orders_pro_lauf") or 0)
        if max_orders > 0:
            orders = await asyncio.to_thread(_zaehle_platziere_order, handle.run_id)
            if orders > max_orders:
                hinweis = (
                    f"Limit überschritten: {orders} platziere_order-Aufrufe "
                    f"(max. {max_orders} erlaubt)."
                )
    eintrag: dict[str, Any] = {
        "zeit": datetime.now(UTC).isoformat(),
        "status": status,
        "kurz": kurz,
        "run_id": handle.run_id,
    }
    if hinweis is not None:
        eintrag["hinweis"] = hinweis
    kurzergebnis = f"{hinweis} {kurz}".strip() if hinweis is not None else kurz
    _emitte_agent_cron(aufgabe, handle.run_id, status, kurzergebnis)
    return status, eintrag


async def tick_agent_cron(
    *,
    erzwingen: bool = False,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Agent-Cron-Tick: feuert fällige + aktivierte Aufgaben als Agent-Runs.

    - Fälligkeit je Aufgabe: ``jetzt >= letzter Lauf + intervall_min`` (letzter
      Lauf aus der Settings-Sektion); ``erzwingen=True`` (``run-now``) feuert
      alle aktivierten Aufgaben sofort. Das ``nur_handelszeiten``-Gate bleibt
      vom Erzwingen unberührt — Orders außerhalb 08:00–22:00 (Europe/Berlin)
      sind ohnehin nicht ausführbar.
    - Skip: läuft bereits ein (User-)Agent-Run, wird der komplette Tick
      übersprungen — eine gleichzeitige LLM-Sitzung gibt es nicht (E081).
    - Ohne Aufgaben Netto-No-Op; deshalb startet ``starte_scheduler`` den Job
      immer und neue Aufgaben greifen ohne Sidecar-Neustart.

    ``now`` ist injizierbar (Tests steuern Fälligkeit/Handelszeiten deterministisch).
    """
    jetzt = (now or _standard_now)()
    einstellungen = await asyncio.to_thread(agent_cron_einstellungen)
    aufgaben = einstellungen["aufgaben"]
    if not aufgaben:
        return {
            "ok": True,
            "aufgaben": 0,
            "hinweis": "Keine Agent-Zeitpläne konfiguriert.",
        }
    if _agent_laeuft():
        return {
            "ok": True,
            "aufgaben": len(aufgaben),
            "geskippt": True,
            "hinweis": "Ein Agent-Run läuft bereits — Cron-Tick übersprungen.",
        }
    letzte_laeufe = einstellungen["letzte_laeufe"]
    ausgefuehrt: list[str] = []
    abgelehnt: list[str] = []
    uebersprungen: list[str] = []
    ausser_handelszeiten: list[str] = []
    nicht_faellig: list[str] = []
    for aufgabe in aufgaben:
        aufgabe_id = str(aufgabe.get("id") or "")
        if not aufgabe_id or not bool(aufgabe.get("enabled", True)):
            continue  # deaktiviert/defekt — kein Status-Eintrag, kein Run
        if bool(aufgabe.get("nur_handelszeiten")) and not _in_handelszeiten(jetzt):
            ausser_handelszeiten.append(aufgabe_id)
            continue
        if not erzwingen:
            letzter_zeit = _letzter_lauf_zeit(letzte_laeufe.get(aufgabe_id))
            if letzter_zeit is not None and jetzt < letzter_zeit + timedelta(
                minutes=_intervall_min_von(aufgabe)
            ):
                nicht_faellig.append(aufgabe_id)
                continue
        status, eintrag = await _fuehre_aufgabe_aus(aufgabe, jetzt)
        await asyncio.to_thread(_letzten_lauf_speichern, aufgabe_id, eintrag)
        if status == "abgelehnt":
            abgelehnt.append(aufgabe_id)
        elif status == "uebersprungen":
            uebersprungen.append(aufgabe_id)
        else:
            ausgefuehrt.append(aufgabe_id)
    return {
        "ok": True,
        "aufgaben": len(aufgaben),
        "ausgefuehrt": ausgefuehrt,
        "abgelehnt": abgelehnt,
        "uebersprungen": uebersprungen,
        "ausser_handelszeiten": ausser_handelszeiten,
        "nicht_faellig": nicht_faellig,
    }


#: Tick-Funktionen je verfügbarer Job-Art (``else``-Zweck rein defensiv).
async def _tick_nicht_verfuegbar(job: str) -> dict[str, Any]:  # pragma: no cover — defensiv
    raise ProviderError(
        f"Job-Art „{job}“ ist nicht verfügbar.",
        422,
        "job_nicht_verfuegbar",
    )


async def tick_download_retry(
    *,
    erzwingen: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Auto-Retry-Tick (E089/T-G3): startet **fällige** Download-Wiederholungen.

    Delegiert an :func:`quantlab.services.downloads.faellige_retries_starten`
    — dort leben Plan-Lesung, Gruppierung, ``submit_job`` und das Warten auf
    die Retry-Jobs. ``erzwingen=True`` (``run_now("download_retry")``) umgeht
    das Fälligkeits-Gate (alle offenen Plan-Zeilen sofort anstoßen);
    ``now`` ist injizierbar (Tests steuern Fälligkeit/Backoff deterministisch,
    Muster :func:`tick_agent_cron`). Ohne fällige Zeilen Netto-No-Op.
    """
    return await downloads.faellige_retries_starten(now=now, erzwingen=erzwingen)


def _planspiel_client() -> PlanspielClient:
    """Client-Fabrik des Planspiel-Recorders (Injektionspunkt für Tests).

    Muster :func:`_standard_client`: Prozess-Singleton
    (:func:`quantlab.services.planspiel.hole_client`) — Token nur im Speicher,
    Pacing ≥ 1 s je Request (E068) gilt damit automatisch für Sweeps.
    """
    from quantlab.services import planspiel as planspiel_service

    return planspiel_service.hole_client()


async def tick_planspiel_recorder(
    *,
    erzwingen: bool = False,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """PSB-Live-Recorder-Tick (E096): ein Sweep → 1m-Bars (quelle ``psb``).

    - **Deaktiviert** (``planspiel.recorder_aktiv``, Default ``False``):
      Netto-No-Op mit ``deaktiviert``-Flag — kein Netz-Request (der Nutzer
      aktiviert den Recorder bewusst; der Schalter greift ohne Neustart).
    - **Handelszeiten-Gate** 08:00–22:00 Europe/Berlin (:func:`_in_handelszeiten`,
      ``now`` injizierbar, Muster :func:`tick_agent_cron`): außerhalb
      **sofort-Rückkehr** mit ``ausser_handelszeiten``-Flag — bewusst KEINE
      Drossel-Schleife im Tick (E096). Das Gate bleibt wie beim Agent-Cron
      vom Erzwingen unberührt — außerhalb der Handelszeiten entstehen ohnehin
      keine Kurse.
    - Sonst ein Sweep (:func:`quantlab.services.planspiel.recorder.sweep`,
      Client über :func:`_planspiel_client`) im Thread; ein
      :class:`PlanspielFehler`/Fehler registriert die Backoff-Fehler-Serie
      (:func:`recorder.fehler_registrieren`) und kommt als ``ok=False`` mit
      ``fehler`` zurück — der Job-Tick stürzt nie.
    """
    del erzwingen  # Gate bleibt unberührt (dokumentiert); Tick ist sonst idempotent
    jetzt = (now or _standard_now)()
    einst = await asyncio.to_thread(planspiel_recorder_service.einstellungen)
    if not einst["aktiv"]:
        return {
            "ok": True,
            "deaktiviert": True,
            "hinweis": (
                "Planspiel-Recorder ist deaktiviert (planspiel.recorder_aktiv) "
                "— kein Sweep."
            ),
        }
    if not _in_handelszeiten(jetzt):
        return {
            "ok": True,
            "ausser_handelszeiten": True,
            "hinweis": (
                "Außerhalb der Handelszeiten (08:00–22:00 Europe/Berlin) — "
                "Recorder-Sweep übersprungen."
            ),
        }
    client = await asyncio.to_thread(_planspiel_client)
    try:
        details = await asyncio.to_thread(
            planspiel_recorder_service.sweep, client, jetzt=jetzt
        )
    except asyncio.CancelledError:
        raise
    except planspiel_fehler.PlanspielFehler as exc:
        serie = await asyncio.to_thread(planspiel_recorder_service.fehler_registrieren)
        logger.warning(
            "Planspiel-Recorder-Sweep fehlgeschlagen (%d. Serie): %s", serie, exc.message
        )
        return {
            "ok": False,
            "fehler": exc.message,
            "gedrosselt": exc.code == "rate_limit",
            "backoff_stufe": serie,
        }
    except Exception as exc:  # noqa: BLE001 — Tick-Fehler: als Ergebnis, nie als Absturz
        serie = await asyncio.to_thread(planspiel_recorder_service.fehler_registrieren)
        logger.exception("Planspiel-Recorder-Sweep fehlgeschlagen (%d. Serie)", serie)
        return {
            "ok": False,
            "fehler": str(exc) or exc.__class__.__name__,
            "gedrosselt": False,
            "backoff_stufe": serie,
        }
    return {
        **details,
        "ok": True,
        "naechstes_intervall_s": planspiel_recorder_service.naechstes_intervall_s(),
    }


def _alter_tage(iso: str) -> float:
    """Alter eines ISO-Markers in Tagen (defekte Marker gelten als uralt)."""
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return float("inf")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (datetime.now(UTC) - moment) / timedelta(days=1)


async def _tick_ausfuehren(
    job: str, *, erzwingen: bool = False, naechster_lauf_s: float | None = None
) -> dict[str, Any]:
    """Führt einen Tick aus, misst die Dauer und emittiert ``scheduler_tick``.

    Single-Flight je Job-Art (:func:`_job_lock`): Läuft bereits ein Tick
    derselben Art, wartet dieser Aufruf auf dessen Ende — zwei gleichzeitige
    Durchgänge einer Job-Art gibt es unter keinem Interleaving. Messung und
    Event beginnen erst nach Lock-Erwerb (``durationMs`` = reine Tick-Zeit).

    Rückgabe: SSE-Payload (§C.5) + ``details`` (Tick-spezifische Auskunft).
    """
    async with _job_lock(job):
        start = datetime.now(UTC)
        fehler: str | None = None
        details: dict[str, Any] = {}
        try:
            if job == "data_update":
                details = await tick_data_update()
            elif job == "fundamentals":
                details = await tick_fundamentals(erzwingen=erzwingen)
            elif job == "quotes":
                details = await quotes_service.tick_quotes(erzwingen=erzwingen)
            elif job == "alerts":
                details = await alerts_service.tick_alerts(erzwingen=erzwingen)
            elif job == "agent_cron":
                details = await tick_agent_cron(erzwingen=erzwingen)
            elif job == "download_retry":
                details = await tick_download_retry(erzwingen=erzwingen)
            elif job == "planspiel_recorder":
                details = await tick_planspiel_recorder(erzwingen=erzwingen)
            else:  # defensiv — Loop startet nur VERFUEGBARE_JOBS
                details = await _tick_nicht_verfuegbar(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — Tick-Fehler: ins Event, nie in den Absturz
            logger.exception("Scheduler-Tick %s fehlgeschlagen", job)
            fehler = str(exc) or exc.__class__.__name__

        dauer_ms = (datetime.now(UTC) - start) / timedelta(milliseconds=1)
        ok = fehler is None and bool(details.get("ok", True))
        teilerfolg: str | None = None
        if fehler is None:
            angemerkt = details.get("fehler")
            if isinstance(angemerkt, str) and angemerkt:
                teilerfolg = angemerkt
        payload: dict[str, Any] = {
            "job": job,
            "ok": ok,
            "durationMs": round(dauer_ms, 1),
            "ranAt": start.isoformat(),
        }
        if fehler is not None:
            payload["error"] = fehler
        elif teilerfolg is not None:
            payload["error"] = teilerfolg  # Teilerfolg: Tick ok, einzelne Symbole fehlerhaft
        if naechster_lauf_s is not None:
            payload["nextRunAt"] = (
                datetime.now(UTC) + timedelta(seconds=naechster_lauf_s)
            ).isoformat()
        events.publizieren(SCHEDULER_TICK, payload)
        return {**payload, "details": details}


# ---------------------------------------------------------------------------
# Run-now (§C.4): genau ein Tick, synchron erwartbar (Tests/Agent)
# ---------------------------------------------------------------------------


async def run_now(job: str) -> dict[str, Any]:
    """Löst genau einen Tick der Job-Art aus und wartet auf sein Ende.

    Läuft bereits ein Tick derselben Job-Art (Loop-Tick oder anderes
    ``run_now``), wird **gewartet** und der eigene Tick danach ausgeführt —
    Single-Flight je Job-Art, nie ein paralleler Doppel-Durchgang
    (s. :func:`_job_lock`).
    ``fundamentals``/``quotes``/``alerts`` laufen mit Refresh-Override („jetzt
    holen/prüfen“ — Quotes überspringen damit den TTL-Cache). Unbekannte
    Job-Art → 422 ``unbekannter_job``.
    """
    if job not in BEKANNTE_JOBS:
        raise ProviderError(
            f"Unbekannte Job-Art „{job}“.",
            422,
            "unbekannter_job",
            hint=f"Gültige Job-Arten: {', '.join(BEKANNTE_JOBS)}.",
        )
    if job not in VERFUEGBARE_JOBS:
        raise ProviderError(
            f"Job-Art „{job}“ ist noch nicht verfügbar (kommt mit T33).",
            422,
            "job_nicht_verfuegbar",
            hint=f"Derzeit verfügbar: {', '.join(VERFUEGBARE_JOBS)}.",
        )
    return await _tick_ausfuehren(job, erzwingen=True)


# ---------------------------------------------------------------------------
# Lifespan-Verwaltung (ein Task je aktiver Job-Art)
# ---------------------------------------------------------------------------


async def _job_schleife(job: str) -> None:
    """Task-Leib eines Jobs: sofortiger erster Tick, danach Intervall-Sleep.

    Die Einstellungen werden **je Runde** neu gelesen — Intervall-Änderungen
    greifen ohne Neustart. Abbruch über Task-Cancel (``stoppe_scheduler``):
    ``asyncio.sleep`` ist kooperativ, der Tick selbst läuft i. d. R. schnell
    in ``to_thread``-Wartepunkte.
    """
    while True:
        einstellungen = await asyncio.to_thread(scheduler_einstellungen)
        intervall_s = await _intervall_s(job, einstellungen)
        await _tick_ausfuehren(job, naechster_lauf_s=intervall_s)
        if job == "planspiel_recorder":
            # E096: adaptiv — der Schlaf liest das Intervall NACH dem Tick neu,
            # damit ein soeben fehlgeschlagener Sweep den Schlaf DIESER Runde
            # verlängert (exponentieller Backoff, Obergrenze im Recorder).
            intervall_s = await _intervall_s(job, einstellungen)
        await asyncio.sleep(intervall_s)


async def _intervall_s(job: str, einstellungen: dict[str, Any]) -> float:
    """Intervall eines Jobs in Sekunden (bereits geklemmt)."""
    if job == "data_update":
        return einstellungen["data_update_interval_h"] * 3600.0
    if job == "fundamentals":
        return einstellungen["fundamentals_refresh_days"] * 86400.0
    if job == "quotes":
        # E024: Poll-Takt aus der live-Sektion (15–300 s, dort geklemmt) —
        # je Runde gelesen, Änderungen greifen ohne Neustart. Der SQLite-Read
        # läuft wie oben über ``asyncio.to_thread`` (Loop bleibt frei).
        live = await asyncio.to_thread(quotes_service.live_einstellungen)
        return live["quotes_poll_s"]
    if job == "alerts":
        return float(einstellungen["alerts_check_s"])
    if job == "agent_cron":
        # T-F1: fester Poll — die Fälligkeit (Intervall je Aufgabe gegen den
        # letzten Lauf) liegt im Tick, nicht in der Schleife; ohne fällige
        # Aufgabe ist der Tick ein Netto-No-Op.
        return _AGENT_CRON_POLL_S
    if job == "download_retry":
        # E089/T-G3: fester Poll — die Fälligkeit (naechster_versuch_us je
        # Plan-Zeile) liegt im Tick; ohne fällige Zeilen Netto-No-Op.
        return _DOWNLOAD_RETRY_POLL_S
    if job == "planspiel_recorder":
        # E096/H4: adaptiver Takt aus der planspiel-Sektion (30–3600 s, dort
        # geklemmt) mit exponentiellem Backoff nach fehlgeschlagenen Sweeps —
        # Zustand im Recorder-Modul, je Runde neu gelesen (kein Neustart nötig).
        return float(await asyncio.to_thread(planspiel_recorder_service.naechstes_intervall_s))
    return 3600.0


async def starte_scheduler() -> list[str]:
    """Startet die Tasks der aktivierten Jobs im aktuellen Loop (idempotent).

    ``data_update_enabled=false`` startet **keinen** data_update-Task (DoD c);
    analog ``fundamentals_enabled`` und — seit T22/E024 — ``live.quotes_enabled``.
    ``alerts`` (T33) startet immer: der Tick ist ohne relevante Alerts ein
    Netto-No-Op (kein Quotes-Abruf), ein Disable-Schalter ist im Schema §D.2
    nicht vorgesehen. ``agent_cron`` (T-F1) startet genauso immer — der Tick
    ist ohne fällige/aktivierte Aufgabe ein No-Op, neue Aufgaben greifen damit
    ohne Sidecar-Neustart. ``download_retry`` (E089/T-G3) ebenso: ohne fällige
    Wiederholungen No-Op — wartende Plan-Zeilen werden ohne Neustart bedient.
    ``planspiel_recorder`` (E096/H4) ebenso: ohne ``planspiel.recorder_aktiv``
    (Default **false**) No-Op mit Flag — die Aktivierung greift ohne Neustart.
    Rückgabe: gestartete Job-Namen.
    """
    zustand = _loop_zustand()
    if zustand.tasks:
        return sorted(job for job, task in zustand.tasks.items() if not task.done())
    einstellungen = await asyncio.to_thread(scheduler_einstellungen)
    live = await asyncio.to_thread(quotes_service.live_einstellungen)
    gewuenscht: list[str] = []
    if einstellungen["data_update_enabled"]:
        gewuenscht.append("data_update")
    if einstellungen["fundamentals_enabled"]:
        gewuenscht.append("fundamentals")
    if live["quotes_enabled"]:
        gewuenscht.append("quotes")
    gewuenscht.append("alerts")  # T33: immer (No-Op ohne aktive Alerts)
    gewuenscht.append("agent_cron")  # T-F1: immer (No-Op ohne fällige Aufgaben)
    gewuenscht.append("download_retry")  # E089/T-G3: immer (No-Op ohne Fällige)
    gewuenscht.append("planspiel_recorder")  # E096/H4: immer (No-Op deaktiviert)
    for job in gewuenscht:
        zustand.tasks[job] = asyncio.create_task(_job_schleife(job), name=f"scheduler-{job}")
    return sorted(gewuenscht)


async def stoppe_scheduler() -> None:
    """Bricht alle Scheduler-Tasks ab und wartet ihr Ende ab (sauberer Shutdown).

    Kein „Task was destroyed but it is pending!“: Cancel + gather im selben
    Loop wie die Tasks (Lifespan-Finally).
    """
    zustand = _LOOPZUSTAENDE.get(asyncio.get_running_loop())
    if zustand is None:
        return
    tasks = [task for task in zustand.tasks.values() if not task.done()]
    zustand.tasks.clear()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "AGENT_CRON_EVENT",
    "AGENT_CRON_PRESETS",
    "BEKANNTE_JOBS",
    "DOMAIN_DAILY_BARS",
    "SCHEDULER_TICK",
    "VERFUEGBARE_JOBS",
    "agent_cron_einstellungen",
    "laufende_jobs",
    "run_now",
    "scheduler_einstellungen",
    "starte_scheduler",
    "stoppe_scheduler",
    "tick_agent_cron",
    "tick_data_update",
    "tick_download_retry",
    "tick_fundamentals",
    "tick_planspiel_recorder",
    "watchlist_symbole",
]
