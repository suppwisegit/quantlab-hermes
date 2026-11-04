"""Download-Job-Manager (T4): asyncio-Queue, Inkrement, Abbruch, SSE-Fortschritt.

Ablauf eines Jobs (Status nur vorwärts, C.1: ``queued→running→done|error|cancelled``):

1. :func:`submit_job` (async) validiert Eingaben, legt die Job-Zeile via
   :mod:`quantlab.storage.repos.downloads_repo` an (``queued``) und startet den
   Job-Task im aktuellen Event-Loop; die API antwortet sofort mit ``{job_id}``.
2. Jobs laufen **nacheinander** (FIFO-Schleuse je Loop — Downloads serialisieren
   schont die Yahoo-Grenzen, R2): Status ``running``, dann je Symbol den
   inkrementellen Start berechnen, Provider-Fetch und Parquet-Upsert je in
   einem Executor-Thread (:mod:`asyncio.to_thread` — der Store macht blocking
   I/O) und ``data_changed``/``download_progress`` auf den globalen SSE-Bus
   (:mod:`quantlab.events`).
3. **Inkrementell (§D.2) vs. Bereichs-Merge (E090/T-G4):** existiert
   ``bar_meta`` für (Symbol, Intervall) und der Nutzer hat keinen expliziten
   Zeitraum gewählt (``start`` ODER ``period`` — der Watchlist-Tick sendet
   beides nicht), wird ab ``last_ts − 5 Balken`` geladen (Überlappung fängt
   Korrekturen/Adjustierungen ab; Dedup last-write-wins regelt Konflikte).
   Bei ausdrücklicher Zeitraum-Wahl gilt das Merge-Gate: Fetch ab
   ``min(basis_start, first_ts)`` — der Bestand wird nicht abgeschnitten,
   fehlende Vergangenheit/Mittellücken füllt der fenster-additive Upsert.
4. **Fehlerisolation:** wirft ein Symbol, erhalten die anderen des Batches
   trotzdem Daten; der Fehler landet in der Job-Zusammenfassung. Alle Symbole
   fehlgeschlagen → ``error``; sonst ``done`` (Teilerfolg sichtbar im
   ``error``-Feld). Null-Download schreibt nichts (§D.2).
5. **Abbruch:** kooperativ — :func:`cancel_job_anfordern` setzt den Status (mit
   Vorwärts-Guard) in der DB, :func:`abbruch_signalieren` weckt den Job; der
   prüft zwischen Symbolen und bricht dann ein (Status bleibt ``cancelled``,
   der Monotonie-Guard verhindert jede Rückwirkung).

Der Provider kommt aus :func:`get_provider` — dieser Injektionspunkt wird in
Tests durch einen Fake ersetzt (E013: reguläre Suite netzfrei).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from quantlab import db, events
from quantlab.providers import rate_limit
from quantlab.providers.base import (
    DAILY_INTERVALS,
    DataProvider,
    ProviderError,
    validate_interval,
)
from quantlab.providers.byteworx import ByteworxProvider
from quantlab.providers.fallback_provider import FallbackProvider
from quantlab.providers.leeway_provider import LeewayProvider
from quantlab.providers.symbols import validate_symbol
from quantlab.providers.yahoo_direct import (
    YahooDirectProvider,
    unterstuetzt_chunk_fortschritt,
)
from quantlab.providers.yfinance_provider import (
    YfinanceProvider,
    jetzt_us,
    pruefe_intraday_fenster,
)
from quantlab.storage import parquet_store
from quantlab.storage.repos import (
    download_retries_repo,
    downloads_repo,
    settings_repo,
    symbols_repo,
)

logger = logging.getLogger(__name__)

#: Mikrosekunden pro Tag.
_TAG_US = parquet_store.US_PER_DAY

#: Überlappung beim Inkrement in Balken (§D.2 — fängt Korrekturen ab).
_UEBERLAPP_BALKEN = 5

#: Geschätzte Balkendauer je Intervall (für die Überlappung in µs; grob je
#: Monats-/Quartalsintervallen, exakt bei Minuten/Tagen).
_BALKEN_DAUER_US: dict[str, int] = {
    "1m": 60 * 1_000_000,
    "2m": 2 * 60 * 1_000_000,
    "5m": 5 * 60 * 1_000_000,
    "15m": 15 * 60 * 1_000_000,
    "30m": 30 * 60 * 1_000_000,
    "60m": 60 * 60 * 1_000_000,
    "90m": 90 * 60 * 1_000_000,
    "1h": 60 * 60 * 1_000_000,
    "1d": _TAG_US,
    "5d": 5 * _TAG_US,
    "1wk": 7 * _TAG_US,
    "1mo": 30 * _TAG_US,
    "3mo": 91 * _TAG_US,
}

#: Von yfinance dokumentierte Perioden (RECHERCHE Block 2, Fakt 2) → Tage.
_PERIODE_TAGE: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "3mo": 91,
    "6mo": 182,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
}

#: Maximale Länge der Fehlerzusammenfassung im Job-Eintrag.
_MAX_FEHLER_LAENGE = 500

_TERMINAL: frozenset[str] = frozenset(
    {
        downloads_repo.STATUS_DONE,
        downloads_repo.STATUS_ERROR,
        downloads_repo.STATUS_CANCELLED,
    }
)

#: Status, bei denen :func:`warte_auf_abschluss` zurückkehrt (E089/T-G3):
#: Terminal-Status **plus** ``wartet_auf_wiederholung`` — ein wartender Job
#: blockiert seine Caller (Agent-Tool, ``tick_data_update``) bewusst NICHT
#: bis zum Timeout; die Wiederholung läuft autonom über den Retry-Tick.
_ABGESCHLOSSEN: frozenset[str] = _TERMINAL | {
    downloads_repo.STATUS_WARTET_AUF_WIEDERHOLUNG
}

# ---------------------------------------------------------------------------
# Auto-Retry (E089/T-G3): Klassifizierung + Backoff
# ---------------------------------------------------------------------------

#: Transiente HTTP-Status — der Versuch wird automatisch wiederholt. 404/422
#: (delisted, unbekanntes Symbol, Intraday-Limit, ungültige Argumente) sind
#: terminal und werden NIE wiederholt (Delisted-Schutz; Bestands-Test
#: ``test_download_service.py`` ProviderError 404 → ``error`` bleibt grün).
_TRANSIENTE_HTTP_STATUS: frozenset[int] = frozenset({429, 502})

#: Transiente Fachcodes (Quelle-blockiert): Stooq hinter Browser-Verifizierung
#: bzw. Tageslimit ist ein Availability-Problem, kein Datenurteil.
_TRANSIENTE_FEHLER_CODES: frozenset[str] = frozenset(
    {"stooq_blockiert", "stooq_limit"}
)

#: Backoff-Abstände in Sekunden nach Fehlversuch n (2/5/10/20/30 min für
#: Versuch 1..5); der 6. Fehlversuch beendet die automatische Wiederholung.
_RETRY_BACKOFF_S: tuple[float, ...] = (
    2 * 60.0,
    5 * 60.0,
    10 * 60.0,
    20 * 60.0,
    30 * 60.0,
)

#: Puffer über der Circuit-Restzeit: nächster Versuch = ``max(Backoff,
#: direct_circuit_rest_s + Puffer)`` — der Wiederholungsversuch soll nicht in
#: den noch offenen Cool-down laufen.
_CIRCUIT_PUFFER_S = 5.0

#: Maximale Fehlversuche je Symbol, bevor die Wiederholung terminal endet
#: (E089: „max 6 Versuche → terminal + deutsch").
_MAX_RETRIES_VERSUCHE = download_retries_repo.MAX_VERSUCHE

#: Timeout des Retry-Ticks auf einen angestoßenen Retry-Job (Muster
#: ``_JOB_TIMEOUT_S`` des Schedulers; Downloads sind durch die FIFO-Schleuse
#: serialisiert, der Timeout ist ein Netz gegen Hänger).
_RETRY_JOB_TIMEOUT_S = 600.0


def _jetzt() -> datetime:
    """Injizierbare Uhr der Retry-Planung (Tests monkeypatchen diese Funktion)."""
    return datetime.now(UTC)


def _datetime_zu_us(moment: datetime) -> int:
    """``datetime`` (tz-aware, naive gilt als UTC) → int64 µs UTC (E006)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


def _ist_transient(panne: BaseException) -> bool:
    """Fehlerklassifizierung der Auto-Wiederholung (E089; Liste dokumentiert).

    Transient (= Wiederholung geplant): :class:`ProviderError` mit
    ``status_code`` 429/502 oder Fachcode ``stooq_blockiert``/``stooq_limit``.
    Alles andere (404/422/…, unbekannte Ausnahmen) ist terminal.
    """
    if not isinstance(panne, ProviderError):
        return False
    return (
        panne.status_code in _TRANSIENTE_HTTP_STATUS
        or panne.code in _TRANSIENTE_FEHLER_CODES
    )


def _original_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Original-Request-Kwargs eines Jobs (für ``submit_job`` des Retry-Ticks).

    Nur die vom Aufrufer gewählten Werte — NICHT die abgeleiteten ``basis_*``
    (recomputes ``create_job``) und NICHT ``retry_von`` (wird frisch gesetzt).
    """
    kwargs: dict[str, Any] = {}
    for schluessel in ("period", "start", "end"):
        wert = params.get(schluessel)
        if wert is not None:
            kwargs[schluessel] = wert
    for schluessel in ("repair", "force_full"):
        if params.get(schluessel):
            kwargs[schluessel] = True
    return kwargs


# ---------------------------------------------------------------------------
# Provider-Bezug (Injektionspunkt für Tests)
# ---------------------------------------------------------------------------

_PROVIDER: DataProvider | None = None


def get_provider() -> DataProvider:
    """Aktiver Marktdaten-Provider (lazy: erst beim ersten Zugriff gebaut).

    Seit T-F9 (E085) die Drei-Bein-Kette: **YahooDirect** (httpx auf
    v8/chart) primär für alle Intervalle → **yfinance** als zweites Yahoo-Bein
    (erstmals auch für Intraday erreichbar) → **Stooq** als reiner Daily-
    Fallback — inkl. Pacing/Circuit-Breaker vor jedem Yahoo-Call (E035).

    Seit E087 mit **getrennten Circuits** je Bein: das Direct-Bein läuft auf
    :data:`rate_limit.RATE_LIMITER_DIRECT`, das yfinance-Bein auf
    :data:`rate_limit.RATE_LIMITER_YF` — ein 429 im einen Bein sperrt das
    andere nicht mehr (Anlass-Fall Welle G). Die Restzeit des Direct-Circuits
    liest :func:`direct_circuit_rest_s` (Lesepunkt für die Auto-Retry-
    Planung, E089/T-G3 — hier bewusst ohne Retry-Logik).

    Seit E088 (T-G2) mit **viertem Bein**: ist in der Settings-Sektion
    ``daten`` ein nicht-leerer ``leeway_api_key`` hinterlegt, läuft die
    Tagesdaten-Kette zusätzlich über
    :class:`quantlab.providers.leeway_provider.LeewayProvider` (zwischen
    yfinance und Stooq; eigener 429-Schutz, nie die Yahoo-Circuits). Der
    Provider wird lazy **einmal** gebaut — eine Key-Änderung in den
    Einstellungen wirkt daher erst mit Sidecar-Neustart bzw. ``_PROVIDER =
    None`` (Tests monkeypatchen ``get_provider`` ohnehin). Ein leerer Key
    oder eine unlesbare Settings-DB bedeutet: Bein aus, Kette exakt wie im
    Drei-Bein-Bestand.
    """
    global _PROVIDER
    if _PROVIDER is None:
        leeway_key = ""
        try:
            sektion = settings_repo.get_section("daten")
            roh = sektion.get("leeway_api_key")
            if isinstance(roh, str):
                leeway_key = roh.strip()
        except Exception:  # noqa: BLE001 — Settings-DB fehlt (Tests) → Bein aus
            leeway_key = ""
        _PROVIDER = FallbackProvider(
            yahoo_direct=YahooDirectProvider(
                rate_limiter=rate_limit.RATE_LIMITER_DIRECT
            ),
            yahoo=YfinanceProvider(rate_limiter=rate_limit.RATE_LIMITER_YF),
            quelle2=LeewayProvider(leeway_key) if leeway_key else None,
            direct_limiter=rate_limit.RATE_LIMITER_DIRECT,
            yahoo_limiter=rate_limit.RATE_LIMITER_YF,
        )
    return _PROVIDER


def direct_circuit_rest_s() -> float:
    """Restdauer des Direct-Circuits in Sekunden (``0.0``, wenn kein Cool-down).

    Strukturierte Lesung für die Auto-Retry-Planung (E089/T-G3-Besitz — hier
    bewusst KEINE Retry-Logik): nächster Versuch = ``max(Circuit-Rest, Backoff)``.
    Zugriffspfad: ``rate_limit.RATE_LIMITER_DIRECT.gesperrt_fehler().rest_s`` —
    die :class:`~quantlab.providers.rate_limit.YahooGedrosseltError` trägt die
    Restzeit strukturiert im Attribut ``rest_s`` (E087).
    """
    return rate_limit.RATE_LIMITER_DIRECT.gesperrt_fehler().rest_s


# ---------------------------------------------------------------------------
# ISIN-Bein (E094, Welle H2): ISIN-förmige Symbole → Byteworx VOR der Kette
# ---------------------------------------------------------------------------


#: ISIN-Form (ISO 6166, ohne Checksummen-Prüfung): 2 Länderbuchstaben + 9
#: NSIN-Zeichen + 1 Prüfziffer. Bewusstes **Duplikat** von
#: ``agent/tools/daten.py`` (``_ISIN_MUSTER``/``ist_isin``, Z. 61–68): ein
#: Import von dort würde einen Zirkelbezug aufspannen (``daten`` importiert
#: ``downloads``) — Wiederverwendung ist hier nicht sauber möglich (DoD H2).
#: Die Agent-Tools lösen ISINs weiterhin zu Yahoo-Kürzeln auf; was als ISIN
#: bis hier durchschlägt, läuft ausschließlich über Byteworx (E094).
_ISIN_MUSTER = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def ist_isin(wert: str) -> bool:
    """``True``, wenn ``wert`` (getrimmt, großgeschrieben) der ISIN-Form entspricht."""
    if not isinstance(wert, str):
        return False
    return _ISIN_MUSTER.match(wert.strip().upper()) is not None


_BYTEWORX: ByteworxProvider | None = None


def _byteworx_provider() -> ByteworxProvider:
    """Das ISIN-native Byteworx-Bein (lazy, einmal je Prozess; E094/H2).

    Kein Key/keine Settings nötig — der Endpunkt ist frei zugänglich
    (Modul-Docstring :mod:`quantlab.providers.byteworx`); das Bein ist deshalb
    **immer** verdrahtet. Tests ersetzen die Factory per ``monkeypatch``
    (MockTransport, E013).
    """
    global _BYTEWORX
    if _BYTEWORX is None:
        _BYTEWORX = ByteworxProvider()
    return _BYTEWORX


# ---------------------------------------------------------------------------
# bar_meta.source (E022/E023 — Herkunft der Balken; 'yahoo' | 'stooq')
# ---------------------------------------------------------------------------


def bar_meta_source(symbol: str, interval: str) -> str:
    """Liest ``bar_meta.source``; ``'yahoo'`` wenn Zeile/Spalte fehlt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT source FROM bar_meta WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()
    finally:
        conn.close()
    return "yahoo" if row is None else str(row["source"])


def _bar_meta_source_setzen(symbol: str, interval: str, quelle: str) -> None:
    """Setzt ``bar_meta.source`` (letzer Schreiber gewinnt — grobe Herkunftsmarke).

    Bewusster SQL-Zeilen-Write statt Store-/Repo-Änderung: Parquet-Store und
    Repos sind Sperrgebiet paralleler Tasks (T20/T18); die Source-Pflege ist
    reine Job-Manager-Aufgabe (E023 „Download-Service merkt nur bar_meta.source“).
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE bar_meta SET source = ? WHERE symbol = ? AND interval = ?",
            (quelle, symbol, interval),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Gap-Repair (T19): Lückenerkennung im Bestandsfenster
# ---------------------------------------------------------------------------


def pruefe_lücken(symbol: str, interval: str) -> list[dict[str, Any]]:
    """Fehlende Handelstage im Bestandsfenster (Heuristik, T19): Mo–Fr ohne Bar.

    Heuristik bewusst simpel (PLAN §T19): zwischen ``first_ts`` und ``last_ts``
    zählen Wochentage (Montag–Freitag, UTC-Kalender), zu denen kein Bar existiert
    — US-Feiertage melden sich damit als „Lücke“ (dokumentierte Übermeldung;
    der Repair-Fetch ist idempotent und korrigiert nur im Fenster). Rückgabe:
    zusammenhängende Zeiträume als ``[{"von": "YYYY-MM-DD", "bis": "YYYY-MM-DD",
    "fehltage": n}, …]``. Nur für ``1d`` sinnvoll/relevant — andere Intervalle
    liefern ``[]``; unbekanntes Symbol ebenfalls ``[]``.
    """
    meta = parquet_store.get_bar_meta(symbol, interval)
    if meta is None or meta.first_ts is None or meta.last_ts is None:
        return []
    if interval != "1d":
        return []
    bars = parquet_store.read_bars(symbol, interval)
    vorhanden = {int(ts) // _TAG_US for ts in bars["ts"]}
    start_tag = meta.first_ts // _TAG_US
    ende_tag = meta.last_ts // _TAG_US

    lücken: list[dict[str, Any]] = []
    laufende: list[int] = []  # Tagesorte (Tage seit Epoche) der aktuellen Lücke
    for tag in range(start_tag, ende_tag + 1):
        moment = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=tag)
        ist_wochenende = moment.weekday() >= 5
        if ist_wochenende or tag in vorhanden:
            if laufende:
                lücken.append(_luecke_als_dict(laufende))
                laufende = []
            continue
        laufende.append(tag)
    if laufende:
        lücken.append(_luecke_als_dict(laufende))
    return lücken


def _luecke_als_dict(tagesorte: list[int]) -> dict[str, Any]:
    """Zusammenhängende Tagesorte → ``{"von", "bis", "fehltage"}`` (ISO-Daten)."""

    def iso(tag: int) -> str:
        moment = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=tag)
        return moment.date().isoformat()

    return {
        "von": iso(tagesorte[0]),
        "bis": iso(tagesorte[-1]),
        "fehltage": len(tagesorte),
    }


# ---------------------------------------------------------------------------
# Zeit-/Zeitraum-Helfer
# ---------------------------------------------------------------------------


def iso_zeitpunkt_zu_us(text: str) -> int:
    """ISO-Datum/Zeit (``YYYY-MM-DD`` genügt; naive = UTC) → int64 µs UTC.

    ``start`` fällt so auf Mitternacht UTC (inklusiv), ``end`` auf Mitternacht
    des Folgetags (exklusiv) — konsistent mit der yfinance-Konvention.
    """
    if not isinstance(text, str) or not text.strip():
        raise ProviderError(
            "Zeitangabe fehlt oder ist leer.", 422, "ungueltiges_datum"
        )
    try:
        moment = datetime.fromisoformat(text.strip())
    except ValueError as exc:
        raise ProviderError(
            f"Ungültige Zeitangabe „{text}“.",
            422,
            "ungueltiges_datum",
            hint="Format YYYY-MM-DD oder ISO-8601 (naive Angaben gelten als UTC).",
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


def _basis_fenster(
    period: str | None, start: str | None, end: str | None
) -> tuple[int | None, int | None]:
    """Request-Parameter → (basis_start_us, basis_end_us), ``None`` = offen.

    Explizite ``start``/``end``-Angaben schlagen ``period`` vor; ohne alles
    gilt ``1mo`` (yfinance-Default). ``max``/``ytd`` werden unterstützt.
    """
    if start is not None or end is not None:
        return (
            iso_zeitpunkt_zu_us(start) if start is not None else None,
            iso_zeitpunkt_zu_us(end) if end is not None else None,
        )
    periode = (period or "1mo").strip()
    if periode == "max":
        return None, None
    if periode == "ytd":
        jahresanfang = datetime(datetime.now(UTC).year, 1, 1, tzinfo=UTC)
        return (jahresanfang - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(
            microseconds=1
        ), None
    tage = _PERIODE_TAGE.get(periode)
    if tage is None:
        raise ProviderError(
            f"Unbekannte Periode „{periode}“.",
            422,
            "unbekannte_periode",
            hint="Gültig: " + ", ".join(_PERIODE_TAGE) + ", ytd, max.",
        )
    return jetzt_us() - tage * _TAG_US, None


def _inkrement_start(interval: str, last_ts: int) -> int:
    """Inkrementeller Fetch-Start: ``last_ts − 5 Balken`` (§D.2, nicht < 0)."""
    dauer = _BALKEN_DAUER_US.get(interval, _TAG_US)
    return max(0, last_ts - _UEBERLAPP_BALKEN * dauer)


# ---------------------------------------------------------------------------
# Job-Verwaltung (je Event-Loop: eine Queue + ein Worker)
# ---------------------------------------------------------------------------


@dataclass
class _JobHandle:
    """Laufzeit-Handle eines Jobs (Abbruch-Event + Task-Referenz)."""

    abbruch: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


@dataclass
class _LoopZustand:
    """Serialisierung je Event-Loop: Jobs laufen nacheinander (FIFO-Schleuse).

    Die Schleuse ist die Queue: ``asyncio.Lock`` bedient wartende Jobs in
    Einreihreihenfolge — ein Download nach dem anderen schont die Yahoo-Grenzen
    (R2). Bewusst **kein** persistenter Worker-Task: der würde einen beendeten
    Event-Loop (Tests) als „pending“ überleben; ein Task je Job endet stets.
    """

    schleuse: asyncio.Lock = field(default_factory=asyncio.Lock)


_LOOPZUSTAENDE: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopZustand] = (
    weakref.WeakKeyDictionary()
)

#: Offene Handles je job_id (nur nicht-terminaler Lebensabschnitt).
_JOBS: dict[str, _JobHandle] = {}


def _loop_zustand() -> _LoopZustand:
    """Zustand (Schleuse) des laufenden Loops (Sidecar hat genau einen; Tests je einen)."""
    loop = asyncio.get_running_loop()
    zustand = _LOOPZUSTAENDE.get(loop)
    if zustand is None:
        zustand = _LoopZustand()
        _LOOPZUSTAENDE[loop] = zustand
    return zustand


async def _job_lauf(job_id: str, schleuse: asyncio.Lock) -> None:
    """Task-Leib eines Jobs: serialisiert via Schleuse, fängt Restfehler ab."""
    try:
        async with schleuse:
            await _job_ausfuehren(job_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # letztes Netz: Job als error verbuchen
        logger.exception("Download-Job %s unerwartet fehlgeschlagen", job_id)
        await asyncio.to_thread(
            _status_vorwaerts,
            job_id,
            downloads_repo.STATUS_ERROR,
            error=f"Unerwarteter Fehler im Download-Job: {exc}",
        )


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def create_job(
    symbols: list[str],
    interval: str,
    *,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    repair: bool = False,
    force_full: bool = False,
    retry_von: list[str] | None = None,
) -> downloads_repo.Download:
    """Validiert und legt den Job an (Status ``queued``); reiht noch NICHT ein.

    Sync-Funktion (DB-Write) — vom async-Endpunkt via ``asyncio.to_thread``
    bzw. direkt aus :func:`submit_job` zu benutzen. Fachliche Fehler als
    :class:`ProviderError` (422): leere Symbol-Liste, ungültiges Symbol/
    Intervall, unbekannte Periode, Intraday-Fenster jenseits 60 Tage.
    ``repair=True`` (T19): Fetch mit ``repair=True`` **über das gesamte
    Bestandsfenster** ( Korrektur statt Inkrement — Details ``_symbol_laden``).
    ``force_full=True`` (MINOR 6, T20-Fix): unterdrückt den Inkrement — intern
    wird je Symbol ab ``first_ts`` des Bestands geladen (wie ein expliziter
    ``start``, rein additiv; Details ``_symbol_laden``).
    ``retry_von`` (E089/T-G3, intern): Job-IDs der Ursprungs-Jobs, deren
    Wiederholungsplan dieser Retry-Job ausführt — der Marker schaltet die
    Plan-Pflege im Job-Abschluss um (Retry-Jobs enden klassisch terminal und
    schreiben die Plan-Zeilen der Ursprungs-Jobs fort). Ohne Marker bleibt
    ``params`` byte-identisch zum Alt-Bestand.
    """
    geprueftes_intervall = validate_interval(interval)
    eindeutig: list[str] = []
    for roh in symbols:
        kandidat = validate_symbol(roh)
        if kandidat not in eindeutig:
            eindeutig.append(kandidat)
    if not eindeutig:
        raise ProviderError(
            "Mindestens ein Symbol angeben.", 422, "leere_symbol_liste"
        )
    basis_start, basis_ende = _basis_fenster(period, start, end)
    if geprueftes_intervall not in DAILY_INTERVALS:
        # Früh ablehnen statt einem zum Scheitern verurteilten Job (gleiche
        # Fachmeldung wie der Provider).
        pruefe_intraday_fenster(basis_start, basis_ende)
    params: dict[str, Any] = {
        "period": period,
        "start": start,
        "end": end,
        "basis_start_us": basis_start,
        "basis_end_us": basis_ende,
        "repair": bool(repair),
        "force_full": bool(force_full),
    }
    if retry_von:
        params["retry_von"] = [str(job_id) for job_id in retry_von]
    job_id = f"job-{uuid.uuid4().hex}"
    return downloads_repo.create_download(
        job_id, eindeutig, geprueftes_intervall, params=params
    )


async def submit_job(
    symbols: list[str],
    interval: str,
    *,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    repair: bool = False,
    force_full: bool = False,
    retry_von: list[str] | None = None,
) -> downloads_repo.Download:
    """Legt den Job an (Thread) und startet seine Verarbeitung im aktuellen Loop.

    Mehrere Jobs laufen nacheinander (FIFO-Schleuse je Event-Loop) — Details:
    :class:`_LoopZustand`. ``retry_von`` (intern, E089/T-G3): Retry-Job-Marker,
    durchgereicht an :func:`create_job`.
    """
    zeile = await asyncio.to_thread(
        create_job, symbols, interval, period=period, start=start, end=end,
        repair=repair, force_full=force_full, retry_von=retry_von,
    )
    handle = _JobHandle()
    _JOBS[zeile.job_id] = handle
    handle.task = asyncio.get_running_loop().create_task(
        _job_lauf(zeile.job_id, _loop_zustand().schleuse)
    )
    return zeile


def cancel_job_anfordern(job_id: str) -> downloads_repo.Download | None:
    """DB-Seite des Abbruchs (thread-safe): Status nur vorwärts auf ``cancelled``.

    Läuft über ``asyncio.to_thread``; das Wecken des Jobs macht danach
    :func:`abbruch_signalieren` im Loop-Thread. ``None`` bei unbekannter ID;
    bereits terminale Jobs bleiben unverändert (Rückgabe mit altem Status).
    """
    job = downloads_repo.get_download(job_id)
    if job is None:
        return None
    if job.status not in _TERMINAL:
        _status_vorwaerts(job_id, downloads_repo.STATUS_CANCELLED)
    return downloads_repo.get_download(job_id)


def abbruch_signalieren(job_id: str) -> None:
    """Setzt das Abbruch-Event (muss im Loop-Thread des Jobs laufen)."""
    handle = _JOBS.get(job_id)
    if handle is not None:
        handle.abbruch.set()


async def warte_auf_abschluss(
    job_id: str, timeout_s: float = 120.0
) -> downloads_repo.Download | None:
    """Wartet, bis ein Job „abgeschlossen“ ist (Hilfe für Agent-Tools und Tests).

    Seit E089/T-G3 kehrt die Funktion auch bei ``wartet_auf_wiederholung``
    zurück (statt bis zum Timeout zu pollten): Ein Job, dessen Symbole im
    Wiederholungsplan ruhen, blockiert seine Caller nicht — der Agent schreibt
    seinen Bericht mit ``wiederholung_um`` und der Retry-Tick übernimmt.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        job = await asyncio.to_thread(downloads_repo.get_download, job_id)
        if job is not None and job.status in _ABGESCHLOSSEN:
            return job
        await asyncio.sleep(0.02)
    return None


# ---------------------------------------------------------------------------
# Job-Ausführung
# ---------------------------------------------------------------------------


def _status_vorwaerts(
    job_id: str, neuer_status: str, *, error: str | None = None
) -> bool:
    """Setzt den Status nur aufwärts (Monotonie-Guard, §T4 „nur vorwärts").

    Delegiert an :func:`downloads_repo.update_status_vorwaerts` — der
    Rang-Vergleich läuft SQL-seitig im ``UPDATE`` und damit **atomar in einer
    ``BEGIN IMMEDIATE``-Transaktion** (kein Read-modify-write über zwei
    Verbindungen): gleichzeitiges Cancel und Job-Ende sind kritikfrei, wer
    zuerst schreibt, gewinnt; ein akzeptiertes ``cancelled`` kann nie mehr zu
    ``done`` werden.
    """
    return downloads_repo.update_status_vorwaerts(job_id, neuer_status, error=error)


async def _job_ausfuehren(job_id: str) -> None:
    """Führt einen dequeue-ten Job aus (Details: Modul-Docstring).

    Das Laufzeit-Handle wird im ``finally`` geräumt — auch bei Fehlerpfaden
    (fehlgeschlagene DB-Lese, Ausnahmen im Loop) bleibt kein Leicheintrag in
    :data:`_JOBS` zurück.
    """
    try:
        job = await asyncio.to_thread(downloads_repo.get_download, job_id)
        if job is None:
            return
        handle = _JOBS.get(job_id)
        abbruch = handle.abbruch if handle is not None else None

        if job.status in _TERMINAL:
            # z. B. Cancel während der Job noch in der Queue war
            events.download_finished(
                job_id, job.status == downloads_repo.STATUS_DONE, job.error
            )
            return

        await asyncio.to_thread(
            _status_vorwaerts, job_id, downloads_repo.STATUS_RUNNING
        )
        provider = get_provider()
        params = job.params or {}
        fehler: dict[str, str] = {}
        # REV-G MAJOR-3: generische (nicht-ProviderError-)Ausnahmen zählen
        # ebenfalls als Panne — ein Symbol ohne Daten gilt nie als Erfolg
        # und fällt nie still aus der Wiederholungsplan-Pflege (dort läuft
        # es über die TERMINALE Klassifikation, „unbekannte Ausnahmen sind
        # terminal“, E089-Doku).
        pannen: dict[str, BaseException] = {}
        erfolgreich = 0
        total = len(job.symbols)
        letztes_symbol = ""
        letzte_position = 0

        for position, symbol in enumerate(job.symbols, start=1):
            if abbruch is not None and abbruch.is_set():
                break
            letztes_symbol = symbol
            letzte_position = position
            try:
                if await _symbol_laden(
                    provider,
                    symbol,
                    job.interval,
                    params,
                    job_id=job_id,
                    position=position,
                    symbole_gesamt=total,
                ):
                    events.data_changed(
                        symbol, job.interval, events.REASON_DOWNLOAD
                    )
                erfolgreich += 1
            except ProviderError as exc:
                fehler[symbol] = exc.message
                pannen[symbol] = exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Isolation: ein Symbol kippt nie den Batch
                fehler[symbol] = str(exc) or exc.__class__.__name__
                # REV-G MAJOR-3: generische Panne wie eine terminale führen —
                # sonst gelte das Symbol als „erfolgreich“ (nicht in
                # ``pannen``): seine Plan-Zeile würde geräumt und ein
                # Ursprungs-Job könnte auf ``done`` ohne Daten gehen.
                pannen[symbol] = exc
            events.download_progress(
                job_id, symbol, position, total, downloads_repo.STATUS_RUNNING
            )

        abgebrochen = abbruch is not None and abbruch.is_set()
        ist_retry = bool(params.get("retry_von"))

        # E089/T-G3: Wiederholungsplan pflegen, BEVOR der Status gewählt wird —
        # offene Plan-Zeilen entscheiden über „wartet_auf_wiederholung“.
        erfolgreiche = [s for s in job.symbols if s not in pannen]
        offene_wiederholungen = False
        wiederholung_beendet = False
        if (pannen or erfolgreiche) and not abgebrochen:
            offene_wiederholungen, wiederholung_beendet = await asyncio.to_thread(
                _wiederholungsplan_pflegen,
                job_id,
                job.interval,
                params,
                pannen,
                erfolgreiche,
            )

        if abgebrochen:
            await asyncio.to_thread(
                _status_vorwaerts, job_id, downloads_repo.STATUS_CANCELLED
            )
        elif not ist_retry and erfolgreich == 0 and offene_wiederholungen:
            # Früheste Fälligkeit der Plan-Zeilen VOR dem Status-Übergang in die
            # Job-params legen (E089/T-G3, Quelle der FE-Anzeige „wartet auf
            # Wiederholung (um HH:MM)“): Beobachter des wartenden Status sehen
            # die Zeitangabe damit stets bereits gesetzt.
            naechster_us = await asyncio.to_thread(
                download_retries_repo.frueheste_offene_us, job_id
            )
            if naechster_us is not None:
                await asyncio.to_thread(
                    downloads_repo.ergaenze_params,
                    job_id,
                    {"retry_naechster_us": naechster_us},
                )
            # Alle Symbole transient fehlgeschlagen → der Job ruht im
            # Wiederholungsplan (nicht-terminal; Cancel bleibt möglich).
            await asyncio.to_thread(
                _status_vorwaerts,
                job_id,
                downloads_repo.STATUS_WARTET_AUF_WIEDERHOLUNG,
                error=_zusammenfassung(fehler),
            )
        elif fehler and erfolgreich == 0:
            fehler_text = _zusammenfassung(fehler)
            if wiederholung_beendet:
                fehler_text = _kappe(
                    "Automatische Wiederholung beendet — " + (fehler_text or "")
                )
            await asyncio.to_thread(
                _status_vorwaerts,
                job_id,
                downloads_repo.STATUS_ERROR,
                error=fehler_text,
            )
        else:
            await asyncio.to_thread(
                _status_vorwaerts,
                job_id,
                downloads_repo.STATUS_DONE,
                error=_zusammenfassung(fehler),
            )

        if ist_retry and not abgebrochen:
            # Retry-Job (Marker ``retry_von``) endet klassisch terminal; seine
            # Plan-Pflege lief unter den Ursprungs-Job-IDs — gelöste Ursprünge
            # werden hier abgeschlossen (done bzw. error mit Schlusstext).
            await _finalisiere_vorgaenger(params.get("retry_von") or [])

        frisch = await asyncio.to_thread(downloads_repo.get_download, job_id)
        if frisch is not None:
            if (
                not abgebrochen
                and not ist_retry
                and frisch.status == downloads_repo.STATUS_WARTET_AUF_WIEDERHOLUNG
            ):
                # E089/T-G3 (SSE-Wahl, dokumentiert): der Übergang auf „wartet“
                # läuft über das BESTEHENDE ``download_progress`` mit dem neuen
                # Statuswert (freier String, events.py unverändert) — KEIN
                # ``download_finished``, sonst schließt der Frontend-Toast den
                # noch wartenden Job sofort.
                events.download_progress(
                    job_id,
                    letztes_symbol,
                    letzte_position,
                    total,
                    downloads_repo.STATUS_WARTET_AUF_WIEDERHOLUNG,
                )
            else:
                events.download_finished(
                    job_id, frisch.status == downloads_repo.STATUS_DONE, frisch.error
                )
    finally:
        _JOBS.pop(job_id, None)


def _chunk_fortschritt(
    job_id: str, symbol: str, position: int, symbole_gesamt: int
) -> Callable[[int, int], None]:
    """Baut den Chunk-Progress-Callback für einen Symbol-Fetch (T-F10).

    Der Callback läuft im Provider-Thread (``asyncio.to_thread``) und meldet
    jeden abgeschlossenen ``1m``-Chunk als **zusätzliches** ``download_progress``
    mit unveränderten Symbol-Zählern plus ``chunk_done``/``chunk_total``
    (1-basiert, „Fenster x/y“). ``gesamt <= 1`` (nicht gechunkte Abrufe, z. B.
    ``1d``) meldet sich nicht — Alt-Payloads bleiben byte-identisch, die
    je-Symbol-Emission in :func:`_job_ausfuehren` bleibt bestehen.

    Die Veröffentlichung erfolgt via ``call_soon_threadsafe``: der SSE-Bus
    (:mod:`quantlab.events`) ist Loop-gebunden, der Callback kommt aus einem
    Executor-Thread. Pannen im Callback selbst schluckt der Provider
    (:meth:`YahooDirectProvider._fortschritt_melden`) — der Fetch stirbt nie am
    Progress.
    """
    loop = asyncio.get_running_loop()

    def fortschritt(fertig: int, gesamt: int) -> None:
        if gesamt <= 1:
            return
        loop.call_soon_threadsafe(
            functools.partial(
                events.download_progress,
                job_id,
                symbol,
                position,
                symbole_gesamt,
                downloads_repo.STATUS_RUNNING,
                chunk_done=fertig,
                chunk_total=gesamt,
            )
        )

    return fortschritt


async def _symbol_laden(
    provider: DataProvider,
    symbol: str,
    interval: str,
    params: dict[str, Any],
    *,
    job_id: str | None = None,
    position: int = 0,
    symbole_gesamt: int = 0,
) -> bool:
    """Lädt und sichert ein Symbol; ``True``, wenn neue Balken dazukamen.

    Inkrementeller Start nur, wenn der Nutzer keinen expliziten Zeitraum
    übergeben hat (``start`` ODER ``period`` — siehe Merge-Gate unten;
    explizit = bewusste Abdeckungs-Anfrage, sonst schlaue Speicherung).
    ``repair=True`` (T19): Fetch mit ``repair=True`` ab ``first_ts`` des
    Bestands (statt Inkrement ab ``last_ts``) — das gesamte Bestandsfenster
    wird neu geladen und via Upsert ersetzt; Balken außerhalb des gelieferten
    Fensters bleiben unangetastet. ``force_full=True`` (MINOR 6, T20-Fix):
    derselbe Vollfenster-Start **ohne** repair-Flag — unterdrückt den
    Inkrement wie ein expliziter ``start=first_ts``. Merge-Gate
    (E090/T-G4): bei ausdrücklicher Zeitraum-Wahl (``period`` ODER ``start``)
    und existierendem Bestand läuft der Fetch ab ``min(fetch_start,
    first_ts)`` — der angefragte Bereich wird vollständig abgedeckt, ohne den
    älteren Bestand zu verwerfen (Mittellücken schließt der fenster-additive
    Upsert).

    Herkunft: nach dem Sichern wird ``bar_meta.source`` auf die Quelle des
    Fetch gesetzt (``'yahoo' | 'stooq'``, E023) — Composite-Provider melden
    sie via ``letzte_quelle()``, schlichte Provider gelten als ``'yahoo'``.
    Seit E094/H2 laufen ISIN-förmige Symbole (ISO-6166-Form, :func:`ist_isin`)
    VOR der Fallback-Kette **ausschließlich** über das Byteworx-Bein
    (``bar_meta.source = 'byteworx'``) — Fenster-, Upsert- und Meta-Mechanik
    bleiben identisch; der Nicht-ISIN-Pfad hält exakt den Alt-Fluss.

    Chunk-Progress (T-F10): mit ``job_id``/``position``/``symbole_gesamt``
    baut der Manager einen Callback und reicht ihn — duck-geoptimer
    Vertragszusatz, gelesen wie ``letzte_quelle`` (
    :func:`quantlab.providers.yahoo_direct.unterstuetzt_chunk_fortschritt`) —
    an den Provider durch. Provider ohne den Zusatz (yfinance-/Stooq-Bein,
    Test-Fakes) erhalten den klassischen Call ohne Callback; das yfinance-Bein
    hat bewusst keine Chunk-Progress-Pflicht (T-F10: nur YahooDirect chunkt
    sichtbar, das Bein ist reiner Fallback).
    """
    if ist_isin(symbol):
        # E094/H2 (REV-Fix): ISINs store-kanonisch in UPPER — psb_repo und
        # der Bulk-Backfill (H5) schreiben UPPER; sonst Split-Brain je
        # Nutzereingabe (Parquet-Key, bar_meta, Events).
        symbol = symbol.strip().upper()
    meta_vorher = await asyncio.to_thread(parquet_store.get_bar_meta, symbol, interval)
    basis_start = params.get("basis_start_us")
    basis_ende = params.get("basis_end_us")
    expliziter_start = params.get("start") is not None
    repair = bool(params.get("repair"))
    force_full = bool(params.get("force_full"))
    # Ausdrückliche Zeitraum-Wahl (E090/T-G4): force_full/repair (Vollfenster)
    # oder period/start (angefragte Abdeckung). Nur OHNE alles gilt der
    # Inkrement (Watchlist-Tick sendet period=None, keinen start).
    zeitraum_gewaehlt = (
        force_full or repair or params.get("period") is not None or expliziter_start
    )
    fetch_start = basis_start
    if (
        not zeitraum_gewaehlt
        and meta_vorher is not None
        and meta_vorher.last_ts is not None
    ):
        fetch_start = _inkrement_start(interval, meta_vorher.last_ts)
    if (
        zeitraum_gewaehlt
        and meta_vorher is not None
        and meta_vorher.first_ts is not None
    ):
        # E090/T-G4 Merge-Gate — generalisiert den force_full-Zweig (MINOR 6)
        # und repair (T19): Fetch ab ``min(basis_start, first_ts)`` (Voll-
        # fenster). Der Bestandsanfang dehnt den angefragten Bereich höchstens
        # aus (Anlass-Fall: AAPL-Bestand 2,7 J. + „5y“ lief sonst als Inkrement
        # ab ``last_ts`` — die fehlende Vergangenheit blieb dauerhaft leer und
        # das Ergebnis verschwieg die Lücke). Mittellücken sind automatisch
        # abgedeckt, weil ab ``fetch_start`` bis ``basis_ende`` geliefert und
        # per Upsert fenster-additiv gemerged wird.
        fetch_start = (
            min(basis_start, meta_vorher.first_ts)
            if basis_start is not None
            else meta_vorher.first_ts
        )

    # E094/H2: ISIN-förmige Symbole laufen VOR der Fallback-Kette ausschließlich
    # über das Byteworx-Bein (autoritative Wettbewerbskurse, Modul-Docstring
    # byteworx.py) — der Nicht-ISIN-Pfad bleibt byte-identisch (Alt-Fluss).
    aktiver_provider: DataProvider = provider
    if ist_isin(symbol):
        aktiver_provider = _byteworx_provider()

    fortschritt = (
        _chunk_fortschritt(job_id, symbol, position, symbole_gesamt)
        if job_id is not None and unterstuetzt_chunk_fortschritt(aktiver_provider)
        else None
    )
    if fortschritt is not None:
        balken = await asyncio.to_thread(
            aktiver_provider.fetch_ohlcv,
            symbol,
            interval,
            fetch_start,
            basis_ende,
            repair,
            fortschritt=fortschritt,
        )
    elif repair:
        balken = await asyncio.to_thread(
            aktiver_provider.fetch_ohlcv, symbol, interval, fetch_start, basis_ende, True
        )
    else:
        balken = await asyncio.to_thread(
            aktiver_provider.fetch_ohlcv, symbol, interval, fetch_start, basis_ende
        )
    meta = await asyncio.to_thread(parquet_store.upsert_bars, symbol, interval, balken)

    if not balken.empty:
        quelle = "yahoo"
        letzte_quelle = getattr(aktiver_provider, "letzte_quelle", None)
        if callable(letzte_quelle):
            quelle = letzte_quelle()
        await asyncio.to_thread(_bar_meta_source_setzen, symbol, interval, quelle)

    await _registry_metadata_pflegen(aktiver_provider, symbol)

    if meta_vorher is None:
        return not balken.empty
    return meta.row_count > meta_vorher.row_count


async def _registry_metadata_pflegen(provider: DataProvider, symbol: str) -> None:
    """Exchange-TZ & Co. in ``symbols`` nachziehen (best-effort, einmal je Symbol)."""
    try:
        vorhanden = await asyncio.to_thread(symbols_repo.get_symbol, symbol)
        if vorhanden is not None and vorhanden.exchange_tz:
            return
        info = await asyncio.to_thread(provider.resolve_symbol, symbol)
    except Exception:  # Metadaten dürfen einen Download nie scheitern lassen
        return
    await asyncio.to_thread(
        symbols_repo.upsert_symbol,
        symbol,
        name=info.name,
        asset_type=info.asset_type,
        exchange_tz=info.exchange_tz,
        currency=info.currency,
    )


def _zusammenfassung(fehler: dict[str, str]) -> str | None:
    """Kompakte deutsche Fehlerzusammenfassung (``Symbol: Meldung``-Paare)."""
    if not fehler:
        return None
    text = "; ".join(f"{symbol}: {meldung}" for symbol, meldung in sorted(fehler.items()))
    return _kappe(text)


def _kappe(text: str) -> str:
    """Kappt eine Meldung auf die Job-Feldlänge (:data:`_MAX_FEHLER_LAENGE`)."""
    if len(text) > _MAX_FEHLER_LAENGE:
        return text[: _MAX_FEHLER_LAENGE - 3] + "…"
    return text


# ---------------------------------------------------------------------------
# Auto-Retry (E089/T-G3): Plan-Pflege + Retry-Tick
# ---------------------------------------------------------------------------


def _wiederholungsplan_pflegen(
    job_id: str,
    interval: str,
    params: dict[str, Any],
    pannen: dict[str, BaseException],
    erfolgreiche: list[str],
) -> tuple[bool, bool]:
    """Schreibt den Wiederholungsplan nach einem Job-Lauf (sync — Thread).

    Plan-Zeilen leben unter der **Original-Job-ID** (bei Retry-Jobs unter den
    ``retry_von``-Vorgängern, korreliert über ``params["retry_von"]``):
    erfolgreiche Symbole räumen ihre Zeile, transient fehlgeschlagene werden
    mit ``versuch + 1`` und ``naechster_versuch_us = jetzt +
    max(Backoff, Circuit-Rest + Puffer)`` fortgeschrieben. Ab Versuch 6
    (:data:`_MAX_RETRIES_VERSUCHE`) endet die Wiederholung — die Zeile bleibt
    als Tombstone (``versuch >= 6``) liegen, ebenso terminal fehlgeschlagene
    Symbole (404/422: Delisted-Schutz — nie wiederholt, aber als „beendet
    ohne Erfolg" sichtbar für die Ursprungs-Job-Finalisierung). Generische
    (nicht-:class:`ProviderError`-)Ausnahmen laufen über dieselbe terminale
    Klassifikation (REV-G MAJOR-3: „unbekannte Ausnahmen sind terminal" —
    sie dürfen nie als Erfolg gelten und nie still aus dem Plan fallen).

    Rückgabe: ``(offene_wiederholungen, wiederholung_beendet)``.
    """
    vorgaenger = params.get("retry_von")
    plan_job_ids = (
        [str(jid) for jid in vorgaenger] if vorgaenger else [job_id]
    )
    jetzt_us = _datetime_zu_us(_jetzt())
    rest_s = direct_circuit_rest_s()
    offene = False
    beendet = False
    for symbol in erfolgreiche:
        for jid in plan_job_ids:
            download_retries_repo.loesche_zeile(jid, symbol, interval)
    for symbol, panne in pannen.items():
        code = panne.code if isinstance(panne, ProviderError) else None
        if _ist_transient(panne):
            versuch = 0
            for jid in plan_job_ids:
                zeile = download_retries_repo.get_zeile(jid, symbol, interval)
                if zeile is not None:
                    versuch = max(versuch, zeile.versuch)
            versuch += 1
            if versuch >= _MAX_RETRIES_VERSUCHE:
                # Tombstone: Versuchslimit erreicht — Wiederholung beendet.
                beendet = True
                naechster_us = jetzt_us
            else:
                delay_s = max(
                    _RETRY_BACKOFF_S[min(versuch, len(_RETRY_BACKOFF_S)) - 1],
                    rest_s + _CIRCUIT_PUFFER_S,
                )
                naechster_us = jetzt_us + int(delay_s * 1_000_000)
                offene = True
            for jid in plan_job_ids:
                download_retries_repo.plane_wiederholung(
                    jid,
                    symbol,
                    interval,
                    versuch=versuch,
                    naechster_versuch_us=naechster_us,
                    fehler_code=code,
                    angelegt_us=jetzt_us,
                )
        else:
            # Terminaler Fehler (404/422 … bzw. generische Ausnahme — REV-G
            # MAJOR-3): nie wiederholt. Bei Retry-Läufen bleibt ein Tombstone
            # (sonst könnte der Ursprungs-Job nie abschließen); bei Erstanlage
            # genügt die Job-Fehlermeldung.
            if vorgaenger:
                beendet = True
                for jid in plan_job_ids:
                    if (
                        download_retries_repo.get_zeile(jid, symbol, interval)
                        is not None
                    ):
                        download_retries_repo.plane_wiederholung(
                            jid,
                            symbol,
                            interval,
                            versuch=_MAX_RETRIES_VERSUCHE,
                            naechster_versuch_us=jetzt_us,
                            fehler_code=code,
                            angelegt_us=jetzt_us,
                        )
    return offene, beendet


async def _finalisiere_vorgaenger(vorgaenger: list[Any]) -> None:
    """Schließt wartende Ursprungs-Jobs ab, deren Wiederholungsplan erledigt ist.

    Ohne offene Plan-Zeilen geht der Ursprungs-Job von
    ``wartet_auf_wiederholung`` terminal: ``error`` mit deutschem Schlusstext
    („Automatische Wiederholung beendet — …“), wenn Tombstones existieren
    (Versuchslimit bzw. terminaler Fehler), sonst ``done`` (alle Symbole
    geladen — der Plan ist geräumt). Läuft über den Monotonie-Guard: ein
    längst gecancelter Ursprungs-Job bleibt unangetastet. Terminale Übergänge
    emittieren reguläres ``download_finished`` (der wartende Toast wird
    geschlossen).
    """
    for roh in vorgaenger:
        job_id_alt = str(roh)
        offen = await asyncio.to_thread(
            download_retries_repo.zaehle_offen, job_id_alt
        )
        if offen > 0:
            continue
        alt = await asyncio.to_thread(downloads_repo.get_download, job_id_alt)
        if alt is None or alt.status != downloads_repo.STATUS_WARTET_AUF_WIEDERHOLUNG:
            continue
        tombsteine = await asyncio.to_thread(
            download_retries_repo.zaehle_beendet_ohne_erfolg, job_id_alt
        )
        if tombsteine > 0:
            text = _kappe(
                "Automatische Wiederholung beendet — "
                + (alt.error or "Symbole konnten auch nach mehreren Versuchen "
                   "nicht geladen werden.")
            )
            akzeptiert = await asyncio.to_thread(
                _status_vorwaerts,
                job_id_alt,
                downloads_repo.STATUS_ERROR,
                error=text,
            )
            if akzeptiert:
                events.download_finished(job_id_alt, False, text)
        else:
            akzeptiert = await asyncio.to_thread(
                _status_vorwaerts,
                job_id_alt,
                downloads_repo.STATUS_DONE,
                error=None,
            )
            if akzeptiert:
                events.download_finished(job_id_alt, True)


def _retry_gruppen(jetzt_us: int, *, erzwingen: bool) -> list[dict[str, Any]]:
    """Baut Retry-Gruppen aus fälligen Plan-Zeilen (sync — Thread).

    Eine Gruppe = ein kommender Retry-Job: gleiche ``interval``-/
    Original-Parameter-Kombination (``period``/``start``/``end``/``repair``/
    ``force_full`` aus dem Ursprungs-Job, ohne ``basis_*``). Gefiltert werden
    Zeilen, deren Symbol bereits in einem offenen (``queued``/``running``)
    Download-Job läuft (In-Flight-Schutz gegen Doppel-Versuche) sowie
    gecancelte/missing Ursprungs-Jobs. ``erzwingen=True`` (``run-now``) nimmt
    ALLE offenen Zeilen — die Fälligkeit wird umgangen.
    """
    if erzwingen:
        zeilen = download_retries_repo.alle_offenen()
    else:
        zeilen = download_retries_repo.faellige(jetzt_us)
    if not zeilen:
        return []
    in_flight: set[str] = set()
    for status in (downloads_repo.STATUS_QUEUED, downloads_repo.STATUS_RUNNING):
        for laufend in downloads_repo.list_nach_status(status, limit=200):
            for symbol in laufend.symbols:
                in_flight.add(f"{symbol}|{laufend.interval}")
    gruppen: dict[tuple[str, str], dict[str, Any]] = {}
    for zeile in zeilen:
        if f"{zeile.symbol}|{zeile.interval}" in in_flight:
            continue
        ursprung = downloads_repo.get_download(zeile.job_id)
        if ursprung is None or ursprung.status == downloads_repo.STATUS_CANCELLED:
            continue
        kwargs = _original_kwargs(ursprung.params or {})
        schluessel = (
            zeile.interval,
            json.dumps(kwargs, sort_keys=True, ensure_ascii=False),
        )
        gruppe = gruppen.get(schluessel)
        if gruppe is None:
            gruppe = {
                "interval": zeile.interval,
                "kwargs": kwargs,
                "symbole": [],
                "vorgaenger": [],
            }
            gruppen[schluessel] = gruppe
        if zeile.symbol not in gruppe["symbole"]:
            gruppe["symbole"].append(zeile.symbol)
        if zeile.job_id not in gruppe["vorgaenger"]:
            gruppe["vorgaenger"].append(zeile.job_id)
    return list(gruppen.values())


async def faellige_retries_starten(
    *,
    now: datetime | None = None,
    erzwingen: bool = False,
) -> dict[str, Any]:
    """Retry-Tick der Auto-Wiederholung (E089/T-G3): fällige Versuche anstoßen.

    Holt die fälligen Plan-Zeilen (``naechster_versuch_us <= now``;
    ``erzwingen=True`` aka ``run_now("download_retry")`` umgeht das Fälligkeits-
    Gate) und legt je (Intervall, Parameter)-Gruppe **einen** Retry-Job über
    :func:`submit_job` an — mit den Original-Parametern und dem internen Marker
    ``retry_von``. Der Versuch läuft dadurch durch den UNVERÄNDERTEN
    ``_job_ausfuehren``/``_symbol_laden``-Apparat (kein zweiter Fetch-Pfad);
    der Tick wartet auf das Job-Ende — die Plan-Fortschreibung und die
    Finalisierung der Ursprungs-Jobs macht dessen Abschluss-Logik.

    ``now`` ist injizierbar (Tests steuern Fälligkeit/Backoff deterministisch,
    Muster ``tick_agent_cron``). Ohne fällige Zeilen: Netto-No-Op.
    """
    jetzt = now if now is not None else _jetzt()
    gruppen = await asyncio.to_thread(
        _retry_gruppen, _datetime_zu_us(jetzt), erzwingen=erzwingen
    )
    if not gruppen:
        return {
            "ok": True,
            "geplant": 0,
            "hinweis": "Keine fälligen Wiederholungen.",
        }
    jobs: list[dict[str, Any]] = []
    for gruppe in gruppen:
        job = await submit_job(
            gruppe["symbole"],
            gruppe["interval"],
            retry_von=gruppe["vorgaenger"],
            **gruppe["kwargs"],
        )
        fertig = await warte_auf_abschluss(job.job_id, _RETRY_JOB_TIMEOUT_S)
        # Auf das vollständige Task-Ende warten, nicht nur auf den Status-
        # übergang: SSE-Emmission und die Finalisierung der Ursprungs-Jobs
        # laufen NACH dem Status-Write im selben Task (das Handle wird im
        # ``finally`` geräumt). So ist bei Tick-Rückkehr der Ursprungs-Job
        # sicher schon abgeschlossen (done bzw. error mit Schlusstext).
        loop = asyncio.get_running_loop()
        ende = loop.time() + _RETRY_JOB_TIMEOUT_S
        while job.job_id in _JOBS and loop.time() < ende:
            await asyncio.sleep(0.01)
        jobs.append(
            {
                "job_id": job.job_id,
                "symbole": gruppe["symbole"],
                "interval": gruppe["interval"],
                "vorgaenger": gruppe["vorgaenger"],
                "status": fertig.status if fertig is not None else None,
            }
        )
    return {"ok": True, "geplant": len(jobs), "jobs": jobs}
