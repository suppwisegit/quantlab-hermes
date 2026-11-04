"""Daten-/Markt-Tools (T14 §E 1–4, T34-Erweiterung): Speicher lesen und füllen,
Live-Quotes, Suche, Fundamentals, Corporate Actions, Watchlists, Scheduler.

Alle Handler laufen laut Registry-Vertrag ``async`` und lagern DB-/pandas-Arbeit
über ``asyncio.to_thread`` in den Threadpool (Muster ``quantlab.db``). Fachliche
Fehler (unbekanntes Symbol, keine Daten, ungültige Argumente) werden zu
:class:`ToolFehler` → ``ok=False``-Tool-Ergebnis (§E: kein Run-Abbruch).

T34-Neuzugänge: ``search_symbols`` (yf.Search mit T19-Cache), ``get_quote`` auf
den T22-Quotes-Service umgestellt (Live-Quote mit Parquet-Fallback, mehrere
Symbole je Abruf), ``get_fundamentals``/``get_corporate_actions`` (nur lokale
Bestände), ``manage_watchlists`` (T28-Repos), ``get_scheduler_status``/
``trigger_data_update`` (T21-Scheduler, ``run_now``).
"""

from __future__ import annotations

import asyncio
import re
from bisect import bisect_right
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, daten_fehler, pruefe_args, schema_von
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.services import data_service, downloads, scheduler
from quantlab.services import fundamentals as fundamentals_service
from quantlab.services import quotes as quotes_service
from quantlab.storage import parquet_store
from quantlab.storage.repos import (
    download_retries_repo,
    downloads_repo,
    symbols_repo,
    watchlists_repo,
)

_US_PER_TAG = parquet_store.US_PER_DAY

#: Maximale Symbolanzahl je ``get_quote``-Batch (Kontext-Schonung).
_MAX_QUOTE_SYMBOLE = 20

#: Ausspielbare Scheduler-Job-Arten (T21/T33-Vertrag ``run_now``).
_SCHEDULER_JOBS: tuple[str, ...] = tuple(scheduler.VERFUEGBARE_JOBS)


# ---------------------------------------------------------------------------
# 0) ISIN-Brücke (E091/T-G5): Erkennung + Auflösung ISIN → Yahoo-Kürzel
# ---------------------------------------------------------------------------

#: ISIN-Form (ISO 6166, ohne Checksummen-Prüfung): 2 Länderbuchstaben + 9
#: alphanumerische NSIN-Zeichen + 1 Prüfziffer. Die Erkennung ist bewusst
#: EIGEN und läuft VOR ``validate_symbol`` — dessen Whitelist
#: (``providers/symbols.py``:30) akzeptiert reine ISINs bereits als Yahoo-
#: Kürzel (E091-Falle); ohne Vorab-Erkennung würde die ISIN still in den
#: Download-Job geschrieben (Anlass-Fund Welle G).
_ISIN_MUSTER = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def ist_isin(wert: str) -> bool:
    """``True``, wenn ``wert`` (getrimmt, großgeschrieben) der ISIN-Form entspricht."""
    if not isinstance(wert, str):
        return False
    return _ISIN_MUSTER.match(wert.strip().upper()) is not None


#: Lokale Seed-Map ISIN → Yahoo-Kürzel (statisch; DAX + US-Blue-Chips, E091).
#: Bewusst als Modul-Dict im Tool-Layer (:mod:`daten`) statt in
#: ``providers/symbols.py``: Es sind Fachdaten des Tool-Auflösungswegs — kein
#: Provider-Konzept, kein Netz — und leben genau dort, wo der Aufrufpfad liegt
#: (``download_data``/``search_symbols``; T-G4 erbt die Stelle). Die Registry
#: (``symbols.isin``) hat Vorrang und füllt sich im Betrieb. Alle ISINs sind
#: ISO-6166-Prüfziffer-verifiziert (inkl. Live-Gegenprüfung Deutsche Börse/
#: Yahoo, 2026-09-21: ENR.DE = Siemens Energy, EOAN.DE = E.ON SE, Hensoldt
#: DE000HAG0005, Hapag-Lloyd DE000HLAG475 nach Kapitalerhöhung 2024).
ISIN_SEEDS: dict[str, str] = {
    # DAX-Kern (Anlass-Fall SAP + Welle-G-DoD)
    "DE0007164600": "SAP.DE",  # SAP SE
    "DE0007236101": "SIE.DE",  # Siemens AG
    "DE0008404005": "ALV.DE",  # Allianz SE
    "DE0005557508": "DTE.DE",  # Deutsche Telekom AG
    "DE000ENER6Y0": "ENR.DE",  # Siemens Energy AG
    "DE000ENAG999": "EOAN.DE",  # E.ON SE
    "DE000SHL1006": "SHL.DE",  # Siemens Healthineers AG
    "DE0008430026": "MRK.DE",  # Merck KGaA
    "DE0005810055": "DB1.DE",  # Deutsche Börse AG
    "DE000HLAG475": "HLAG.DE",  # Hapag-Lloyd AG (ISIN ab Kapitalerhöhung 2024)
    "DE000HAG0005": "HAG.DE",  # Hensoldt AG
    "DE000WCH8881": "WCH.DE",  # Wacker Chemie AG
    # Weitere DAX-/Blue-Chip-Beine (statisch, verifizierbare Standard-ISINs)
    "DE0007037129": "RWE.DE",  # RWE AG
    "DE000BASF111": "BAS.DE",  # BASF SE
    "DE000BAY0017": "BAY.DE",  # Bayer AG
    "DE0005190003": "BMW.DE",  # BMW AG
    "DE0007100000": "MBG.DE",  # Mercedes-Benz Group AG
    "DE0007664039": "VOW3.DE",  # Volkswagen AG (Vz.)
    "DE0005140008": "DBK.DE",  # Deutsche Bank AG
    "DE000CBK1001": "CBK.DE",  # Commerzbank AG
    "DE0006231004": "IFX.DE",  # Infineon Technologies AG
    "DE000ZAL1111": "ZAL.DE",  # Zalando SE
    "NL0000235190": "AIR.DE",  # Airbus SE (niederländische ISIN)
    # US-Blue-Chips
    "US5949181045": "MSFT",  # Microsoft
    "US0378331005": "AAPL",  # Apple
}

#: Hook-Stelle dritte Auflösungsstufe (dokumentiert, NICHT implementiert —
#: DoD T-G5): Sobald das T-G2-Bein (leeway.tech) plus Settings-Key verfügbar
#: ist, kann hier eine ISIN-Suche über die Quelle eingehängt werden (RECHERCHE
#: §Welle G, G-0a: „ISIN direkt als {symbol} zulässig“). Die Kette bleibt:
#: Registry → Seed-Map → [Quelle-Suche] → deutscher Fachfehler.


def _loese_isin(isin: str) -> tuple[str | None, str | None]:
    """Eine ISIN auflösen → (Kürzel, Quelle) — Registry, dann Seed-Map.

    Kein Netz-Call (lokal, best effort). Die Rückgabe-``quelle`` dient nur der
    Transparenz im Tool-Ergebnis („registry“ | „seed“ | ``None``).
    """
    normalisiert = isin.strip().upper()
    if not _ISIN_MUSTER.match(normalisiert):
        return None, None
    treffer = symbols_repo.get_symbol_by_isin(normalisiert)
    if treffer is not None:
        return treffer.symbol, "registry"
    seed = ISIN_SEEDS.get(normalisiert)
    return (seed, "seed") if seed else (None, None)


def loese_isin_zu_kuerzel(isin: str) -> str | None:
    """Eine ISIN → Yahoo-Kürzel (Registry → Seed-Map), ``None`` wenn unbekannt.

    Öffentliche Vorlage für lesende Tools (:mod:`planspiel`, ``yahoo_kuerzel``)
    — bewusst netzfrei und damit gefahrlos in Lese-Pfaden nutzbar.
    """
    kuerzel, _quelle = _loese_isin(isin)
    return kuerzel


def _loese_symbole(
    args_symbole: list[str],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Erkennt ISINs VOR ``validate_symbol`` und löst sie zu Kürzeln auf (E091).

    Rückgabe: (Symbol-Liste für den Job, ``aufloesung``-Block je ISIN). Reine
    Kürzel laufen wörtlich durch — der ``aufloesung``-Block bleibt leer, damit
    Alt-Ergebnisse ohne ISINs byte-identisch bleiben. Nicht auflösbare ISINs
    führen zu :class:`ToolFehler` (deutsch, mit Lösungshinweis) — es wird
    bewusst KEIN Job mit geratenen Kürzeln angelegt (Anlass-Fund E091: der
    Agent riet „SAP“/„SIEGY“ und 7 Symbole scheiterten).
    """
    job_symbole: list[str] = []
    aufloesung: dict[str, dict[str, Any]] = {}
    unbekannt: list[str] = []
    for roh in args_symbole:
        eintrag = str(roh).strip()
        if not ist_isin(eintrag):
            job_symbole.append(roh)  # unverändert — Alt-Pfad byte-identisch
            continue
        isin = eintrag.upper()
        kuerzel, quelle = _loese_isin(isin)
        if kuerzel is None:
            unbekannt.append(isin)
            continue
        job_symbole.append(kuerzel)
        aufloesung[isin] = {"symbol": kuerzel, "quelle": quelle}
    if unbekannt:
        raise ToolFehler(
            f"ISIN unbekannt — Symbol in der Suche suchen oder Registry pflegen "
            f"(search_symbols mit der ISIN aufrufen oder das Kürzel in der "
            f"Symbol-Registry hinterlegen): {', '.join(unbekannt)}"
        )
    return job_symbole, aufloesung


# ---------------------------------------------------------------------------
# 1) get_market_data
# ---------------------------------------------------------------------------


class GetMarketDataArgs(BaseModel):
    """Argumente von ``get_market_data``."""

    symbol: str = Field(min_length=1, description='Symbol-Kürzel, z. B. "AAPL" oder "^GSPC".')
    interval: str = Field(default="1d", description='yfinance-Intervall, z. B. "1d", "1h", "1wk".')
    start: str | None = Field(
        default=None, description='Startdatum inklusive, z. B. "2024-01-01" (leer = ab Anfang).'
    )
    end: str | None = Field(
        default=None, description='Enddatum exklusiv, z. B. "2024-12-31" (leer = bis aktuell).'
    )
    limit: int = Field(
        default=260,
        ge=1,
        le=2000,
        description="Maximale Anzahl Balken (jüngste), Default 260 — etwa 1 Handelsjahr.",
    )


def _iso_kompakt(ts_us: int) -> str:
    """µs UTC → kompakter Zeitstempel (Datum; bei Intraday mit HH:MM)."""
    moment = datetime.fromtimestamp(ts_us / 1_000_000, tz=UTC)
    if (moment.hour, moment.minute, moment.second) == (0, 0, 0):
        return moment.date().isoformat()
    return moment.strftime("%Y-%m-%dT%H:%M")


def _balken_kompakt(balken: pd.DataFrame) -> list[list[Any]]:
    """Balken-Frame → kompakte Zeilen ``[zeit, open, high, low, close, volume]``."""
    zeilen: list[list[Any]] = []
    for bar in balken.itertuples(index=False):
        zeilen.append(
            [
                _iso_kompakt(int(bar.ts)),
                round(float(bar.open), 4),
                round(float(bar.high), 4),
                round(float(bar.low), 4),
                round(float(bar.close), 4),
                round(float(bar.volume), 0),
            ]
        )
    return zeilen


async def get_market_data(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Liest gespeicherte OHLCV-Balken: Summary-Statistik + kompakte Balkenliste."""
    args = pruefe_args(GetMarketDataArgs, roh_args, "get_market_data")

    def _lesen() -> tuple[str, pd.DataFrame]:
        start_us = downloads.iso_zeitpunkt_zu_us(args.start) if args.start else None
        end_us = downloads.iso_zeitpunkt_zu_us(args.end) if args.end else None
        balken = data_service.read_ohlcv(
            args.symbol, args.interval, start=start_us, end=end_us, limit=args.limit
        )
        return args.interval, balken

    try:
        interval, balken = await asyncio.to_thread(_lesen)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    if balken.empty:
        raise ToolFehler(
            f"Für „{args.symbol}“ ({interval}) liegen im gewählten Zeitraum keine Balken vor. "
            "Zeitraum anpassen oder download_data verwenden."
        )

    letzte = balken.iloc[-1]
    zusammenfassung = {
        "symbol": args.symbol,
        "interval": interval,
        "anzahl_balken": int(len(balken)),
        "zeitraum": {
            "von": _iso_kompakt(int(balken.iloc[0]["ts"])),
            "bis": _iso_kompakt(int(letzte["ts"])),
        },
        "letzter_kurs": round(float(letzte["close"]), 4),
        "hoch": round(float(balken["high"].max()), 4),
        "tief": round(float(balken["low"].min()), 4),
        "durchschnitts_volumen": round(float(balken["volume"].mean()), 0),
        "format_balken": "[zeit, open, high, low, close, volume]",
    }
    # Bewusst Reihenfolge: Zusammenfassung VOR den Balken — beim Result-Kapping
    # (registry.TOOL_RESULT_MAX_CHARS) gehen nur die rohen Balken verloren.
    return {"zusammenfassung": zusammenfassung, "balken": _balken_kompakt(balken)}


# ---------------------------------------------------------------------------
# 2) list_symbols
# ---------------------------------------------------------------------------


class ListSymbolsArgs(BaseModel):
    """Argumente von ``list_symbols`` (bewusst leer — keine Parameter)."""


async def list_symbols(_roh_args: dict[str, Any]) -> dict[str, Any]:
    """Listet gespeicherte Symbole mit Datenabdeckung je Intervall auf."""

    def _lesen() -> list[dict[str, Any]]:
        resultate: list[dict[str, Any]] = []
        for symbol in symbols_repo.list_symbols():
            abdeckung: dict[str, dict[str, Any]] = {}
            for meta in parquet_store.list_bar_meta(symbol.symbol):
                abdeckung[meta.interval] = {
                    "row_count": meta.row_count,
                    "first_ts": _iso_kompakt(meta.first_ts) if meta.first_ts else None,
                    "last_ts": _iso_kompakt(meta.last_ts) if meta.last_ts else None,
                }
            resultate.append(
                {
                    "symbol": symbol.symbol,
                    "name": symbol.name,
                    "abdeckung": abdeckung,
                }
            )
        return resultate

    symbole = await asyncio.to_thread(_lesen)
    return {"anzahl": len(symbole), "symbole": symbole}


# ---------------------------------------------------------------------------
# 3) download_data
# ---------------------------------------------------------------------------


class DownloadDataArgs(BaseModel):
    """Argumente von ``download_data`` (E090/T-G4: start/force_full/repair)."""

    symbols: list[str] = Field(
        min_length=1,
        description=(
            'Symbol-Kürzel, z. B. ["AAPL", "MSFT"] — auch ISINs (z. B. '
            '"DE0007164600") sind erlaubt und werden vor dem Job aufgelöst.'
        ),
    )
    interval: str = Field(default="1d", description='yfinance-Intervall, z. B. "1d" oder "1h".')
    period: str | None = Field(
        default="1y",
        description=(
            'Zeitraum, z. B. "1y" (Default), "6mo", "2y", "5y", "max", "ytd"; '
            "null = reiner Inkrement-Update (nur neue Balken, Watchlist-Stil, "
            "kein Coverage-Bericht). Ein vorhandener Bestand bleibt erhalten — "
            "es wird nur das Fehlende ergänzt; die echte Abdeckung steht im "
            "Ergebnisfeld coverage."
        ),
    )
    start: str | None = Field(
        default=None,
        description=(
            'Startdatum JJJJ-MM-TT (z. B. "2021-09-01") für einen gezielten '
            "Zeitraum ab diesem Datum. Schlägt period vor — sind beide "
            "gesetzt, gilt start und period wird ignoriert (REST-Konvention)."
        ),
    )
    force_full: bool = Field(
        default=False,
        description=(
            "True = vorhandenen Bestand vollständig neu laden (ab dem ersten "
            "gespeicherten Balken), statt nur zu ergänzen. Default False."
        ),
    )
    repair: bool = Field(
        default=False,
        description=(
            "True = Bestandsfenster mit Korrektur-Flag der Quelle neu laden "
            "(schließt Lücken und korrigiert Balken innerhalb der Historie). "
            "Default False."
        ),
    )


#: Eigenes Warte-Limit (s) — bewusst unter dem Dispatch-Timeout von 120 s,
#: damit das Tool noch berichten kann, statt selbst gekappt zu werden.
_DOWNLOAD_TIMEOUT_S = 110.0


# ---------------------------------------------------------------------------
# 3a) download_data-Helfer (E089/T-G3 + E090/T-G4): Wiederholungen,
#     Start-Validierung, Laufenden-Job-Dedup, Coverage
# ---------------------------------------------------------------------------

#: Toleranz der Warnung „geliefert < angefragt“: Ein period/start-Anfang darf
#: auf ein Wochenende/Feiertag fallen; liefert die Quelle ab dem ersten
#: Handelstag danach, ist das KEINE ungelieferte Lücke.
_COVERAGE_TOLERANZ_US = 7 * _US_PER_TAG

#: Mikrosekunden eines Jahres (365 Tage) für die Warnungs-Spannenangabe.
_JAHR_US = 365 * _US_PER_TAG


def _warnungs_spannen(
    geliefert_us: int, angefragt_us: int
) -> tuple[str, str, str]:
    """(geliefert, angefragt, Einheit) der Coverage-Warnung.

    Ab 1 Jahr angefragter Spanne in Jahren mit deutschem Dezimalkomma
    („2,7“ / „5,0“ / „Jahren“), darunter in Tagen — beide Werte stets in
    derselben Einheit (geliefert ≤ angefragt)."""
    if angefragt_us >= _JAHR_US:
        return (
            f"{geliefert_us / _JAHR_US:.1f}".replace(".", ","),
            f"{angefragt_us / _JAHR_US:.1f}".replace(".", ","),
            "Jahren",
        )
    return (
        str(max(1, geliefert_us // _US_PER_TAG)),
        str(max(1, angefragt_us // _US_PER_TAG)),
        "Tagen",
    )


def _validiere_start_datum(start: str | None) -> str | None:
    """``start`` deutsch validieren (JJJJ-MM-TT genügt; ISO-8601 erlaubt).

    Kommt VOR dem Job-Anlegen — ein Tippfehler im Datum ist ein deutscher
    Tool-Fehler (``ok=False``), kein 422-Provider-Umweg mit englischem Hint.
    """
    if start is None or not start.strip():
        return start
    try:
        downloads.iso_zeitpunkt_zu_us(start)
    except ProviderError as exc:
        raise ToolFehler(
            f'Ungültiges Startdatum „{start}“ — bitte im Format JJJJ-MM-TT '
            'angeben (z. B. "2021-09-01").'
        ) from exc
    return start.strip()


def _laufenden_job_finden(
    symbole: list[str], interval: str
) -> downloads_repo.Download | None:
    """Offenen Job mit identischem Symbol-Set + Intervall finden (E090/T-G4).

    Das Tool wartet max. ~110 s und meldet sonst „läuft noch“ — der typische
    Folgeruf des Agenten darf dann NICHT erneut einreihen (Doppel-Job,
    doppelter Netz-Traffic, doppelte Yahoo-Last), sondern bekommt DIESEN Job
    gemeldet (``status: "laeuft_bereits"``). ``queued``/``running`` zählen als
    offen („running“ zuerst — näher am Ende); der Symbol-Vergleich läuft als
    Menge (Reihenfolge egal, Duplikate dedupliziert ``create_job`` ohnehin).
    Best effort: endet der Job zwischen Prüfung und Rückkehr, wird regulär
    neu eingereiht (harmlos).
    """
    wunsch = {str(symbol) for symbol in symbole}
    for status in (downloads_repo.STATUS_RUNNING, downloads_repo.STATUS_QUEUED):
        for job in downloads_repo.list_nach_status(status, limit=200):
            if job.interval == interval and set(job.symbols) == wunsch:
                return job
    return None


def _coverage_lesen(
    ergebnis: downloads_repo.Download, params: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Echte Abdeckung je Symbol aus ``bar_meta`` nach Job-Abschluss (E090).

    ``{first_ts, last_ts, rows, quelle}`` — ``quelle`` aus ``bar_meta.source``
    (bewusst ``None`` ohne Bestand: das Alt-Default ``'yahoo'`` würde eine
    Herkunft vortäuschen — genau die Konfabulations-Falle E090). ``warnung``
    (im Symbol-Eintrag), wenn die gelieferte Spanne kürzer ist als der
    angefragte Zeitraum (``params["basis_start_us"]`` vs. ``first_ts``; 7-Tage-
    Toleranz gegen Wochenend-Anfänge): „Nur 2,7 von 5,0 Jahren verfügbar —
    die Quelle lieferte kein Älteres.“ Sync-Funktion (Thread-Pfad).
    """
    angefragt_start = params.get("basis_start_us") if params else None
    abdeckung: dict[str, dict[str, Any]] = {}
    for symbol in ergebnis.symbols:
        meta = parquet_store.get_bar_meta(symbol, ergebnis.interval)
        if meta is None:
            abdeckung[symbol] = {
                "first_ts": None,
                "last_ts": None,
                "rows": None,
                "quelle": None,
            }
            continue
        zeile: dict[str, Any] = {
            "first_ts": _iso_kompakt(meta.first_ts) if meta.first_ts else None,
            "last_ts": _iso_kompakt(meta.last_ts) if meta.last_ts else None,
            "rows": meta.row_count,
            "quelle": downloads.bar_meta_source(symbol, ergebnis.interval),
        }
        if (
            meta.first_ts is not None
            and meta.last_ts is not None
            and isinstance(angefragt_start, int)
            and meta.first_ts - angefragt_start > _COVERAGE_TOLERANZ_US
        ):
            geliefert, angefragt, einheit = _warnungs_spannen(
                meta.last_ts - meta.first_ts, meta.last_ts - angefragt_start
            )
            zeile["warnung"] = (
                f"Nur {geliefert} von {angefragt} {einheit} verfügbar — "
                "die Quelle lieferte kein Älteres."
            )
        abdeckung[symbol] = zeile
    return abdeckung


def _wiederholungen_lesen(job_id: str) -> list[dict[str, Any]]:
    """Wiederholungsplan-Zeilen eines wartenden Jobs → strukturiert je Symbol.

    E089/T-G3: ``{symbol, code, wiederholung_um (ISO-UTC), versuch}`` — Quelle
    sind die offenen Plan-Zeilen (``download_retries``); Tombstones
    (``versuch >= 6``) zählen nicht — der Job ist dann längst terminal.
    Aufsteigend nach Symbol (deterministische Präsentation).
    """
    zeilen = [
        zeile
        for zeile in download_retries_repo.zeilen_fuer_job(job_id)
        if zeile.versuch < download_retries_repo.MAX_VERSUCHE
    ]
    return [
        {
            "symbol": zeile.symbol,
            "code": zeile.letzte_fehler_code,
            "wiederholung_um": datetime.fromtimestamp(
                zeile.naechster_versuch_us / 1_000_000, tz=UTC
            ).isoformat(),
            "versuch": zeile.versuch,
        }
        for zeile in sorted(zeilen, key=lambda zeile: zeile.symbol)
    ]


async def download_data(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Stößt einen Download-Job an und wartet auf das Ende (max. ~110 s).

    E090/T-G4: Bei ausdrücklicher Zeitraum-Wahl (``period`` oder ``start``)
    meldet das Ergebnis zusätzlich ``coverage`` je Symbol (first_ts/last_ts/
    rows/quelle + ``warnung``, wenn die Quelle weniger liefert als angefragt).
    Läuft für dasselbe Symbol-Set + Intervall bereits ein Job, wird KEIN
    zweiter eingereiht — das Ergebnis meldet diesen Job
    (``status: "laeuft_bereits"``, ``job_laueft_weiter: true``).
    """
    args = pruefe_args(DownloadDataArgs, roh_args, "download_data")
    # E091: ISINs werden VOR submit_job aufgelöst (Registry → Seed-Map) —
    # ``validate_symbol`` (im Job-Manager) ließe sie sonst still als Kürzel
    # durch. Nicht auflösbare ISINs werfen hier den deutschen Fachfehler,
    # BEVOR ein Job mit geratenen Kürzeln angelegt wird.
    symbole, aufloesung = await asyncio.to_thread(_loese_symbole, args.symbols)
    start = _validiere_start_datum(args.start)
    # E090/T-G4 (Timeout-/Doppelruf-Pfad): identisches Symbol-Set + Intervall
    # während eines laufenden Jobs → DIESEN Job melden statt erneut einzureihen.
    vorhanden = await asyncio.to_thread(_laufenden_job_finden, symbole, args.interval)
    if vorhanden is not None:
        antwort: dict[str, Any] = {
            "status": "laeuft_bereits",
            "job_id": vorhanden.job_id,
            "job_laueft_weiter": True,
            "meldung": (
                "Für dieses Symbol-Set und Intervall läuft bereits ein "
                "Download-Job — keine zweite Einreihung. Er läuft weiter; "
                "bitte später erneut mit get_market_data prüfen."
            ),
        }
        if aufloesung:
            antwort["aufloesung"] = aufloesung
        return antwort
    try:
        job = await downloads.submit_job(
            symbole,
            args.interval,
            period=args.period,
            start=start,
            repair=args.repair,
            force_full=args.force_full,
        )
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    ergebnis = await downloads.warte_auf_abschluss(job.job_id, timeout_s=_DOWNLOAD_TIMEOUT_S)
    if ergebnis is None:
        antwort = {
            "status": "laeuft_noch",
            "job_id": job.job_id,
            "meldung": (
                "Der Download läuft noch (Timeout der Wartezeit erreicht). "
                "Er läuft im Hintergrund weiter; bitte später erneut mit "
                "get_market_data prüfen."
            ),
        }
        if aufloesung:
            antwort["aufloesung"] = aufloesung
        return antwort

    def _zeilen() -> dict[str, Any]:
        zaehler: dict[str, int | None] = {}
        for symbol in ergebnis.symbols:
            meta = parquet_store.get_bar_meta(symbol, ergebnis.interval)
            zaehler[symbol] = meta.row_count if meta is not None else None
        return zaehler

    row_counts = await asyncio.to_thread(_zeilen)
    antwort = {
        "status": ergebnis.status,
        "job_id": ergebnis.job_id,
        "interval": ergebnis.interval,
        "row_counts": row_counts,
    }
    if args.period is not None or start is not None:
        # E090/T-G4: echte Abdeckung je Symbol (additiv) — nur bei
        # ausdrücklicher Zeitraum-Wahl. Ohne period/start (Inkrement-Pfad,
        # Watchlist-Stil) bleibt das Alt-Ergebnis ohne coverage-Feld.
        antwort["coverage"] = await asyncio.to_thread(
            _coverage_lesen, ergebnis, ergebnis.params
        )
    if ergebnis.status == downloads_repo.STATUS_WARTET_AUF_WIEDERHOLUNG:
        # E089/T-G3: wartende Symbole strukturiert je Symbol melden (Feld nur
        # in diesem Fall — der Alt-Pfad bleibt byte-identisch).
        antwort["wiederholungen"] = await asyncio.to_thread(
            _wiederholungen_lesen, ergebnis.job_id
        )
    if ergebnis.error:
        antwort["hinweis"] = ergebnis.error
    if aufloesung:  # additiv (E091): nur bei ISIN-Input — Alt-Form bleibt identisch
        antwort["aufloesung"] = aufloesung
    return antwort


# ---------------------------------------------------------------------------
# 4) get_quote (T34: T22-Quotes-Service + Parquet-Fallback, mehrere Symbole)
# ---------------------------------------------------------------------------


class GetQuoteArgs(BaseModel):
    """Argumente von ``get_quote`` — ``symbol`` (eins) oder ``symbols`` (Batch)."""

    symbol: str | None = Field(
        default=None,
        min_length=1,
        description='Ein Symbol-Kürzel, z. B. "MSFT" (oder „symbols“ angeben).',
    )
    symbols: list[str] | None = Field(
        default=None,
        description='Mehrere Symbol-Kürzel, z. B. ["AAPL", "MSFT"] (max. 20 je Abruf).',
    )


def _tages_balken(symbol: str) -> tuple[str | None, pd.DataFrame]:
    """Beste gespeicherte Serie für Tagesbetrachtungen (1d bevorzugt)."""
    metas = parquet_store.list_bar_meta(symbol)
    if not metas:
        return None, pd.DataFrame()
    interval = next((m.interval for m in metas if m.interval == "1d"), metas[0].interval)
    try:
        return interval, data_service.read_ohlcv(symbol, interval, limit=400)
    except ProviderError:
        return interval, pd.DataFrame()


def _veraenderungen(balken: pd.DataFrame) -> tuple[int, float, dict[str, float | None]]:
    """Balken → (letzter ts µs, letzter Kurs, %-Veränderung über 1T/1W/1M)."""
    ts_liste = balken["ts"].astype("int64").tolist()
    closes = balken["close"].astype(float).tolist()
    letzter_ts = int(ts_liste[-1])
    letzter_kurs = float(closes[-1])

    def _veraenderung(tage: float) -> float | None:
        """Ø-Change gegenüber dem letzten Balken vor ``tage`` Tagen (oder ``None``)."""
        ziel_us = letzter_ts - int(tage * _US_PER_TAG)
        index = bisect_right(ts_liste, ziel_us) - 1
        if index < 0:
            return None
        basis = float(closes[index])
        if basis == 0.0:
            return None
        return round((letzter_kurs / basis - 1.0) * 100.0, 2)

    return (
        letzter_ts,
        letzter_kurs,
        {"1t": _veraenderung(1.0), "1w": _veraenderung(7.0), "1m": _veraenderung(30.0)},
    )


def _quote_zeile(symbol: str, eintrag: dict[str, Any] | None) -> dict[str, Any]:
    """Ein Symbol → Quote-Zeile: Live-Quote (T22) mit gespeicherter Kursbasis."""
    fehler: str | None = None
    live: dict[str, Any] | None = None
    if eintrag is not None:
        preis = eintrag.get("price")
        if "error" not in eintrag and isinstance(preis, (int, float)) and preis > 0:
            live = eintrag
        else:
            panne = eintrag.get("error") or {}
            fehler = (
                f"{panne.get('code', 'provider_fehler')}: "
                f"{panne.get('message', 'kein Live-Quote')}"
            )

    interval, balken = _tages_balken(symbol)
    kurs: float | None = None
    zeitpunkt: str | None = None
    aenderungen: dict[str, float | None] = {"1t": None, "1w": None, "1m": None}
    if not balken.empty:
        letzter_ts, letzter_kurs, aenderungen = _veraenderungen(balken)
        kurs = round(letzter_kurs, 4)
        zeitpunkt = _iso_kompakt(letzter_ts)

    zeile: dict[str, Any] = {"symbol": symbol}
    if live is not None:
        quote_ts = live.get("quote_ts")
        zeile.update(
            preis=round(float(live["price"]), 4),
            quelle="live",
            waehrung=live.get("currency"),
            vorheriger_schluss=(
                round(float(live["previous_close"]), 4)
                if isinstance(live.get("previous_close"), (int, float))
                else None
            ),
            zeitpunkt=_iso_kompakt(int(quote_ts))
            if isinstance(quote_ts, int)
            else zeitpunkt,
            veraenderung_prozent=aenderungen,
            interval=interval,
        )
        return zeile
    if kurs is not None:
        zeile.update(
            preis=kurs,
            quelle="speicher",
            waehrung=None,
            vorheriger_schluss=None,
            zeitpunkt=zeitpunkt,
            veraenderung_prozent=aenderungen,
            interval=interval,
        )
        if fehler is not None:
            zeile["fehler"] = fehler
        return zeile
    zeile.update(
        preis=None,
        quelle=None,
        zeitpunkt=None,
        veraenderung_prozent=aenderungen,
        fehler=fehler or "Keine Kursdaten (weder Live-Quote noch gespeicherte Balken).",
    )
    zeile["hinweis"] = "download_data kann die Daten zuerst laden."
    return zeile


async def get_quote(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Letzte Kurse (bevorzugt Live, sonst gespeichert) + Veränderung 1T/1W/1M.

    Fehler je Symbol werden im Ergebnisfeld ``fehler`` gemeldet; nur wenn
    **kein** Symbol lieferbar ist, wird ein :class:`ToolFehler` geworfen.
    """
    args = pruefe_args(GetQuoteArgs, roh_args, "get_quote")
    symbole: list[str] = []
    if args.symbol and args.symbol.strip():
        symbole.append(args.symbol.strip())
    if args.symbols:
        symbole.extend(s.strip() for s in args.symbols if s and s.strip())
    symbole = list(dict.fromkeys(symbole))
    if not symbole:
        raise ToolFehler(
            'Bitte „symbol“ (z. B. "MSFT") oder „symbols“ (Liste) angeben.'
        )
    if len(symbole) > _MAX_QUOTE_SYMBOLE:
        raise ToolFehler(
            f"Zu viele Symbole ({len(symbole)}) — maximal {_MAX_QUOTE_SYMBOLE} "
            "je Abruf."
        )

    def _lesen() -> tuple[list[dict[str, Any]], list[str]]:
        eintraege = quotes_service.get_service().liefere(symbole)
        je_symbol = {
            str(eintrag.get("symbol")): eintrag
            for eintrag in eintraege
            if eintrag.get("symbol")
        }
        zeilen = [_quote_zeile(symbol, je_symbol.get(symbol)) for symbol in symbole]
        ohne_daten = [
            zeile["symbol"]
            for zeile in zeilen
            if zeile.get("quelle") is None
        ]
        return zeilen, ohne_daten

    try:
        zeilen, ohne_daten = await asyncio.to_thread(_lesen)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    if len(ohne_daten) == len(zeilen):
        raise ToolFehler(
            "Kein Symbol lieferte einen Kurs: "
            + "; ".join(f"{z['symbol']}: {z.get('fehler')}" for z in zeilen)
        )
    return {"anzahl": len(zeilen), "zitate": zeilen}


# ---------------------------------------------------------------------------
# 5) search_symbols (T34: yf.Search-Wrapper über den T19-Cache)
# ---------------------------------------------------------------------------


class SearchSymbolsArgs(BaseModel):
    """Argumente von ``search_symbols``."""

    query: str = Field(
        min_length=1,
        max_length=100,
        description=(
            'Suchbegriff (Name oder Kürzel), z. B. "Apple" oder "AAPL" — auch '
            'eine ISIN (z. B. "DE0007164600") ist erlaubt; bei eindeutigem '
            "Treffer wird die ISIN in der Symbol-Registry gepflegt."
        ),
    )
    limit: int = Field(
        default=8, ge=1, le=20, description="Maximale Trefferzahl, Default 8."
    )


def _pflege_isin_in_registry(
    isin: str, treffer: list[dict[str, Any]]
) -> list[str]:
    """Best effort (E091): gefundenem Symbol die ISIN in die Registry schreiben.

    Ziel-Kürzel: die ISIN ist bereits auflösbar (Registry/Seed-Map) und liegt
    unter den Treffern — nur dieser exakte Match wird gepflegt. REV-G MINOR-5:
    ein einzelner Fuzzy-Treffer ohne Auflösung schreibt KEIN ISIN mehr
    (sonst sticky falsches Mapping möglich); mehrdeutige Listen bleiben
    unverändert. Rückgabe: gepflegte Kürzel (leer = nichts geschrieben).
    """
    symbole = [
        str(t.get("symbol"))
        for t in treffer
        if isinstance(t, dict) and t.get("symbol")
    ]
    ziel: str | None = None
    kuerzel = loese_isin_zu_kuerzel(isin)
    if kuerzel:
        ziel = next((s for s in symbole if s.casefold() == kuerzel.casefold()), None)
    if ziel is None:
        return []
    symbols_repo.upsert_symbol(ziel, isin=isin.strip().upper())
    return [ziel]


async def search_symbols(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Sucht Symbols-Kürzel bei Yahoo (24-h-Cache, Rate-Limit-Guard)."""
    args = pruefe_args(SearchSymbolsArgs, roh_args, "search_symbols")
    # E091: ISIN-Form wird normalisiert (trim/groß) und wie bisher an die
    # Suchlogik weitergereicht — Yahoo-Search beantwortet ISIN-Queries.
    isin_form = ist_isin(args.query)
    query = args.query.strip().upper() if isin_form else args.query

    def _suche() -> tuple[list[dict[str, Any]], bool]:
        # ANNAHME: die Suche ist bewusst ein Router-Konsum ohne HTTP-Umweg —
        # T19 (Cache + Circuit-Guard + „erschöpfte Queries“) lebt als
        # Modulfunktion in ``api/routers/data.py``; dieselbe Funktionsreferenz
        # nutzt auch der Symbol-Add-Pfad. Duplizieren würde zwei Cache-
        # Semantiken erzeugen.
        from quantlab.api.routers.data import _symbolsuche

        return _symbolsuche(query, args.limit)

    try:
        treffer, aus_cache = await asyncio.to_thread(_suche)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    ergebnis: dict[str, Any] = {
        "query": query,
        "anzahl": len(treffer),
        "aus_cache": aus_cache,
        "ergebnisse": treffer,
        "hinweis": "Daten zu einem Treffer lädt download_data; der Quote-Typ "
        "(quoteType) hilft bei der Auswahl.",
    }
    if isin_form:  # additiv (E091): Registry-Pflege nur bei ISIN-Query sichtbar
        ergebnis["registry_gepflegt"] = await asyncio.to_thread(
            _pflege_isin_in_registry, query, treffer
        )
    return ergebnis


# ---------------------------------------------------------------------------
# 6) get_fundamentals (T34: lokale Bestände, kein Netz)
# ---------------------------------------------------------------------------

#: Statements des Fundamentals-Schemas (§D.2) und wie viele Perioden das Tool
#: je Statement maximal zurückgibt (Kontext-Schonung; Rest via Kapping eh weg).
_STATEMENTS: tuple[str, ...] = ("income", "balance", "cashflow")
_MAX_PERIODEN = 8


class GetFundamentalsArgs(BaseModel):
    """Argumente von ``get_fundamentals``."""

    symbol: str = Field(min_length=1, description='Symbol-Kürzel, z. B. "AAPL".')
    freq: Literal["annual", "quarterly"] = Field(
        default="annual",
        description='Frequenz: "annual" (Default, fällt bei Leer auf quarterly '
        "zurück) oder \"quarterly\".",
    )


async def get_fundamentals(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Key-Ratios + Statement-Auszüge aus dem gespeicherten Fundamentals-Bestand."""
    args = pruefe_args(GetFundamentalsArgs, roh_args, "get_fundamentals")

    def _lesen() -> dict[str, Any]:
        symbol = validate_symbol(args.symbol)  # ProviderError → daten_fehler
        if not fundamentals_service.hat_daten(symbol):
            return {
                "symbol": symbol,
                "leer": True,
                "hinweis": (
                    "Es sind keine Fundamentals gespeichert — trigger_data_update "
                    'mit job="fundamentals" (Watchlist) kann sie laden.'
                ),
            }
        ratios = fundamentals_service.berechne_ratios(symbol, freq=args.freq)
        statements: dict[str, Any] = {}
        for statement in _STATEMENTS:
            pivot = fundamentals_service.lies_statement_pivot(symbol, statement, args.freq)
            perioden = pivot["periods"][-_MAX_PERIODEN:]
            offset = len(pivot["periods"]) - len(perioden)
            statements[statement] = {
                "line_items": pivot["line_items"],
                "periods": perioden,
                "values": [zeile[offset:] for zeile in pivot["values"]],
                "leer": pivot["leer"],
            }
        return {
            "symbol": symbol,
            "freq": ratios.get("freq", args.freq),
            "basis": ratios.get("basis"),
            "ratios": ratios.get("ratios", []),
            "leer": bool(ratios.get("leer")),
            "statements": statements,
            "hinweis": "Werte stammen aus dem lokalen Bestand (kein Netzabruf).",
        }

    try:
        return await asyncio.to_thread(_lesen)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc


# ---------------------------------------------------------------------------
# 7) get_corporate_actions (T34: lokale Events, E022)
# ---------------------------------------------------------------------------


class GetCorporateActionsArgs(BaseModel):
    """Argumente von ``get_corporate_actions``."""

    symbol: str = Field(min_length=1, description='Symbol-Kürzel, z. B. "AAPL".')
    start: str | None = Field(
        default=None, description='Startdatum inklusive, z. B. "2024-01-01" (leer = ab Anfang).'
    )
    end: str | None = Field(
        default=None, description='Enddatum exklusiv, z. B. "2025-01-01" (leer = bis aktuell).'
    )
    limit: int = Field(
        default=100, ge=1, le=500,
        description="Maximale Anzahl Events (jüngste), Default 100.",
    )


async def get_corporate_actions(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Gespeicherte Splits/Dividenden (E022) — nur lokaler Bestand, kein Netz."""
    args = pruefe_args(GetCorporateActionsArgs, roh_args, "get_corporate_actions")

    def _lesen() -> list[dict[str, Any]]:
        symbol = validate_symbol(args.symbol)
        if not data_service.actions_symbol_bekannt(symbol):
            raise ToolFehler(
                f"Für „{symbol}“ sind weder Kursdaten noch Corporate-Actions "
                "gespeichert. (Hinweis: download_data mit actions lädt beide.)"
            )
        start_us = downloads.iso_zeitpunkt_zu_us(args.start) if args.start else None
        end_us = downloads.iso_zeitpunkt_zu_us(args.end) if args.end else None
        aktionen = data_service.lese_actions(symbol, start_us, end_us)
        return [
            {
                "datum": aktion.datum,
                "action": aktion.action,
                "value": aktion.value,
                "currency": aktion.currency,
            }
            for aktion in aktionen[-args.limit :]
        ]

    try:
        aktionen = await asyncio.to_thread(_lesen)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    dividenden_summe = round(
        sum(a["value"] for a in aktionen if a["action"] == "dividend"), 6
    )
    return {
        "symbol": args.symbol,
        "anzahl": len(aktionen),
        "dividenden_summe_fenster": dividenden_summe,
        "aktionen": aktionen,
        "hinweis": (
            "action=split (value=Verhältnis) bzw. dividend (value=Betrag je "
            "Aktie). Events existieren nur für Fenster, die mit actions=True "
            "geladen wurden."
        ),
    }


# ---------------------------------------------------------------------------
# 8) manage_watchlists (T34: T28-Repos)
# ---------------------------------------------------------------------------


class ManageWatchlistsArgs(BaseModel):
    """Argumente von ``manage_watchlists``."""

    action: Literal["list", "create", "add", "remove", "rename", "delete"] = Field(
        description='Aktion: "list", "create" (braucht name), "add"/"remove" '
        '(brauchen watchlist_id + symbol), "rename" (watchlist_id + name), '
        '"delete" (watchlist_id).'
    )
    watchlist_id: int | None = Field(
        default=None, ge=1, description="Listen-ID, z. B. 1 (Standard-Liste)."
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description='Listenname (create/rename), z. B. "Dividenden".',
    )
    symbol: str | None = Field(
        default=None, min_length=1, max_length=24, description='Symbol-Kürzel, z. B. "AAPL".'
    )


def _watchlist_or_fehler(watchlist_id: int | None) -> int:
    if watchlist_id is None:
        raise ToolFehler('Diese Aktion benötigt eine "watchlist_id".')
    return watchlist_id


def _name_or_fehler(name: str | None) -> str:
    if name is None or not name.strip():
        raise ToolFehler('Diese Aktion benötigt einen nicht-leeren "name".')
    return name.strip()


def _symbol_or_fehler(symbol: str | None) -> str:
    if symbol is None or not symbol.strip():
        raise ToolFehler('Diese Aktion benötigt ein "symbol".')
    return symbol.strip()


async def manage_watchlists(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Watchlists auflisten, anlegen, umbenennen, löschen; Symbole auf/abnehmen."""
    args = pruefe_args(ManageWatchlistsArgs, roh_args, "manage_watchlists")

    def _ausfuehren() -> dict[str, Any]:
        if args.action == "list":
            listen = [
                {
                    "id": liste.id,
                    "name": liste.name,
                    "position": liste.position,
                    "symbole": [
                        eintrag.symbol
                        for eintrag in watchlists_repo.list_symbols(liste.id)
                    ],
                }
                for liste in watchlists_repo.list_watchlists()
            ]
            return {"action": "list", "anzahl": len(listen), "watchlists": listen}
        if args.action == "create":
            liste = watchlists_repo.create_watchlist(_name_or_fehler(args.name))
            return {
                "action": "create",
                "watchlist": {"id": liste.id, "name": liste.name, "symbole": []},
            }
        watchlist_id = _watchlist_or_fehler(args.watchlist_id)
        if args.action == "add":
            neu = watchlists_repo.add_symbol(watchlist_id, _symbol_or_fehler(args.symbol))
            return {
                "action": "add",
                "watchlist_id": watchlist_id,
                "neu_aufgenommen": neu,
                "hinweis": None if neu else "Symbol stand bereits in der Liste.",
            }
        if args.action == "remove":
            entfernt = watchlists_repo.remove_symbol(
                watchlist_id, _symbol_or_fehler(args.symbol)
            )
            return {
                "action": "remove",
                "watchlist_id": watchlist_id,
                "entfernt": entfernt,
                "hinweis": None if entfernt else "Symbol war nicht in der Liste.",
            }
        if args.action == "rename":
            liste = watchlists_repo.rename_watchlist(
                watchlist_id, _name_or_fehler(args.name)
            )
            if liste is None:
                raise ToolFehler(f"Watchlist {watchlist_id} existiert nicht.")
            return {"action": "rename", "watchlist": {"id": liste.id, "name": liste.name}}
        # delete — Standard-Liste 1 schützen (Quotes/Downloads/Scheduler-Konvention).
        if watchlist_id == watchlists_repo.DEFAULT_WATCHLIST_ID:
            raise ToolFehler(
                "Die Standard-Watchlist (ID 1) kann nicht gelöscht werden — sie "
                "trägt die migrierte v1-Liste und den Download-/Quote-Zyklus."
            )
        geloescht = watchlists_repo.delete_watchlist(watchlist_id)
        if not geloescht:
            raise ToolFehler(f"Watchlist {watchlist_id} existiert nicht.")
        return {"action": "delete", "watchlist_id": watchlist_id, "geloescht": True}

    try:
        return await asyncio.to_thread(_ausfuehren)
    except ValueError as exc:
        raise ToolFehler(str(exc)) from exc


# ---------------------------------------------------------------------------
# 9) get_scheduler_status / trigger_data_update (T34: T21-Scheduler)
# ---------------------------------------------------------------------------


class SchedulerStatusArgs(BaseModel):
    """Argumente von ``get_scheduler_status`` (bewusst leer — keine Parameter)."""


async def get_scheduler_status(_roh_args: dict[str, Any]) -> dict[str, Any]:
    """Scheduler-Einstellungen + laufende Jobs (Diagnose: „Warum sind Daten alt?“)."""
    einstellungen = await asyncio.to_thread(scheduler.scheduler_einstellungen)
    return {
        "einstellungen": einstellungen,
        "verfuegbare_jobs": list(_SCHEDULER_JOBS),
        "laufende_jobs": scheduler.laufende_jobs(),
        "hinweis": "trigger_data_update stößt einen Job sofort an (run-now).",
    }


class TriggerDataUpdateArgs(BaseModel):
    """Argumente von ``trigger_data_update``."""

    job: Literal["data_update", "fundamentals", "quotes", "alerts"] = Field(
        default="data_update",
        description='Job-Art: "data_update" (Default: 1d-Download der Watchlist), '
        '"fundamentals" (Statements-Refresh), "quotes" (Live-Quotes), '
        '"alerts" (Alert-Prüfung).',
    )


async def trigger_data_update(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Stößt einen Scheduler-Job sofort an und wartet auf sein Ende (run-now)."""
    args = pruefe_args(TriggerDataUpdateArgs, roh_args, "trigger_data_update")
    try:
        ergebnis = await scheduler.run_now(args.job)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    details = ergebnis.get("details")
    return {
        "job": ergebnis.get("job", args.job),
        "ok": ergebnis.get("ok"),
        "dauer_ms": ergebnis.get("durationMs"),
        "fehler": ergebnis.get("error"),
        "details": details if isinstance(details, dict) else {},
    }


# ---------------------------------------------------------------------------
# Registrierung (§E-Reihenfolge 1–4, dann T34-Zugänge)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="get_market_data",
            description=(
                "Liest gespeicherte Kursdaten (OHLCV) aus dem lokalen Speicher: "
                "Summary (letzter Kurs, Hoch/Tief, Ø-Volumen) plus kompakte Balkenliste. "
                'Beispiel: {"symbol": "AAPL", "interval": "1d", "limit": 60} — für große '
                "Zeiträume start/end setzen, das Ergebnis wird bei Übergröße gekappt."
            ),
            parameters=schema_von(GetMarketDataArgs),
            handler=get_market_data,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="list_symbols",
            description=(
                "Listet alle lokal gespeicherten Symbole inkl. Datenabdeckung je Intervall "
                "(Balkenanzahl, erster/letzter Zeitpunkt). Keine Parameter: {}."
            ),
            parameters=schema_von(ListSymbolsArgs),
            handler=list_symbols,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="download_data",
            description=(
                "Lädt Kursdaten aus dem Netz (yfinance) und speichert sie lokal; wartet "
                "auf das Job-Ende und meldet Balkenanzahl sowie coverage je Symbol "
                "(erste/letzte Zeit, Quelle; warnung, wenn die Quelle weniger liefert "
                "als angefragt). ISINs (z. B. \"DE0007164600\") werden vorab zu "
                "Yahoo-Kürzeln aufgelöst (Registry + lokale Referenz; unbekannte ISINs "
                "führen zu einem Fachfehler, statt zu raten). Vorhandene Bestände "
                "bleiben erhalten — es wird nur das Fehlende ergänzt. Zeitraum über "
                'period (z. B. "5y") oder start (JJJJ-MM-TT; schlägt period vor); '
                "force_full lädt den Bestand komplett neu, repair korrigiert Lücken "
                "in der Historie. Läuft derselbe Download bereits, wird kein zweiter "
                "Job gestartet (status \"laeuft_bereits\"). "
                'Beispiel: {"symbols": ["AAPL"], "interval": "1d", "period": "5y"}.'
            ),
            parameters=schema_von(DownloadDataArgs),
            handler=download_data,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="get_quote",
            description=(
                "Liefert letzte Kurse je Symbol — bevorzugt den Live-Quote "
                "(Watchlist-Service mit TTL-Cache), sonst den letzten gespeicherten "
                "Schlusskurs — plus prozentuale Veränderung über 1 Tag, 1 Woche und "
                "1 Monat (aus gespeicherten Tagesbalken). Nimmt „symbol“ (eins) oder "
                "„symbols“ (bis 20) entgegen. "
                'Beispiel: {"symbols": ["MSFT", "AAPL"]}.'
            ),
            parameters=schema_von(GetQuoteArgs),
            handler=get_quote,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="search_symbols",
            description=(
                "Sucht Symbol-Kürzel bei Yahoo (Name oder Kürzel, fuzzy) — Ergebnisse "
                "mit exchange/quoteType; identische Suchen kommen aus dem 24-h-Cache. "
                "Auch ISINs (z. B. \"DE0007164600\") sind erlaubt: bei eindeutigem "
                "Treffer wird die ISIN in der Symbol-Registry gepflegt. "
                'Beispiel: {"query": "Apple", "limit": 5}.'
            ),
            parameters=schema_von(SearchSymbolsArgs),
            handler=search_symbols,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="get_fundamentals",
            description=(
                "Liest gespeicherte Fundamentals: Key-Ratios (Netto-Marge, ROE, ROA, "
                "EBIT-Marge, Umsatzwachstum YoY, …) plus Statement-Auszüge (income, "
                "balance, cashflow; letzte 8 Perioden). Rein lokal — ohne Bestand "
                "kommt ein Lade-Hinweis. "
                'Beispiel: {"symbol": "AAPL", "freq": "annual"}.'
            ),
            parameters=schema_von(GetFundamentalsArgs),
            handler=get_fundamentals,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="get_corporate_actions",
            description=(
                "Liest gespeicherte Splits und Dividenden eines Symbols (optional mit "
                "start/end-Fenster), inklusive Dividenden-Summe im Fenster. "
                'Beispiel: {"symbol": "AAPL", "start": "2024-01-01"}.'
            ),
            parameters=schema_von(GetCorporateActionsArgs),
            handler=get_corporate_actions,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="manage_watchlists",
            description=(
                "Verwaltet Watchlists: auflisten (action=\"list\"), anlegen "
                "(action=\"create\", name=…), Symbol aufnehmen/entfernen "
                "(action=\"add\"|\"remove\", watchlist_id=…, symbol=…), umbenennen "
                "(action=\"rename\") oder löschen (action=\"delete\", nicht Liste 1). "
                'Beispiel: {"action": "add", "watchlist_id": 1, "symbol": "MSFT"}.'
            ),
            parameters=schema_von(ManageWatchlistsArgs),
            handler=manage_watchlists,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="get_scheduler_status",
            description=(
                "Zeigt den Stand der Hintergrund-Jobs: Intervalle/Einstellungen, "
                "verfügbare Job-Arten und aktuell laufende Ticks — nützlich, wenn "
                "Daten alt wirken. Keine Parameter: {}."
            ),
            parameters=schema_von(SchedulerStatusArgs),
            handler=get_scheduler_status,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="trigger_data_update",
            description=(
                "Stößt einen Hintergrund-Job sofort an und wartet auf das Ende: "
                "„data_update“ lädt Tageskurse der Watchlist, „fundamentals“ "
                "aktualisiert Statements, „quotes“ holt Live-Quotes, „alerts“ prüft "
                "Alert-Bedingungen. "
                'Beispiel: {"job": "data_update"}.'
            ),
            parameters=schema_von(TriggerDataUpdateArgs),
            handler=trigger_data_update,
        )
    )


_registrieren()

#: ANNAHME: „1T/1W/1M“ in ``get_quote`` sind Kalenderbetrachtungen (1/7/30 Tage);
#: der Lookup nimmt den letzten Balken vor dem Zielzeitpunkt — interval-unabhängig.
#: ANNAHME (T34): ``search_symbols`` konsumiert die T19-Suchfunktion direkt aus
#: dem Data-Router (Funktionsimport, kein HTTP), damit Cache-/Rate-Limit-Semantik
#: genau einmal existiert; der Quotes-Abruf in ``get_quote`` darf je Symbol in
#: einen Netz-Fetch münden (TTL-Cache/Single-Flight im T22-Service).
