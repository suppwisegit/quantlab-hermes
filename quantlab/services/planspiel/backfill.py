"""Bulk-Historien-Backfill (Welle H5, E097): 5Y/1d fürs gesamte PSB-Universum.

Der Lauf holt für JEDES Instrument aus :func:`psb_repo.liste` die Tages-
Historie (Byteworx ``timeSpan=5Y`` → Intervall ``1d``, E094) und sichert sie
über :func:`parquet_store.upsert_bars` (Herkunftsmarke ``bar_meta.source =
'byteworx'`` — Muster ``downloads._symbol_laden``). Der Lauf ist bewusst
**streng serial** (keine eigene Parallelität): das Pacing kommt ausschließlich
aus dem prozessweiten Byteworx-Slot des Providers (≥ 0,5 s je Request) — ein
parallel laufender Download-Job teilt sich denselben Slot automatisch.

**Resumabel (E097):** der Fortschritt je (ISIN, Zeitraum) liegt in
``psb_backfill_fortschritt`` (Migration 9) — ein Abbruch oder Backend-Neustart
verliert nur die gerade laufende ISIN. Beim Start werden ``laeuft``-Zeilen auf
``offen`` zurückgesetzt (Crash-Recovery), ``fertig``- und ``fehler``-Zeilen
werden übersprungen. Fehlversuche je ISIN werden **neustartfest** gezählt
(max. 3 → ``fehler`` mit deutschem Grund; der Lauf läuft weiter).

**Start/Pause (asyncio-Hintergrundjob, KEIN scheduler.py):** Der Zustand ist
Router-unabhängig in diesem Modul (Desktop-App, ein Prozess; Muster
``routers/planspiel.py`` „Universum-Import“). Ein zweiter Start während eines
Laufs wirft :class:`BackfillLaeuftBereits` (Router → 409); ``pause`` setzt das
Stop-Signal, lässt die aktuelle ISIN sauber auslaufen und wartet auf den Task
(kein Zombie-Task).

**Ergebnis-Vertrag / Ehrlichkeit der Schätzung:** :func:`snapshot` meldet
``{aktiv, insgesamt, fertig, fehler, laeuft, offen, aktueller_isin, start_us,
geschaetzt_rest_s, hinweis, isins}``. ``geschaetzt_rest_s`` ist eine
**Untergrenze** (Rest-ISINs × 0,5 s Pacing — Netzzeit/-ausfälle kommen on
top): ein Voll-Lauf über das reale Universum (~800 ISINs, RECHERCHE H.2)
dauert daher mehrere Minuten, mindestens ~7 Minuten allein an Pacing — die
UI sagt das explizit. Keine Protokoll-Ausgabe im Paket: unerwartete Abbrüche
liegen
als deutscher ``hinweis`` im Ergebnis (Paket-Konvention, QA P1).

Konsumiert :mod:`quantlab.storage.repos.psb_repo` (Universum) + Parquet-Store;
die Fortschritts-Tabelle ist H5-eigenes Schema und wird hier direkt gepflegt
(``psb_repo`` bleibt Domäne der Universum-/Kennzahlen-Zeilen — Besitzmatrix
Welle H-3). Komplett netzfrei testbar: Provider ist injizierbar
(:class:`httpx.MockTransport`-Fakes), kein Scheduler, kein echtes Netz.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from quantlab import db
from quantlab.providers.byteworx import ByteworxProvider
from quantlab.storage import parquet_store
from quantlab.storage.repos import psb_repo

#: Geführter Zeitraum (Byteworx-timeSpan der 1d-Serie) — Migration-9-Konstante.
ZEITRAUM = "5Y"

#: Intervall der gelieferten Serie (kanonische Balken-Einheit).
INTERVALL = "1d"

#: Herkunftsmarke der gesicherten Balken (E094 — Byteworx-Bein).
QUELLE = "byteworx"

#: Fehlversuche je ISIN, bevor die ISIN als ``fehler`` aufgegeben wird
#: (neustartfest in der Fortschritts-Tabelle gezählt).
MAX_FEHLVERSUCHE = 3

#: Pacing-Annahme der Restschätzung (Sekunden je Rest-ISIN) — Spiegel des
#: prozessweiten Byteworx-Mindestabstands (``byteworx``: ≥ 0,5 s; dort
#: privat, daher hier als ehrliche Konstante dokumentiert). Die Schätzung
#: ist bewusst eine **Untergrenze** (nur Pacing, keine Netzzeit).
PACING_ANNAHME_S = 0.5


class BackfillLaeuftBereits(Exception):
    """Ein Bulk-Lauf läuft bereits — Dedup-Fall (Router: 409, deutsch)."""


# ---------------------------------------------------------------------------
# Fortschritts-Zeile (Tabelle psb_backfill_fortschritt, Migration 9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FortschrittZeile:
    """Eine Buchhaltungszeile je (ISIN, Zeitraum) — Spiegel der Tabelle."""

    isin: str
    zeitraum: str
    status: str  # offen | laeuft | fertig | fehler
    punkte: int
    fehlversuche: int
    #: Letzter Fehlergrund (deutsch); nach Erfolg wieder leer (None).
    grund: str | None
    aktualisiert_am: str


def _zu_zeile(row: sqlite3.Row) -> FortschrittZeile:
    return FortschrittZeile(
        isin=row["isin"],
        zeitraum=row["zeitraum"],
        status=row["status"],
        punkte=int(row["punkte"]),
        fehlversuche=int(row["fehlversuche"]),
        grund=row["grund"],
        aktualisiert_am=row["aktualisiert_am"],
    )


def _zeile_lesen(isin: str) -> FortschrittZeile | None:
    """Buchhaltungszeile der ISIN (UPPER-Kanon) für den Lauf-Zeitraum."""
    geprueft = (isin or "").strip().upper()
    if not geprueft:
        return None
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM psb_backfill_fortschritt WHERE isin = ? AND zeitraum = ?",
            (geprueft, ZEITRAUM),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_zeile(row)


def _zeilen_lesen() -> list[FortschrittZeile]:
    """Alle Buchhaltungszeilen des Lauf-Zeitraums (nach ISIN sortiert)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM psb_backfill_fortschritt WHERE zeitraum = ? ORDER BY isin",
            (ZEITRAUM,),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_zeile(row) for row in rows]


def _zeilen_anlegen(isins: Iterable[str]) -> None:
    """Legt fehlende Zeilen als ``offen`` an (idempotent, eine Transaktion).

    ISIN-Kanonik: Zeilen, die nicht dem UPPER-Kanon entsprechen (nur durch
    externe Eingriffe möglich — QuantLab schreibt stets UPPER), werden
    verworfen; ihr Fortschritt wird neu gefahren (Upserts sind idempotent,
    kein Datenverlust im Store).
    """
    jetzt = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM psb_backfill_fortschritt "
            "WHERE zeitraum = ? AND isin <> upper(isin)",
            (ZEITRAUM,),
        )
        for isin in isins:
            geprueft = (isin or "").strip().upper()
            if not geprueft:
                continue
            conn.execute(
                "INSERT INTO psb_backfill_fortschritt("
                "isin, zeitraum, status, punkte, fehlversuche, grund, aktualisiert_am) "
                "VALUES (?, ?, 'offen', 0, 0, NULL, ?) "
                "ON CONFLICT(isin, zeitraum) DO NOTHING",
                (geprueft, ZEITRAUM, jetzt),
            )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _fortschritt_upsert(
    isin: str,
    *,
    status: str,
    punkte: int = 0,
    fehlversuche: int = 0,
    grund: str | None = None,
) -> None:
    """Schreibt den kompletten Lauf-Zustand einer ISIN (UPPER-Kanon, Upsert).

    Bewusst **ohne** COALESCE-Teil-Updates: SQLite prüft NOT NULL vor dem
    Konflikt-Pfad, und der Aufrufer (Lauf-Schleife) kennt den Vorzustand aus
    :func:`_zeile_lesen` ohnehin — Vollwerte sind der einfachste ehrliche
    Vertrag (keine impliziten Default-Resets).
    """
    geprueft = (isin or "").strip().upper()
    if not geprueft:
        raise ValueError("backfill._fortschritt_upsert ohne ISIN ist nicht möglich.")
    jetzt = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO psb_backfill_fortschritt("
            "isin, zeitraum, status, punkte, fehlversuche, grund, aktualisiert_am) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(isin, zeitraum) DO UPDATE SET "
            "status = excluded.status, "
            "punkte = excluded.punkte, "
            "fehlversuche = excluded.fehlversuche, "
            "grund = excluded.grund, "
            "aktualisiert_am = excluded.aktualisiert_am",
            (geprueft, ZEITRAUM, status, int(punkte), int(fehlversuche), grund, jetzt),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _laeuft_zu_offen() -> int:
    """Setzt hängende ``laeuft``-Zeilen auf ``offen`` (Crash-Recovery, H5).

    Läuft der Prozess hart ab (Backend-Neustart mitten im Fetch), bleibt die
    aktuelle ISIN als ``laeuft`` stehen — der nächste Start setzt sie zurück.
    Liefert die Anzahl zurückgesetzter Zeilen.
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE psb_backfill_fortschritt SET status = 'offen', "
            "aktualisiert_am = ? WHERE zeitraum = ? AND status = 'laeuft'",
            (db.utc_now_iso(), ZEITRAUM),
        )
        anzahl = cursor.rowcount
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return max(0, int(anzahl))


def _bar_meta_quelle_setzen(symbol: str, interval: str, quelle: str) -> None:
    """Setzt ``bar_meta.source`` (Muster ``downloads._bar_meta_source_setzen``).

    Bewusste Kopie im H5-Modul: ``downloads`` ist nicht im Besitz dieser Welle
    und die Source-Pflege gehört zum jeweiligen Schreibpfad (E023).
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
# Der resumable Lauf (synchron — Aufrufer: asyncio.to_thread im Job-Task)
# ---------------------------------------------------------------------------


def _fehlermeldung(exc: Exception) -> str:
    """Deutsche Fehlermeldung eines Fehlversuchs (ehrlicher Grund-Text)."""
    return str(exc).strip() or exc.__class__.__name__


def fuehre_backfill_aus(
    provider: Any,
    *,
    stopp: threading.Event | None = None,
) -> dict[str, int]:
    """Fährt den Bulk-Lauf aus (synchron, serial je ISIN, resumabel).

    Lesefolge: :func:`psb_repo.liste` (Name-Sortierung). Je ISIN gilt:

    - ``fertig``/``fehler`` → überspringen (Resumabilität/3-Fehler-Grenze);
    - sonst Fetch (``INTERVALL``) → :func:`parquet_store.upsert_bars` →
      Quelle ``QUELLE`` in ``bar_meta`` → ``fertig`` mit Punktezahl;
    - Fehler → Fehlversuch zählen; ab :data:`MAX_FEHLVERSUCHE` Status
      ``fehler`` mit deutschem Grund — **der Lauf läuft weiter**.

    Ein gesetztes ``stopp``-Ereignis (Pause) wird zwischen zwei ISINs
    geprüft; die laufende ISIN läuft sauber aus (kein halber Zustand — die
    Buchhaltung kennt nur ganze ISIN-Ergebnisse). Rückgabe: Zähler dieses
    Laufs; der Gesamtstand lebt in der Tabelle (:func:`snapshot`).
    """
    ereignis = stopp if stopp is not None else _STOP_EVENT
    _laeuft_zu_offen()  # Wiederaufnahme: Crash-Überbleibsel sind „offen“
    instrumente = psb_repo.liste()
    _zeilen_anlegen(instrument.isin for instrument in instrumente)
    fertig = 0
    fehlerhaft = 0
    verarbeitet = 0
    for instrument in instrumente:
        if ereignis.is_set():
            break
        isin = instrument.isin.strip().upper()
        zeile = _zeile_lesen(isin)
        if zeile is not None and zeile.status in ("fertig", "fehler"):
            continue  # Resumabel: fertige/aufgegebene ISINs kosten keinen Fetch
        _fortschritt_upsert(
            isin,
            status="laeuft",
            punkte=zeile.punkte if zeile is not None else 0,
            fehlversuche=zeile.fehlversuche if zeile is not None else 0,
        )
        try:
            balken = provider.fetch_ohlcv(isin, INTERVALL)
            meta = parquet_store.upsert_bars(isin, INTERVALL, balken)
            if not balken.empty:
                _bar_meta_quelle_setzen(isin, INTERVALL, QUELLE)
            _fortschritt_upsert(
                isin,
                status="fertig",
                punkte=meta.row_count,
                fehlversuche=0,
                grund=None,
            )
            fertig += 1
        except Exception as exc:  # noqa: BLE001 — je ISIN zählen, Lauf läuft weiter
            versuche = (zeile.fehlversuche if zeile is not None else 0) + 1
            aufgegeben = versuche >= MAX_FEHLVERSUCHE
            _fortschritt_upsert(
                isin,
                status="fehler" if aufgegeben else "offen",
                punkte=zeile.punkte if zeile is not None else 0,
                fehlversuche=versuche,
                grund=_fehlermeldung(exc),
            )
            if aufgegeben:
                fehlerhaft += 1
        verarbeitet += 1
    _laeuft_zu_offen()  # Pause traf die Buchhaltung nie halb — Sicherheitshalber
    return {
        "insgesamt": len(instrumente),
        "fertig": fertig,
        "fehler": fehlerhaft,
        "verarbeitet": verarbeitet,
        "abgebrochen": int(ereignis.is_set()),
    }


# ---------------------------------------------------------------------------
# Start/Pause (asyncio-Hintergrundjob) + Fortschritts-Snapshot
# ---------------------------------------------------------------------------

_ZUSTAND_LOCK = threading.Lock()
_STOP_EVENT = threading.Event()
_AKTIV: bool = False
_START_US: int | None = None
#: Deutscher Hinweis, falls der Lauf-Task unerwartet abbrach (Paket-Konvention:
#: Hinweise im Ergebnis statt Log-Ausgabe); normalerweise ``None``.
_HINWEIS: dict[str, str] | None = None
#: Starke Referenz auf den Lauf-Task (GC-Schutz; Muster routers/planspiel.py).
_LAUF_TASK: asyncio.Task[None] | None = None


def zustand_zuruecksetzen() -> None:
    """Setzt den Laufzeit-Zustand zurück (Test-Hygiene — simuliert Neustart).

    Die Fortschritts-Tabelle bleibt bewusst stehen: genau sie macht den Lauf
    resumabel über Prozessgrenzen.
    """
    global _AKTIV, _START_US, _LAUF_TASK, _HINWEIS
    with _ZUSTAND_LOCK:
        _AKTIV = False
        _START_US = None
        _LAUF_TASK = None
        _HINWEIS = None
    _STOP_EVENT.set()
    _STOP_EVENT.clear()


async def _task_hauptlauf(provider: Any) -> None:
    """Job-Task: Lauf im Thread, Endzustand garantiert, Hygiene garantiert.

    Ein unerwarteter Fehler (nicht je-ISIN behandelbar, z. B. Store-/DB-Panne)
    wird als deutscher ``hinweis`` niedergelegt (Snapshot) — der Task stürzt
    nicht still, und die Buchhaltung bleibt resumabel.
    """
    global _AKTIV, _START_US, _HINWEIS
    try:
        await asyncio.to_thread(fuehre_backfill_aus, provider, stopp=_STOP_EVENT)
    except Exception as exc:  # noqa: BLE001 — Job-Endzustand ist Pflicht (Muster H1)
        with _ZUSTAND_LOCK:
            _HINWEIS = {
                "code": "backfill_abgebrochen",
                "message": (
                    f"Der Historien-Backfill brach ab: "
                    f"{str(exc).strip() or exc.__class__.__name__} — ein neuer "
                    "Start setzt fort (fertige ISINs werden übersprungen)."
                ),
            }
    finally:
        schliessen = getattr(provider, "close", None)
        if callable(schliessen):
            schliessen()  # per-Lauf-Provider (Byteworx) freigeben — Hygiene
        with _ZUSTAND_LOCK:
            _AKTIV = False
            _START_US = None


async def starte_backfill(provider: Any | None = None) -> dict[str, Any]:
    """Startet (oder nimmt wieder auf) — Dedup via :class:`BackfillLaeuftBereits`.

    Ohne ``provider`` läuft ein frischer :class:`ByteworxProvider` (E094: kein
    Key, keine Settings); Tests injizieren Fakes bzw. monkeypatchen den
    Modul-Namen ``ByteworxProvider``. Rückgabe: :func:`snapshot` direkt nach
    dem Start (``aktiv=True``).
    """
    global _AKTIV, _START_US, _LAUF_TASK, _HINWEIS
    laeuft_bereits = False
    with _ZUSTAND_LOCK:
        if _AKTIV:
            laeuft_bereits = True
        else:
            _AKTIV = True
            _START_US = int(time.time() * 1_000_000)  # µs UTC (E006)
            _HINWEIS = None
            _STOP_EVENT.clear()
    if not laeuft_bereits:
        if provider is None:
            provider = ByteworxProvider()  # Modul-Name zur Laufzeit — testbar
        _LAUF_TASK = asyncio.create_task(_task_hauptlauf(provider))
        return snapshot()
    # Dedup-Fall: aktueller ISIN-Hinweis erst außerhalb des Locks ermitteln.
    zeile = next((z for z in _zeilen_lesen() if z.status == "laeuft"), None)
    aktuell = f" (aktuell ISIN {zeile.isin})" if zeile is not None else ""
    raise BackfillLaeuftBereits("Ein Historien-Backfill läuft bereits" + aktuell + ".")


async def pausiere_backfill() -> dict[str, Any]:
    """Pausiert höflich: Stop-Signal, aktuelle ISIN läuft aus, Task endet.

    Idempotent — ohne Lauf liefert der Aufruf einfach den Snapshot. Nach der
    Rückgabe ist der Task garantiert beendet (kein Zombie-Task).
    """
    _STOP_EVENT.set()
    with _ZUSTAND_LOCK:
        task = _LAUF_TASK
    if task is not None and not task.done():
        await asyncio.gather(task, return_exceptions=True)
    return snapshot()


def snapshot() -> dict[str, Any]:
    """Fortschritts-Snapshot (E097-Vertrag) + je-ISIN-Kompaktstatistik.

    ``aktiv=False`` projiziert Crash-Überbleibsel ehrlich weg: eine in der
    Tabelle noch ``laeuft``-markierte Zeile wird als ``offen`` berichtet (die
    autoritative Rücksetzung macht der nächste Start); ``start_us``/
    ``geschaetzt_rest_s``/``aktueller_isin`` laufen nur während eines Laufs.
    ``hinweis`` trägt einen deutschen Grund, falls der Lauf-Task unerwartet
    abbrach (Paket-Konvention: Hinweise im Ergebnis statt Log-Ausgabe).
    """
    with _ZUSTAND_LOCK:
        aktiv = _AKTIV
        start_us = _START_US
        hinweis = _HINWEIS
    zeilen = _zeilen_lesen()
    if not aktiv:
        zeilen = [
            replace(zeile, status="offen") if zeile.status == "laeuft" else zeile
            for zeile in zeilen
        ]
    insgesamt = len(zeilen)
    fertig = sum(1 for z in zeilen if z.status == "fertig")
    fehler = sum(1 for z in zeilen if z.status == "fehler")
    laeuft = sum(1 for z in zeilen if z.status == "laeuft")
    aktueller = next((z.isin for z in zeilen if z.status == "laeuft"), None)
    geschaetzt: float | None = None
    if aktiv:
        rest = insgesamt - fertig - fehler  # Rest-ISINs brauchen je ≥ 1 Fetch
        geschaetzt = round(rest * PACING_ANNAHME_S, 1)
    return {
        "aktiv": aktiv,
        "insgesamt": insgesamt,
        "fertig": fertig,
        "fehler": fehler,
        "laeuft": laeuft,
        "offen": insgesamt - fertig - fehler - laeuft,
        "aktueller_isin": aktueller,
        "start_us": start_us if aktiv else None,
        "geschaetzt_rest_s": geschaetzt,
        "hinweis": hinweis,
        "isins": [
            {
                "isin": z.isin,
                "status": z.status,
                "punkte": z.punkte,
                "fehlversuche": z.fehlversuche,
                "grund": z.grund,
                "aktualisiert_am": z.aktualisiert_am,
            }
            for z in zeilen
        ],
    }


__all__ = [
    "INTERVALL",
    "MAX_FEHLVERSUCHE",
    "PACING_ANNAHME_S",
    "QUELLE",
    "BackfillLaeuftBereits",
    "FortschrittZeile",
    "ZEITRAUM",
    "fuehre_backfill_aus",
    "pausiere_backfill",
    "snapshot",
    "starte_backfill",
    "zustand_zuruecksetzen",
]
