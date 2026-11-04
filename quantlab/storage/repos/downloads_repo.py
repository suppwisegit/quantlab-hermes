"""Downloads-Repository: Job-Buchhaltung (Tabelle ``downloads``, D.1) — nur DB-Schicht.

Der asyncio-Job-Manager (Queue, Abbruch, SSE-Fortschritt) kommt in T4
(``services/downloads.py``) und nutzt ausschließlich dieses Repo zur
Persistenz. Statuswerte sind hier als Konstanten verankert; Terminal-Status
setzen ``finished_at`` (nicht-Terminal-Updates lassen es unberührt).

SQLite-Zugriffe nach dem Verbindungs-Muster aus :mod:`quantlab.db`
(``BEGIN IMMEDIATE``, konditionales ``ROLLBACK``, Verbindung pro Operation).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from quantlab import db

#: Job-Lebenszyklus (C.1: queued→running→done | error | cancelled); seit
#: E089/T-G3 mit Zwischenstufe ``wartet_auf_wiederholung`` (nicht-terminal:
#: ein Job mit transient fehlgeschlagenen Symbolen ruht im Wiederholungsplan,
#: Cancel bleibt möglich).
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_WARTET_AUF_WIEDERHOLUNG = "wartet_auf_wiederholung"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

_TERMINAL: frozenset[str] = frozenset({STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED})

#: Rang je Status — Übergänge sind nur aufwärts erlaubt (Monotonie, §T4, seit
#: E089/T-G3 auf 4 Stufen):
#: ``queued (0) → running (1) → wartet_auf_wiederholung (2) →
#: done | error | cancelled (3)``. Legal damit u. a.: ``wartet → error``
#: (Wiederholung erschöpft) und ``wartet → done`` (Erfolg räumt den Plan);
#: Terminal-Status bleiben unumkehrbar.
_STATUS_RANG: dict[str, int] = {
    STATUS_QUEUED: 0,
    STATUS_RUNNING: 1,
    STATUS_WARTET_AUF_WIEDERHOLUNG: 2,
    STATUS_DONE: 3,
    STATUS_ERROR: 3,
    STATUS_CANCELLED: 3,
}


@dataclass(frozen=True)
class Download:
    """Ein Eintrag der Downloads-Tabelle (Job-Buchhaltung, ohne Laufzeitlogik)."""

    job_id: str
    symbols: list[str] = field(default_factory=list)
    interval: str = "1d"
    params: dict[str, Any] | None = None
    status: str = STATUS_QUEUED
    error: str | None = None
    created_at: str = ""
    finished_at: str | None = None


def _row_to_download(row: sqlite3.Row) -> Download:
    """sqlite3.Row → :class:`Download` (JSON-Spalten tolerant dekodiert)."""
    try:
        symbols = json.loads(row["symbols_json"])
    except json.JSONDecodeError:
        symbols = []  # defekte Zeile → wie leere Symbol-Liste behandeln
    params_raw = row["params_json"]
    try:
        params = json.loads(params_raw) if params_raw else None
    except json.JSONDecodeError:
        params = None
    return Download(
        job_id=row["job_id"],
        symbols=symbols if isinstance(symbols, list) else [],
        params=params if isinstance(params, dict) else None,
        interval=row["interval"],
        status=row["status"],
        error=row["error"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
    )


def create_download(
    job_id: str,
    symbols: list[str],
    interval: str,
    *,
    params: dict[str, Any] | None = None,
    status: str = STATUS_QUEUED,
) -> Download:
    """Legt einen Download-Job an (Status-Default ``queued``) und liefert ihn zurück."""
    created_at = db.utc_now_iso()
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — INSERT ist zwar atomar, aber
        # das einheitliche Transaktionsmuster kostet nichts und hält die
        # Schreibpfade konsistent.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO downloads(job_id, symbols_json, interval, params_json, "
            "status, error, created_at, finished_at) VALUES (?, ?, ?, ?, ?, NULL, ?, NULL)",
            (
                job_id,
                json.dumps(list(symbols), ensure_ascii=False),
                interval,
                json.dumps(params, ensure_ascii=False, sort_keys=True) if params else None,
                status,
                created_at,
            ),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return Download(
        job_id=job_id,
        symbols=list(symbols),
        interval=interval,
        params=params,
        status=status,
        error=None,
        created_at=created_at,
        finished_at=None,
    )


def get_download(job_id: str) -> Download | None:
    """Ein Job anhand seiner ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM downloads WHERE job_id = ?", (job_id,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _row_to_download(row)


def list_downloads(limit: int = 50) -> list[Download]:
    """Jobs, neueste zuerst (``created_at`` DESC; ``rowid`` als deterministischer Tiebreaker)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM downloads ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_download(row) for row in rows]


def list_nach_status(status: str, limit: int = 50) -> list[Download]:
    """Jobs eines Status, neueste zuerst (additiv, Muster :func:`list_downloads`).

    Gelesen u. a. von der Auto-Retry-Planung (E089/T-G3: offene — also
    ``queued``/``running`` — Download-Jobs gegen Doppel-Versuche schützen) und
    vom Download-Tool-Vertrag (E090/T-G4: Timeout-Pfad ohne Neueinreihung).
    Unbekannter Status → ``ValueError`` (Tippfehler früh sichtbar).
    """
    if status not in _STATUS_RANG:
        raise ValueError(f"Unbekannter Download-Status: {status!r}")
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM downloads WHERE status = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (status, int(limit)),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_download(row) for row in rows]


def update_status(job_id: str, status: str, *, error: str | None = None) -> bool:
    """Setzt Status (und optional Fehlermeldung); ``True`` wenn der Job existierte.

    Terminal-Status (done/error/cancelled) schreiben ``finished_at``; sonstige
    Updates lassen es unverändert — ein späterer Restart-Marker kann ein
    Terminal-Datum nicht verwischen.
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE downloads SET status = ?, error = ?, "
            "finished_at = CASE WHEN ? = 1 THEN ? ELSE finished_at END "
            "WHERE job_id = ?",
            (
                status,
                error,
                1 if status in _TERMINAL else 0,
                db.utc_now_iso(),
                job_id,
            ),
        )
        updated = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return updated


def ergaenze_params(job_id: str, zusatz: dict[str, Any]) -> bool:
    """Ergänzt ``params_json`` eines Jobs um Schlüssel (Merge; ``True`` bei Treffer).

    Schlüsselweise: nur NICHT-vorhandene bzw. übergebene Keys werden gesetzt —
    bestehende Einträge des Zusatz-Keys werden überschrieben, alle anderen
    Parameter bleiben unangetastet. E089/T-G3-Lesepunkt: der Übergang auf
    ``wartet_auf_wiederholung`` schreibt ``retry_naechster_us`` (früheste
    Fälligkeit der Plan-Zeilen) — Quelle der FE-Anzeige „um HH:MM“ (REST
    ``GET /api/data/downloads`` führt ``params`` unverändert fort).
    Unbekannte Job-ID → ``False``.
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT params_json FROM downloads WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return False
        try:
            params = json.loads(row["params_json"]) if row["params_json"] else {}
        except json.JSONDecodeError:
            params = {}
        if not isinstance(params, dict):
            params = {}
        params.update(zusatz)
        conn.execute(
            "UPDATE downloads SET params_json = ? WHERE job_id = ?",
            (
                json.dumps(params, ensure_ascii=False, sort_keys=True),
                job_id,
            ),
        )
        conn.execute("COMMIT")
        return True
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def update_status_vorwaerts(job_id: str, status: str, *, error: str | None = None) -> bool:
    """Setzt den Status nur aufwärts — Monotonie-Guard **in derselben Transaktion**.

    Der Rang-Vergleich läuft als ``CASE`` direkt in der ``WHERE``-Klausel des
    ``UPDATE``: Prüfung und Schreiben sind damit **eine** atomare Operation
    (``BEGIN IMMEDIATE``, Muster: :mod:`quantlab.db`). Ein gleichzeitiges
    Cancel (anderer Thread/Verbindung) und das Job-Ende können sich nicht mehr
    überholen — wer zuerst schreibt, gewinnt; kein ``done`` nach einem
    akzeptierten ``cancelled`` (und umgekehrt, gleichrangig geht nicht mehr).

    ``True`` nur, wenn der Übergang akzeptiert wurde (Job existiert **und**
    neuer Status hat strikt höheren Rang); Terminal-Status sind unumkehrbar.
    Unbekannter Ziel-Status → ``ValueError``.
    """
    ziel_rang = _STATUS_RANG.get(status)
    if ziel_rang is None:
        raise ValueError(f"Unbekannter Download-Status: {status!r}")
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE downloads SET status = ?, error = ?, "
            "finished_at = CASE WHEN ? = 1 THEN ? ELSE finished_at END "
            "WHERE job_id = ? AND ("
            "  CASE status WHEN ? THEN 0 WHEN ? THEN 1 WHEN ? THEN 2 ELSE 3 END"
            ") < ?",
            (
                status,
                error,
                1 if status in _TERMINAL else 0,
                db.utc_now_iso(),
                job_id,
                STATUS_QUEUED,
                STATUS_RUNNING,
                STATUS_WARTET_AUF_WIEDERHOLUNG,
                ziel_rang,
            ),
        )
        akzeptiert = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return akzeptiert
