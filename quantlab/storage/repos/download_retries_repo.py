"""Wiederholungsplan-Repository (Tabelle ``download_retries``, E089/T-G3) — nur DB-Schicht.

Eine Zeile je (``job_id``, ``symbol``, ``interval``): Ein Download-Job, dessen
Symbol mit **transientem** Fehler (429/502/``stooq_blockiert``/``stooq_limit``)
scheitert, bekommt hier seinen Versuchszähler und den fälligen nächsten Versuch
(``naechster_versuch_us``, int64 µs UTC — E006). ``job_id`` ist dabei der
Job, dessen Fehlversuch die Zeile **zuletzt schrieb** (Erstversuch: der
Original-Job; Retry-Versuche schreiben unter derselben Original-Job-ID weiter,
korreliert über den ``retry_von``-Marker des Retry-Jobs — Details
:mod:`quantlab.services.downloads`).

Zeilen mit ``versuch >= 6`` sind Tombstones (Wiederholung beendet bzw.
terminaler Fehler beim Retry) — sie zählen weder als fällig noch als offen,
bleiben aber zur Diagnose liegen.

SQLite-Zugriffe nach dem Verbindungs-Muster aus :mod:`quantlab.db`
(``BEGIN IMMEDIATE``, konditionales ``ROLLBACK``, Verbindung pro Operation).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from quantlab import db

#: Fehlversuche, nach denen die automatische Wiederholung beendet ist
#: (E089: „max 6 Versuche → terminal"). Selbst hier nur als Tombstone-Grenze
#: dokumentiert — die Fachlogik liegt beim Job-Manager.
MAX_VERSUCHE = 6


@dataclass(frozen=True)
class RetryPlanZeile:
    """Ein Eintrag der Wiederholungsplan-Tabelle (ohne Laufzeitlogik)."""

    job_id: str
    symbol: str
    interval: str
    versuch: int
    naechster_versuch_us: int
    letzte_fehler_code: str | None
    angelegt_us: int


def _row_to_zeile(row: sqlite3.Row) -> RetryPlanZeile:
    return RetryPlanZeile(
        job_id=row["job_id"],
        symbol=row["symbol"],
        interval=row["interval"],
        versuch=int(row["versuch"]),
        naechster_versuch_us=int(row["naechster_versuch_us"]),
        letzte_fehler_code=row["letzte_fehler_code"],
        angelegt_us=int(row["angelegt_us"]),
    )


def plane_wiederholung(
    job_id: str,
    symbol: str,
    interval: str,
    *,
    versuch: int,
    naechster_versuch_us: int,
    fehler_code: str | None,
    angelegt_us: int,
) -> None:
    """Legt die Plan-Zeile an oder aktualisiert sie (Upsert, „letzer Fehlversuch gewinnt“).

    Der Versuchszähler wird vom Aufrufer (Job-Manager) **inkrementiert
    übergeben** — das Repo ist bewusst zähler-dumm, damit Klassifizierung und
    Backoff an genau einer Stelle (Job-Manager) leben.
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO download_retries(job_id, symbol, interval, versuch, "
            "naechster_versuch_us, letzte_fehler_code, angelegt_us) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(job_id, symbol, interval) DO UPDATE SET "
            "versuch = excluded.versuch, "
            "naechster_versuch_us = excluded.naechster_versuch_us, "
            "letzte_fehler_code = excluded.letzte_fehler_code",
            (
                job_id,
                symbol,
                interval,
                int(versuch),
                int(naechster_versuch_us),
                fehler_code,
                int(angelegt_us),
            ),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def get_zeile(job_id: str, symbol: str, interval: str) -> RetryPlanZeile | None:
    """Eine Plan-Zeile — ``None`` wenn nicht geplant."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM download_retries "
            "WHERE job_id = ? AND symbol = ? AND interval = ?",
            (job_id, symbol, interval),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _row_to_zeile(row)


def loesche_zeile(job_id: str, symbol: str, interval: str) -> bool:
    """Räumt eine Plan-Zeile (Erfolg des Symbols); ``True`` wenn gelöscht."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM download_retries "
            "WHERE job_id = ? AND symbol = ? AND interval = ?",
            (job_id, symbol, interval),
        )
        geloescht = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return geloescht


def faellige(now_us: int, *, limit: int = 200) -> list[RetryPlanZeile]:
    """Fällige Wiederholungen (``1 <= versuch < 6`` und ``naechster <= now``).

    Tombstones (``versuch >= 6``) sind nie fällig. Aufsteigend nach Fälligkeit,
    dann ``id`` (deterministische Reihenfolge für den Retry-Tick).
    """
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM download_retries "
            "WHERE versuch >= 1 AND versuch < ? AND naechster_versuch_us <= ? "
            "ORDER BY naechster_versuch_us, id LIMIT ?",
            (MAX_VERSUCHE, int(now_us), int(limit)),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_zeile(row) for row in rows]


def alle_offenen(*, limit: int = 200) -> list[RetryPlanZeile]:
    """Alle ausstehenden Plan-Zeilen (``1 <= versuch < 6``, egal wann fällig).

    Grundlage für ``run-now`` des ``download_retry``-Jobs (Fälligkeits-Gate
    umgehen — „jetzt alle wartenden Wiederholungen anstoßen“).
    """
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM download_retries "
            "WHERE versuch >= 1 AND versuch < ? "
            "ORDER BY naechster_versuch_us, id LIMIT ?",
            (MAX_VERSUCHE, int(limit)),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_zeile(row) for row in rows]


def zeilen_fuer_job(job_id: str, *, limit: int = 200) -> list[RetryPlanZeile]:
    """Plan-Zeilen eines Jobs (auch Tombstones) — Quelle des Tool-Ergebnisses."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM download_retries WHERE job_id = ? "
            "ORDER BY naechster_versuch_us, id LIMIT ?",
            (job_id, int(limit)),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_zeile(row) for row in rows]


def zaehle_offen(job_id: str) -> int:
    """Ausstehende Zeilen eines Jobs (``versuch < 6``) — Finalisierungs-Gate."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM download_retries WHERE job_id = ? AND versuch < ?",
            (job_id, MAX_VERSUCHE),
        ).fetchone()
    finally:
        conn.close()
    return int(row["n"]) if row is not None else 0


def frueheste_offene_us(job_id: str) -> int | None:
    """Früheste Fälligkeit der offenen Zeilen eines Jobs (``None`` = keine).

    Lesepunkt für den Übergang auf ``wartet_auf_wiederholung`` (E089/T-G3):
    der Wert wandert als ``params.retry_naechster_us`` in die Job-Zeile und
    gibt der Frontend-Toast die geplante Uhrzeit („wartet auf Wiederholung
    (um HH:MM)“). Tombstones zählen nicht.
    """
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT MIN(naechster_versuch_us) AS frueheste FROM download_retries "
            "WHERE job_id = ? AND versuch < ?",
            (job_id, MAX_VERSUCHE),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["frueheste"] is None:
        return None
    return int(row["frueheste"])


def zaehle_beendet_ohne_erfolg(job_id: str) -> int:
    """Tombstone-Zeilen eines Jobs (``versuch >= 6``) — „Wiederholung beendet ohne Erfolg“."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM download_retries WHERE job_id = ? AND versuch >= ?",
            (job_id, MAX_VERSUCHE),
        ).fetchone()
    finally:
        conn.close()
    return int(row["n"]) if row is not None else 0
