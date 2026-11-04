"""Freshness-Repository: ``data_freshness`` (v2 §D.2, T21/T22/E035).

Staleness-Marker: Wann wurde eine Domain (``daily_bars``, ``fundamentals``,
``quotes``, …) zuletzt für einen Schlüssel (z. B. Symbol) aktualisiert? Der
Scheduler (T21) liest diese Marker, um netzarme, idempotente Ticks zu fahren
und „Warum sind meine Daten alt?“ beantwortbar zu machen. Wie beim Search-Cache
liegt jede TTL-/Alters-Bewertung beim Aufrufer — hier steht nur der Zeitstempel.

Muster: :mod:`quantlab.db` — eine Verbindung pro Operation, ``BEGIN IMMEDIATE``
für Schreib-Upserts, ``ROLLBACK`` nur bei offener Transaktion.
"""

from __future__ import annotations

from quantlab import db


def mark_fresh(domain: str, key: str, *, fetched_at: str | None = None) -> str:
    """Setzt/aktualisiert den Freshness-Marker (Upsert); liefert den Zeitstempel.

    ``fetched_at=None`` trägt „jetzt“ ein (:func:`db.utc_now_iso`); ein
    expliziter Wert erlaubt Rückdatierung aus Provider-Antworten (T21).
    """
    zeitstempel = fetched_at if fetched_at is not None else db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO data_freshness(domain, key, fetched_at) VALUES (?, ?, ?) "
            "ON CONFLICT(domain, key) DO UPDATE SET fetched_at = excluded.fetched_at",
            (domain, key, zeitstempel),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return zeitstempel


def get_freshness(domain: str, key: str) -> str | None:
    """Zeitstempel eines Markers — ``None`` wenn die Domain/den Schlüssel noch nicht markiert."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT fetched_at FROM data_freshness WHERE domain = ? AND key = ?",
            (domain, key),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else str(row["fetched_at"])


def list_domain(domain: str) -> dict[str, str]:
    """Alle Marker einer Domain als ``{key: fetched_at}``."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT key, fetched_at FROM data_freshness WHERE domain = ?", (domain,)
        ).fetchall()
    finally:
        conn.close()
    return {str(row["key"]): str(row["fetched_at"]) for row in rows}


def delete_entry(domain: str, key: str) -> bool:
    """Entfernt einen Marker; ``True`` wenn er existierte."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM data_freshness WHERE domain = ? AND key = ?", (domain, key)
        )
        deleted = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return deleted
