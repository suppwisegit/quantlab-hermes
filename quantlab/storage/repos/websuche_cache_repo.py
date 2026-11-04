"""Websuche-Cache-Repository: ``websuche_cache`` (Welle F, T-F6/E083).

Cache für Agent-Websuch-Ergebnisse (``query`` → ``results`` + ``fetched_at``),
Aufbau analog :mod:`quantlab.storage.repos.search_cache_repo` (T19). Die
TTL-Bewertung (24 h) liegt bewusst beim Aufrufer
(:mod:`quantlab.services.websuche`): Das Repo speichert nur den Zeitstempel
und bleibt damit frei von Zeit-Politik — Tests und künftige TTL-Änderungen
greifen ohne Schema-Anpassung.

Muster: :mod:`quantlab.db` — eine Verbindung pro Operation, ``BEGIN IMMEDIATE``
für Schreib-Upserts, ``ROLLBACK`` nur bei offener Transaktion.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from quantlab import db


@dataclass(frozen=True)
class CachedWebsuche:
    """Zeile aus ``websuche_cache`` (``results`` = geparstes ``results_json``)."""

    query: str
    results: list[Any]
    fetched_at: str


def _zu_cached(row: sqlite3.Row) -> CachedWebsuche:
    try:
        results = json.loads(row["results_json"])
    except (json.JSONDecodeError, TypeError):
        results = []  # defekte Zeile → wie „nicht im Cache“ behandeln
    return CachedWebsuche(
        query=str(row["query"]),
        results=results if isinstance(results, list) else [],
        fetched_at=str(row["fetched_at"]),
    )


def save(query: str, results: list[Any]) -> CachedWebsuche:
    """Legt die Ergebnisse einer Suche ab (Upsert — gleiche Query überschreibt)."""
    fetched_at = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO websuche_cache(query, results_json, fetched_at) "
            "VALUES (?, ?, ?) ON CONFLICT(query) DO UPDATE SET "
            "results_json = excluded.results_json, fetched_at = excluded.fetched_at",
            (query, json.dumps(list(results), ensure_ascii=False), fetched_at),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return CachedWebsuche(
        query=query, results=list(results), fetched_at=fetched_at
    )


def get(query: str) -> CachedWebsuche | None:
    """Gecachte Ergebnisse einer Query — ``None`` wenn (noch) nicht gecacht."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM websuche_cache WHERE query = ?", (query,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_cached(row)


def delete(query: str) -> bool:
    """Wirft eine Query aus dem Cache; ``True`` wenn sie gecacht war."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM websuche_cache WHERE query = ?", (query,)
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
