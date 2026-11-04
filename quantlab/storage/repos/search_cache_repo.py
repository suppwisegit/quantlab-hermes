"""Search-Cache-Repository: ``symbol_search_cache`` (v2 §D.2, T19).

Cache für Symbolsuche-Ergebnisse (``query`` → ``results`` + ``fetched_at``).
Die TTL-Bewertung (24 h, T19) liegt bewusst beim Aufrufer: Das Repo speichert
nur den Zeitstempel und bleibt damit frei von Zeit-Politik — Tests und künftige
TTL-Änderungen greifen ohne Schema-Anpassung.

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
class CachedSearch:
    """Zeile aus ``symbol_search_cache`` (``results`` = geparstes ``results_json``)."""

    query: str
    results: list[Any]
    fetched_at: str


def _zu_cached(row: sqlite3.Row) -> CachedSearch:
    try:
        results = json.loads(row["results_json"])
    except (json.JSONDecodeError, TypeError):
        results = []  # defekte Zeile → wie „nicht im Cache“ behandeln
    return CachedSearch(
        query=str(row["query"]),
        results=results if isinstance(results, list) else [],
        fetched_at=str(row["fetched_at"]),
    )


def save(query: str, results: list[Any]) -> CachedSearch:
    """Legt die Ergebnisse einer Suche ab (Upsert — gleiche Query überschreibt)."""
    fetched_at = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO symbol_search_cache(query, results_json, fetched_at) "
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
    return CachedSearch(
        query=query, results=list(results), fetched_at=fetched_at
    )


def get(query: str) -> CachedSearch | None:
    """Gecachte Ergebnisse einer Query — ``None`` wenn (noch) nicht gecacht."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM symbol_search_cache WHERE query = ?", (query,)
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
            "DELETE FROM symbol_search_cache WHERE query = ?", (query,)
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
