"""Symbols-Repository: Symbol-Registry (Tabelle ``symbols``) — nur DB-Logik (T2).

Validierung/Normalisierung von Yahoo-Tickern (``^GSPC``, ``EURUSD=X``, …) und
Provider-Zugriffe kommen in T3/T4 (``providers/symbols.py``). Das Löschen räumt
konsistent die reinen DB-Abhängigkeiten mit auf (``watchlist_symbols``, alle
Listen — E033; ``bar_meta``) — Parquet-Dateien räumt die darüberliegende
Datenschicht (T3/T4) ab.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from quantlab import db


@dataclass(frozen=True)
class Symbol:
    """Ein Eintrag der Symbol-Registry (Zeile aus ``symbols``)."""

    symbol: str
    name: str | None
    asset_type: str | None
    exchange_tz: str | None
    currency: str | None
    added_at: str
    # E091/T-G5: ISIN (Migration 6, nullable) — Alt-Bestand bleibt NULL.
    # Mit Default ``None`` angehängt, damit positionale Konstruktionen des
    # Bestands unverändert weiterlaufen.
    isin: str | None = None


def _to_symbol(row: sqlite3.Row) -> Symbol:
    return Symbol(
        symbol=row["symbol"],
        name=row["name"],
        asset_type=row["asset_type"],
        exchange_tz=row["exchange_tz"],
        currency=row["currency"],
        added_at=row["added_at"],
        isin=row["isin"],
    )


def upsert_symbol(
    symbol: str,
    *,
    name: str | None = None,
    asset_type: str | None = None,
    exchange_tz: str | None = None,
    currency: str | None = None,
    isin: str | None = None,
) -> Symbol:
    """Legt das Symbol an oder aktualisiert die Metadaten (idempotent).

    ``added_at`` bleibt beim Update erhalten; ``None``-Metadaten überschreiben
    bestehende Werte nicht (COALESCE) — partielle Updates sind so gefahrlos.
    ``isin`` (E091) folgt derselben Regel: Alt-Aufrufe ohne den Parameter
    lassen eine gepflegte ISIN unangetastet.
    """
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — siehe settings_repo.update_section.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO symbols(symbol, name, asset_type, exchange_tz, currency, isin, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET "
            "name = COALESCE(excluded.name, symbols.name), "
            "asset_type = COALESCE(excluded.asset_type, symbols.asset_type), "
            "exchange_tz = COALESCE(excluded.exchange_tz, symbols.exchange_tz), "
            "currency = COALESCE(excluded.currency, symbols.currency), "
            "isin = COALESCE(excluded.isin, symbols.isin)",
            (symbol, name, asset_type, exchange_tz, currency, isin, db.utc_now_iso()),
        )
        row = conn.execute("SELECT * FROM symbols WHERE symbol = ?", (symbol,)).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt/aktualisiert — immer lesbar
    return _to_symbol(row)


def get_symbol(symbol: str) -> Symbol | None:
    """Ein Symbol anhand seines Kürzels (``None`` wenn unbekannt)."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM symbols WHERE symbol = ?", (symbol,)).fetchone()
    finally:
        conn.close()
    return _to_symbol(row) if row is not None else None


def get_symbol_by_isin(isin: str) -> Symbol | None:
    """Ein Symbol anhand seiner ISIN (``None`` wenn unbekannt/leer).

    E091-Auflösungsstufe 1: die Registry füllt sich im Betrieb (search_symbols-
    Pflege) und hat Vorrang vor der Seed-Map im Tool-Layer.
    """
    geprueft = (isin or "").strip().upper()
    if not geprueft:
        return None
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM symbols WHERE isin = ?", (geprueft,)).fetchone()
    finally:
        conn.close()
    return _to_symbol(row) if row is not None else None


def list_symbols() -> list[Symbol]:
    """Alle gespeicherten Symbole, alphabetisch sortiert."""
    conn = db.connect()
    try:
        rows = conn.execute("SELECT * FROM symbols ORDER BY symbol").fetchall()
    finally:
        conn.close()
    return [_to_symbol(row) for row in rows]


def delete_symbol(symbol: str) -> bool:
    """Löscht das Symbol samt Watchlist-/bar_meta-Zeilen; ``True`` wenn gelöscht.

    E033: Das Symbol wird aus **allen** Watchlists entfernt (rohes SQL in dieser
    Transaktion statt ``watchlists_repo.remove_symbol_from_all`` — das Löschen
    aller DB-Abhängigkeiten bleibt so eine atomare Operation, Muster wie bei
    ``bar_meta``).
    """
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — siehe settings_repo.update_section.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM watchlist_symbols WHERE symbol = ?", (symbol,))
        conn.execute("DELETE FROM bar_meta WHERE symbol = ?", (symbol,))
        cursor = conn.execute("DELETE FROM symbols WHERE symbol = ?", (symbol,))
        deleted = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return deleted
