"""Watchlists-Repository: ``watchlists`` + ``watchlist_symbols`` (v2 §D.2, E033).

Mehrere benannte Listen; Symbole je Liste mit Position (klein = oben) und
``added_at``. Migration 3 legt die **Default-Liste** (id=1 „Watchlist“) an und
führt die v1-Global-Watchlist dorthin über — die v1-Zugriffsstellen (Dashboard,
Purge, Symbol-Löschen) laufen über :data:`DEFAULT_WATCHLIST_ID`.

Muster: :mod:`quantlab.db` (bindend) — eine Verbindung pro Operation,
Schreib-Transaktionen mit ``BEGIN IMMEDIATE``, ``ROLLBACK`` nur bei offener
Transaktion, sync-Funktionen (FastAPI-Handler als normale ``def``).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from quantlab import db

#: Default-Liste der Migration 3 (E033) — Heimat der migrierten v1-Watchlist
#: und Antwort von ``GET /api/data/symbols`` (``watchlist_position``).
DEFAULT_WATCHLIST_ID = 1


@dataclass(frozen=True)
class Watchlist:
    """Zeile aus ``watchlists``."""

    id: int
    name: str
    position: int
    created_at: str


@dataclass(frozen=True)
class WatchlistSymbol:
    """Zeile aus ``watchlist_symbols`` (``position`` klein = oben)."""

    watchlist_id: int
    symbol: str
    position: int
    added_at: str


def _zu_watchlist(row: sqlite3.Row) -> Watchlist:
    return Watchlist(
        id=int(row["id"]),
        name=str(row["name"]),
        position=int(row["position"]),
        created_at=str(row["created_at"]),
    )


def _zu_watchlist_symbol(row: sqlite3.Row) -> WatchlistSymbol:
    return WatchlistSymbol(
        watchlist_id=int(row["watchlist_id"]),
        symbol=str(row["symbol"]),
        position=int(row["position"]),
        added_at=str(row["added_at"]),
    )


# ---------------------------------------------------------------------------
# Listen (CRUD)
# ---------------------------------------------------------------------------


def create_watchlist(name: str, *, position: int | None = None) -> Watchlist:
    """Legt eine Liste an (ohne ``position``: ans Ende) und liefert sie zurück.

    ``ValueError`` bei leerem Namen.
    """
    if not name.strip():
        raise ValueError("Watchlist-Name darf nicht leer sein.")
    created_at = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if position is None:
            # Ohne explizite Position ans Ende (MAX+1 in derselben Transaktion;
            # NOT NULL-Spalte direkt im INSERT berechnen).
            cursor = conn.execute(
                "INSERT INTO watchlists(name, position, created_at) VALUES (?, "
                "COALESCE((SELECT MAX(position) FROM watchlists), 0) + 1, ?)",
                (name.strip(), created_at),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO watchlists(name, position, created_at) VALUES (?, ?, ?)",
                (name.strip(), position, created_at),
            )
        row = conn.execute(
            "SELECT * FROM watchlists WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_watchlist(row)


def get_watchlist(watchlist_id: int) -> Watchlist | None:
    """Eine Liste anhand ihrer ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM watchlists WHERE id = ?", (int(watchlist_id),)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_watchlist(row)


def list_watchlists() -> list[Watchlist]:
    """Alle Listen (``position`` aufsteigend, ``id`` als Tiebreaker)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM watchlists ORDER BY position, id"
        ).fetchall()
    finally:
        conn.close()
    return [_zu_watchlist(row) for row in rows]


def rename_watchlist(watchlist_id: int, name: str) -> Watchlist | None:
    """Benennt eine Liste um — ``None`` wenn unbekannt, ``ValueError`` wenn leer."""
    if not name.strip():
        raise ValueError("Watchlist-Name darf nicht leer sein.")
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE watchlists SET name = ? WHERE id = ?", (name.strip(), int(watchlist_id))
        )
        row = (
            conn.execute(
                "SELECT * FROM watchlists WHERE id = ?", (int(watchlist_id),)
            ).fetchone()
            if cursor.rowcount > 0
            else None
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return None if row is None else _zu_watchlist(row)


def delete_watchlist(watchlist_id: int) -> bool:
    """Löscht eine Liste samt ihrer Symbole; ``True`` wenn sie existierte."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM watchlist_symbols WHERE watchlist_id = ?", (int(watchlist_id),)
        )
        cursor = conn.execute(
            "DELETE FROM watchlists WHERE id = ?", (int(watchlist_id),)
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


# ---------------------------------------------------------------------------
# Symbole in Listen
# ---------------------------------------------------------------------------


def add_symbol(watchlist_id: int, symbol: str) -> bool:
    """Nimmt das Symbol ans Ende der Liste auf (idempotent).

    ``True`` wenn es neu war, ``False`` wenn es bereits drin stand (Position
    unverändert). ``ValueError`` wenn die Liste nicht existiert — E033 garantiert
    Liste 1 per Migration, alle weiteren Listen verwaltet T28.
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        vorhanden = conn.execute(
            "SELECT 1 FROM watchlists WHERE id = ?", (int(watchlist_id),)
        ).fetchone()
        if vorhanden is None:
            raise ValueError(f"Watchlist {int(watchlist_id)} existiert nicht.")
        cursor = conn.execute(
            "INSERT INTO watchlist_symbols(watchlist_id, symbol, position, added_at) "
            "VALUES (?, ?, "
            "COALESCE((SELECT MAX(position) FROM watchlist_symbols WHERE watchlist_id = ?), 0)"
            " + 1, ?) "
            "ON CONFLICT(watchlist_id, symbol) DO NOTHING",
            (int(watchlist_id), symbol, int(watchlist_id), db.utc_now_iso()),
        )
        neu = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return neu


def remove_symbol(watchlist_id: int, symbol: str) -> bool:
    """Entfernt das Symbol aus genau einer Liste; ``True`` wenn vorhanden."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM watchlist_symbols WHERE watchlist_id = ? AND symbol = ?",
            (int(watchlist_id), symbol),
        )
        entfernt = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return entfernt


def remove_symbol_from_all(symbol: str) -> int:
    """Entfernt das Symbol aus **allen** Listen; Anzahl entfernte Zeilen.

    Aufräum-Hook beim Löschen eines Symbols (Purge/Registry-Delete) — die
    Zeilen sollen in keiner Liste dangling bleiben (E033).
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM watchlist_symbols WHERE symbol = ?", (symbol,)
        )
        entfernt = cursor.rowcount
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return entfernt


def list_symbols(watchlist_id: int) -> list[WatchlistSymbol]:
    """Symbole einer Liste (``position`` aufsteigend) — leer bei unbekannter Liste."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM watchlist_symbols WHERE watchlist_id = ? ORDER BY position, symbol",
            (int(watchlist_id),),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_watchlist_symbol(row) for row in rows]


def positionen(watchlist_id: int) -> dict[str, int]:
    """``{symbol: position}`` einer Liste — leer bei unbekannter Liste.

    Response-Kompatibilität v1: ``GET /api/data/symbols`` bezieht
    ``watchlist_position`` hieraus (Default-Liste 1, klein = oben).
    """
    return {eintrag.symbol: eintrag.position for eintrag in list_symbols(watchlist_id)}


def reorder_symbols(watchlist_id: int, symbols: list[str]) -> bool:
    """Setzt die Reihenfolge der Listen-Symbole neu (PATCH-Semantik T28).

    Vertrag: Die Liste definiert die neue Ordnung der **genannten** Mitglieder
    (Position 1..n in der gegebenen Reihenfolge); nicht erwähnte Mitglieder
    rücken dahinter nach (bisherige Reihenfolge bleibt erhalten); nicht in der
    Liste enthaltene Symbole werden ignoriert. ``False`` wenn die Watchlist
    nicht existiert. Kollisionsfrei durch vollständige Neunumerierung.
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        vorhanden = conn.execute(
            "SELECT 1 FROM watchlists WHERE id = ?", (int(watchlist_id),)
        ).fetchone()
        if vorhanden is None:
            conn.execute("COMMIT")
            return False
        aktuelle = [
            str(row["symbol"])
            for row in conn.execute(
                "SELECT symbol FROM watchlist_symbols WHERE watchlist_id = ? "
                "ORDER BY position, symbol",
                (int(watchlist_id),),
            )
        ]
        mitglieder = set(aktuelle)
        genannt = [symbol for symbol in symbols if symbol in mitglieder]
        rest = [symbol for symbol in aktuelle if symbol not in set(genannt)]
        for position, symbol in enumerate(genannt + rest, start=1):
            conn.execute(
                "UPDATE watchlist_symbols SET position = ? "
                "WHERE watchlist_id = ? AND symbol = ?",
                (position, int(watchlist_id), symbol),
            )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return True
