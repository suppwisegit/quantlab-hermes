"""Strategies-Repository: ``user_strategies`` + ``user_strategy_versions`` (E026).

Eigene Strategien als versionierter Python-Code: Die Strategie (Name,
Beschreibung) ist die Identität; jede Speicherung erzeugt eine **neue Version**
(``code`` unveränderlich, Versionen laufen monoton 1, 2, 3, …). Backtests
referenzieren ``"user:<id>@<version>"`` (E026). Sandbox/Validierung macht
``services/sandbox.py`` (T30) — dieses Repo persistiert nur.

Muster: :mod:`quantlab.db` — eine Verbindung pro Operation, ``BEGIN IMMEDIATE``,
``ROLLBACK`` nur bei offener Transaktion.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from quantlab import db


@dataclass(frozen=True)
class Strategy:
    """Zeile aus ``user_strategies`` (``name`` eindeutig)."""

    id: int
    name: str
    description: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StrategyVersion:
    """Zeile aus ``user_strategy_versions`` (unveränderlicher Code-Schnappschuss)."""

    strategy_id: int
    version: int
    code: str
    created_at: str


def _zu_strategy(row: sqlite3.Row) -> Strategy:
    return Strategy(
        id=int(row["id"]),
        name=str(row["name"]),
        description=str(row["description"] or ""),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _zu_version(row: sqlite3.Row) -> StrategyVersion:
    return StrategyVersion(
        strategy_id=int(row["strategy_id"]),
        version=int(row["version"]),
        code=str(row["code"]),
        created_at=str(row["created_at"]),
    )


# ---------------------------------------------------------------------------
# Strategien (CRUD)
# ---------------------------------------------------------------------------


def create_strategy(name: str, *, description: str = "") -> Strategy:
    """Legt eine Strategie an (noch ohne Versionen); ``ValueError`` bei leerem/doppeltem Namen."""
    if not name.strip():
        raise ValueError("Strategie-Name darf nicht leer sein.")
    now = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT INTO user_strategies(name, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (name.strip(), description, now, now),
        )
        row = conn.execute(
            "SELECT * FROM user_strategies WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.execute("COMMIT")
    except sqlite3.IntegrityError as exc:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise ValueError(f"Der Strategie-Name „{name.strip()}“ ist bereits vergeben.") from exc
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_strategy(row)


def get_strategy(strategy_id: int) -> Strategy | None:
    """Eine Strategie anhand ihrer ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM user_strategies WHERE id = ?", (int(strategy_id),)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_strategy(row)


def list_strategies() -> list[Strategy]:
    """Alle Strategien, nach Name sortiert."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM user_strategies ORDER BY name COLLATE NOCASE, id"
        ).fetchall()
    finally:
        conn.close()
    return [_zu_strategy(row) for row in rows]


def update_strategy(
    strategy_id: int, *, name: str | None = None, description: str | None = None
) -> Strategy | None:
    """Aktualisiert Name/Beschreibung (``None`` = unverändert), stampft ``updated_at``.

    ``None`` wenn unbekannt; ``ValueError`` bei leerem oder schon vergebenem Namen.
    """
    if name is not None and not name.strip():
        raise ValueError("Strategie-Name darf nicht leer sein.")
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM user_strategies WHERE id = ?", (int(strategy_id),)
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        try:
            conn.execute(
                "UPDATE user_strategies SET name = ?, description = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    row["name"] if name is None else name.strip(),
                    row["description"] if description is None else description,
                    db.utc_now_iso(),
                    int(strategy_id),
                ),
            )
        except sqlite3.IntegrityError as exc:
            # UNIQUE-Verstoß (Name doppelt) → fachlicher Fehler; das äußere
            # except übernimmt den ROLLBACK (Muster: quantlab.db).
            raise ValueError(
                f"Der Strategie-Name „{name.strip() if name else row['name']}“ "
                "ist bereits vergeben."
            ) from exc
        updated = conn.execute(
            "SELECT * FROM user_strategies WHERE id = ?", (int(strategy_id),)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_strategy(updated)


def delete_strategy(strategy_id: int) -> bool:
    """Löscht eine Strategie samt aller Versionen; ``True`` wenn sie existierte."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM user_strategy_versions WHERE strategy_id = ?", (int(strategy_id),)
        )
        cursor = conn.execute(
            "DELETE FROM user_strategies WHERE id = ?", (int(strategy_id),)
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
# Versionen
# ---------------------------------------------------------------------------


def add_version(strategy_id: int, code: str) -> StrategyVersion:
    """Hängt die nächste Version an (monoton, atomar) und liefert sie zurück.

    Die Versionsnummer wird in derselben ``BEGIN IMMEDIATE``-Transaktion aus
    ``MAX(version)`` abgeleitet — parallele Speicherungen können keine
    doppelte Version erzeugen. ``ValueError`` wenn die Strategie fehlt oder
    ``code`` leer ist.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Strategie-Code darf nicht leer sein.")
    created_at = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        vorhanden = conn.execute(
            "SELECT 1 FROM user_strategies WHERE id = ?", (int(strategy_id),)
        ).fetchone()
        if vorhanden is None:
            raise ValueError(f"Strategie {int(strategy_id)} existiert nicht.")
        # Version in der Transaktion aus MAX ableiten (rowid/lastrowid ist bei
        # zusammengesetztem PK (strategy_id, version) NICHT die Versionsnummer).
        hoechste = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM user_strategy_versions "
            "WHERE strategy_id = ?",
            (int(strategy_id),),
        ).fetchone()
        version = int(hoechste[0]) + 1
        conn.execute(
            "INSERT INTO user_strategy_versions(strategy_id, version, code, created_at) "
            "VALUES (?, ?, ?, ?)",
            (int(strategy_id), version, code, created_at),
        )
        row = conn.execute(
            "SELECT * FROM user_strategy_versions WHERE strategy_id = ? AND version = ?",
            (int(strategy_id), version),
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_version(row)


def get_version(strategy_id: int, version: int) -> StrategyVersion | None:
    """Eine konkrete Version — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM user_strategy_versions WHERE strategy_id = ? AND version = ?",
            (int(strategy_id), int(version)),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_version(row)


def list_versions(strategy_id: int) -> list[StrategyVersion]:
    """Alle Versionen einer Strategie, aufsteigend (älteste zuerst)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM user_strategy_versions WHERE strategy_id = ? ORDER BY version",
            (int(strategy_id),),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_version(row) for row in rows]


def latest_version(strategy_id: int) -> StrategyVersion | None:
    """Die neueste Version — ``None`` wenn die Strategie (noch) keine Version hat."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM user_strategy_versions WHERE strategy_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (int(strategy_id),),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_version(row)
