"""Screener-Repository: ``screener_presets`` (v2 §D.2, E027/T32).

Gespeicherte Regel-Sets als Regel-AST (``rules_json``: Liste von
``{feld, op, wert}``-Regeln, AND-verknüpft). Die Ausführung über das lokale
Universum macht ``services/screener.py`` (T32) — dieses Repo ist reine
Preset-Persistenz.

Muster: :mod:`quantlab.db` — eine Verbindung pro Operation, ``BEGIN IMMEDIATE``,
``ROLLBACK`` nur bei offener Transaktion.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from quantlab import db


@dataclass(frozen=True)
class ScreenerPreset:
    """Zeile aus ``screener_presets`` (``rules`` = geparstes ``rules_json``)."""

    id: int
    name: str
    rules: list[Any]
    created_at: str
    updated_at: str


def _lade_regeln(text: str) -> list[Any]:
    """``rules_json`` tolerant dekodieren (defekte Zeile → leere Liste)."""
    try:
        rules = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    return rules if isinstance(rules, list) else []


def _zu_preset(row: sqlite3.Row) -> ScreenerPreset:
    return ScreenerPreset(
        id=int(row["id"]),
        name=str(row["name"]),
        rules=_lade_regeln(row["rules_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _pruefe_name(name: str) -> str:
    if not name.strip():
        raise ValueError("Preset-Name darf nicht leer sein.")
    return name.strip()


def create_preset(name: str, rules: list[Any]) -> ScreenerPreset:
    """Legt ein Preset an; ``ValueError`` bei leerem Namen oder nicht-Liste-Regeln."""
    if not isinstance(rules, list):
        raise ValueError("Screener-Regeln müssen eine Liste sein.")
    now = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT INTO screener_presets(name, rules_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (_pruefe_name(name), json.dumps(rules, ensure_ascii=False), now, now),
        )
        row = conn.execute(
            "SELECT * FROM screener_presets WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_preset(row)


def get_preset(preset_id: int) -> ScreenerPreset | None:
    """Ein Preset anhand seiner ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM screener_presets WHERE id = ?", (int(preset_id),)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_preset(row)


def list_presets() -> list[ScreenerPreset]:
    """Alle Presets, nach Name sortiert."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM screener_presets ORDER BY name COLLATE NOCASE, id"
        ).fetchall()
    finally:
        conn.close()
    return [_zu_preset(row) for row in rows]


def update_preset(
    preset_id: int, *, name: str | None = None, rules: list[Any] | None = None
) -> ScreenerPreset | None:
    """Aktualisiert Name/Regeln (``None`` = unverändert), stampft ``updated_at``.

    ``None`` wenn das Preset nicht existiert; ``ValueError`` bei leerem Namen
    oder nicht-Liste-Regeln.
    """
    if name is not None:
        name = _pruefe_name(name)
    if rules is not None and not isinstance(rules, list):
        raise ValueError("Screener-Regeln müssen eine Liste sein.")
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM screener_presets WHERE id = ?", (int(preset_id),)
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        conn.execute(
            "UPDATE screener_presets SET name = ?, rules_json = ?, updated_at = ? "
            "WHERE id = ?",
            (
                row["name"] if name is None else name,
                row["rules_json"]
                if rules is None
                else json.dumps(rules, ensure_ascii=False),
                db.utc_now_iso(),
                int(preset_id),
            ),
        )
        updated = conn.execute(
            "SELECT * FROM screener_presets WHERE id = ?", (int(preset_id),)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_preset(updated)


def delete_preset(preset_id: int) -> bool:
    """Löscht ein Preset; ``True`` wenn es existierte."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM screener_presets WHERE id = ?", (int(preset_id),)
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
