"""Charts-Repository: ``chart_drawings`` (v2 §D.2, E034).

Persistenz der Zeichentools: Eine Zeichnung gehört zu einem Chart
(``chart_key = "<SYMBOL>|<interval>"``, z. B. ``"AAPL|1d"``), trägt Anker-
punkte ``[{time: int64 µs UTC, price: float}, …]`` (E006/E034) und optionales
Styling (``{color, width, fontSize? …}``). Rendering/Interaktion liegt im
Renderer (T26) — dieses Repo speichert nur rohe Punkte/Styling-Dokumente.

Muster: :mod:`quantlab.db` — eine Verbindung pro Operation, ``BEGIN IMMEDIATE``,
``ROLLBACK`` nur bei offener Transaktion.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from quantlab import db

#: Zulässige Zeichentools (CHECK-Constraint §D.2, E034).
DRAWING_TOOLS: frozenset[str] = frozenset(
    {
        "trendline",
        "ray",
        "hline",
        "vline",
        "fib_retracement",
        "rectangle",
        "ellipse",
        "brush",
        "text",
    }
)


@dataclass(frozen=True)
class ChartDrawing:
    """Zeile aus ``chart_drawings`` (``points``/``style`` = geparste JSON-Spalten)."""

    id: int
    chart_key: str
    tool: str
    points: list[dict[str, Any]]
    style: dict[str, Any] | None
    created_at: str
    updated_at: str


def _lade_json(text: str, default: Any) -> Any:
    """JSON-Spalte tolerant dekodieren (defekte Zeile wirft nicht)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _zu_drawing(row: sqlite3.Row) -> ChartDrawing:
    points = _lade_json(row["points_json"], [])
    style_raw = row["style_json"]
    style = _lade_json(style_raw, None) if style_raw else None
    return ChartDrawing(
        id=int(row["id"]),
        chart_key=str(row["chart_key"]),
        tool=str(row["tool"]),
        points=points if isinstance(points, list) else [],
        style=style if isinstance(style, dict) else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _pruefe_anker(tool: str, points: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Validiert Tool + Ankerpunkte — ``ValueError`` mit deutscher Meldung."""
    if tool not in DRAWING_TOOLS:
        raise ValueError(f"Unbekanntes Zeichentool: {tool!r}")
    if not isinstance(points, list) or not points:
        raise ValueError("Eine Zeichnung braucht mindestens einen Ankerpunkt.")
    return list(points)


def _pruefe_anker_ohne_tool(points: list[dict[str, Any]]) -> None:
    """Anker-Validierung ohne Tool-Kontext (Update: Tool ist unveränderlich)."""
    if not isinstance(points, list) or not points:
        raise ValueError("Eine Zeichnung braucht mindestens einen Ankerpunkt.")


def create_drawing(
    chart_key: str,
    tool: str,
    points: list[dict[str, Any]],
    *,
    style: dict[str, Any] | None = None,
) -> ChartDrawing:
    """Legt eine Zeichnung für ``chart_key`` an; ``ValueError`` bei ungültigem Tool/Ankern."""
    anker = _pruefe_anker(tool, points)
    now = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT INTO chart_drawings(chart_key, tool, points_json, style_json, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                chart_key,
                tool,
                json.dumps(anker, ensure_ascii=False),
                json.dumps(style, ensure_ascii=False) if style is not None else None,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM chart_drawings WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_drawing(row)


def get_drawing(drawing_id: int) -> ChartDrawing | None:
    """Eine Zeichnung anhand ihrer ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM chart_drawings WHERE id = ?", (int(drawing_id),)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_drawing(row)


def list_drawings(chart_key: str) -> list[ChartDrawing]:
    """Alle Zeichnungen eines Charts, in Anlege-Reihenfolge (``id`` aufsteigend).

    Leer bei unbekanntem ``chart_key`` (Lese-Pfad bleibt robust).
    """
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM chart_drawings WHERE chart_key = ? ORDER BY id",
            (chart_key,),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_drawing(row) for row in rows]


def update_drawing(
    drawing_id: int,
    *,
    points: list[dict[str, Any]] | None = None,
    style: dict[str, Any] | None = None,
) -> ChartDrawing | None:
    """Aktualisiert Anker und/oder Styling (``None`` = unverändert), stampft ``updated_at``.

    ``style=None`` bedeutet unverändert — Styling lässt sich per Update nicht
    auf NULL zurücksetzen (leeres Objekt ``{}`` statt dessen speichern). ``None``
    Rückgabe wenn die Zeichnung nicht existiert; ``ValueError`` bei leeren Ankern.
    """
    if points is not None:
        _pruefe_anker_ohne_tool(points)
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM chart_drawings WHERE id = ?", (int(drawing_id),)
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        conn.execute(
            "UPDATE chart_drawings SET points_json = ?, style_json = ?, updated_at = ? "
            "WHERE id = ?",
            (
                row["points_json"]
                if points is None
                else json.dumps(list(points), ensure_ascii=False),
                row["style_json"]
                if style is None
                else json.dumps(style, ensure_ascii=False),
                db.utc_now_iso(),
                int(drawing_id),
            ),
        )
        updated = conn.execute(
            "SELECT * FROM chart_drawings WHERE id = ?", (int(drawing_id),)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_drawing(updated)


def delete_drawing(drawing_id: int) -> bool:
    """Löscht eine Zeichnung; ``True`` wenn sie existierte."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM chart_drawings WHERE id = ?", (int(drawing_id),)
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


def delete_drawings(chart_key: str) -> int:
    """Löscht **alle** Zeichnungen eines Charts („alle löschen“, E034); Anzahl gelöschter."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM chart_drawings WHERE chart_key = ?", (chart_key,)
        )
        deleted = cursor.rowcount
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return deleted
