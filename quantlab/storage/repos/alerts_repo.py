"""Alerts-Repository: ``alerts`` + ``alert_events`` (v2 §D.2, T33).

Ein Alert ist eine Bedingung über Kurs/Metrik eines Symbols (``params_json``,
z. B. ``{"price": 150.0}`` bei ``price_above``); die Evaluation (Zustands-
maschine, Re-Arm nach ``cooldown_min``) macht ``services/alerts.py`` (T33) —
dieses Repo persistiert nur Alert-Definitionen und Auslöse-Historie.

Muster: :mod:`quantlab.db` — eine Verbindung pro Operation, ``BEGIN IMMEDIATE``,
``ROLLBACK`` nur bei offener Transaktion.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from quantlab import db

#: Zulässige Alert-Arten (CHECK-Constraint §D.2).
ALERT_KINDS: frozenset[str] = frozenset(
    {
        "price_above",
        "price_below",
        "price_crossing_up",
        "price_crossing_down",
        "percent_move",
    }
)

#: Zulässige Alert-Zustände (CHECK-Constraint §D.2; Default ``active``).
ALERT_STATES: frozenset[str] = frozenset({"active", "paused", "triggered"})


@dataclass(frozen=True)
class Alert:
    """Zeile aus ``alerts`` (``params`` = geparstes ``params_json``)."""

    id: int
    symbol: str
    kind: str
    params: dict[str, Any]
    state: str  # active | paused | triggered
    note: str | None
    cooldown_min: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AlertEvent:
    """Zeile aus ``alert_events`` (Auslöse-Historie, T33)."""

    id: int
    alert_id: int
    triggered_at: str
    value: float
    message: str
    quote_ts: str | None


def _lade_params(text: str) -> dict[str, Any]:
    """``params_json`` tolerant dekodieren (defekte Zeile → leeres Dict)."""
    try:
        params = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return params if isinstance(params, dict) else {}


def _zu_alert(row: sqlite3.Row) -> Alert:
    return Alert(
        id=int(row["id"]),
        symbol=str(row["symbol"]),
        kind=str(row["kind"]),
        params=_lade_params(row["params_json"]),
        state=str(row["state"]),
        note=row["note"],
        cooldown_min=int(row["cooldown_min"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _zu_event(row: sqlite3.Row) -> AlertEvent:
    return AlertEvent(
        id=int(row["id"]),
        alert_id=int(row["alert_id"]),
        triggered_at=str(row["triggered_at"]),
        value=float(row["value"]),
        message=str(row["message"]),
        quote_ts=row["quote_ts"],
    )


# ---------------------------------------------------------------------------
# Alerts (CRUD)
# ---------------------------------------------------------------------------


def create_alert(
    symbol: str,
    kind: str,
    params: dict[str, Any],
    *,
    note: str | None = None,
    cooldown_min: int = 60,
) -> Alert:
    """Legt einen aktiven Alert an; ``ValueError`` bei unbekannter Art/nicht-Objekt-Params."""
    if kind not in ALERT_KINDS:
        raise ValueError(f"Unbekannte Alert-Art: {kind!r}")
    if not isinstance(params, dict):
        raise ValueError("Alert-Parameter müssen ein Objekt (dict) sein.")
    now = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT INTO alerts(symbol, kind, params_json, state, note, cooldown_min, "
            "created_at, updated_at) VALUES (?, ?, ?, 'active', ?, ?, ?, ?)",
            (
                symbol,
                kind,
                json.dumps(params, ensure_ascii=False, sort_keys=True),
                note,
                int(cooldown_min),
                now,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (cursor.lastrowid,)).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_alert(row)


def get_alert(alert_id: int) -> Alert | None:
    """Ein Alert anhand seiner ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (int(alert_id),)).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_alert(row)


def list_alerts(
    *, state: str | None = None, symbol: str | None = None
) -> list[Alert]:
    """Alerts, neueste zuerst — optional nach Zustand und/oder Symbol gefiltert.

    ``ValueError`` bei unbekanntem Filter-Zustand (Trennung Lese-/Filterfehler).
    """
    if state is not None and state not in ALERT_STATES:
        raise ValueError(f"Unbekannter Alert-Zustand: {state!r}")
    query = "SELECT * FROM alerts"
    clauses: list[str] = []
    args: list[Any] = []
    if state is not None:
        clauses.append("state = ?")
        args.append(state)
    if symbol is not None:
        clauses.append("symbol = ?")
        args.append(symbol)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC, id DESC"
    conn = db.connect()
    try:
        rows = conn.execute(query, args).fetchall()
    finally:
        conn.close()
    return [_zu_alert(row) for row in rows]


def update_alert(
    alert_id: int,
    *,
    state: str | None = None,
    note: str | None = None,
    params: dict[str, Any] | None = None,
    cooldown_min: int | None = None,
) -> Alert | None:
    """Aktualisiert Felder schlüsselweise (``None`` = unverändert), stampft ``updated_at``.

    §C.4 patcht öffentlich nur ``state`` (active/paused) — das Repo kann mehr
    (T33-Auslöser setzt z. B. ``state='triggered'``). ``None`` wenn unbekannt;
    ``ValueError`` bei ungültigem Zielzustand/Params.
    """
    if state is not None and state not in ALERT_STATES:
        raise ValueError(f"Unbekannter Alert-Zustand: {state!r}")
    if params is not None and not isinstance(params, dict):
        raise ValueError("Alert-Parameter müssen ein Objekt (dict) sein.")
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (int(alert_id),)).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        conn.execute(
            "UPDATE alerts SET state = ?, note = ?, params_json = ?, cooldown_min = ?, "
            "updated_at = ? WHERE id = ?",
            (
                row["state"] if state is None else state,
                row["note"] if note is None else note,
                row["params_json"]
                if params is None
                else json.dumps(params, ensure_ascii=False, sort_keys=True),
                row["cooldown_min"] if cooldown_min is None else int(cooldown_min),
                db.utc_now_iso(),
                int(alert_id),
            ),
        )
        updated = conn.execute(
            "SELECT * FROM alerts WHERE id = ?", (int(alert_id),)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_alert(updated)


def delete_alert(alert_id: int) -> bool:
    """Löscht einen Alert samt Historie; ``True`` wenn er existierte."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM alert_events WHERE alert_id = ?", (int(alert_id),))
        cursor = conn.execute("DELETE FROM alerts WHERE id = ?", (int(alert_id),))
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
# Auslöse-Historie (alert_events, T33)
# ---------------------------------------------------------------------------


def add_event(
    alert_id: int,
    *,
    value: float,
    message: str,
    quote_ts: str | None = None,
    triggered_at: str | None = None,
) -> AlertEvent:
    """Protokolliert ein Auslösen; ``ValueError`` wenn der Alert nicht existiert."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        vorhanden = conn.execute(
            "SELECT 1 FROM alerts WHERE id = ?", (int(alert_id),)
        ).fetchone()
        if vorhanden is None:
            raise ValueError(f"Alert {int(alert_id)} existiert nicht.")
        cursor = conn.execute(
            "INSERT INTO alert_events(alert_id, triggered_at, value, message, quote_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                int(alert_id),
                triggered_at if triggered_at is not None else db.utc_now_iso(),
                float(value),
                message,
                quote_ts,
            ),
        )
        row = conn.execute(
            "SELECT * FROM alert_events WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_event(row)


def list_events(alert_id: int, limit: int = 100) -> list[AlertEvent]:
    """Historie eines Alerts, neueste zuerst (``triggered_at`` DESC, ``id`` als Tiebreaker)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM alert_events WHERE alert_id = ? "
            "ORDER BY triggered_at DESC, id DESC LIMIT ?",
            (int(alert_id), int(limit)),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_event(row) for row in rows]
