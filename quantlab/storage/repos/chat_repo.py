"""Chat-Repository: ``chat_runs`` + ``chat_messages`` (+ Kontext-Lesezugriff) — T12.

Fachliche Grundlage des Agent-Verlaufs (E009): Ein **Run** ist ein Loop-Lauf
(eine Nutzernachricht inkl. aller Tool-Runden bis zur finalen Antwort); die
**Konversation** ist die Menge aller Runs desselben Projekts (bzw. der globale
Chat bei ``project_id IS NULL``). Folge-Runs im selben Projekt bauen ihren
LLM-Request auf dem vollständigen Konversationsverlauf auf
(:func:`list_conversation_messages`).

Muster: ``quantlab.db`` (bindend) — Verbindung pro Operation, Schreib-Transaktionen
mit ``BEGIN IMMEDIATE``, ``ROLLBACK`` nur bei offener Transaktion, sync-Funktionen
(Async-Aufrufer: ``asyncio.to_thread``).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from quantlab import db

#: Zulässige Run-Status (CHECK-Constraint §D.1).
RUN_STATUS: frozenset[str] = frozenset({"running", "done", "cancelled", "error"})

#: Terminal-Status — ihr Setzen stampft ``finished_at``.
_TERMINAL_STATUS: frozenset[str] = frozenset({"done", "cancelled", "error"})


@dataclass(frozen=True)
class ChatRun:
    """Zeile aus ``chat_runs``."""

    id: str
    project_id: int | None
    title: str | None
    model: str
    thinking_type: str
    reasoning_effort: str | None
    status: str
    error: str | None
    input_tokens: int
    output_tokens: int
    created_at: str
    finished_at: str | None


@dataclass(frozen=True)
class ChatMessage:
    """Zeile aus ``chat_messages`` (``seq`` = Reihenfolge innerhalb des Runs)."""

    id: int
    run_id: str
    seq: int
    role: str
    content: str | None
    reasoning_content: str | None
    tool_calls_json: str | None
    tool_call_id: str | None
    name: str | None
    created_at: str

    def tool_calls(self) -> list[dict[str, Any]] | None:
        """Geparste ``tool_calls_json`` (``None`` wenn keine; defektes JSON → ``None``)."""
        if not self.tool_calls_json:
            return None
        try:
            calls = json.loads(self.tool_calls_json)
        except json.JSONDecodeError:
            return None
        return calls if isinstance(calls, list) else None


def _zu_run(row: sqlite3.Row) -> ChatRun:
    return ChatRun(
        id=str(row["id"]),
        project_id=row["project_id"],
        title=row["title"],
        model=str(row["model"]),
        thinking_type=str(row["thinking_type"]),
        reasoning_effort=row["reasoning_effort"],
        status=str(row["status"]),
        error=row["error"],
        input_tokens=int(row["input_tokens"] or 0),
        output_tokens=int(row["output_tokens"] or 0),
        created_at=str(row["created_at"]),
        finished_at=row["finished_at"],
    )


def _zu_message(row: sqlite3.Row) -> ChatMessage:
    return ChatMessage(
        id=int(row["id"]),
        run_id=str(row["run_id"]),
        seq=int(row["seq"]),
        role=str(row["role"]),
        content=row["content"],
        reasoning_content=row["reasoning_content"],
        tool_calls_json=row["tool_calls_json"],
        tool_call_id=row["tool_call_id"],
        name=row["name"],
        created_at=str(row["created_at"]),
    )


# ---------------------------------------------------------------------------
# chat_runs
# ---------------------------------------------------------------------------


def create_run(
    *,
    run_id: str,
    model: str,
    thinking_type: str,
    reasoning_effort: str | None = None,
    project_id: int | None = None,
    title: str | None = None,
) -> ChatRun:
    """Legt den Run mit Status ``running`` an."""
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO chat_runs(id, project_id, title, model, thinking_type, "
            "reasoning_effort, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
            (
                run_id,
                project_id,
                title,
                model,
                thinking_type,
                reasoning_effort,
                db.utc_now_iso(),
            ),
        )
        row = conn.execute("SELECT * FROM chat_runs WHERE id = ?", (run_id,)).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_run(row)


def get_run(run_id: str) -> ChatRun | None:
    """Ein Run anhand seiner ID (``None`` wenn unbekannt)."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM chat_runs WHERE id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    return _zu_run(row) if row is not None else None


def list_runs(project_id: int | None = None) -> list[ChatRun]:
    """Runs, neueste zuerst; mit ``project_id`` gefiltert, ohne Filter alle."""
    conn = db.connect()
    try:
        if project_id is None:
            rows = conn.execute(
                "SELECT * FROM chat_runs ORDER BY created_at DESC, id DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_runs WHERE project_id = ? "
                "ORDER BY created_at DESC, id DESC",
                (project_id,),
            ).fetchall()
    finally:
        conn.close()
    return [_zu_run(row) for row in rows]


def set_run_status(run_id: str, status: str, *, error: str | None = None) -> None:
    """Setzt den Status; Terminal-Status stampfen ``finished_at``.

    ``error`` wird nur zusammen mit Status ``error`` gespeichert (sonst ignoriert).
    """
    if status not in RUN_STATUS:
        raise ValueError(f"Unbekannter Run-Status: {status!r}")
    if status != "error":
        error = None
    finished_at = db.utc_now_iso() if status in _TERMINAL_STATUS else None
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        if status in _TERMINAL_STATUS:
            conn.execute(
                "UPDATE chat_runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, finished_at, run_id),
            )
        else:
            conn.execute("UPDATE chat_runs SET status = ? WHERE id = ?", (status, run_id))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def add_run_usage(run_id: str, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
    """Zählt Verbrauch inkrementell auf (crash-sicher: je Iteration sofort persistiert)."""
    if input_tokens == 0 and output_tokens == 0:
        return
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE chat_runs SET input_tokens = input_tokens + ?, "
            "output_tokens = output_tokens + ? WHERE id = ?",
            (max(input_tokens, 0), max(output_tokens, 0), run_id),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# chat_messages
# ---------------------------------------------------------------------------


def add_message(
    run_id: str,
    *,
    role: str,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
    name: str | None = None,
) -> ChatMessage:
    """Hängt eine Message an den Run (``seq`` = fortlaufend, transaktionell vergeben)."""
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"Unbekannte Message-Rolle: {role!r}")
    tool_calls_json = (
        json.dumps(tool_calls, ensure_ascii=False) if tool_calls is not None else None
    )
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — seq-Zuweisung ist Read-modify-write.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM chat_messages WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row is not None
        seq = int(row["next_seq"])
        cursor = conn.execute(
            "INSERT INTO chat_messages(run_id, seq, role, content, reasoning_content, "
            "tool_calls_json, tool_call_id, name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                seq,
                role,
                content,
                reasoning_content,
                tool_calls_json,
                tool_call_id,
                name,
                db.utc_now_iso(),
            ),
        )
        message_id = int(cursor.lastrowid)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    message = get_message(message_id)
    assert message is not None  # frisch eingefügt — immer lesbar
    return message


def get_message(message_id: int) -> ChatMessage | None:
    """Eine Message anhand ihrer ID (``None`` wenn unbekannt)."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM chat_messages WHERE id = ?", (message_id,)
        ).fetchone()
    finally:
        conn.close()
    return _zu_message(row) if row is not None else None


def list_messages(run_id: str) -> list[ChatMessage]:
    """Alle Messages eines Runs in ``seq``-Reihenfolge."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
    finally:
        conn.close()
    return [_zu_message(row) for row in rows]


def list_conversation_messages(
    project_id: int | None, *, tiefe: int, ohne_run_id: str | None = None
) -> list[ChatMessage]:
    """Konversationsverlauf (Vorläufer-Runs) für den LLM-Request.

    - Konversation = alle Runs mit gleichem ``project_id`` (inkl. globaler Chat
      bei ``NULL``); System-Messages bleiben außen vor (der Projektkontext wird
      je Run frisch aus ``project_context`` generiert, keine Doppelungen).
    - Älteste zuerst (``chat_runs.created_at``, dann ``seq``); die letzten
      ``tiefe`` Messages bleiben übrig (Settings ``historien_tiefe``).
    - ``ohne_run_id`` nimmt den laufenden Run heraus (dessen Messages kommen
      separat und stets vollständig hinzu).
    """
    tiefe = max(int(tiefe), 0)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT m.* FROM chat_messages m "
            "JOIN chat_runs r ON m.run_id = r.id "
            "WHERE r.project_id IS ? AND m.role != 'system' AND m.run_id != ? "
            "ORDER BY r.created_at ASC, m.seq ASC",
            (project_id, ohne_run_id or ""),
        ).fetchall()
    finally:
        conn.close()
    messages = [_zu_message(row) for row in rows]
    return messages[-tiefe:] if tiefe > 0 else []


# ---------------------------------------------------------------------------
# project_context (Lesezugriff für die Kontext-Injektion; CRUD kommt mit T7)
# ---------------------------------------------------------------------------


def read_project_context(project_id: int) -> str:
    """Projektkontext-Text (``''`` wenn kein Eintrag existiert)."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT content FROM project_context WHERE project_id = ?", (project_id,)
        ).fetchone()
    finally:
        conn.close()
    return str(row["content"]) if row is not None else ""
