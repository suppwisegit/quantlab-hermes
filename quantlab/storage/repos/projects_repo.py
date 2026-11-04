"""Projekte-Repository: ``projects``, ``tasks``, ``notes``, ``project_context``,
``project_symbols`` und ``artifacts`` (T7, §D.1).

Fachliche Grundlage des Projekt-Systems: Ein **Projekt** bündelt Tasks/Todos,
Markdown-Notizen, den **Agent-Kontext** (wird in Agent-Runs injiziert —
Lesezugriff dafür liegt bei ``chat_repo.read_project_context``), verknüpfte
Markt-Symbole und **Artefakte** (Reports/Charts/Backtests/Analysen, die Agent
oder Analyse-Views ablegen — Payload-Formate definiert T14, hier nur
JSON-Grundzugriff).

Muster: ``quantlab.db`` (bindend) — Verbindung pro Operation, Schreib-
Transaktionen mit ``BEGIN IMMEDIATE``, ``ROLLBACK`` nur bei offener
Transaktion, sync-Funktionen (FastAPI-Endpunkte als normale ``def``).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from quantlab import db

#: Zulässige Task-Status (CHECK-Constraint §D.1) — Zyklus todo→doing→done.
TASK_STATUS: frozenset[str] = frozenset({"todo", "doing", "done"})

#: Zulässige Artefakt-Arten (CHECK-Constraint §D.1).
ARTIFACT_KINDS: frozenset[str] = frozenset({"report", "chart", "backtest", "analysis"})


@dataclass(frozen=True)
class Project:
    """Zeile aus ``projects``."""

    id: int
    name: str
    description: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Task:
    """Zeile aus ``tasks`` (``position`` steuert die Reihenfolge in der UI)."""

    id: int
    project_id: int
    title: str
    status: str  # todo | doing | done
    position: int
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True)
class Note:
    """Zeile aus ``notes`` (Markdown-Body)."""

    id: int
    project_id: int
    title: str
    body_md: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ProjectContext:
    """Zeile aus ``project_context`` (1:1 zum Projekt)."""

    project_id: int
    content: str
    updated_at: str


@dataclass(frozen=True)
class Artifact:
    """Zeile aus ``artifacts`` (``payload`` = geparstes ``payload_json``)."""

    id: int
    project_id: int
    kind: str  # report | chart | backtest | analysis
    title: str
    payload: dict[str, Any]
    created_at: str


def _zu_project(row: sqlite3.Row) -> Project:
    return Project(
        id=int(row["id"]),
        name=str(row["name"]),
        description=str(row["description"] or ""),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _zu_task(row: sqlite3.Row) -> Task:
    return Task(
        id=int(row["id"]),
        project_id=int(row["project_id"]),
        title=str(row["title"]),
        status=str(row["status"]),
        position=int(row["position"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        completed_at=row["completed_at"],
    )


def _zu_note(row: sqlite3.Row) -> Note:
    return Note(
        id=int(row["id"]),
        project_id=int(row["project_id"]),
        title=str(row["title"]),
        body_md=str(row["body_md"] or ""),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _zu_context(row: sqlite3.Row) -> ProjectContext:
    return ProjectContext(
        project_id=int(row["project_id"]),
        content=str(row["content"] or ""),
        updated_at=str(row["updated_at"]),
    )


def _zu_artifact(row: sqlite3.Row) -> Artifact:
    try:
        payload = json.loads(row["payload_json"])
    except json.JSONDecodeError:
        payload = {}  # defekte Zeile → leerer Payload (robust gegen Schema-Drift)
    if not isinstance(payload, dict):
        payload = {}
    return Artifact(
        id=int(row["id"]),
        project_id=int(row["project_id"]),
        kind=str(row["kind"]),
        title=str(row["title"]),
        payload=payload,
        created_at=str(row["created_at"]),
    )


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------


def create_project(name: str, *, description: str = "") -> Project:
    """Legt ein Projekt an (``created_at`` = ``updated_at`` = jetzt)."""
    now = db.utc_now_iso()
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — siehe settings_repo.update_section.
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT INTO projects(name, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (name, description, now, now),
        )
        project_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_project(row)


def get_project(project_id: int) -> Project | None:
    """Ein Projekt anhand seiner ID (``None`` wenn unbekannt)."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    finally:
        conn.close()
    return _zu_project(row) if row is not None else None


def list_projects() -> list[Project]:
    """Alle Projekte, zuletzt aktualisierte zuerst (Listen-Sicht T7)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC, id DESC"
        ).fetchall()
    finally:
        conn.close()
    return [_zu_project(row) for row in rows]


def update_project(
    project_id: int, *, name: str | None = None, description: str | None = None
) -> Project | None:
    """Aktualisiert Felder schlüsselweise (``None`` = unverändert) und stampft ``updated_at``.

    ``None`` wenn das Projekt nicht existiert.
    """
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — Read-modify-write serialisiert sauber.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        new_name = row["name"] if name is None else name
        new_description = row["description"] if description is None else description
        conn.execute(
            "UPDATE projects SET name = ?, description = ?, updated_at = ? WHERE id = ?",
            (new_name, new_description, db.utc_now_iso(), project_id),
        )
        updated = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_project(updated)


def delete_project(project_id: int) -> bool:
    """Löscht das Projekt samt abhängiger Zeilen (Tasks, Notes, Kontext, Symbole, Artefakte).

    Die Tabellen tragen untereinander keine FOREIGN KEYs (§D.1) — der Kaskaden-
    Löschpfad läuft deshalb bewusst explizit in einer Transaktion.
    """
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM tasks WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM notes WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM project_context WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM project_symbols WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM artifacts WHERE project_id = ?", (project_id,))
        cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
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
# tasks (Status-Zyklus todo→doing→done, Position für Re-Order)
# ---------------------------------------------------------------------------


def create_task(project_id: int, title: str, *, status: str = "todo") -> Task:
    """Legt einen Task ans Ende der Liste (``position`` = MAX + 1, transaktional)."""
    if status not in TASK_STATUS:
        raise ValueError(f"Unbekannter Task-Status: {status!r}")
    completed_at = db.utc_now_iso() if status == "done" else None
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — Positionszuweisung ist Read-modify-write.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 AS next_position "
            "FROM tasks WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        assert row is not None
        position = int(row["next_position"])
        cursor = conn.execute(
            "INSERT INTO tasks(project_id, title, status, position, created_at, "
            "updated_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, title, status, position, db.utc_now_iso(), db.utc_now_iso(), completed_at),
        )
        task_id = int(cursor.lastrowid)
        created = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert created is not None
    return _zu_task(created)


def list_tasks(project_id: int) -> list[Task]:
    """Tasks des Projekts in Positions-Reihenfolge (Gleichstand: älteste zuerst)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE project_id = ? ORDER BY position, id",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_task(row) for row in rows]


def get_task(project_id: int, task_id: int) -> Task | None:
    """Ein Task innerhalb seines Projekts (``None`` wenn unbekannt/fremd)."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ? AND project_id = ?", (task_id, project_id)
        ).fetchone()
    finally:
        conn.close()
    return _zu_task(row) if row is not None else None


def _positionen_normalisieren(
    conn: sqlite3.Connection, project_id: int, task_id: int, ziel: int
) -> None:
    """Positionen des Projekts lückenlos 0..n-1 (M6 — nach Positions-Änderung).

    Move-Semantik: der gepatchte Task rückt auf Slot ``min(ziel, n−1)``; alle
    übrigen Tasks behalten ihre bisherige ``(position, id)``-Reihenfolge und
    füllen die restlichen Slots. Läuft **in derselben** ``BEGIN IMMEDIATE``-
    Transaktion wie das auslösende Update — wiederholte Re-Orders können keine
    Duplikate oder Lücken hinterlassen.
    """
    zeilen = conn.execute(
        "SELECT id FROM tasks WHERE project_id = ? ORDER BY position, id",
        (project_id,),
    ).fetchall()
    reihenfolge = [int(row["id"]) for row in zeilen if int(row["id"]) != task_id]
    slot = max(0, min(int(ziel), len(reihenfolge)))
    reihenfolge.insert(slot, task_id)
    for index, eintrag in enumerate(reihenfolge):
        conn.execute("UPDATE tasks SET position = ? WHERE id = ?", (index, eintrag))


def update_task(
    project_id: int,
    task_id: int,
    *,
    title: str | None = None,
    status: str | None = None,
    position: int | None = None,
) -> Task | None:
    """Aktualisiert Felder schlüsselweise; jeder Change stampft ``updated_at``.

    Status-Wechsel pflegen ``completed_at``: Sprung auf ``done`` setzt den
    Zeitstempel, Weg von ``done`` räumt ihn ab (§D.1). Eine Positions-Änderung
    normalisiert direkt anschließend (gleiche Transaktion) alle Positionen des
    Projekts auf 0..n-1 (:func:`_positionen_normalisieren`). ``None`` wenn der
    Task nicht (in diesem Projekt) existiert; unbekannter Status → ``ValueError``.
    """
    if status is not None and status not in TASK_STATUS:
        raise ValueError(f"Unbekannter Task-Status: {status!r}")
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — Read-modify-write serialisiert sauber.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ? AND project_id = ?", (task_id, project_id)
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        new_title = row["title"] if title is None else title
        new_status = row["status"] if status is None else status
        new_position = row["position"] if position is None else position
        if status is not None and status == "done":
            new_completed = row["completed_at"] or db.utc_now_iso()
        elif status is not None:
            new_completed = None
        else:
            new_completed = row["completed_at"]
        conn.execute(
            "UPDATE tasks SET title = ?, status = ?, position = ?, updated_at = ?, "
            "completed_at = ? WHERE id = ?",
            (new_title, new_status, new_position, db.utc_now_iso(), new_completed, task_id),
        )
        if position is not None:
            # M6: nach jeder Positions-Änderung lückenlos normalisieren —
            # innerhalb derselben Transaktion (atomar, siehe oben).
            _positionen_normalisieren(conn, project_id, task_id, new_position)
        updated = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_task(updated)


def delete_task(project_id: int, task_id: int) -> bool:
    """Löscht den Task; ``True`` wenn gelöscht (``False`` = nicht gefunden/fremd)."""
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM tasks WHERE id = ? AND project_id = ?", (task_id, project_id)
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
# notes (Markdown)
# ---------------------------------------------------------------------------


def create_note(project_id: int, title: str, *, body_md: str = "") -> Note:
    """Legt eine Notiz an (``created_at`` = ``updated_at`` = jetzt)."""
    now = db.utc_now_iso()
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT INTO notes(project_id, title, body_md, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, title, body_md, now, now),
        )
        note_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None
    return _zu_note(row)


def list_notes(project_id: int) -> list[Note]:
    """Notizen des Projekts, zuletzt bearbeitete zuerst."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM notes WHERE project_id = ? ORDER BY updated_at DESC, id DESC",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_note(row) for row in rows]


def get_note(project_id: int, note_id: int) -> Note | None:
    """Eine Notiz innerhalb ihres Projekts (``None`` wenn unbekannt/fremd)."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ? AND project_id = ?", (note_id, project_id)
        ).fetchone()
    finally:
        conn.close()
    return _zu_note(row) if row is not None else None


def update_note(
    project_id: int, note_id: int, *, title: str | None = None, body_md: str | None = None
) -> Note | None:
    """Aktualisiert Felder schlüsselweise und stampft ``updated_at`` (DoD T7).

    ``None`` wenn die Notiz nicht (in diesem Projekt) existiert.
    """
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ? AND project_id = ?", (note_id, project_id)
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        new_title = row["title"] if title is None else title
        new_body = row["body_md"] if body_md is None else body_md
        conn.execute(
            "UPDATE notes SET title = ?, body_md = ?, updated_at = ? WHERE id = ?",
            (new_title, new_body, db.utc_now_iso(), note_id),
        )
        updated = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_note(updated)


def delete_note(project_id: int, note_id: int) -> bool:
    """Löscht die Notiz; ``True`` wenn gelöscht (``False`` = nicht gefunden/fremd)."""
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM notes WHERE id = ? AND project_id = ?", (note_id, project_id)
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
# project_context (Agent-Kontext, 1:1 zum Projekt)
# ---------------------------------------------------------------------------


def get_context(project_id: int) -> ProjectContext | None:
    """Kontext-Zeile des Projekts (``None`` wenn noch nie gespeichert)."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM project_context WHERE project_id = ?", (project_id,)
        ).fetchone()
    finally:
        conn.close()
    return _zu_context(row) if row is not None else None


def set_context(project_id: int, content: str) -> ProjectContext:
    """Speichert den Kontext (Upsert) und stampft ``updated_at`` (Roundtrip DoD T7)."""
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO project_context(project_id, content, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(project_id) DO UPDATE SET "
            "content = excluded.content, updated_at = excluded.updated_at",
            (project_id, content, db.utc_now_iso()),
        )
        row = conn.execute(
            "SELECT * FROM project_context WHERE project_id = ?", (project_id,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None
    return _zu_context(row)


# ---------------------------------------------------------------------------
# project_symbols (idempotente Verknüpfung Projekt ↔ Symbol)
# ---------------------------------------------------------------------------


def add_project_symbol(project_id: int, symbol: str) -> bool:
    """Verknüpft ein Symbol mit dem Projekt (idempotent — DoD T7).

    ``True`` nur beim erstmaligen Anlegen; erneutes Add ist ein No-Op.
    """
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT OR IGNORE INTO project_symbols(project_id, symbol) VALUES (?, ?)",
            (project_id, symbol),
        )
        added = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return added


def remove_project_symbol(project_id: int, symbol: str) -> bool:
    """Entfernt die Verknüpfung; ``True`` wenn gelöscht."""
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM project_symbols WHERE project_id = ? AND symbol = ?",
            (project_id, symbol),
        )
        removed = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return removed


def list_project_symbols(project_id: int) -> list[str]:
    """Verknüpfte Symbole, alphabetisch sortiert."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT symbol FROM project_symbols WHERE project_id = ? ORDER BY symbol",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()
    return [str(row["symbol"]) for row in rows]


# ---------------------------------------------------------------------------
# artifacts (Payload-Formate definiert T14 — hier nur JSON-Grundzugriff)
# ---------------------------------------------------------------------------


def create_artifact(project_id: int, kind: str, title: str, payload: dict[str, Any]) -> Artifact:
    """Legt ein Artefakt an (``payload`` wird als JSON serialisiert)."""
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"Unbekannte Artefakt-Art: {kind!r}")
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT INTO artifacts(project_id, kind, title, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, kind, title, json.dumps(payload, ensure_ascii=False), db.utc_now_iso()),
        )
        artifact_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None
    return _zu_artifact(row)


def list_artifacts(project_id: int) -> list[Artifact]:
    """Artefakte des Projekts, neueste zuerst (Payload inklusive — lokale DB)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE project_id = ? ORDER BY created_at DESC, id DESC",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_artifact(row) for row in rows]


def get_artifact(project_id: int, artifact_id: int) -> Artifact | None:
    """Ein Artefakt innerhalb seines Projekts (``None`` wenn unbekannt/fremd)."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM artifacts WHERE id = ? AND project_id = ?",
            (artifact_id, project_id),
        ).fetchone()
    finally:
        conn.close()
    return _zu_artifact(row) if row is not None else None


def delete_artifact(project_id: int, artifact_id: int) -> bool:
    """Löscht das Artefakt; ``True`` wenn gelöscht (``False`` = nicht gefunden/fremd)."""
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db).
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM artifacts WHERE id = ? AND project_id = ?",
            (artifact_id, project_id),
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
