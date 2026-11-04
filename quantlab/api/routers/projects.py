"""Projekt-System-Endpunkte (C.1, T7): Projects-, Tasks-, Notes-, Context-,
Project-Symbols- und Artifacts-CRUD.

Alle Endpunkte sind bewusst normale ``def`` (nicht ``async``): die SQLite-
Aufrufe des Repos laufen synchron und werden von FastAPI im Threadpool
ausgeführt — der Event-Loop blockiert nicht (Muster: ``api/routers/settings.py``).

Validierung: Strikte Feldregeln (Leerlängen, Status-Enum, Zeichenlimit) liegen
in den Pydantic-Schemas — Verletzungen laufen über den zentralen
RequestValidationError-Handler als 422 mit deutscher Meldung (``errors.py``).
Fachliche 404-Fälle (Projekt/Task/Notiz/Artefakt unbekannt oder fremd zum
Projekt) werfen :class:`ApiError` mit deutscher Nutzermeldung.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field, StringConstraints

from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.storage.repos import projects_repo

router = APIRouter(tags=["projects"])

#: Zeichenlimit für den Agent-Kontext (§T7: „Zeichenlimit-Anzeige“ — das Limit
#: wird backendseitig erzwungen und frontendseitig angezeigt; ANNAHME: 20 000
#: Zeichen sind für Projekt-Hintergründe großzügig, halten Runs aber schlank).
CONTEXT_MAX_CHARS = 20_000

TaskStatus = Literal["todo", "doing", "done"]
ArtifactKind = Literal["report", "chart", "backtest", "analysis"]

#: Textfeld-Regel: Rand-Whitespace wird entfernt, danach gilt Mindestlänge 1 —
#: „   " ist damit kein gültiger Projekt-/Task-/Notiz-Titel (422-Pfad).
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# ---------------------------------------------------------------------------
# Request-Schemas (422-Pfad: zentraler Validierungs-Handler)
# ---------------------------------------------------------------------------


class ProjectCreateRequest(BaseModel):
    """Body von ``POST /api/projects``."""

    name: NonEmptyText = Field(max_length=200)
    description: str = Field(default="", max_length=5000)


class ProjectUpdateRequest(BaseModel):
    """Body von ``PUT /api/projects/{id}`` — alle Felder optional (Merge)."""

    name: NonEmptyText | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=5000)


class TaskCreateRequest(BaseModel):
    """Body von ``POST /api/projects/{id}/tasks``."""

    title: NonEmptyText = Field(max_length=500)
    status: TaskStatus = "todo"


class TaskUpdateRequest(BaseModel):
    """Body von ``PATCH /api/projects/{id}/tasks/{task_id}`` — Re-Order via ``position``."""

    title: NonEmptyText | None = Field(default=None, max_length=500)
    status: TaskStatus | None = None
    position: int | None = Field(default=None, ge=0)


class NoteCreateRequest(BaseModel):
    """Body von ``POST /api/projects/{id}/notes``."""

    title: NonEmptyText = Field(max_length=200)
    body_md: str = Field(default="", max_length=100_000)


class NoteUpdateRequest(BaseModel):
    """Body von ``PUT /api/projects/{id}/notes/{note_id}`` (Autosave T7)."""

    title: NonEmptyText | None = Field(default=None, max_length=200)
    body_md: str | None = Field(default=None, max_length=100_000)


class ContextUpdateRequest(BaseModel):
    """Body von ``PUT /api/projects/{id}/context`` (Agent-Kontext)."""

    content: str = Field(max_length=CONTEXT_MAX_CHARS)


class ArtifactCreateRequest(BaseModel):
    """Body von ``POST /api/projects/{id}/artifacts`` (Grundgerüst — T14 füllt Payloads)."""

    kind: ArtifactKind
    title: str = Field(min_length=1, max_length=200)
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Response-Mapper + 404-Helper
# ---------------------------------------------------------------------------


def _project_als_dict(project: projects_repo.Project) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
    }


def _task_als_dict(task: projects_repo.Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "project_id": task.project_id,
        "title": task.title,
        "status": task.status,
        "position": task.position,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "completed_at": task.completed_at,
    }


def _note_als_dict(note: projects_repo.Note) -> dict[str, Any]:
    return {
        "id": note.id,
        "project_id": note.project_id,
        "title": note.title,
        "body_md": note.body_md,
        "created_at": note.created_at,
        "updated_at": note.updated_at,
    }


def _context_als_dict(
    context: projects_repo.ProjectContext | None, project_id: int
) -> dict[str, Any]:
    """Kontext-Form: ohne Zeile immer ``content: ""`` (GET vor dem ersten PUT)."""
    if context is None:
        return {"project_id": project_id, "content": "", "updated_at": None}
    return {
        "project_id": context.project_id,
        "content": context.content,
        "updated_at": context.updated_at,
    }


def _artifact_als_dict(artifact: projects_repo.Artifact) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "project_id": artifact.project_id,
        "kind": artifact.kind,
        "title": artifact.title,
        "payload": artifact.payload,
        "created_at": artifact.created_at,
    }


def _project_or_404(project_id: int) -> projects_repo.Project:
    project = projects_repo.get_project(project_id)
    if project is None:
        raise ApiError(
            404,
            "projekt_nicht_gefunden",
            f"Projekt mit ID {project_id} nicht gefunden.",
        )
    return project


def _task_or_404(project_id: int, task_id: int) -> projects_repo.Task:
    task = projects_repo.get_task(project_id, task_id)
    if task is None:
        raise ApiError(
            404,
            "task_nicht_gefunden",
            f"Task mit ID {task_id} wurde im Projekt {project_id} nicht gefunden.",
        )
    return task


def _note_or_404(project_id: int, note_id: int) -> projects_repo.Note:
    note = projects_repo.get_note(project_id, note_id)
    if note is None:
        raise ApiError(
            404,
            "notiz_nicht_gefunden",
            f"Notiz mit ID {note_id} wurde im Projekt {project_id} nicht gefunden.",
        )
    return note


def _artifact_or_404(project_id: int, artifact_id: int) -> projects_repo.Artifact:
    artifact = projects_repo.get_artifact(project_id, artifact_id)
    if artifact is None:
        raise ApiError(
            404,
            "artefakt_nicht_gefunden",
            f"Artefakt mit ID {artifact_id} wurde im Projekt {project_id} nicht gefunden.",
        )
    return artifact


def _symbol_validieren(symbol: str) -> str:
    """Yahoo-Kürzel validieren (T3-Regeln) und in den ApiError-Vertrag mappen."""
    try:
        return validate_symbol(symbol)
    except ProviderError as exc:
        raise exc.to_api_error() from exc


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------


@router.get("/projects")
def list_projects() -> dict[str, Any]:
    """Alle Projekte, zuletzt aktualisierte zuerst."""
    return {"projects": [_project_als_dict(p) for p in projects_repo.list_projects()]}


@router.post("/projects", status_code=201)
def create_project(payload: ProjectCreateRequest) -> dict[str, Any]:
    """Legt ein Projekt an (201 + vollständiges Projekt-Objekt)."""
    project = projects_repo.create_project(payload.name, description=payload.description)
    return _project_als_dict(project)


@router.get("/projects/{project_id}")
def get_project(project_id: int) -> dict[str, Any]:
    """Ein Projekt anhand seiner ID (404 wenn unbekannt)."""
    return _project_als_dict(_project_or_404(project_id))


@router.put("/projects/{project_id}")
def update_project(project_id: int, payload: ProjectUpdateRequest) -> dict[str, Any]:
    """Aktualisiert Name/Beschreibung (Merge) und stampft ``updated_at``."""
    _project_or_404(project_id)
    updated = projects_repo.update_project(
        project_id, name=payload.name, description=payload.description
    )
    assert updated is not None  # Existenz oben geprüft
    return _project_als_dict(updated)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: int) -> Response:
    """Löscht das Projekt samt Tasks/Notizen/Kontext/Symbolen/Artefakten."""
    _project_or_404(project_id)
    projects_repo.delete_project(project_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# tasks (Status todo|doing|done + Position für Re-Order)
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/tasks")
def list_tasks(project_id: int) -> dict[str, Any]:
    """Tasks des Projekts in Positions-Reihenfolge."""
    _project_or_404(project_id)
    return {"tasks": [_task_als_dict(t) for t in projects_repo.list_tasks(project_id)]}


@router.post("/projects/{project_id}/tasks", status_code=201)
def create_task(project_id: int, payload: TaskCreateRequest) -> dict[str, Any]:
    """Legt einen Task ans Ende der Liste (Position = MAX + 1)."""
    _project_or_404(project_id)
    task = projects_repo.create_task(project_id, payload.title, status=payload.status)
    return _task_als_dict(task)


@router.patch("/projects/{project_id}/tasks/{task_id}")
def update_task(project_id: int, task_id: int, payload: TaskUpdateRequest) -> dict[str, Any]:
    """Aktualisiert Titel/Status/Position (Merge) — Status-Zyklus steuert die UI."""
    _task_or_404(project_id, task_id)
    updated = projects_repo.update_task(
        project_id,
        task_id,
        title=payload.title,
        status=payload.status,
        position=payload.position,
    )
    assert updated is not None
    return _task_als_dict(updated)


@router.delete("/projects/{project_id}/tasks/{task_id}", status_code=204)
def delete_task(project_id: int, task_id: int) -> Response:
    """Löscht den Task."""
    _task_or_404(project_id, task_id)
    projects_repo.delete_task(project_id, task_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# notes (Markdown)
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/notes")
def list_notes(project_id: int) -> dict[str, Any]:
    """Notizen des Projekts, zuletzt bearbeitete zuerst."""
    _project_or_404(project_id)
    return {"notes": [_note_als_dict(n) for n in projects_repo.list_notes(project_id)]}


@router.post("/projects/{project_id}/notes", status_code=201)
def create_note(project_id: int, payload: NoteCreateRequest) -> dict[str, Any]:
    """Legt eine Notiz an."""
    _project_or_404(project_id)
    note = projects_repo.create_note(project_id, payload.title, body_md=payload.body_md)
    return _note_als_dict(note)


@router.put("/projects/{project_id}/notes/{note_id}")
def update_note(project_id: int, note_id: int, payload: NoteUpdateRequest) -> dict[str, Any]:
    """Aktualisiert Titel/Markdown-Body — jedes Update stampft ``updated_at`` (Autosave)."""
    _note_or_404(project_id, note_id)
    updated = projects_repo.update_note(
        project_id, note_id, title=payload.title, body_md=payload.body_md
    )
    assert updated is not None
    return _note_als_dict(updated)


@router.delete("/projects/{project_id}/notes/{note_id}", status_code=204)
def delete_note(project_id: int, note_id: int) -> Response:
    """Löscht die Notiz."""
    _note_or_404(project_id, note_id)
    projects_repo.delete_note(project_id, note_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# project_context (Agent-Kontext)
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/context")
def get_context(project_id: int) -> dict[str, Any]:
    """Agent-Kontext des Projekts (ohne Zeile: leerer Inhalt)."""
    _project_or_404(project_id)
    return _context_als_dict(projects_repo.get_context(project_id), project_id)


@router.put("/projects/{project_id}/context")
def put_context(project_id: int, payload: ContextUpdateRequest) -> dict[str, Any]:
    """Speichert den Agent-Kontext (Upsert, Roundtrip) — der Agent liest ihn je Run."""
    _project_or_404(project_id)
    return _context_als_dict(projects_repo.set_context(project_id, payload.content), project_id)


# ---------------------------------------------------------------------------
# project_symbols (Verknüpfung Projekt ↔ Markt-Symbol)
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/symbols")
def list_project_symbols(project_id: int) -> dict[str, Any]:
    """Verknüpfte Symbole des Projekts (alphabetisch)."""
    _project_or_404(project_id)
    return {"symbols": projects_repo.list_project_symbols(project_id)}


@router.post("/projects/{project_id}/symbols/{symbol}")
def add_project_symbol(project_id: int, symbol: str) -> dict[str, Any]:
    """Verknüpft ein Symbol (idempotent — erneutes Add ist ein No-Op)."""
    _project_or_404(project_id)
    kuerzel = _symbol_validieren(symbol)
    projects_repo.add_project_symbol(project_id, kuerzel)
    return {"symbols": projects_repo.list_project_symbols(project_id)}


@router.delete("/projects/{project_id}/symbols/{symbol}")
def remove_project_symbol(project_id: int, symbol: str) -> dict[str, Any]:
    """Entfernt die Symbol-Verknüpfung und liefert die Restliste."""
    _project_or_404(project_id)
    kuerzel = _symbol_validieren(symbol)
    projects_repo.remove_project_symbol(project_id, kuerzel)
    return {"symbols": projects_repo.list_project_symbols(project_id)}


# ---------------------------------------------------------------------------
# artifacts (report | chart | backtest | analysis — Grundzugriff, T14 baut aus)
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/artifacts")
def list_artifacts(project_id: int) -> dict[str, Any]:
    """Artefakte des Projekts, neueste zuerst (Payload inklusive — lokale DB)."""
    _project_or_404(project_id)
    return {"artifacts": [_artifact_als_dict(a) for a in projects_repo.list_artifacts(project_id)]}


@router.post("/projects/{project_id}/artifacts", status_code=201)
def create_artifact(project_id: int, payload: ArtifactCreateRequest) -> dict[str, Any]:
    """Legt ein Artefakt an (Payload frei strukturiert — Formate definiert T14)."""
    _project_or_404(project_id)
    artifact = projects_repo.create_artifact(
        project_id, payload.kind, payload.title, payload.payload
    )
    return _artifact_als_dict(artifact)


@router.get("/projects/{project_id}/artifacts/{artifact_id}")
def get_artifact(project_id: int, artifact_id: int) -> dict[str, Any]:
    """Ein Artefakt inklusive Payload."""
    return _artifact_als_dict(_artifact_or_404(project_id, artifact_id))


@router.delete("/projects/{project_id}/artifacts/{artifact_id}", status_code=204)
def delete_artifact(project_id: int, artifact_id: int) -> Response:
    """Löscht das Artefakt."""
    _artifact_or_404(project_id, artifact_id)
    projects_repo.delete_artifact(project_id, artifact_id)
    return Response(status_code=204)
