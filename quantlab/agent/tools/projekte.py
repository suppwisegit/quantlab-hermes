"""Projekt-Tools (T14, PLAN §E 13–15; Welle I/E102): Reports, Tasks,
Projekt-Kontext sowie Projekt-/Notiz-Verwaltung und Artefakt-Zugriff.

Alle Projekt-Aktionen laufen über :mod:`quantlab.storage.repos.projects_repo`
(T7). Fehlt ``project_id`` in den Argumenten, greift der Run-Kontext
(:func:`quantlab.agent.tools.kontext.aktives_projekt`); ist auch dort kein
Projekt gesetzt, gibt es einen deutschen Tool-Fehler (§D.1: Artefakte sind
projektgebunden).

Welle I (E102/I1): ``manage_projects``, ``manage_notes`` und ``list_artifacts``
schließen die Tool-Lücke zum vollständigen Projekt-System (Repos + REST
existieren bereits) — Delegation ausschließlich auf das Repo, Validierung
spiegelt die Regeln des Projects-Routers (``api/routers/projects.py``).
``set_context`` bleibt bewusst KEIN Tool (Prompt-Injektionsfläche, §E102).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, daten_fehler, pruefe_args, schema_von
from quantlab.agent.tools.kontext import aktives_projekt
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.storage.repos import projects_repo

#: Wie viele Notizen ``read_project_context`` maximal mitliefert (Kontext-Schonung).
_MAX_NOTIZEN = 5

#: Maximale Länge eines Notiz-Körpers im Tool-Ergebnis (Zeichen).
_MAX_NOTIZ_BODY = 600


def _projekt_oder_fehler(projekt_id: int | None) -> int:
    """Projekt-ID auflösen (Argument schlägt Run-Kontext); sonst deutscher Fehler."""
    if projekt_id is not None:
        return projekt_id
    aus_kontext = aktives_projekt()
    if aus_kontext is not None:
        return aus_kontext
    raise ToolFehler(
        "Für diese Aktion muss ein Projekt gewählt sein — bitte eine project_id "
        "angeben oder im Chat ein Projekt auswählen."
    )


def _projekt_or_404(projekt_id: int) -> projects_repo.Project:
    projekt = projects_repo.get_project(projekt_id)
    if projekt is None:
        raise ToolFehler(f"Projekt mit ID {projekt_id} nicht gefunden.")
    return projekt


# ---------------------------------------------------------------------------
# 13) save_report
# ---------------------------------------------------------------------------


class SaveReportArgs(BaseModel):
    """Argumente von ``save_report``."""

    project_id: int | None = Field(
        default=None,
        ge=1,
        description="Ziel-Projekt (im Chat gewähltes Projekt wird automatisch verwendet).",
    )
    title: str = Field(
        min_length=1, max_length=200, description='Report-Titel, z. B. "MSFT Analyse Q3".'
    )
    content_md: str = Field(
        min_length=1, max_length=100_000, description="Vollständiger Report als Markdown."
    )
    recommendation: Literal["kaufen", "halten", "verkaufen"] = Field(
        description='Empfehlung: "kaufen", "halten" oder "verkaufen".'
    )
    symbols: list[str] = Field(
        min_length=1, description='Bewertete Symbole, z. B. ["MSFT"] oder ["AAPL", "MSFT"].'
    )
    rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Optionale Bewertung 1–5 (1 = sehr schlecht, 5 = sehr attraktiv).",
    )


async def save_report(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Speichert einen Aktien-Report (Markdown + Empfehlung) als Projekt-Artefakt."""
    args = pruefe_args(SaveReportArgs, roh_args, "save_report")
    projekt_id = _projekt_oder_fehler(args.project_id)

    def _speichern() -> dict[str, Any]:
        _projekt_or_404(projekt_id)
        artefakt = projects_repo.create_artifact(
            projekt_id,
            "report",
            args.title.strip(),
            {
                "content_md": args.content_md,
                "recommendation": args.recommendation,
                "symbols": args.symbols,
                "rating": args.rating,
            },
        )
        return {
            "status": "gespeichert",
            "artefakt_id": artefakt.id,
            "project_id": projekt_id,
            "titel": artefakt.title,
            "empfehlung": args.recommendation,
            "hinweis": "Der Report erscheint im Artefakte-Panel und in der Projekt-Ansicht.",
        }

    return await asyncio.to_thread(_speichern)


# ---------------------------------------------------------------------------
# 14) manage_tasks
# ---------------------------------------------------------------------------


class ManageTasksArgs(BaseModel):
    """Argumente von ``manage_tasks``."""

    action: Literal["list", "create", "set_status"] = Field(
        description='Aktion: "list" (Todos auflisten), "create" (neuer Todo, braucht title) '
        'oder "set_status" (braucht task_id und status).'
    )
    project_id: int | None = Field(
        default=None, ge=1, description="Projekt (Default: das im Chat gewählte Projekt)."
    )
    task_id: int | None = Field(
        default=None, ge=1, description="Task-ID (nur für set_status), z. B. 7."
    )
    title: str | None = Field(
        default=None,
        max_length=500,
        description='Titel (nur für create), z. B. "Ergebnisbericht schreiben".',
    )
    status: Literal["todo", "doing", "done"] | None = Field(
        default=None, description='Ziel-Status (nur für set_status): "todo", "doing" oder "done".'
    )


async def manage_tasks(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Projekt-Todos listen, anlegen oder ihren Status setzen."""
    args = pruefe_args(ManageTasksArgs, roh_args, "manage_tasks")
    projekt_id = _projekt_oder_fehler(args.project_id)

    def _task_als_dict(task: projects_repo.Task) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "status": task.status,
            "position": task.position,
        }

    def _ausfuehren() -> dict[str, Any]:
        _projekt_or_404(projekt_id)
        if args.action == "list":
            tasks = projects_repo.list_tasks(projekt_id)
            return {
                "action": "list",
                "project_id": projekt_id,
                "tasks": [_task_als_dict(t) for t in tasks],
            }
        if args.action == "create":
            if args.title is None or not args.title.strip():
                raise ToolFehler('„create“ benötigt einen nicht-leeren "title".')
            task = projects_repo.create_task(projekt_id, args.title.strip())
            return {"action": "create", "project_id": projekt_id, "task": _task_als_dict(task)}
        # set_status
        if args.task_id is None:
            raise ToolFehler('„set_status“ benötigt eine "task_id".')
        if args.status is None:
            raise ToolFehler('„set_status“ benötigt einen "status" (todo|doing|done).')
        task = projects_repo.update_task(projekt_id, args.task_id, status=args.status)
        if task is None:
            raise ToolFehler(
                f"Task {args.task_id} wurde im Projekt {projekt_id} nicht gefunden."
            )
        return {"action": "set_status", "project_id": projekt_id, "task": _task_als_dict(task)}

    return await asyncio.to_thread(_ausfuehren)


# ---------------------------------------------------------------------------
# 15) read_project_context
# ---------------------------------------------------------------------------


class ReadProjectContextArgs(BaseModel):
    """Argumente von ``read_project_context``."""

    project_id: int | None = Field(
        default=None, ge=1, description="Projekt (Default: das im Chat gewählte Projekt)."
    )


async def read_project_context(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Projekt-Kontext (vom Nutzer gepflegt) + die letzten Notizen lesen."""
    args = pruefe_args(ReadProjectContextArgs, roh_args, "read_project_context")
    projekt_id = _projekt_oder_fehler(args.project_id)

    def _lesen() -> dict[str, Any]:
        projekt = _projekt_or_404(projekt_id)
        kontext = projects_repo.get_context(projekt_id)
        notizen = [
            {
                "id": note.id,
                "title": note.title,
                "body_md": (
                    note.body_md[:_MAX_NOTIZ_BODY] + "…"
                    if len(note.body_md) > _MAX_NOTIZ_BODY
                    else note.body_md
                ),
                "updated_at": note.updated_at,
            }
            for note in projects_repo.list_notes(projekt_id)[:_MAX_NOTIZEN]
        ]
        return {
            "project_id": projekt_id,
            "name": projekt.name,
            "beschreibung": projekt.description,
            "kontext": kontext.content if kontext is not None else "",
            "letzte_notizen": notizen,
        }

    return await asyncio.to_thread(_lesen)


# ---------------------------------------------------------------------------
# Welle I (E102, I1) 16) manage_projects
# ---------------------------------------------------------------------------


class ManageProjectsArgs(BaseModel):
    """Argumente von ``manage_projects``."""

    aktion: Literal["liste", "erstellen", "umbenennen", "beschreiben", "loeschen",
                    "symbol_hinzufuegen"] = Field(
        description='Aktion: "liste" (Projekte mit Task-/Notiz-/Artefakt-Zählern), '
        '"erstellen" (braucht name), "umbenennen" (project_id + name), "beschreiben" '
        '(project_id + description), "loeschen" (project_id + bestaetigung=true) '
        'oder "symbol_hinzufuegen" (project_id + symbol).'
    )
    project_id: int | None = Field(
        default=None, ge=1, description="Projekt (Default: das im Chat gewählte Projekt)."
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description='Projektname (erstellen Pflicht, umbenennen), z. B. "SAP-Analyse Q4".',
    )
    description: str | None = Field(
        default=None,
        max_length=2000,
        description="Projekt-Beschreibung (erstellen optional, beschreiben Pflicht), "
        'z. B. "Dividenden-Research für Q4".',
    )
    symbol: str | None = Field(
        default=None,
        min_length=1,
        max_length=24,
        description='Markt-Symbol (nur symbol_hinzufuegen), z. B. "SAP.DE" oder "MSFT".',
    )
    bestaetigung: bool = Field(
        default=False,
        description="Nur für loeschen: true — aber nur, wenn der Nutzer im Chat "
        "ausdrücklich zugestimmt hat (Projekt wird samt Tasks/Notizen/Artefakten gelöscht).",
    )


def _name_or_fehler(name: str | None, aktion: str, *, feld: str = "name") -> str:
    """Projekt-/Notiz-Titel erzwingen (Rand-Whitespace entfernt — Muster Projects-Router)."""
    if name is None or not name.strip():
        raise ToolFehler(f'„{aktion}“ benötigt einen nicht-leeren "{feld}".')
    return name.strip()


def _projekt_symbol_validieren(symbol: str | None) -> str:
    """Yahoo-Kürzel validieren (T3-Regeln) — Muster Projects-Router ``_symbol_validieren``.

    Reine ISINs passieren die Whitelist bewusst ebenfalls (E091 — siehe
    ``providers/symbols.py``); wie im REST-Router wird ohne ISIN-Auflösung
    gespeichert. ``ProviderError`` (Meldung bereits deutsch) → :class:`ToolFehler`.
    """
    if symbol is None or not symbol.strip():
        raise ToolFehler('Diese Aktion benötigt ein "symbol".')
    try:
        return validate_symbol(symbol)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc


def _projekt_als_dict(projekt: projects_repo.Project) -> dict[str, Any]:
    return {
        "id": projekt.id,
        "name": projekt.name,
        "description": projekt.description,
        "created_at": projekt.created_at,
        "updated_at": projekt.updated_at,
    }


async def manage_projects(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Projekte auflisten, anlegen, umbenennen, beschreiben, löschen; Symbole verknüpfen."""
    args = pruefe_args(ManageProjectsArgs, roh_args, "manage_projects")

    def _ausfuehren() -> dict[str, Any]:
        if args.aktion == "liste":
            projekte = [
                {
                    **_projekt_als_dict(projekt),
                    "anzahl_tasks": len(projects_repo.list_tasks(projekt.id)),
                    "anzahl_notizen": len(projects_repo.list_notes(projekt.id)),
                    "anzahl_artefakte": len(projects_repo.list_artifacts(projekt.id)),
                }
                for projekt in projects_repo.list_projects()
            ]  # Repo sortiert bereits nach updated_at DESC (zuletzt bearbeitet zuerst)
            return {"aktion": "liste", "anzahl": len(projekte), "projekte": projekte}
        if args.aktion == "erstellen":
            projekt = projects_repo.create_project(
                _name_or_fehler(args.name, "erstellen"), description=args.description or ""
            )
            return {"aktion": "erstellen", "projekt": _projekt_als_dict(projekt)}

        projekt_id = _projekt_oder_fehler(args.project_id)
        projekt = _projekt_or_404(projekt_id)
        if args.aktion == "umbenennen":
            aktualisiert = projects_repo.update_project(
                projekt_id, name=_name_or_fehler(args.name, "umbenennen")
            )
            assert aktualisiert is not None  # Existenz oben geprüft
            return {"aktion": "umbenennen", "projekt": _projekt_als_dict(aktualisiert)}
        if args.aktion == "beschreiben":
            if args.description is None:
                raise ToolFehler('„beschreiben“ benötigt eine "description".')
            aktualisiert = projects_repo.update_project(
                projekt_id, description=args.description
            )
            assert aktualisiert is not None
            return {"aktion": "beschreiben", "projekt": _projekt_als_dict(aktualisiert)}
        if args.aktion == "symbol_hinzufuegen":
            kuerzel = _projekt_symbol_validieren(args.symbol)
            neu = projects_repo.add_project_symbol(projekt_id, kuerzel)
            return {
                "aktion": "symbol_hinzufuegen",
                "project_id": projekt_id,
                "symbol": kuerzel,
                "neu_aufgenommen": neu,
                "symbole": projects_repo.list_project_symbols(projekt_id),
                "hinweis": None if neu else "Symbol war bereits mit dem Projekt verknüpft.",
            }
        # loeschen — Kaskade (Tasks, Notizen, Kontext, Symbole, Artefakte), nur
        # mit ausdrücklicher Bestätigung (Muster „depot-loeschen“/bestaetigung-Arg).
        if not args.bestaetigung:
            raise ToolFehler(
                f"Das Projekt „{projekt.name}“ (ID {projekt_id}) wird mitsamt allen "
                "Tasks, Notizen, Kontexten, Symbolen und Artefakten unwiderruflich "
                "gelöscht. Bitte den Nutzer im Chat ausdrücklich um Bestätigung "
                "bitten; nach seiner Zustimmung den Aufruf mit bestaetigung=true "
                "wiederholen."
            )
        projects_repo.delete_project(projekt_id)
        return {"aktion": "loeschen", "project_id": projekt_id, "geloescht": True}

    return await asyncio.to_thread(_ausfuehren)


# ---------------------------------------------------------------------------
# Welle I (E102, I1) 17) manage_notes
# ---------------------------------------------------------------------------


class ManageNotesArgs(BaseModel):
    """Argumente von ``manage_notes``."""

    aktion: Literal["liste", "erstellen", "aktualisieren", "loeschen"] = Field(
        description='Aktion: "liste" (vollständige Notizen, neueste zuerst), "erstellen" '
        '(braucht title + body_md als Markdown), "aktualisieren" (note_id, title und/oder '
        'body_md) oder "loeschen" (note_id + bestaetigung=true).'
    )
    project_id: int | None = Field(
        default=None, ge=1, description="Projekt (Default: das im Chat gewählte Projekt)."
    )
    note_id: int | None = Field(
        default=None, ge=1, description="Notiz-ID (nur aktualisieren/loeschen), z. B. 3."
    )
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description='Notiz-Titel (erstellen Pflicht, aktualisieren optional), '
        'z. B. "Bewertungs-Thesen".',
    )
    body_md: str | None = Field(
        default=None,
        max_length=100_000,
        description="Notiz-Inhalt als Markdown (erstellen Pflicht, aktualisieren "
        'optional — ersetzt den bisherigen Text), z. B. "## These\\n\\n…".',
    )
    bestaetigung: bool = Field(
        default=False,
        description="Nur für loeschen: true — aber nur, wenn der Nutzer im Chat "
        "ausdrücklich zugestimmt hat.",
    )


def _notiz_als_dict(note: projects_repo.Note) -> dict[str, Any]:
    return {
        "id": note.id,
        "project_id": note.project_id,
        "title": note.title,
        "body_md": note.body_md,
        "created_at": note.created_at,
        "updated_at": note.updated_at,
    }


async def manage_notes(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Projekt-Notizen vollständig lesen, anlegen, Teil-aktualisieren oder löschen."""
    args = pruefe_args(ManageNotesArgs, roh_args, "manage_notes")
    projekt_id = _projekt_oder_fehler(args.project_id)

    def _ausfuehren() -> dict[str, Any]:
        _projekt_or_404(projekt_id)
        if args.aktion == "liste":
            notizen = [
                _notiz_als_dict(note)
                for note in projects_repo.list_notes(projekt_id)
            ]  # Repo sortiert nach updated_at DESC; VOLLSTÄNDIG (kein 5×600-Kapping)
            return {
                "aktion": "liste",
                "project_id": projekt_id,
                "anzahl": len(notizen),
                "notizen": notizen,
            }
        if args.aktion == "erstellen":
            titel = _name_or_fehler(args.title, "erstellen", feld="title")
            if args.body_md is None or not args.body_md.strip():
                raise ToolFehler(
                    '„erstellen“ benötigt einen nicht-leeren "body_md" (Markdown).'
                )
            note = projects_repo.create_note(projekt_id, titel, body_md=args.body_md)
            return {
                "aktion": "erstellen",
                "project_id": projekt_id,
                "notiz": _notiz_als_dict(note),
            }

        if args.note_id is None:
            raise ToolFehler(f'„{args.aktion}“ benötigt eine "note_id".')
        if args.aktion == "aktualisieren":
            if args.title is None and args.body_md is None:
                raise ToolFehler(
                    '„aktualisieren“ benötigt "title" oder "body_md" (Teil-Update).'
                )
            if args.title is not None and not args.title.strip():
                raise ToolFehler('Der "title" einer Notiz darf nicht leer sein.')
            aktualisiert = projects_repo.update_note(
                projekt_id,
                args.note_id,
                title=None if args.title is None else args.title.strip(),
                body_md=args.body_md,
            )
            if aktualisiert is None:
                raise ToolFehler(
                    f"Notiz {args.note_id} wurde im Projekt {projekt_id} nicht gefunden."
                )
            return {
                "aktion": "aktualisieren",
                "project_id": projekt_id,
                "notiz": _notiz_als_dict(aktualisiert),
            }
        # loeschen
        if not args.bestaetigung:
            raise ToolFehler(
                f"Notiz {args.note_id} löschen ist unwiderruflich. Bitte den Nutzer "
                "im Chat ausdrücklich um Bestätigung bitten; nach seiner Zustimmung "
                "den Aufruf mit bestaetigung=true wiederholen."
            )
        geloescht = projects_repo.delete_note(projekt_id, args.note_id)
        if not geloescht:
            raise ToolFehler(
                f"Notiz {args.note_id} wurde im Projekt {projekt_id} nicht gefunden."
            )
        return {
            "aktion": "loeschen",
            "project_id": projekt_id,
            "note_id": args.note_id,
            "geloescht": True,
        }

    return await asyncio.to_thread(_ausfuehren)


# ---------------------------------------------------------------------------
# Welle I (E102, I1) 18) list_artifacts
# ---------------------------------------------------------------------------


class ListArtifactsArgs(BaseModel):
    """Argumente von ``list_artifacts``."""

    aktion: Literal["liste", "lesen"] = Field(
        description='Aktion: "liste" (Artefakte des Projekts: id/kind/titel/created_at) '
        'oder "lesen" (artifact_id → vollständiger Inhalt).'
    )
    project_id: int | None = Field(
        default=None, ge=1, description="Projekt (Default: das im Chat gewählte Projekt)."
    )
    artifact_id: int | None = Field(
        default=None, ge=1, description="Artefakt-ID (nur lesen), z. B. 12."
    )


async def list_artifacts(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Artefakte des Projekts auflisten oder ein Artefakt vollständig lesen."""
    args = pruefe_args(ListArtifactsArgs, roh_args, "list_artifacts")
    projekt_id = _projekt_oder_fehler(args.project_id)

    def _ausfuehren() -> dict[str, Any]:
        _projekt_or_404(projekt_id)
        if args.aktion == "liste":
            artefakte = [
                {
                    "id": artefakt.id,
                    "kind": artefakt.kind,
                    "titel": artefakt.title,
                    "created_at": artefakt.created_at,
                }
                for artefakt in projects_repo.list_artifacts(projekt_id)
            ]  # Repo sortiert bereits neueste zuerst; Payload nur via „lesen“
            return {
                "aktion": "liste",
                "project_id": projekt_id,
                "anzahl": len(artefakte),
                "artefakte": artefakte,
            }
        if args.artifact_id is None:
            raise ToolFehler('„lesen“ benötigt eine "artifact_id".')
        artefakt = projects_repo.get_artifact(projekt_id, args.artifact_id)
        if artefakt is None:
            raise ToolFehler(
                f"Artefakt {args.artifact_id} wurde im Projekt {projekt_id} nicht "
                "gefunden."
            )
        return {
            "aktion": "lesen",
            "project_id": projekt_id,
            "artefakt": {
                "id": artefakt.id,
                "kind": artefakt.kind,
                "titel": artefakt.title,
                "payload": artefakt.payload,
                "created_at": artefakt.created_at,
            },
        }

    return await asyncio.to_thread(_ausfuehren)


# ---------------------------------------------------------------------------
# Registrierung (§E-Reihenfolge 13–15; Welle I/E102 additiv 16–18)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="save_report",
            description=(
                "Speichert einen vollständigen Aktien-Report (Markdown) mit klarer "
                "Empfehlung als Artefakt im Projekt — erscheint im Artefakte-Panel. "
                'Beispiel: {"title": "MSFT Analyse", "content_md": "# Microsoft …", '
                '"recommendation": "kaufen", "symbols": ["MSFT"], "rating": 4}.'
            ),
            parameters=schema_von(SaveReportArgs),
            handler=save_report,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="manage_tasks",
            description=(
                "Verwaltet die Projekt-Todos: auflisten (action=\"list\"), anlegen "
                "(action=\"create\", title=…) oder Status setzen (action=\"set_status\", "
                "task_id=…, status=\"todo\"|\"doing\"|\"done\"). "
                'Beispiel: {"action": "create", "title": "Nvidia-Bericht prüfen"}.'
            ),
            parameters=schema_von(ManageTasksArgs),
            handler=manage_tasks,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="read_project_context",
            description=(
                "Liest den vom Nutzer gepflegten Projekt-Kontext (Hintergründe, Ziele) "
                "und die letzten Notizen des Projekts — ideal als erste Orientierung. "
                'Beispiel: {"project_id": 1} (ohne project_id: das gewählte Projekt).'
            ),
            parameters=schema_von(ReadProjectContextArgs),
            handler=read_project_context,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="manage_projects",
            description=(
                "Verwaltet die Projekte: auflisten mit Task-/Notiz-/Artefakt-Zählern "
                "(aktion=\"liste\"), anlegen (aktion=\"erstellen\"), umbenennen, "
                "beschreiben, löschen (nur mit bestaetigung=true) und Markt-Symbole "
                "verknüpfen (aktion=\"symbol_hinzufuegen\"). "
                'Beispiel: {"aktion": "erstellen", "name": "SAP-Analyse Q4"}.'
            ),
            parameters=schema_von(ManageProjectsArgs),
            handler=manage_projects,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="manage_notes",
            description=(
                "Verwaltet die Projekt-Notizen: vollständig lesen (aktion=\"liste\", "
                "neueste zuerst), anlegen (aktion=\"erstellen\" mit title und "
                "body_md als Markdown), Teil-Update (aktion=\"aktualisieren\") oder "
                "löschen (nur mit bestaetigung=true). "
                'Beispiel: {"aktion": "erstellen", "title": "Thesen", '
                '"body_md": "## These 1\\n\\n…"}.'
            ),
            parameters=schema_von(ManageNotesArgs),
            handler=manage_notes,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="list_artifacts",
            description=(
                "Findet gespeicherte Projekt-Artefakte wieder: Reports, Charts, "
                "Backtests und Analysen auflisten (aktion=\"liste\" mit id/kind/"
                "titel/created_at) oder ein Artefakt vollständig lesen "
                "(aktion=\"lesen\"). Beispiel: {\"aktion\": \"liste\"} — danach "
                'z. B. {"aktion": "lesen", "artifact_id": 12}.'
            ),
            parameters=schema_von(ListArtifactsArgs),
            handler=list_artifacts,
        )
    )


_registrieren()
