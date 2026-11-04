"""Zeichentools-Persistenz (T26, §C.4/E034): CRUD für ``chart_drawings``.

Eine Zeichnung gehört zu einem Chart (``chart_key = "<SYMBOL>|<interval>"``,
z. B. ``"AAPL|1d"``), trägt Ankerpunkte ``[{time: int64 µs UTC, price: float}]``
(E006) und optionales Styling (``{color, width, fontSize?, text? …}``). Das
Repo (``storage/repos/charts_repo.py``, T18) validiert Tool/Anker fachlich —
der Router mappt Feldregeln auf Pydantic (422-Pfad über den zentralen
RequestValidationError-Handler, ``errors.py``) und unbekannte IDs auf 404.

Alle Endpunkte sind bewusst normale ``def`` (nicht ``async``): die SQLite-
Aufrufe des Repos laufen synchron und werden von FastAPI im Threadpool
ausgeführt (Muster: ``api/routers/projects.py``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel, Field, StringConstraints

from quantlab.errors import ApiError
from quantlab.storage.repos import charts_repo

router = APIRouter(tags=["charts"])

#: Textfeld-Regel (Muster projects.py): Rand-Whitespace entfernt, Mindestlänge 1.
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: ``chart_key``-Format ``"<SYMBOL>|<interval>"`` (E034) — serverseitig erzwungen
#: (REV MINOR 11: Müll-/Injection-Keys → 422 statt stummer 201; der Key bleibt
#: ein opaque Datenwert per parametrisiertem SQL, das Pattern härkt nur das Format).
#: Symbolgrammatik wie validate_symbol (inkl. ^-Indizes, =F/=X).
CHART_KEY_MUSTER = r"^[A-Za-z0-9_.^=-]{1,24}\|(1d|1h|15m|5m|1m)$"

#: ``chart_key`` als Query-Parameter (Format + Längenlimit direkt im Annotated).
ChartKeyQuery = Annotated[NonEmptyText, Query(max_length=100, pattern=CHART_KEY_MUSTER)]

#: Zulässige Zeichentools (CHECK-Constraint §D.2 — Spiegel des Repos).
DrawingTool = Literal[
    "trendline",
    "ray",
    "hline",
    "vline",
    "fib_retracement",
    "rectangle",
    "ellipse",
    "brush",
    "text",
]

#: Obergrenze Ankerpunkte je Zeichnung — Brush freihändig kann viele Punkte
#: sammeln, die DB soll aber keine Megabyte-JSOBs sehen (ANNAHME: 2000 sind
#: großzügig über alle Tools).
MAX_PUNKTE = 2000


#: Obergrenze für ``time`` (REV MINOR 12): 2^53 − 1 = Number.MAX_SAFE_INTEGER —
#: oberhalb verliert float64 (JS-Client) Präzision, und µs-Werte jenseits des
#: Jahres ~2255 sind fachlich bedeutungslos; schützt zugleich vor Overflow.
TIME_MAX = 2**53 - 1


class AnkerRequest(BaseModel):
    """Ein Ankerpunkt: ``time`` = int64 µs UTC (E006), ``price`` = Rohkurs."""

    time: int = Field(ge=0, le=TIME_MAX)
    price: float = Field(allow_inf_nan=False)


class DrawingCreateRequest(BaseModel):
    """Body von ``POST /api/charts/drawings``."""

    chart_key: NonEmptyText = Field(max_length=100, pattern=CHART_KEY_MUSTER)
    tool: DrawingTool
    points: list[AnkerRequest] = Field(min_length=1, max_length=MAX_PUNKTE)
    style: dict[str, Any] | None = None


class DrawingUpdateRequest(BaseModel):
    """Body von ``PUT /api/charts/drawings/{id}`` — Punkte und/oder Styling."""

    points: list[AnkerRequest] | None = Field(default=None, min_length=1, max_length=MAX_PUNKTE)
    style: dict[str, Any] | None = None


def _drawing_als_dict(drawing: charts_repo.ChartDrawing) -> dict[str, Any]:
    return {
        "id": drawing.id,
        "chart_key": drawing.chart_key,
        "tool": drawing.tool,
        "points": drawing.points,
        "style": drawing.style,
        "created_at": drawing.created_at,
        "updated_at": drawing.updated_at,
    }


def _drawing_or_404(drawing_id: int) -> charts_repo.ChartDrawing:
    drawing = charts_repo.get_drawing(drawing_id)
    if drawing is None:
        raise ApiError(
            404,
            "zeichnung_nicht_gefunden",
            f"Zeichnung mit ID {drawing_id} nicht gefunden.",
        )
    return drawing


def _punkte_roh(points: list[AnkerRequest] | None) -> list[dict[str, Any]] | None:
    """Pydantic-Anker → rohe Punktdokumente für das Repo (``None`` = unverändert)."""
    if points is None:
        return None
    return [{"time": punkt.time, "price": punkt.price} for punkt in points]


@router.get("/charts/drawings")
def list_drawings(chart_key: ChartKeyQuery) -> dict[str, Any]:
    """Alle Zeichnungen eines Charts (leer bei unbekanntem ``chart_key``)."""
    zeichnungen = charts_repo.list_drawings(chart_key)
    return {"chart_key": chart_key, "drawings": [_drawing_als_dict(z) for z in zeichnungen]}


@router.post("/charts/drawings", status_code=201)
def create_drawing(payload: DrawingCreateRequest) -> dict[str, Any]:
    """Legt eine Zeichnung an (201 + vollständiges Zeichnungs-Objekt)."""
    try:
        drawing = charts_repo.create_drawing(
            payload.chart_key,
            payload.tool,
            _punkte_roh(payload.points) or [],
            style=payload.style,
        )
    except ValueError as exc:
        raise ApiError(422, "zeichnung_ungueltig", str(exc)) from exc
    return _drawing_als_dict(drawing)


@router.put("/charts/drawings/{drawing_id}")
def update_drawing(drawing_id: int, payload: DrawingUpdateRequest) -> dict[str, Any]:
    """Aktualisiert Anker und/oder Styling (Tool ist unveränderlich)."""
    _drawing_or_404(drawing_id)
    try:
        updated = charts_repo.update_drawing(
            drawing_id,
            points=_punkte_roh(payload.points),
            style=payload.style,
        )
    except ValueError as exc:
        raise ApiError(422, "zeichnung_ungueltig", str(exc)) from exc
    assert updated is not None  # Existenz oben geprüft
    return _drawing_als_dict(updated)


@router.delete("/charts/drawings/{drawing_id}", status_code=204)
def delete_drawing(drawing_id: int) -> Response:
    """Löscht eine einzelne Zeichnung."""
    _drawing_or_404(drawing_id)
    charts_repo.delete_drawing(drawing_id)
    return Response(status_code=204)


@router.delete("/charts/drawings")
def delete_all_drawings(chart_key: ChartKeyQuery) -> dict[str, Any]:
    """Löscht **alle** Zeichnungen eines Charts („Alle löschen“)."""
    deleted = charts_repo.delete_drawings(chart_key)
    return {"chart_key": chart_key, "deleted": deleted}


__all__ = ["MAX_PUNKTE", "router"]
