"""Alerts-Endpunkte (T33, §C.4): CRUD, Historie, evaluate-now.

- ``GET /api/alerts?state=&symbol=`` — Liste (neueste zuerst), je Eintrag mit
  ``letzter_trigger`` (jüngstes ``alert_events``-Datum, ``null`` ohne Historie)
  für das Alert-Center.
- ``POST /api/alerts`` — anlegen (``price > 0``, ``bars ≥ 1``, 422 deutsch;
  Validierung :func:`quantlab.services.alerts.validiere_params`).
- ``PATCH /api/alerts/{id}`` — **nur** ``state`` (active/paused, §C.4);
  ``triggered`` ist rein backend-seitig (Evaluator).
- ``DELETE /api/alerts/{id}`` — löschen samt Historie (alerts_repo).
- ``GET /api/alerts/{id}/events`` — Auslöse-Historie (neueste zuerst).
- ``POST /api/alerts/{id}/evaluate`` — „Jetzt prüfen“ (Test-Trigger für
  UI/Agent): frischer Quote, dieselbe Zustandsmaschine wie der Scheduler-Tick
  (ein manueller Auslöser startet den Cooldown — kein Doppel-Trigger).

Konventionen wie ``routers/fundamentals``: Sync-Handler im Threadpool für
DB-Pfade, ``async`` beim evaluate (Quotes-Abruf); Fehler im Vertrag
``{detail: {code, message(de), hint?}}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.services import alerts as alerts_service
from quantlab.storage.repos import alerts_repo

router = APIRouter(tags=["alerts"])

#: Obergrenze der Historie je Abruf (Schutz vor Riesen-Responses).
MAX_EVENTS_LIMIT = 500


def _alert_zu_dict(
    alert: alerts_repo.Alert, *, letzter_trigger: str | None = None
) -> dict[str, Any]:
    """Repo-Alert → API-Dict (Felder 1:1, ``letzter_trigger`` ergänzt)."""
    return {
        "id": alert.id,
        "symbol": alert.symbol,
        "kind": alert.kind,
        "params": alert.params,
        "state": alert.state,
        "note": alert.note,
        "cooldown_min": alert.cooldown_min,
        "created_at": alert.created_at,
        "updated_at": alert.updated_at,
        "letzter_trigger": letzter_trigger,
    }


def _event_zu_dict(ereignis: alerts_repo.AlertEvent) -> dict[str, Any]:
    return {
        "id": ereignis.id,
        "alert_id": ereignis.alert_id,
        "triggered_at": ereignis.triggered_at,
        "value": ereignis.value,
        "message": ereignis.message,
        "quote_ts": ereignis.quote_ts,
    }


def _lade_alert(alert_id: int) -> alerts_repo.Alert:
    """Alert laden oder 404 ``alert_nicht_gefunden`` (deutsch)."""
    alert = alerts_repo.get_alert(int(alert_id))
    if alert is None:
        raise ApiError(
            404,
            "alert_nicht_gefunden",
            f"Der Alert {int(alert_id)} existiert nicht (mehr).",
        )
    return alert


# ---------------------------------------------------------------------------
# GET · POST /api/alerts
# ---------------------------------------------------------------------------


@router.get("/alerts")
def list_alerts(state: str | None = None, symbol: str | None = None) -> dict[str, Any]:
    """Alle Alerts (neueste zuerst), optional nach Zustand/Symbol gefiltert."""
    try:
        alerts = alerts_repo.list_alerts(state=state, symbol=symbol)
    except ValueError as exc:
        raise ApiError(422, "ungueltiger_zustand", str(exc)) from exc
    eintraege = []
    for alert in alerts:
        letztes = alerts_repo.list_events(alert.id, limit=1)
        eintraege.append(
            _alert_zu_dict(alert, letzter_trigger=letztes[0].triggered_at if letztes else None)
        )
    return {"alerts": eintraege, "anzahl": len(eintraege)}


class CreateAlertRequest(BaseModel):
    """Body von ``POST /api/alerts`` (§C.4)."""

    symbol: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    params: dict[str, Any]
    note: str | None = None
    cooldown_min: int = 60


@router.post("/alerts", status_code=201)
def create_alert(payload: CreateAlertRequest) -> dict[str, Any]:
    """Legt einen aktiven Alert an; Parameter werden fachlich validiert (422)."""
    try:
        symbol = validate_symbol(payload.symbol)
        params = alerts_service.validiere_params(payload.kind, payload.params)
        if payload.cooldown_min < 1:
            raise ValueError(
                f"„cooldown_min“ muss mindestens 1 sein (erhalten: {payload.cooldown_min})."
            )
        alert = alerts_repo.create_alert(
            symbol,
            payload.kind,
            params,
            note=payload.note,
            cooldown_min=payload.cooldown_min,
        )
    except ValueError as exc:
        raise ApiError(422, "ungueltige_alert_parameter", str(exc)) from exc
    except ProviderError as exc:
        raise exc.to_api_error() from exc
    return _alert_zu_dict(alert)


# ---------------------------------------------------------------------------
# PATCH · DELETE /api/alerts/{id}
# ---------------------------------------------------------------------------


class PatchAlertRequest(BaseModel):
    """Body von ``PATCH /api/alerts/{id}`` — §C.4: nur ``state`` (active/paused)."""

    state: str


@router.patch("/alerts/{alert_id}")
def patch_alert(alert_id: int, payload: PatchAlertRequest) -> dict[str, Any]:
    """Pausiert/aktiviert einen Alert (``triggered`` ist backend-intern → 422)."""
    if payload.state not in ("active", "paused"):
        raise ApiError(
            422,
            "ungueltiger_zustand",
            f"Unbekannter Alert-Zustand „{payload.state}“. "
            "Erlaubt: active, paused („triggered“ setzt der Auslöser).",
        )
    try:
        alert = alerts_repo.update_alert(int(alert_id), state=payload.state)
    except ValueError as exc:
        raise ApiError(422, "ungueltiger_zustand", str(exc)) from exc
    if alert is None:
        raise ApiError(
            404,
            "alert_nicht_gefunden",
            f"Der Alert {int(alert_id)} existiert nicht (mehr).",
        )
    return _alert_zu_dict(alert)


@router.delete("/alerts/{alert_id}")
def delete_alert(alert_id: int) -> dict[str, Any]:
    """Löscht einen Alert samt Historie."""
    geloescht = alerts_repo.delete_alert(int(alert_id))
    if not geloescht:
        raise ApiError(
            404,
            "alert_nicht_gefunden",
            f"Der Alert {int(alert_id)} existiert nicht (mehr).",
        )
    return {"id": int(alert_id), "geloescht": True}


# ---------------------------------------------------------------------------
# GET /api/alerts/{id}/events · POST /api/alerts/{id}/evaluate
# ---------------------------------------------------------------------------


@router.get("/alerts/{alert_id}/events")
def list_events(alert_id: int, limit: int = 100) -> dict[str, Any]:
    """Auslöse-Historie eines Alerts (neueste zuerst, ``limit`` gekappt)."""
    _lade_alert(alert_id)
    if limit < 1 or limit > MAX_EVENTS_LIMIT:
        raise ApiError(
            422,
            "ungueltiges_limit",
            f"„limit“ muss zwischen 1 und {MAX_EVENTS_LIMIT} liegen (erhalten: {limit}).",
        )
    ereignisse = alerts_repo.list_events(int(alert_id), limit=limit)
    return {
        "alert_id": int(alert_id),
        "anzahl": len(ereignisse),
        "ereignisse": [_event_zu_dict(ereignis) for ereignis in ereignisse],
    }


@router.post("/alerts/{alert_id}/evaluate")
async def evaluate_alert(alert_id: int) -> dict[str, Any]:
    """„Jetzt prüfen“: frischer Quote, dieselbe Zustandsmaschine wie der Tick."""
    try:
        return await alerts_service.evaluate_now(int(alert_id))
    except LookupError as exc:
        raise ApiError(
            404,
            "alert_nicht_gefunden",
            f"Der Alert {int(alert_id)} existiert nicht (mehr).",
        ) from exc


__all__ = ["MAX_EVENTS_LIMIT", "router"]
