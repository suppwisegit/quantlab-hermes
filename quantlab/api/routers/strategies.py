"""Strategien-Endpunkte (§C.4, T30): CRUD + Versionen + validate + run.

Eigene Strategien (E026) leben als versionierter Python-Code in
``user_strategies``/``user_strategy_versions`` (:mod:`strategies_repo`, T18);
die Sandbox-Ausführung macht :mod:`quantlab.services.sandbox` (E030).

Regeln:

- **Speichern erzeugt Versionen:** jedes PUT/POST mit ``code`` hängt eine
  neue monotone Version an (Repo-Garantie: ``BEGIN IMMEDIATE`` + MAX+1).
  Gespeicherter Code muss die AST-Prüfung bestehen — jede gespeicherte
  Version ist damit ausführbar (Grundlage für ``user:<id>@<version>`` in
  Backtests, E026). Anmerkung (ANNAHME, dokumentiert): Zwischenstände ohne
  gültigen Code bleiben bewusst ungespeichert; der Editor validiert vorher.
- ``POST /api/strategies/validate`` prüft **ohne Ausführung** → 200 mit
  ``{ok, verstoesse: []}`` bzw. 422 mit der strukturierten Verstoß-Liste
  (Zeilennummern) im Body.
- ``POST /api/strategies/{id}/run`` führt die gewählte Version in der Sandbox
  gegen gespeicherte Kursdaten aus → 200 mit Testlauf-Report bzw. 422 mit
  deutschem Fehler (``sandbox_timeout``, ``sandbox_speicher``, ``sandbox_cpu``,
  ``sandbox_laufzeit``, …). Resourcen-Fehler (Strategie/Version/Symbol
  unbekannt) → 404.

Alle Endpunkte sind bewusst normale ``def`` (nicht ``async``): SQLite-, AST-
und Sandbox-Aufrufe laufen synchron im FastAPI-Threadpool (Muster:
``api/routers/projects.py``) — der Event-Loop blockiert nicht.
"""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.services import data_service, sandbox
from quantlab.storage.repos import strategies_repo

router = APIRouter(tags=["strategies"])

#: Textfeld-Regel (Muster projects-Router): Rand-Whitespace weg, Mindestlänge 1.
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: Obergrenze für Strategie-Code (Zeichen) — Schutz vor Pathologien.
CODE_MAX_CHARS = sandbox.MAX_CODE_CHARS

#: Maximal angezeigte/übertragene Signalpunkte im Run-Report (Zusammenfassung).
_SIGNALE_CAP = 1000

#: Erlaubtes Testlauf-Budget (Wall-Clock des Gastcodes, Sekunden).
_TIMEOUT_MIN_S = 1
_TIMEOUT_MAX_S = 120


# ---------------------------------------------------------------------------
# Request-Schemas
# ---------------------------------------------------------------------------


class StrategyCreateRequest(BaseModel):
    """Body von ``POST /api/strategies`` (``code`` optional = erste Version)."""

    name: NonEmptyText = Field(max_length=200)
    description: str = Field(default="", max_length=5000)
    code: str | None = Field(default=None, max_length=CODE_MAX_CHARS)


class StrategyUpdateRequest(BaseModel):
    """Body von ``PUT /api/strategies/{id}`` — ``code`` → neue Version."""

    name: NonEmptyText | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    code: str | None = Field(default=None, max_length=CODE_MAX_CHARS)


class VersionCreateRequest(BaseModel):
    """Body von ``POST /api/strategies/{id}/versionen`` (E026: neues Snapshot)."""

    code: str = Field(min_length=1, max_length=CODE_MAX_CHARS)


class ValidateRequest(BaseModel):
    """Body von ``POST /api/strategies/validate`` (AST ohne Ausführung)."""

    code: str = Field(min_length=1, max_length=CODE_MAX_CHARS)


class StrategyRunRequest(BaseModel):
    """Body von ``POST /api/strategies/{id}/run`` (Sandbox-Testlauf)."""

    symbol: NonEmptyText
    interval: str = "1d"
    version: int | None = Field(default=None, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)
    start: str | int | None = None
    end: str | int | None = None
    timeout_s: float | None = Field(default=None, ge=_TIMEOUT_MIN_S, le=_TIMEOUT_MAX_S)


# ---------------------------------------------------------------------------
# Response-Mapper + Helfer
# ---------------------------------------------------------------------------


def _strategie_als_dict(
    strategie: strategies_repo.Strategy, mit_versionen: bool = False
) -> dict[str, Any]:
    neueste = strategies_repo.latest_version(strategie.id)
    dict_: dict[str, Any] = {
        "id": strategie.id,
        "name": strategie.name,
        "description": strategie.description,
        "created_at": strategie.created_at,
        "updated_at": strategie.updated_at,
        "latest_version": neueste.version if neueste else None,
        "version_anzahl": len(strategies_repo.list_versions(strategie.id)),
    }
    if neueste is not None:
        dict_["code"] = neueste.code
    if mit_versionen:
        dict_["versionen"] = _versionen_als_dict(strategie.id)
    return dict_


def _versionen_als_dict(strategy_id: int) -> list[dict[str, Any]]:
    return [
        {"version": v.version, "code": v.code, "created_at": v.created_at}
        for v in strategies_repo.list_versions(strategy_id)
    ]


def _strategie_or_404(strategy_id: int) -> strategies_repo.Strategy:
    strategie = strategies_repo.get_strategy(strategy_id)
    if strategie is None:
        raise ApiError(
            404,
            "strategie_nicht_gefunden",
            f"Strategie mit ID {strategy_id} nicht gefunden.",
        )
    return strategie


def _version_or_404(
    strategie: strategies_repo.Strategy, version: int | None
) -> strategies_repo.StrategyVersion:
    """Gewählte Version oder die neueste (Default) — 404 bei fehlender Version."""
    if version is None:
        neueste = strategies_repo.latest_version(strategie.id)
        if neueste is None:
            raise ApiError(
                422,
                "keine_version",
                f"Die Strategie „{strategie.name}“ hat noch keine gespeicherte Version.",
                hint="Erst Code speichern (neue Version), dann ausführen.",
            )
        return neueste
    gefundene = strategies_repo.get_version(strategie.id, version)
    if gefundene is None:
        raise ApiError(
            404,
            "strategie_version_nicht_gefunden",
            f"Version {version} der Strategie „{strategie.name}“ (ID {strategie.id}) "
            f"existiert nicht.",
        )
    return gefundene


def _verstoss_antwort(verstoesse: list[dict[str, Any]]) -> JSONResponse:
    """Verstoß-Liste als 422 im Fehlervertrag + strukturierten Zusatzfeldern.

    Einheitliche Form für validate **und** alle Speicherpfade — der Editor
    erhält so nach einer abgelehnten Speicherung direkt die Zeilennummern.
    """
    erste = verstoesse[0]
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "strategie_code_abgelehnt",
                "message": (
                    f"Der Strategie-Code wurde abgelehnt ({len(verstoesse)} "
                    f"Verstoß/Verstöße) — Zeile {erste.get('zeile')}: "
                    f"{erste.get('meldung')}"
                ),
            },
            "ok": False,
            "verstoesse": verstoesse,
        },
    )


def _pruefe_oder_antwort(code: str) -> JSONResponse | None:
    """AST-Prüfung für Speicherpfade: Verstöße → 422-Antwort, sonst ``None``."""
    verstoesse = sandbox.pruefe_code_als_dict(code)
    return _verstoss_antwort(verstoesse) if verstoesse else None


def _sandbox_422(ergebnis: sandbox.SandboxErgebnis) -> JSONResponse:
    """Sandbox-Fehlausgang als 422 im Fehlervertrag + strukturierten Zusatzfeldern."""
    art = ergebnis.error_kind or "intern"
    hinweis: str | None = None
    if ergebnis.traceback_zeilen:
        erste = ergebnis.traceback_zeilen[0]
        hinweis = f"Zeile {erste.get('zeile')}: {erste.get('text')}"
    elif ergebnis.verstoesse:
        erste = ergebnis.verstoesse[0]
        hinweis = f"Zeile {erste.get('zeile')}: {erste.get('meldung')}"
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": f"sandbox_{art}",
                "message": ergebnis.meldung or "Die Strategie-Ausführung ist fehlgeschlagen.",
                **({"hint": hinweis} if hinweis else {}),
            },
            "error_kind": art,
            "verstoesse": ergebnis.verstoesse,
            "traceback_zeilen": ergebnis.traceback_zeilen,
        },
    )


def _zu_us(value: str | int | None, feld: str) -> int | None:
    """ISO-Datum/-Zeit oder µs → int64 µs UTC (Muster routers/backtest.py)."""
    from datetime import UTC, datetime

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ApiError(
            422, "ungueltiger_zeitraum", f"„{feld}“ muss Datum oder µs-Zeitstempel sein."
        )
    if isinstance(value, int):
        return value
    text = value.strip()
    if text.lstrip("-").isdigit():
        return int(text)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApiError(
            422,
            "ungueltiger_zeitraum",
            f"„{feld}“ ist weder ISO-Datum/-Zeit noch µs-Zeitstempel: „{value}“.",
            hint="Beispiele: 2023-01-01, 2023-01-01T09:30:00Z.",
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)  # ANNAHME: naive Angaben sind UTC
    return int(moment.timestamp()) * 1_000_000


def _testlauf_report(
    ergebnis: sandbox.SandboxErgebnis,
    *,
    strategie_id: int,
    version: int,
    symbol: str,
    interval: str,
    params: dict[str, Any],
    daten: pd.DataFrame,
) -> dict[str, Any]:
    """Sandbox-Ergebnis → Testlauf-Report (Zusammenfassung + Signalserie)."""
    report: dict[str, Any] = {
        "ok": ergebnis.ok,
        "strategy_id": strategie_id,
        "version": version,
        "symbol": symbol,
        "interval": interval,
        "params": params,
        "n_balken": ergebnis.n_balken,
        "dauer_s": round(ergebnis.dauer_s, 3),
        "error_kind": ergebnis.error_kind,
        "meldung": ergebnis.meldung,
    }
    if not ergebnis.ok:
        report["verstoesse"] = ergebnis.verstoesse
        report["traceback_zeilen"] = ergebnis.traceback_zeilen
        return report

    ts_spalte = daten["ts"].tolist() if "ts" in daten.columns else list(range(len(daten)))
    signale = [
        {"ts": int(ts), "weight": gewicht}
        for ts, gewicht in zip(ts_spalte, ergebnis.gewichte, strict=True)
    ]
    werte = [w for w in ergebnis.gewichte if w is not None]
    long_balken = sum(1 for w in werte if w > 0)
    report["zusammenfassung"] = {
        "long_balken": long_balken,
        "flat_balken": len(ergebnis.gewichte) - long_balken,
        "gewicht_min": min(werte) if werte else None,
        "gewicht_max": max(werte) if werte else None,
    }
    report["signale_gesamt"] = len(signale)
    report["signale_gekuerzt"] = len(signale) > _SIGNALE_CAP
    report["signale"] = signale[:_SIGNALE_CAP]
    return report


# ---------------------------------------------------------------------------
# validate (AST-Prüfung ohne Ausführung) — VOR den {id}-Routen deklariert
# ---------------------------------------------------------------------------


@router.post("/strategies/validate", response_model=None)
def validate_strategy(payload: ValidateRequest) -> JSONResponse:
    """AST-Prüfung (E030): 200 ``{ok: true}`` oder 422 mit Verstoß-Liste (Zeilen)."""
    verstoesse = sandbox.pruefe_code_als_dict(payload.code)
    if not verstoesse:
        return JSONResponse(status_code=200, content={"ok": True, "verstoesse": []})
    return _verstoss_antwort(verstoesse)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.get("/strategies")
def list_strategies() -> dict[str, Any]:
    """Alle Strategien (Metadaten + neueste Version), nach Name sortiert."""
    return {
        "strategies": [_strategie_als_dict(s) for s in strategies_repo.list_strategies()]
    }


@router.post("/strategies", status_code=201, response_model=None)
def create_strategy(payload: StrategyCreateRequest) -> dict[str, Any] | JSONResponse:
    """Legt eine Strategie an; mit ``code`` zusätzlich als Version 1 (AST-geprüft)."""
    ablehnung = (
        _pruefe_oder_antwort(payload.code) if payload.code is not None else None
    )
    if ablehnung is not None:
        # Erst prüfen, dann anlegen — kein halbes Objekt bei abgelehntem Code.
        return ablehnung
    try:
        strategie = strategies_repo.create_strategy(payload.name, description=payload.description)
    except ValueError as exc:
        raise ApiError(422, "strategie_name", str(exc)) from exc
    if payload.code is not None:
        strategies_repo.add_version(strategie.id, payload.code)
    return _strategie_als_dict(strategie, mit_versionen=True)


@router.get("/strategies/{strategy_id}")
def get_strategy(strategy_id: int) -> dict[str, Any]:
    """Detail einer Strategie inkl. Versionen und neuestem Code."""
    return _strategie_als_dict(_strategie_or_404(strategy_id), mit_versionen=True)


@router.put("/strategies/{strategy_id}", response_model=None)
def update_strategy(
    strategy_id: int, payload: StrategyUpdateRequest
) -> dict[str, Any] | JSONResponse:
    """Aktualisiert Name/Beschreibung; ``code`` erzeugt eine neue Version (monoton)."""
    _strategie_or_404(strategy_id)
    if payload.code is not None:
        ablehnung = _pruefe_oder_antwort(payload.code)
        if ablehnung is not None:
            return ablehnung
    try:
        aktualisiert = strategies_repo.update_strategy(
            strategy_id, name=payload.name, description=payload.description
        )
        assert aktualisiert is not None  # Existenz oben geprüft
        if payload.code is not None:
            strategies_repo.add_version(strategy_id, payload.code)
    except ValueError as exc:
        raise ApiError(422, "strategie_name", str(exc)) from exc
    return _strategie_als_dict(aktualisiert, mit_versionen=True)


@router.delete("/strategies/{strategy_id}", status_code=204)
def delete_strategy(strategy_id: int) -> Response:
    """Löscht die Strategie samt aller Versionen."""
    _strategie_or_404(strategy_id)
    strategies_repo.delete_strategy(strategy_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Versionen
# ---------------------------------------------------------------------------


@router.get("/strategies/{strategy_id}/versionen")
def list_versions(strategy_id: int) -> dict[str, Any]:
    """Alle Versionen (Code inklusive — lokale DB), aufsteigend."""
    _strategie_or_404(strategy_id)
    return {"versionen": _versionen_als_dict(strategy_id)}


@router.post("/strategies/{strategy_id}/versionen", status_code=201, response_model=None)
def create_version(
    strategy_id: int, payload: VersionCreateRequest
) -> dict[str, Any] | JSONResponse:
    """Hängt eine neue Version an (monoton, atomar; AST-geprüft)."""
    _strategie_or_404(strategy_id)
    ablehnung = _pruefe_oder_antwort(payload.code)
    if ablehnung is not None:
        return ablehnung
    try:
        version = strategies_repo.add_version(strategy_id, payload.code)
    except ValueError as exc:
        raise ApiError(422, "strategie_version", str(exc)) from exc
    return {"version": version.version, "created_at": version.created_at}


# ---------------------------------------------------------------------------
# Sandbox-Testlauf (E030)
# ---------------------------------------------------------------------------


@router.post("/strategies/{strategy_id}/run", response_model=None)
def run_strategy(strategy_id: int, payload: StrategyRunRequest) -> dict[str, Any] | JSONResponse:
    """Führt eine Version in der Sandbox gegen gespeicherte Kursdaten aus (E030)."""
    strategie = _strategie_or_404(strategy_id)
    version = _version_or_404(strategie, payload.version)

    start_us = _zu_us(payload.start, "start")
    end_us = _zu_us(payload.end, "end")
    if start_us is not None and end_us is not None and start_us > end_us:
        raise ApiError(
            422,
            "ungueltiger_zeitraum",
            "„start“ liegt nach „end“.",
            hint="start ist inklusive, end exklusiv (Mikrosekunden UTC).",
        )

    try:
        daten: pd.DataFrame = data_service.read_ohlcv(
            payload.symbol, payload.interval, start=start_us, end=end_us
        )
    except ProviderError as exc:
        raise exc.to_api_error() from exc
    if len(daten) < 2:
        raise ApiError(
            422,
            "zu_wenige_balken",
            f"Für „{payload.symbol}“ ({payload.interval}) sind keine Kursdaten "
            f"gespeichert ({len(daten)} Balken).",
            hint="Lade zuerst Daten im Dashboard herunter.",
        )

    ergebnis = sandbox.fuehre_strategie_aus(
        version.code,
        daten,
        payload.params,
        timeout_s=payload.timeout_s if payload.timeout_s is not None else sandbox.DEFAULT_TIMEOUT_S,
    )
    if not ergebnis.ok:
        return _sandbox_422(ergebnis)
    return _testlauf_report(
        ergebnis,
        strategie_id=strategie.id,
        version=version.version,
        symbol=payload.symbol,
        interval=payload.interval,
        params=payload.params,
        daten=daten,
    )
