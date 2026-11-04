"""Fundamentals-, Freshness- und Scheduler-Endpunkte (T21, §C.4/E035).

- ``GET /api/data/fundamentals?symbol&statement&freq`` — Statement als Pivot
  ``{line_items, periods, values}`` (on-read gedreht aus dem Long-Format);
  die Quelle liefert nichts → 200 mit ``leer:true`` statt Fehler (R17).
  Unbekanntes Symbol (weder Registry noch Bestand) → 404 deutsch.
- ``GET /api/data/fundamentals/ratios?symbol`` — berechnete Key-Ratios aus dem
  gespeicherten Bestand (Netto-Marge, ROE, ROA, EBIT-Marge, Umsatzwachstum
  YoY …; Details :mod:`quantlab.services.fundamentals`).
- ``GET /api/data/freshness?domain&key?`` — Staleness-Marker aus
  ``data_freshness`` (inkl. ``alter_tage`` — „Warum sind meine Daten alt?“).
- ``POST /api/scheduler/run-now`` — genau ein sofortiger Tick
  (``data_update | fundamentals | quotes | alerts``; ``quotes``/``alerts``
  liefern bis T22/T33 422 ``job_nicht_verfuegbar``).

Der Scheduler-Endpunkt liegt bewusst in diesem Router (statt eines eigenen
``routers/scheduler.py``): main.py wird so mit **einer** Include-Zeile
erweitert (Welle-B-Abstimmung mit T22) — T22/T33 dürfen ihn ausziehen.

Konventionen wie ``routers/data``: Sync-Handler im Threadpool für DB-Pfade,
``async`` nur beim run-now (wartet auf den Tick); ProviderError → ApiError
1:1 (``to_api_error``), Meldungen deutsch.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.providers.fundamentals import validate_frequenz, validate_statement
from quantlab.providers.symbols import validate_symbol
from quantlab.services import fundamentals as fundamentals_service
from quantlab.services import scheduler
from quantlab.storage.repos import freshness_repo, symbols_repo

router = APIRouter(tags=["fundamentals"])


# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------


def _ohne_provider_fehler[T](  # noqa: UP047 — PEP-695-Generics, Ruff-Style
    func: Callable[..., Awaitable[T] | T]
) -> Callable[..., Awaitable[T] | T]:
    """Mappt ProviderError im Handler in den ApiError-Vertrag (Muster routers.data).

    Behält die Original-Signatur (``functools.wraps``) — sync bleibt sync
    (Threadpool), async bleibt async.
    """

    @functools.wraps(func)
    def wrapper_sync(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)  # type: ignore[return-value]
        except ProviderError as exc:
            raise exc.to_api_error() from exc

    @functools.wraps(func)
    async def wrapper_async(*args: Any, **kwargs: Any) -> T:
        try:
            return await func(*args, **kwargs)  # type: ignore[misc]
        except ProviderError as exc:
            raise exc.to_api_error() from exc

    if inspect.iscoroutinefunction(func):
        return wrapper_async
    return wrapper_sync


def _symbol_bekannt(symbol: str) -> str:
    """Validiert das Kürzel (422) und verlangt Registry-Eintrag oder Bestand (404)."""
    geprueft = validate_symbol(symbol)
    if (
        symbols_repo.get_symbol(geprueft) is None
        and not fundamentals_service.hat_daten(geprueft)
    ):
        raise ApiError(
            404,
            "symbol_nicht_gefunden",
            f"Das Symbol „{geprueft}“ ist nicht gespeichert.",
        )
    return geprueft


def _alter_tage(iso: str) -> float | None:
    """Alter eines Freshness-Markers in Tagen (``None`` bei defektem Zeitstempel)."""
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return round((datetime.now(UTC) - moment) / timedelta(days=1), 3)


# ---------------------------------------------------------------------------
# Statements (Pivot on-read)
# ---------------------------------------------------------------------------


@router.get("/data/fundamentals")
@_ohne_provider_fehler
def read_fundamentals(
    symbol: str,
    statement: str = "income",
    freq: str = "annual",
) -> dict[str, Any]:
    """Statement als ``{line_items, periods, values}`` (§C.4, Pivot on-read).

    ``periods`` aufsteigend, ``values[zeile][spalte]`` mit ``null`` für fehlende
    Kombinationen; ``leer:true`` wenn die Quelle (bisher) nichts geliefert hat
    (kein Fehler, R17). Ungültige ``statement``/``freq`` → 422.
    """
    geprueft = _symbol_bekannt(symbol)
    validate_statement(statement)
    validate_frequenz(freq)
    pivot = fundamentals_service.lies_statement_pivot(geprueft, statement, freq)
    return {
        "symbol": geprueft,
        "statement": statement,
        "freq": freq,
        "line_items": pivot["line_items"],
        "periods": pivot["periods"],
        "values": pivot["values"],
        "currency": pivot["currency"],
        "updated_at": pivot["updated_at"],
        "zeilen": pivot["zeilen"],
        "leer": pivot["leer"],
    }


# ---------------------------------------------------------------------------
# Key-Ratios (berechnet, §C.4)
# ---------------------------------------------------------------------------


@router.get("/data/fundamentals/ratios")
@_ohne_provider_fehler
def read_ratios(symbol: str, freq: str = "annual") -> dict[str, Any]:
    """Berechnete Key-Ratios (Netto-Marge, ROE, ROA, EBIT-Marge, Wachstum YoY …).

    Basis: gespeicherte Statements, neueste Periode; ``freq`` wählt annual
    (Default) bzw. quarterly — fehlt annual vollständig, fällt die Berechnung
    auf quarterly zurück (``freq`` in der Antwort nennt die genutzte Basis).
    Ohne verwertbare Werte → ``leer:true`` (200), kein Fehler.
    """
    geprueft = _symbol_bekannt(symbol)
    validate_frequenz(freq)
    return fundamentals_service.berechne_ratios(geprueft, freq=freq)


# ---------------------------------------------------------------------------
# Datenfrische / Staleness (§C.4, E035-Sichtbarkeit)
# ---------------------------------------------------------------------------


@router.get("/data/freshness")
def read_freshness(domain: str, key: str | None = None) -> dict[str, Any]:
    """Staleness-Marker einer Domain (``key`` schränkt auf ein Symbol ein).

    Antwort: ``{domain, eintraege: [{domain, key, fetched_at, alter_tage}]}`` —
    ``alter_tage`` gerundet auf 3 Stellen (0.0 = frisch); leere Liste heißt
    „noch nie aktualisiert“, kein Fehler.
    """
    domain = domain.strip()
    if not domain:
        raise ApiError(
            422, "ungueltige_domain", "Parameter „domain“ fehlt oder ist leer."
        )
    if key is not None:
        key = key.strip()
        if not key:
            raise ApiError(
                422, "ungueltiger_schluessel", "Parameter „key“ darf nicht leer sein."
            )
        marker = freshness_repo.get_freshness(domain, key)
        eintraege = (
            [
                {
                    "domain": domain,
                    "key": key,
                    "fetched_at": marker,
                    "alter_tage": _alter_tage(marker),
                }
            ]
            if marker is not None
            else []
        )
    else:
        eintraege = [
            {
                "domain": domain,
                "key": schluessel,
                "fetched_at": zeitstempel,
                "alter_tage": _alter_tage(zeitstempel),
            }
            for schluessel, zeitstempel in sorted(
                freshness_repo.list_domain(domain).items()
            )
        ]
    return {"domain": domain, "anzahl": len(eintraege), "eintraege": eintraege}


# ---------------------------------------------------------------------------
# Scheduler run-now (§C.4 — Tests/Agent)
# ---------------------------------------------------------------------------


class RunNowRequest(BaseModel):
    """Body von ``POST /api/scheduler/run-now`` — ``job`` je §C.4."""

    job: str = Field(min_length=1)


@router.post("/scheduler/run-now")
@_ohne_provider_fehler
async def scheduler_run_now(payload: RunNowRequest) -> dict[str, Any]:
    """Löst genau einen sofortigen Tick aus und wartet auf dessen Ende.

    Antwort: das ``scheduler_tick``-Payload-Format (§C.5) plus ``details``
    (z. B. aktualisierte Symbole). ``quotes``/``alerts`` → 422, bis T22/T33
    deren Ticks liefern.
    """
    return await scheduler.run_now(payload.job.strip())
