"""Live-Quotes-Endpunkt (T22, §C.4/E024): ``GET /api/data/quotes?symbols=A,B``.

Batch-Quotes aus dem Quotes-Service — TTL-Cache und Single-Flight sitzen im
Service (ein Fetch je Symbol, parallele Aufrufe inklusive), der Router ist
dünn. Antwort: ``{quotes: [...], fetched_at}`` mit dem E024-Quote-Objekt je
Eintrag (``{symbol, price, previous_close, change_pct, currency, quote_ts}``
plus optionale fast_info-Kennzahlen); ein fehlerhaftes Symbol liefert ein
``error``-Feld im eigenen Eintrag (die anderen unberührt).

Der Handler ist ``async`` + ``asyncio.to_thread`` (Muster Downloads-Router):
Der Batch kann Netz-Pacing (0,75 s/Symbol, E035) enthalten — der Event-Loop
darf dabei nicht blockieren. Fehler folgen dem C.1-Vertrag
(``{detail: {code, message, hint}}``) über ``ProviderError.to_api_error()``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter

from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.providers.yfinance_provider import jetzt_us
from quantlab.services import quotes as quotes_service

router = APIRouter(tags=["data"])

#: Obergrenze der Symbol-Anzahl je Anfrage — Schutz vor aus Versehen riesigen
#: Batches (z. B. Universum statt Watchlist); fachlich nicht Teil des Vertrags.
MAX_SYMBOLS_PRO_ANFRAGE = 100


def _parse_symbols(symbols: str) -> list[str]:
    """Komma-Liste → bereinigte, deduplizierte Symbol-Liste (422 bei leer)."""
    symbole: list[str] = []
    for roh in symbols.split(","):
        symbol = roh.strip()
        if symbol and symbol not in symbole:
            symbole.append(symbol)
    if not symbole:
        raise ApiError(
            422,
            "ungueltige_symbole",
            "Parameter „symbols“ fehlt oder enthält keine Symbole.",
            hint='Format: eine Komma-Liste, z. B. ?symbols=AAPL,MSFT.',
        )
    if len(symbole) > MAX_SYMBOLS_PRO_ANFRAGE:
        raise ApiError(
            422,
            "ungueltige_symbole",
            f"Zu viele Symbole in einer Anfrage (max. {MAX_SYMBOLS_PRO_ANFRAGE}).",
            hint="Bitte die Anfrage aufteilen.",
        )
    return symbole


@router.get("/data/quotes")
async def read_quotes(symbols: str) -> dict[str, Any]:
    """Batch-Quotes (TTL-Cache, Single-Flight; Felder s. E024)."""
    symbole = _parse_symbols(symbols)
    try:
        eintraege = await asyncio.to_thread(
            quotes_service.get_service().liefere, symbole
        )
    except ProviderError as exc:
        raise exc.to_api_error() from exc
    fetched_at = max(
        (
            eintrag["quote_ts"]
            for eintrag in eintraege
            if isinstance(eintrag.get("quote_ts"), int)
        ),
        default=jetzt_us(),
    )
    return {"quotes": eintraege, "fetched_at": fetched_at}


__all__ = ["MAX_SYMBOLS_PRO_ANFRAGE", "router"]
