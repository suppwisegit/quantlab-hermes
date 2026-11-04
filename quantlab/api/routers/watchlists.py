"""Watchlists-Endpunkte (T28, §C.4): Multi-Watchlist-CRUD, Reorder, Symbole.

| Methode + Pfad | Zweck |
|---|---|
| ``GET · POST /api/watchlists`` | Listen lesen/anlegen |
| ``PATCH /api/watchlists/{id}`` | umbenennen/Reorder |
| ``DELETE /api/watchlists/{id}`` | löschen (Default-Liste 1 löschgeschützt, T18/E033) |
| ``POST · DELETE /api/watchlists/{id}/symbols/{symbol}`` | Symbol in Liste aufnehmen/entfernen |

Konventionen:

- **Sync-Handler im Threadpool** (normale ``def``, Muster routers/data.py) —
  die Repos laufen synchron auf SQLite.
- **Reorder-Vertrag (T18-Repo, ``reorder_symbols``):** die genannten Symbole
  bekommen die Positionen 1..n in der gegebenen Reihenfolge; nicht genannte
  Mitglieder rücken dahinter (bisherige Ordnung bleibt); nicht in der Liste
  enthaltene Symbole werden ignoriert.
- **Fehlervertrag (C.1):** ``404`` unbekannte Liste, ``422`` ungültige
  Eingaben (leerer Name, leerer Patch, ungültiges Symbol-Kürzel) sowie der
  Lösch-Schutz der Default-Liste.
- **Response-Kompatibilität:** Listen-Antworten liefern je Symbol die
  Registry-Metadaten (``name``, ``asset_type``, ``currency``) und die
  Delisted-Marke (T20) mit — die Watchlist-UIs (Dashboard/Märkte) brauchen
  dafür keinen zweiten ``GET /api/data/symbols``-Aufruf. Ein Symbol kann in
  einer Watchlist stehen, ohne in der Registry zu sein (TradingView-Stil:
  Aufnahme über die T19-Suche ohne Registry-Zwang) — ANNAHME T28: Metadaten
  sind dann ``null``/``false``, Quotes/Downloads funktionieren unabhängig
  davon.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.services import data_service
from quantlab.storage.repos import symbols_repo, watchlists_repo
from quantlab.storage.repos.watchlists_repo import Watchlist, WatchlistSymbol

router = APIRouter(tags=["watchlists"])


# ---------------------------------------------------------------------------
# Schemas (Pydantic)
# ---------------------------------------------------------------------------


class WatchlistCreateRequest(BaseModel):
    """Body von ``POST /api/watchlists`` (§C.4)."""

    name: str = Field(min_length=1)


class WatchlistPatchRequest(BaseModel):
    """Body von ``PATCH /api/watchlists/{id}`` (§C.4).

    ``name`` benennt um, ``symbols`` setzt die Reihenfolge (Reorder-Vertrag
    siehe Modul-Docstring); beides zusammen ist erlaubt (erst Umbenennen,
    dann Reorder). Weder ``name`` noch ``symbols`` → 422.
    """

    name: str | None = None
    symbols: list[str] | None = None


# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------


def _registry_mappe() -> dict[str, symbols_repo.Symbol]:
    """Registry-Einträge je Symbol (eine Abfrage für alle Listen)."""
    return {eintrag.symbol: eintrag for eintrag in symbols_repo.list_symbols()}


def _symbol_als_dict(
    eintrag: WatchlistSymbol, registry: dict[str, symbols_repo.Symbol]
) -> dict[str, Any]:
    """Watchlist-Symbolzeile + Registry-Metadaten + Delisted-Marke (T20)."""
    bekannt = registry.get(eintrag.symbol)
    return {
        "symbol": eintrag.symbol,
        "position": eintrag.position,
        "added_at": eintrag.added_at,
        "name": bekannt.name if bekannt is not None else None,
        "asset_type": bekannt.asset_type if bekannt is not None else None,
        "currency": bekannt.currency if bekannt is not None else None,
        "delisted": data_service.ist_delisted(eintrag.symbol),
    }


def _liste_als_dict(
    liste: Watchlist,
    registry: dict[str, symbols_repo.Symbol] | None = None,
) -> dict[str, Any]:
    """Eine Watchlist inkl. ihrer Symbole (Position aufsteigend — klein = oben)."""
    if registry is None:
        registry = _registry_mappe()
    symbole = watchlists_repo.list_symbols(liste.id)
    return {
        "id": liste.id,
        "name": liste.name,
        "position": liste.position,
        "created_at": liste.created_at,
        "symbole": [_symbol_als_dict(eintrag, registry) for eintrag in symbole],
    }


def _liste_oder_404(watchlist_id: int) -> Watchlist:
    """Liste laden oder den einheitlichen 404-Fehler werfen."""
    liste = watchlists_repo.get_watchlist(watchlist_id)
    if liste is None:
        raise ApiError(
            404,
            "watchlist_nicht_gefunden",
            f"Keine Watchlist mit der ID {int(watchlist_id)} gefunden.",
        )
    return liste


def _name_fehler_zu_422(exc: ValueError) -> ApiError:
    """Repo-``ValueError`` (leerer Name) → 422 mit deutscher Meldung."""
    return ApiError(422, "ungueltiger_name", str(exc))


# ---------------------------------------------------------------------------
# Listen (§C.4)
# ---------------------------------------------------------------------------


@router.get("/watchlists")
def list_watchlists() -> dict[str, Any]:
    """Alle Watchlists inkl. Symbole (``position`` aufsteigend, klein = oben)."""
    registry = _registry_mappe()
    return {
        "watchlists": [
            _liste_als_dict(liste, registry) for liste in watchlists_repo.list_watchlists()
        ]
    }


@router.post("/watchlists", status_code=201)
def create_watchlist(payload: WatchlistCreateRequest) -> dict[str, Any]:
    """Legt eine neue (leere) Watchlist an → 201 mit der vollen Liste."""
    try:
        liste = watchlists_repo.create_watchlist(payload.name)
    except ValueError as exc:
        raise _name_fehler_zu_422(exc) from exc
    return _liste_als_dict(liste, registry={})


@router.patch("/watchlists/{watchlist_id}")
def patch_watchlist(watchlist_id: int, payload: WatchlistPatchRequest) -> dict[str, Any]:
    """Benennt um und/oder setzt die Symbol-Reihenfolge neu (§C.4).

    Reorder-Vertrag: ``symbols`` nennt die Mitglieder in ihrer neuen
    Reihenfolge (Positionen 1..n); nicht genannte Mitglieder rücken dahinter,
    Fremde werden ignoriert — doppelte Nennungen werden vorab dedupliziert
    (erste Nennung gewinnt).
    """
    if payload.name is None and payload.symbols is None:
        raise ApiError(
            422,
            "ungueltiger_patch",
            "Der Patch enthält weder „name“ noch „symbols“.",
        )
    _liste_oder_404(watchlist_id)
    if payload.name is not None:
        try:
            watchlists_repo.rename_watchlist(watchlist_id, payload.name)
        except ValueError as exc:
            raise _name_fehler_zu_422(exc) from exc
    if payload.symbols is not None:
        gereiht: list[str] = []
        for symbol in payload.symbols:
            if symbol not in gereiht:
                gereiht.append(symbol)
        watchlists_repo.reorder_symbols(watchlist_id, gereiht)
    return _liste_als_dict(_liste_oder_404(watchlist_id))


@router.delete("/watchlists/{watchlist_id}")
def delete_watchlist(watchlist_id: int) -> dict[str, Any]:
    """Löscht die Liste samt Symbol-Einträgen; die Default-Liste (1) ist geschützt."""
    if watchlist_id == watchlists_repo.DEFAULT_WATCHLIST_ID:
        raise ApiError(
            422,
            "standardliste_geschuetzt",
            "Die Standard-Watchlist (Liste 1) kann nicht gelöscht werden.",
        )
    if not watchlists_repo.delete_watchlist(watchlist_id):
        raise ApiError(
            404,
            "watchlist_nicht_gefunden",
            f"Keine Watchlist mit der ID {int(watchlist_id)} gefunden.",
        )
    return {"id": watchlist_id, "geloescht": True}


# ---------------------------------------------------------------------------
# Symbole in Listen (§C.4)
# ---------------------------------------------------------------------------


@router.post("/watchlists/{watchlist_id}/symbols/{symbol}")
def add_watchlist_symbol(watchlist_id: int, symbol: str) -> dict[str, Any]:
    """Nimmt das Symbol ans Ende der Liste auf (idempotent).

    Antwort ``{watchlist_id, symbol, neu}`` — ``neu=false`` bedeutet, das
    Symbol stand bereits in der Liste (Position unverändert). Unbekannte
    Liste → 404, ungültiges Kürzel → 422.
    """
    try:
        geprueft = validate_symbol(symbol)
    except ProviderError as exc:
        raise exc.to_api_error() from exc
    try:
        neu = watchlists_repo.add_symbol(watchlist_id, geprueft)
    except ValueError as exc:
        raise ApiError(404, "watchlist_nicht_gefunden", str(exc)) from exc
    return {"watchlist_id": watchlist_id, "symbol": geprueft, "neu": neu}


@router.delete("/watchlists/{watchlist_id}/symbols/{symbol}")
def remove_watchlist_symbol(watchlist_id: int, symbol: str) -> dict[str, Any]:
    """Entfernt das Symbol aus genau dieser Liste (idempotent).

    Antwort ``{watchlist_id, symbol, entfernt}`` — ``entfernt=false``, wenn
    das Symbol nicht in der Liste stand; die Registry/Parquet-Daten bleiben
    unberührt (das ist die Domäne von ``DELETE /api/data/symbols``).
    """
    _liste_oder_404(watchlist_id)
    entfernt = watchlists_repo.remove_symbol(watchlist_id, symbol)
    return {"watchlist_id": watchlist_id, "symbol": symbol, "entfernt": entfernt}


__all__ = ["router"]
