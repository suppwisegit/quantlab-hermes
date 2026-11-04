"""Märkte-Endpunkte (T28, §C.4): Übersichts-Karten + Top-Mover.

| Methode + Pfad | Zweck |
|---|---|
| ``GET /api/markets/overview`` | Indizes-Karten + Zusammenfassung (nutzt Quotes) |
| ``GET /api/markets/movers?scope=&window=`` | Top-Mover (Change %) über Quotes/Parquet |

Konventionen:

- **overview:** Die Index-Liste kommt aus der Settings-Sektion ``markets``
  (``indizes``, Default ``^GSPC/^DJI/^IXIC/^GDAXI/^FTSE/^N225`` — §T28
  „overview über konfigurierte Index-Liste“). Die Quotes liefert der
  T22-Service (TTL-Cache, Single-Flight); Fehler stehen je Eintrag im
  ``error``-Feld (Batch-Vertrag E024), ein fehlerhafter Index kippt die
  Karten nicht. ``zusammenfassung`` zählt die Karten mit Richtung.
- **movers:** Basis ist der lokale 1d-Parquet-Bestand — ``window`` definiert
  den Rückblick in **Handels-Balken** (``1d``=1, ``1w``=5, ``1m``=21;
  calendarisch robust, dokumentiert im DoD). Der Preis ist der Live-Quote
  (T22), wenn er frisch geliefert wird (``price_quelle: "quote"``), sonst der
  letzte gespeicherte Close (``price_quelle: "parquet"``) — so bleiben die
  Mover auch ohne Netz/Quote-Abdeckung deterministisch. Symbole mit zu
  wenig Balken für das Fenster entfallen (kein erfundenes Change %).
- **scope:** ``watchlist`` = eine Watchlist (Default-Liste 1; ``watchlist_id``
  optional — ANNAHME T28: additive Erweiterung von §C.4, damit die Märkte-
  Ansicht Mover der gerade aktiven Liste zeigen kann), ``universum`` = alle
  Registry-/Bestands-Symbole mit 1d-Balken (E027-Definition
  ``bar_meta.row_count > 0``).
- **Handler:** ``overview`` ist ``async`` + ``asyncio.to_thread`` (Quote-Batch
  kann Netz-Pacing enthalten, Muster routers/quotes.py); ``movers`` läuft als
  normaler ``def`` im Threadpool (nur Parquet/SQLite + gecachte Quotes).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter

from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.providers.yfinance_provider import jetzt_us
from quantlab.services import quotes as quotes_service
from quantlab.storage import parquet_store
from quantlab.storage.repos import settings_repo, symbols_repo, watchlists_repo

router = APIRouter(tags=["markets"])

#: Gültige ``scope``-Werte (§C.4).
MOVER_SCOPES: tuple[str, ...] = ("watchlist", "universum")

#: Gültige ``window``-Werte → Rückblick in 1d-Handels-Balken (§C.4; 1 Woche ≈
#: 5 Handelstage, 1 Monat ≈ 21 Handelstage — Ticker-basiert, keine Kalender).
MOVER_FENSTER: dict[str, int] = {"1d": 1, "1w": 5, "1m": 21}

#: Default-/Obergrenze der Mover-Listen je Richtung.
DEFAULT_MOVER_LIMIT = 10
MAX_MOVER_LIMIT = 50


# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------


def index_symbole() -> list[str]:
    """Settings-Sektion ``markets`` → bereinigte, deduplizierte Index-Liste.

    Defensiv gegen Schema-Drift in der Settings-DB (der Settings-Router prüft
    Typen, aber Bestands-Zeilen können älter sein): Nicht-Strings/Leeres
    wird still herausgefiltert — eine kaputte Einstellung darf die
    Übersichts-Karten nie kippen.
    """
    try:
        sektion = settings_repo.get_section("markets")
    except ValueError:  # pragma: no cover — Defaults sind ab T28 vorhanden
        return []
    roh = sektion.get("indizes", [])
    if not isinstance(roh, list):
        return []
    symbole: list[str] = []
    for eintrag in roh:
        if isinstance(eintrag, str) and eintrag.strip():
            symbol = eintrag.strip()
            if symbol not in symbole:
                symbole.append(symbol)
    return symbole


def _registry_namen() -> dict[str, str]:
    """``{symbol: name}`` der Registry (Namen für Karten/Mover, best-effort)."""
    return {
        eintrag.symbol: eintrag.name
        for eintrag in symbols_repo.list_symbols()
        if eintrag.name
    }


def _fetched_at(eintraege: list[dict[str, Any]]) -> int:
    """Maximaler ``quote_ts`` der Batch-Antwort (Fehler-Einträge ignoriert)."""
    return max(
        (
            eintrag["quote_ts"]
            for eintrag in eintraege
            if isinstance(eintrag.get("quote_ts"), int)
        ),
        default=jetzt_us(),
    )


def _universum_symbole() -> list[str]:
    """Alle Symbole mit 1d-Bestand (E027: ``bar_meta.row_count > 0``), sortiert."""
    return sorted(
        {
            meta.symbol
            for meta in parquet_store.list_bar_meta()
            if meta.interval == "1d" and meta.row_count > 0
        }
    )


def _basis_closes(symbol: str, rueckblick: int) -> tuple[float, float] | None:
    """``(letzter Close, Close ``rueckblick`` Balken davor)`` — ``None`` bei zu wenig Daten.

    Liest genau ``rueckblick + 1`` Balken (Parquet, 1d). Ein leerer Bestand
    liefert ein leeres DataFrame → ``None`` (Read-Pfad wirft für fehlende
    Files keinen Fehler, parquet_store.read_bars).
    """
    balken = parquet_store.read_bars(symbol, "1d", limit=rueckblick + 1)
    if len(balken) < rueckblick + 1:
        return None
    closes = balken["close"].tolist()
    letzter = float(closes[-1])
    bezug = float(closes[-1 - rueckblick])
    if letzter <= 0.0 or bezug <= 0.0:  # keine sinnvolle %-Basis
        return None
    return letzter, bezug


def _quote_preis(eintrag: dict[str, Any] | None) -> float | None:
    """Live-Preis aus einem Quote-Eintrag — ``None`` bei Fehler/Null-Preis."""
    if not isinstance(eintrag, dict):
        return None
    if "error" in eintrag:
        return None
    preis = eintrag.get("price")
    if isinstance(preis, (int, float)) and preis > 0:
        return float(preis)
    return None


# ---------------------------------------------------------------------------
# GET /api/markets/overview (§C.4)
# ---------------------------------------------------------------------------


@router.get("/markets/overview")
async def read_overview() -> dict[str, Any]:
    """Indizes-Karten (Quotes, E024-Felder) + Richtungszusammenfassung.

    Antwort: ``{indizes: [<Quote-Eintrag + name?>], fetched_at,
    zusammenfassung: {anzahl, im_plus, im_minus}}``. Ohne konfigurierte
    Indizes: leere Karten mit 200 (gültiger Leerzustand, UI zeigt Hint).
    """
    symbole = await asyncio.to_thread(index_symbole)
    if not symbole:
        return {
            "indizes": [],
            "fetched_at": jetzt_us(),
            "zusammenfassung": {"anzahl": 0, "im_plus": 0, "im_minus": 0},
        }
    try:
        eintraege = await asyncio.to_thread(quotes_service.get_service().liefere, symbole)
    except ProviderError as exc:
        raise exc.to_api_error() from exc
    namen = await asyncio.to_thread(_registry_namen)
    karten: list[dict[str, Any]] = []
    for eintrag in eintraege:
        karte = dict(eintrag)
        name = namen.get(str(eintrag.get("symbol")))
        if name is not None:
            karte["name"] = name
        karten.append(karte)
    im_plus = sum(
        1
        for karte in karten
        if isinstance(karte.get("change_pct"), (int, float)) and karte["change_pct"] > 0
    )
    im_minus = sum(
        1
        for karte in karten
        if isinstance(karte.get("change_pct"), (int, float)) and karte["change_pct"] < 0
    )
    return {
        "indizes": karten,
        "fetched_at": _fetched_at(eintraege),
        "zusammenfassung": {
            "anzahl": len(karten),
            "im_plus": im_plus,
            "im_minus": im_minus,
        },
    }


# ---------------------------------------------------------------------------
# GET /api/markets/movers (§C.4)
# ---------------------------------------------------------------------------


@router.get("/markets/movers")
def read_movers(
    scope: str = "watchlist",
    window: str = "1d",
    limit: int = DEFAULT_MOVER_LIMIT,
    watchlist_id: int | None = None,
) -> dict[str, Any]:
    """Top-Mover (Change %) je Scope und Fenster.

    Antwort: ``{scope, window, count, gewinner: [...], verlierer: [...]}`` —
    je Liste max. ``limit`` Einträge, sortiert nach Change % (Gewinner
    absteigend, Verlierer aufsteigend). Je Eintrag:
    ``{symbol, name, price, basis_close, change_pct, price_quelle}``.

    ``watchlist_id`` ist nur im Scope ``watchlist`` wirksam (Default: Liste 1);
    im Scope ``universum`` wird der Parameter stillschweigend ignoriert —
    ``scope`` gewinnt (das Frontend sendet ``watchlist_id`` stets mit).
    """
    if scope not in MOVER_SCOPES:
        raise ApiError(
            422,
            "ungueltiger_scope",
            f"Unbekannter Scope „{scope}“.",
            hint=f"Gültige Werte: {', '.join(MOVER_SCOPES)}.",
        )
    if window not in MOVER_FENSTER:
        raise ApiError(
            422,
            "ungueltiges_fenster",
            f"Unbekanntes Fenster „{window}“.",
            hint=f"Gültige Werte: {', '.join(MOVER_FENSTER)}.",
        )
    if limit < 1 or limit > MAX_MOVER_LIMIT:
        raise ApiError(
            422,
            "ungueltiges_limit",
            f"„limit“ muss zwischen 1 und {MAX_MOVER_LIMIT} liegen.",
        )

    if scope == "watchlist":
        liste_id = (
            watchlist_id if watchlist_id is not None else watchlists_repo.DEFAULT_WATCHLIST_ID
        )
        if watchlists_repo.get_watchlist(liste_id) is None:
            raise ApiError(
                404,
                "watchlist_nicht_gefunden",
                f"Keine Watchlist mit der ID {int(liste_id)} gefunden.",
            )
        symbole = [
            eintrag.symbol for eintrag in watchlists_repo.list_symbols(liste_id)
        ]
    else:
        symbole = _universum_symbole()

    rueckblick = MOVER_FENSTER[window]
    # Ein Quote-Batch für alle Kandidaten (TTL-Cache; Fehler je Symbol
    # isoliert im Service — der Mover fällt dann auf Parquet zurück).
    quote_mappe: dict[str, dict[str, Any]] = {}
    if symbole:
        quote_mappe = {
            eintrag["symbol"]: eintrag
            for eintrag in quotes_service.get_service().liefere(symbole)
            if "symbol" in eintrag
        }
    namen = _registry_namen()

    eintraege: list[dict[str, Any]] = []
    for symbol in symbole:
        try:
            basis = _basis_closes(symbol, rueckblick)
        except ProviderError:
            continue  # defekter Bestand — Symbol überspringen, Rest läuft
        if basis is None:
            continue  # zu wenig Balken für das Fenster (dokumentierte Semantik)
        letzter, bezug = basis
        live = _quote_preis(quote_mappe.get(symbol))
        preis = live if live is not None else letzter
        eintraege.append(
            {
                "symbol": symbol,
                "name": namen.get(symbol),
                "price": preis,
                "basis_close": bezug,
                "change_pct": (preis / bezug - 1.0) * 100.0,
                "price_quelle": "quote" if live is not None else "parquet",
            }
        )

    rangfolge = sorted(eintraege, key=lambda eintrag: eintrag["change_pct"], reverse=True)
    return {
        "scope": scope,
        "window": window,
        "count": len(eintraege),
        "gewinner": rangfolge[:limit],
        "verlierer": list(reversed(rangfolge))[:limit],
    }


__all__ = ["MOVER_FENSTER", "MOVER_SCOPES", "index_symbole", "router"]
