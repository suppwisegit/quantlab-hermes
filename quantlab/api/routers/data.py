"""Daten-Endpunkte (T4, PLAN §C.1): Symbol-Registry, OHLC-Read, Download-Jobs.

 Seit T20 (§C.4/E022) additionally:

- ``GET /api/data/ohlc?…&adjust=raw|adjusted`` — Adjustierung **on-read**
  (Default ``raw``; ungültige Werte → 422 deutsch). ``adjusted`` multipliziert
  OHLC mit der je Bar persistierten Spalte ``adj_factor``; bei Stooq-Bestand
  (``bar_meta.source='stooq'``, E023) erscheint stattdessen ein deutsches
  ``hinweis``-Feld in der Response.
- ``GET /api/data/actions`` — Dividenden+Splits aus ``corporate_actions``,
  aufsteigend sortiert, ``start``/``end``-Filter (Tagesauflösung).

 Seit T19 (§C.4/E035/E023) zusätzlich:

- ``GET /api/data/search`` — Symbolsuche via ``yf.Search`` mit 24-h-Cache in
  ``symbol_search_cache`` (TTL wird hier — beim Aufrufer — bewertet, das Repo
  speichert nur ``fetched_at``).
- ``GET /api/data/provider-status`` — Pacing-/Circuit-Zustand von Yahoo plus
  Stooq-Verfügbarkeit.
- ``POST /api/data/symbols`` validiert **suche-basiert** (kein teurer
  ``Ticker.history``-Probe-Call mehr — Antwort < 1 s).
- ``POST /api/data/downloads`` kennt die Option ``repair`` (Gap-Repair).

 Seit Welle F (T-F11) zusätzlich:

- ``GET /api/data/gaps`` — Lücken-Report je Symbol (umschließt
  ``downloads.pruefe_lücken``, nur ``1d``; Details im Handler-Docstring).

Konventionen:

- **Sync-Handler im Threadpool:** alle DB-/Parquet-Zugriffe und die
  Symbolsuche laufen in normalen ``def``-Handlern — FastAPI führt sie in
  seinem Threadpool aus, der Event-Loop bleibt frei. Die beiden ``async``-
  Endpunkte (Job anlegen/abbrechen) lagern ihre DB-Anteile an
  ``asyncio.to_thread`` aus (Muster agent.py).
- **Fehlervertrag (C.1):** :class:`~quantlab.providers.base.ProviderError`
  wird 1:1 über ``to_api_error()`` übernommen — 404 fehlende Daten, 422
  ungültige Eingaben, 502 Quelle nicht erreichbar, 429 Yahoo gedrosselt;
  Meldungen deutsch.
- **Zeit:** Antworten nutzen int64 Mikrosekunden UTC (E006); Query-Parameter
  ``start``/``end`` akzeptieren ``YYYY-MM-DD``/ISO-8601 (naive Angaben = UTC).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantlab import events
from quantlab.errors import ApiError
from quantlab.providers import rate_limit, stooq_provider
from quantlab.providers.base import (
    ProviderError,
    SymbolInfo,
    SymbolNotFoundError,
    validate_interval,
)
from quantlab.providers.symbols import validate_symbol
from quantlab.services import data_service, downloads, indicators
from quantlab.storage import parquet_store
from quantlab.storage.repos import (
    downloads_repo,
    search_cache_repo,
    symbols_repo,
    watchlists_repo,
)

router = APIRouter(tags=["data"])

#: TTL der Suchergebnisse in Sekunden (T19: 24 h).
SUCHE_TTL_S = 24 * 60 * 60

#: Obergrenze für ``limit`` der Symbolsuche (eine Seite reicht zur Auswahl).
SUCHE_MAX_LIMIT = 50


# ---------------------------------------------------------------------------
# Schemas (Pydantic)
# ---------------------------------------------------------------------------


class SymbolAddRequest(BaseModel):
    """Body von ``POST /api/data/symbols`` (§C.1)."""

    symbol: str = Field(min_length=1)


class DownloadCreateRequest(BaseModel):
    """Body von ``POST /api/data/downloads`` (§C.1 + T19 ``repair`` + MINOR 6).

    ``start``/``end`` schlagen ``period`` vor (Format ``YYYY-MM-DD``/ISO-8601,
    naive Angaben = UTC); ohne alles gilt ``1mo``. Intraday-Fenster jenseits
    60 Tage werden bereits beim Anlegen mit 422 abgelehnt. ``repair=True``
    (T19): Fetch mit yfinance ``repair=True`` über das gesamte Bestandsfenster
    (ersetzt nur das Fenster; Balken außerhalb bleiben unangetastet).
    ``force_full=True`` (MINOR 6, T20-Fix): Convenience-Flag — unterdrückt den
    Inkrement und lädt intern ab ``first_ts`` des Bestands (wie ein expliziter
    ``start``, nur ohne ihn selbst angeben zu müssen); rein additiv, Default
    ``False`` = unverändertes Inkrement-Verhalten.
    """

    symbols: list[str] = Field(min_length=1)
    interval: str
    period: str | None = None
    start: str | None = None
    end: str | None = None
    repair: bool = False
    force_full: bool = False


class IndicatorRequest(BaseModel):
    """Ein Indikator-Wunsch innerhalb von ``POST /api/data/indicators`` (§C.1).

    ``type`` ∈ ``sma | ema | rsi | macd | bollinger`` (close-basiert, T6) plus
    der T25-Typen ``atr | obv | vwap | stoch | adx | cci | williamsr |
    ichimoku`` (OHLCV-basiert — high/low/volume kommen aus den gespeicherten
    Balken, der Client schickt nur ``params``). ``params`` sind typ-abhängig
    (Defaults: SMA/EMA ``period=20``, RSI ``period=14``, MACD
    ``fast=12, slow=26, signal=9``, Bollinger ``period=20, std=2``, ATR
    ``period=14``, Stoch ``k=14, smooth=3, d=3``, ADX ``period=14``, CCI
    ``period=20``, Williams %R ``period=14``, Ichimoku ``tenkan=9, kijun=26,
    senkou=52``; OBV/VWAP sind parameterlos). Die fachliche Validierung
    (unbekannter Typ, unbekannte/sinnwidrige Parameter) macht
    :mod:`quantlab.services.indicators` → 422 mit deutscher Meldung.
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------


def _ohne_provider_fehler[T](  # noqa: UP047 — PEP-695-Generics, Ruff-Style
    func: Callable[..., Awaitable[T] | T]
) -> Callable[..., Awaitable[T] | T]:
    """Mappt einen ProviderError im Handler in den ApiError-Vertrag (C.1).

    Behält die Original-Signatur bei (``functools.wraps``), damit FastAPI die
    Parameter-Auflösung unverändert am dekorierten Handler machen kann; sync
    Handler bleiben sync (Threadpool), async bleiben async.
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


def _watchlist_aufnehmen(symbol: str) -> None:
    """Nimmt das Symbol ans Ende der Default-Watchlist auf (idempotent, E033).

    Die Default-Liste (id=1) legt Migration 3 immer an — der v1-Global-Watchlist
    entspricht sie eins zu eins (Positionen/``added_at`` überführt).
    """
    watchlists_repo.add_symbol(watchlists_repo.DEFAULT_WATCHLIST_ID, symbol)


def _watchlist_positionen() -> dict[str, int]:
    """Positionen der Default-Watchlist (klein = oben; E033: Liste 1)."""
    return watchlists_repo.positionen(watchlists_repo.DEFAULT_WATCHLIST_ID)


def _abdeckung_je_intervall(symbol: str) -> dict[str, dict[str, Any]]:
    """``bar_meta`` des Symbols → ``{interval: {first_ts, last_ts, row_count}}`` (µs UTC)."""
    abdeckung: dict[str, dict[str, Any]] = {}
    for meta in parquet_store.list_bar_meta(symbol):
        abdeckung[meta.interval] = {
            "first_ts": meta.first_ts,
            "last_ts": meta.last_ts,
            "row_count": meta.row_count,
        }
    return abdeckung


def _symbol_als_dict(
    symbol: symbols_repo.Symbol, positionen: dict[str, int] | None = None
) -> dict[str, Any]:
    if positionen is None:
        positionen = _watchlist_positionen()
    return {
        "symbol": symbol.symbol,
        "name": symbol.name,
        "asset_type": symbol.asset_type,
        "exchange_tz": symbol.exchange_tz,
        "currency": symbol.currency,
        "added_at": symbol.added_at,
        "watchlist_position": positionen.get(symbol.symbol),
        "abdeckung": _abdeckung_je_intervall(symbol.symbol),
        # T20: Delisted-Marke (leere Yahoo-History bei bekanntem Symbol) —
        # Bestand/Abdeckung bleiben sichtbar, UI/Quotes (T22) können kennzeichnen.
        "delisted": data_service.ist_delisted(symbol.symbol),
    }


def _download_als_dict(job: downloads_repo.Download) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "symbols": job.symbols,
        "interval": job.interval,
        "params": job.params,
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }


# ---------------------------------------------------------------------------
# Symbolsuche (T19, §C.4): yf.Search + 24-h-Cache + suche-basiertes Symbol-Add
# ---------------------------------------------------------------------------


def _suche_cache_frisch(fetched_at: str) -> bool:
    """Ist der Cache-Eintrag jünger als :data:`SUCHE_TTL_S` (24 h)?

    Die TTL-Bewertung liegt bewusst beim Aufrufer (T18-Übergabe): Das Repo
    speichert nur ``fetched_at`` — defekte Zeitstempel gelten als „abgelaufen“.
    """
    try:
        moment = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return datetime.now(UTC) - moment < timedelta(seconds=SUCHE_TTL_S)


def _suchergebnis_aufbereiten(quotes: list[Any]) -> list[dict[str, Any]]:
    """``yf.Search``-Quotes → vertragliche Trefferliste ``{symbol, name,
    exchange, quoteType}`` (§C.4). Einträge ohne Symbol entfallen."""
    treffer: list[dict[str, Any]] = []
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        symbol = quote.get("symbol")
        if not symbol:
            continue
        treffer.append(
            {
                "symbol": str(symbol),
                "name": quote.get("shortname") or quote.get("longname"),
                "exchange": quote.get("exchange"),
                "quoteType": quote.get("quoteType"),
            }
        )
    return treffer


def _yf_suche(query: str, limit: int) -> list[dict[str, Any]]:
    """Ein echter ``yf.Search``-Call (lazy importiert, E013) → Trefferliste.

    Auch die Suche ist ein Yahoo-Call (E035: „vor jedem Yahoo-Call“): Der
    Guard :meth:`YahooRateLimiter.vor_yahoo_aufruf` läuft daher VOR dem
    ``yf.Search`` — bei offenem Circuit fliegt sofort
    :class:`~quantlab.providers.rate_limit.YahooGedrosseltError` (429, deutsch),
    und ein 429 **beim** Search öffnet den Circuit (sichtbar über
    ``GET /api/data/provider-status``). Seit E087 auf dem **LEICHT**-Circuit
    (:data:`rate_limit.RATE_LIMITER_LEICHT`, gemeinsam mit Quotes-Poll und
    Fundamentals) — ein 429 der Suche öffnet nie einen Download-Circuit. Die
    robuste Bedienung bei offenem Circuit (Cache/Registry statt 429, T-F12)
    entscheidet :func:`_symbolsuche` VOR diesem Aufruf — hier bleibt der enge
    Yahoo-Weg.
    """
    import yfinance as yf  # lazy: Import/Testsuite bleibt netzfrei

    try:
        rate_limit.RATE_LIMITER_LEICHT.vor_yahoo_aufruf()
        ergebnis = yf.Search(
            query=query, max_results=limit, enable_fuzzy_query=True, timeout=10
        )
        quotes = ergebnis.quotes or []
    except rate_limit.YahooGedrosseltError:
        raise  # offener Circuit — Guard liefert bereits 429 mit deutscher Meldung
    except Exception as exc:  # noqa: BLE001 — Netz-/Formatpanne der Suche
        if rate_limit.ist_rate_limit_fehler(exc):
            dauer = rate_limit.RATE_LIMITER_LEICHT.verzeichne_rate_limit()
            raise ProviderError(
                "Yahoo drosselt die Anfragen (429 Too Many Requests). Weitere "
                f"Yahoo-Abrufe werden für {dauer:.0f} s ausgesetzt — auch die "
                "Symbolsuche ist danach wieder verfügbar.",
                429,
                "yahoo_gedrosselt",
                hint="Etwas seltener anfragen — der Cool-down verdoppelt sich "
                "bei jeder weiteren Drosselung (max. 900 s).",
            ) from exc
        raise ProviderError(
            f"Symbolsuche für „{query}“ fehlgeschlagen: {exc}",
            502,
            "suche_fehlgeschlagen",
        ) from exc
    return _suchergebnis_aufbereiten(list(quotes))


#: Queries, deren letzter Fetch nachweislich **alles** geliefert hat, was
#: Yahoo kennt (weniger Treffer als angefragt) — ein größerer ``limit``
#: verdient dann keinen Refetch. Prozesslokal (ein Sidecar, E009); wird
#: verworfen, sobald die Zeile aus dem Cache weg ist.
_ERSCHOEPFTE_SUCHE: set[str] = set()

_SUCHE_ZUSTAND_LOCK = threading.Lock()


def _symbolsuche(
    query: str, limit: int, *, marken: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """Sucht Symbole — erst Cache (24 h), sonst Yahoo; liefert (Treffer, aus_cache).

    Yahoo werden ``max_results=limit`` Treffer abverlangt. Der Cache bedient
    nur, was er auch **decken** kann (M1): ``len(results) >= limit`` — ein
    größerer ``limit`` nach kleinem First-Call refetcht also und
    **überschreibt** den Cache (profitiert auch dem internen Symbol-Add-Pfad
    mit ``limit=8``); das ``limit`` steht bewusst NICHT im Schlüssel
    (Fragmentierung). Ausnahme: Der letzte Fetch brachte weniger Treffer als
    angefragt — Yahoo ist für diese Query erschöpft, größere limits finden
    nichts Neues, der Cache bedient sie weiterhin (identische Requests
    bleiben beim genau EIN Search-Call des T19-DoD). Zusätzlicher Slice auch
    auf dem Cache-Pfad, falls ein größerer ``limit`` früher gecacht wurde.

    T-F12 (robuste Datenfindung): Ist der Rate-Limit-Circuit offen
    (:meth:`YahooRateLimiter.ist_gesperrt`) und der frische Cache deckt
    nicht, wird statt des 429 degradiert bedient — erst der **stale**
    Cache-Eintrag (TTL überschritten; ohne die M1-Deckungsregel, denn bei
    offenem Circuit ist ein Refetch ohnehin unmöglich), sonst lokale
    Registry-Treffer (:func:`_registry_treffer`). Nur wenn beides leer
    bleibt, fliegt weiterhin das ehrliche 429. ``marken`` (übergibt nur der
    Router-Handler) wird bei degradierter Bedienung um ``quelle``/
    ``hinweis`` ergänzt — additiv im Response-Vertrag; ohne das Argument
    (z. B. der Agent-Tools-Pfad) bleibt das Verhalten exakt wie zuvor.
    """
    schluessel = query.casefold()
    gecacht = search_cache_repo.get(schluessel)
    if gecacht is None:
        with _SUCHE_ZUSTAND_LOCK:
            _ERSCHOEPFTE_SUCHE.discard(schluessel)  # Zeile weg → Merkung verwerfen
    elif _suche_cache_frisch(gecacht.fetched_at):
        with _SUCHE_ZUSTAND_LOCK:
            erschoepft = schluessel in _ERSCHOEPFTE_SUCHE
        if len(gecacht.results) >= limit or erschoepft:
            ergebnisse = [dict(eintrag) for eintrag in gecacht.results]
            return ergebnisse[:limit], True
    if marken is not None and rate_limit.RATE_LIMITER_LEICHT.ist_gesperrt():
        return _suche_bei_offenem_circuit(schluessel, limit, gecacht, marken)
    ergebnisse = _yf_suche(query, limit)[:limit]
    with _SUCHE_ZUSTAND_LOCK:
        if len(ergebnisse) < limit:
            _ERSCHOEPFTE_SUCHE.add(schluessel)  # Yahoo hatte weniger als limit
        else:
            _ERSCHOEPFTE_SUCHE.discard(schluessel)
    search_cache_repo.save(schluessel, ergebnisse)
    return ergebnisse, False


def _suche_bei_offenem_circuit(
    schluessel: str,
    limit: int,
    gecacht: search_cache_repo.CachedSearch | None,
    marken: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Degradierte Bedienung bei offenem Circuit — staler Cache, sonst Registry.

    T-F12: Der Guard in :func:`_yf_suche` würde hier ohnehin 429 werfen
    („Yahoo drosselt“) — dabei sind lokale Daten verfügbar. Hierarchie:
    (1) der **stale** Cache-Eintrag (TTL überschritten, ehrlich als
    ``quelle="cache_veraltet"`` gekennzeichnet), (2) lokale Registry-Treffer
    (:func:`_registry_treffer`, ``quelle="registry"``), (3) erst dann das
    weiterhin ehrliche 429 (:meth:`YahooRateLimiter.gesperrt_fehler`).
    Kein einziger Netz-Call; der Cache wird nicht überschrieben (es fand ja
    kein Refresh statt).
    """
    if gecacht is not None and gecacht.results:
        ergebnisse = [dict(eintrag) for eintrag in gecacht.results][:limit]
        marken.update(
            {
                "quelle": "cache_veraltet",
                "hinweis": "Yahoo drosselt gerade (Rate-Limit) — Ergebnis aus "
                "dem Suchcache und älter als 24 Stunden.",
            }
        )
        return ergebnisse, True
    treffer = _registry_treffer(schluessel, limit)
    if treffer:
        marken.update(
            {
                "quelle": "registry",
                "hinweis": "Yahoo drosselt gerade (Rate-Limit) — Treffer aus "
                "der lokalen Symbol-Registry (ohne Yahoo-Abgleich).",
            }
        )
        return treffer, True
    raise rate_limit.RATE_LIMITER_LEICHT.gesperrt_fehler()


def _registry_treffer(schluessel: str, limit: int) -> list[dict[str, Any]]:
    """Lokale Registry-Treffer für eine Query (symbols_repo) — ohne Netz (T-F12).

    Gleiche Treffer-Form wie ``yf.Search`` (``{symbol, name, exchange,
    quoteType}``); ``exchange`` ist ``None`` (die Registry hält nur
    ``exchange_tz``, keine Börse), ``quoteType`` best-effort aus dem
    ``asset_type``. Exaktes Kürzel gewinnt; sonst Teilstring in Kürzel oder
    Name (case-insensitiv).
    """
    funde: list[tuple[bool, str, symbols_repo.Symbol]] = []
    for eintrag in symbols_repo.list_symbols():  # Repo-Vertrag: alphabetisch
        symbol_klein = eintrag.symbol.casefold()
        name_klein = (eintrag.name or "").casefold()
        if schluessel in symbol_klein or (name_klein and schluessel in name_klein):
            funde.append((symbol_klein == schluessel, symbol_klein, eintrag))
    funde.sort(key=lambda fund: (not fund[0], fund[1]))  # exaktes Kürzel zuerst
    return [
        {
            "symbol": eintrag.symbol,
            "name": eintrag.name,
            "exchange": None,
            "quoteType": _asset_als_quote_type(eintrag.asset_type),
        }
        for _, _, eintrag in funde[:limit]
    ]


#: quoteType (Yahoo) → asset_type der Registry (klein, Fachtoleranz).
_QUOTE_TYPE_ZU_ASSET: dict[str, str] = {
    "EQUITY": "equity",
    "ETF": "etf",
    "INDEX": "index",
    "CRYPTOCURRENCY": "crypto",
    "CURRENCY": "forex",
    "FUTURE": "future",
    "MUTUALFUND": "fund",
}


def _asset_type_aus_quote_type(quote_type: str | None) -> str | None:
    if not quote_type:
        return None
    return _QUOTE_TYPE_ZU_ASSET.get(str(quote_type).upper(), str(quote_type).lower())


#: Umkehrung von ``_QUOTE_TYPE_ZU_ASSET`` — Registry-``asset_type`` → Yahoo-
#: ``quoteType`` für Registry-Treffer der degradierten Suche (T-F12).
_ASSET_ZU_QUOTE_TYPE: dict[str, str] = {v: k for k, v in _QUOTE_TYPE_ZU_ASSET.items()}


def _asset_als_quote_type(asset_type: str | None) -> str | None:
    """Registry-``asset_type`` → Yahoo-``quoteType`` (Umkehrweg, best-effort)."""
    if not asset_type:
        return None
    return _ASSET_ZU_QUOTE_TYPE.get(asset_type, asset_type.upper())


def _symbol_ueber_suche_validieren(symbol: str) -> SymbolInfo:
    """Validiert das Kürzel über die Symbolsuche statt einer ``Ticker.history``-Probe.

    Metadaten (Name, quoteType) kommen aus dem Suchergebnis — der Download-
    Job-Manager reichert ``exchange_tz`` später best-effort über den Provider
    an (``_registry_metadata_pflegen``). Genauer Treffer = exaktes Symbol
    (case-insensitiv); keines → 404 (deutsch).
    """
    treffer, _ = _symbolsuche(symbol, limit=8)
    for eintrag in treffer:
        if str(eintrag["symbol"]).casefold() == symbol.casefold():
            return SymbolInfo(
                symbol=symbol,
                name=eintrag.get("name"),
                asset_type=_asset_type_aus_quote_type(eintrag.get("quoteType")),
                exchange_tz=None,
                currency=None,
            )
    raise ProviderError(
        f"Symbol „{symbol}“ ist bei Yahoo nicht bekannt oder liefert keine Daten.",
        404,
        "symbol_nicht_gefunden",
        hint="Yahoo-Kürzel prüfen, z. B. „AAPL“, „^GSPC“, „EURUSD=X“, „ES=F“.",
    )


# ---------------------------------------------------------------------------
# Symbol-Registry (§C.1)
# ---------------------------------------------------------------------------


@router.get("/data/symbols")
def list_symbols() -> dict[str, Any]:
    """Gespeicherte Symbole inkl. Abdeckung je Intervall (first/last ts, row_count)."""
    symbole = symbols_repo.list_symbols()
    positionen = _watchlist_positionen()
    return {"symbols": [_symbol_als_dict(symbol, positionen) for symbol in symbole]}


@router.post("/data/symbols")
@_ohne_provider_fehler
def add_symbol(payload: SymbolAddRequest) -> dict[str, Any]:
    """Nimmt das Symbol in Registry/Watchlist auf — Validierung über die Suche.

    Seit T19 ohne die teure ``Ticker.history``-Probe (blockierte bis ~30 s):
    Ein ``yf.Search``-Treffer mit exaktem Symbol genügt; Metadaten (Name,
    quoteType) stammen aus dem Suchergebnis. Die Antwort landet damit weit
    unter 1 s.
    """
    geprueft = validate_symbol(payload.symbol)
    info = _symbol_ueber_suche_validieren(geprueft)
    gespeichert = symbols_repo.upsert_symbol(
        geprueft,
        name=info.name,
        asset_type=info.asset_type,
        exchange_tz=info.exchange_tz,
        currency=info.currency,
    )
    _watchlist_aufnehmen(geprueft)
    return _symbol_als_dict(gespeichert)


@router.delete("/data/symbols/{symbol}")
@_ohne_provider_fehler
def delete_symbol(symbol: str) -> dict[str, Any]:
    """Löscht Symbol samt Watchlist-Eintrag, bar_meta-Zeilen und Parquet-Daten."""
    geprueft = validate_symbol(symbol)
    in_registry = symbols_repo.get_symbol(geprueft) is not None
    hat_meta = bool(parquet_store.list_bar_meta(geprueft))
    if not in_registry and not hat_meta:
        raise ApiError(
            404,
            "symbol_nicht_gefunden",
            f"Das Symbol „{geprueft}“ ist nicht gespeichert.",
        )
    geloeschte_dateien = parquet_store.delete_symbol_data(geprueft)
    symbols_repo.delete_symbol(geprueft)
    # Frontend verwirft alle Caches des Symbols (interval "*" = alle Intervalle).
    events.data_changed(geprueft, "*", events.REASON_GELOESCHT)
    return {"symbol": geprueft, "geloeschte_dateien": geloeschte_dateien}


# ---------------------------------------------------------------------------
# Symbolsuche + Provider-Status (T19, §C.4)
# ---------------------------------------------------------------------------


@router.get("/data/search")
@_ohne_provider_fehler
def search_symbols(q: str, limit: int = 10) -> dict[str, Any]:
    """Symbolsuche via ``yf.Search(enable_fuzzy_query=True)`` mit 24-h-Cache.

    Antwort: ``{query, count, results, aus_cache}`` mit Treffern als
    ``{symbol, name, exchange, quoteType}``. Identische Anfragen innerhalb
    der TTL bedienen sich aus ``symbol_search_cache`` (genau ein Yahoo-Call).

    Seit T-F12 additiv: Ist der Rate-Limit-Circuit offen und der frische
    Cache deckt nicht, wird statt 429 degradiert bedient (staler Cache oder
    lokale Registry-Treffer, kein Netz-Call) — die Antwort trägt dann
    zusätzlich ``quelle`` (``"cache_veraltet"`` | ``"registry"``) und einen
    deutschen ``hinweis``; ``aus_cache`` ist dann ``True`` („lokal bedient,
    kein Yahoo-Call“). Frontends ohne Kenntnis der neuen Felder können sie
    ignorieren; der Normalpfad (Yahoo/frischer Cache) bleibt unverändert.
    """
    query = q.strip()
    if not query:
        raise ProviderError(
            "Suchbegriff fehlt oder ist leer.", 422, "ungueltige_suche"
        )
    if limit < 1 or limit > SUCHE_MAX_LIMIT:
        raise ProviderError(
            f"„limit“ muss zwischen 1 und {SUCHE_MAX_LIMIT} liegen.",
            422,
            "ungueltiges_limit",
        )
    marken: dict[str, Any] = {}
    treffer, aus_cache = _symbolsuche(query, limit, marken=marken)
    antwort: dict[str, Any] = {
        "query": query,
        "count": len(treffer),
        "results": treffer,
        "aus_cache": aus_cache,
    }
    antwort.update(marken)  # additiv: nur bei degradierter Bedienung gefüllt
    return antwort


@router.get("/data/provider-status")
def provider_status() -> dict[str, Any]:
    """Pacing-/Circuit-Zustand je Yahoo-Circuit + Stooq-Verfügbarkeit (E035, §C.4).

    Shape (seit E087 mit getrennten Circuits; Schlüssel ``yahoo`` bleibt aus
    Kompatibilität erhalten und meint das Direct-Bein): ``{"yahoo": …,
    "yfinance": …, "leicht": …, "stooq": …}`` — je Circuit
    ``{"state": "ok"|"cooldown", "cooldown_until"?: ISO-UTC}``, Stooq
    ``{"state": "ok"|"limit", "seit"?: ISO-UTC}``; optionale Felder nur im
    jeweiligen Ausnahmezustand.
    """
    return {
        "yahoo": rate_limit.status_aggregiert(rate_limit.RATE_LIMITER_DIRECT),
        "yfinance": rate_limit.RATE_LIMITER_YF.status(),
        "leicht": rate_limit.RATE_LIMITER_LEICHT.status(),
        "stooq": stooq_provider.status(),
    }


# ---------------------------------------------------------------------------
# OHLC-Read (§C.1)
# ---------------------------------------------------------------------------


@router.get("/data/ohlc")
@_ohne_provider_fehler
def read_ohlc(
    symbol: str,
    interval: str,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    adjust: str = "raw",
) -> dict[str, Any]:
    """Balken ``{ts, open, high, low, close, volume}`` aus Parquet (µs UTC).

    ``start`` inklusive, ``end`` exklusiv (``YYYY-MM-DD``/ISO-8601); ``limit``
    kürzt auf die jüngsten Balken. Unbekanntes Symbol → 404 (deutsch).

    ``adjust`` (T20/E022): ``raw`` (Default) liefert die Rohkurse — die
    Response ist dann formatidentisch zum T4-Bestand; ``adjusted``
    multipliziert OHLC **on-read** mit der je Bar gespeicherten Spalte
    ``adj_factor`` (Volume unverändert) und setzt ``"adjust": "adjusted"``.
    Bei Stooq-Bestand wird nicht multipliziert und ein ``hinweis``-Feld
    ergänzt (Close dort bereits adjustiert, E023); dasselbe ``hinweis``-Feld
    erscheint bei Legacy-Blendung — v1-adjustierte Kurse als raw, einmalig
    mit explizitem ``start`` neu laden (M3, T20-Fix, unabhängig von ``adjust``).
    Ungültige Werte → 422.
    """
    validate_interval(interval)
    start_us = downloads.iso_zeitpunkt_zu_us(start) if start is not None else None
    end_us = downloads.iso_zeitpunkt_zu_us(end) if end is not None else None
    balken = data_service.read_ohlcv(
        symbol, interval, start=start_us, end=end_us, limit=limit, adjust=adjust
    )
    geprueft = validate_symbol(symbol)
    daten = [
        {
            "ts": int(zeile.ts),
            "open": float(zeile.open),
            "high": float(zeile.high),
            "low": float(zeile.low),
            "close": float(zeile.close),
            "volume": float(zeile.volume),
        }
        for zeile in balken.itertuples(index=False)
    ]
    antwort: dict[str, Any] = {
        "symbol": geprueft,
        "interval": interval,
        "count": len(daten),
        "bars": daten,
    }
    if adjust == "adjusted":
        antwort["adjust"] = "adjusted"
    hinweis = data_service.adjustierungs_hinweis(geprueft, interval, adjust)
    if hinweis is not None:
        antwort["hinweis"] = hinweis
    return antwort


# ---------------------------------------------------------------------------
# Corporate Actions (T20, §C.4): GET /api/data/actions
# ---------------------------------------------------------------------------


@router.get("/data/actions")
@_ohne_provider_fehler
def read_actions(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Dividenden und Splits aus ``corporate_actions``, aufsteigend nach Ex-Tag.

    ``start`` inklusive, ``end`` exklusiv (``YYYY-MM-DD``/ISO-8601, Filter auf
    Tagesebene). Antwort: ``{symbol, count, actions}`` mit je Event
    ``{ts, action: "dividend"|"split", value, currency?, source}`` — ``ts``
    ist der Ex-Tag als Mitternacht-UTC-µs (E006), ``value`` die Dividende je
    Stück bzw. die Split-Ratio (z. B. ``4.0``). Ein bekanntes Symbol ohne
    Events liefert ``count: 0``; ein unbekanntes → 404 (deutsch).
    """
    geprueft = validate_symbol(symbol)
    if not data_service.actions_symbol_bekannt(geprueft):
        raise SymbolNotFoundError(
            f"Für das Symbol „{geprueft}“ sind keine Corporate Actions "
            "gespeichert.",
            hint="Dividenden und Splits entstehen automatisch mit dem "
            "nächsten Marktdaten-Download des Symbols.",
        )
    start_us = downloads.iso_zeitpunkt_zu_us(start) if start is not None else None
    end_us = downloads.iso_zeitpunkt_zu_us(end) if end is not None else None
    aktionen = data_service.lese_actions(geprueft, start=start_us, end=end_us)
    daten = [
        {
            "ts": int(aktion.ts_us),
            "action": aktion.action,
            "value": aktion.value,
            "currency": aktion.currency,
            "source": aktion.source,
        }
        for aktion in aktionen
    ]
    return {"symbol": geprueft, "count": len(daten), "actions": daten}


# ---------------------------------------------------------------------------
# Indikatoren (§C.1, T6/E011 — Berechnung im Backend, Werte an genau einem Ort)
# ---------------------------------------------------------------------------


class IndicatorsBody(BaseModel):
    """Body von ``POST /api/data/indicators`` (§C.1)."""

    symbol: str = Field(min_length=1)
    interval: str
    indicators: list[IndicatorRequest] = Field(min_length=1)


def _werte_als_json(serie: pd.Series) -> list[float | None]:
    """Alignment-Vertrag: NaN (führende Lücke) → ``null``, sonst float.

    Die Serie ist damit **zu den OHLC-Balken aligned** — gleiche Länge wie
    ``ts``/``bars``; das Frontend startet die Linie ab dem ersten gültigen
    Wert (``null`` → Whitespace-Punkt im Chart).
    """
    return [None if pd.isna(wert) else float(wert) for wert in serie]


@router.post("/data/indicators")
@_ohne_provider_fehler
def compute_indicators(payload: IndicatorsBody) -> dict[str, Any]:
    """Indikator-Serien zu den gespeicherten OHLC-Balken (E011: Backend-only).

    Body: ``{symbol, interval, indicators: [{type, params}]}`` (Siehe
    :class:`IndicatorsBody`). Antwort:
    ``{symbol, interval, count, ts, indicators}`` mit ``ts`` = Balken-Zeit-
    stempel (int64 µs UTC, E006 — identisch zu ``GET /api/data/ohlc``) und je
    Indikator ``{type, params, id, series: [{key, values}]}``; ``values`` ist
    alignmentstreu zu ``ts`` mit ``null`` für nicht definierte Werte.

    Unbekanntes Symbol/Intervall ohne Daten → 404; unbekannter Typ bzw.
    ungültige Parameter → 422 (deutsch, via Service). Die Indikator-Specs
    werden **vor** dem Datenzugriff geprüft — sinnwidrige Eingaben liefern
    damit 422, auch wenn das Symbol keine Daten hat.
    """
    validate_interval(payload.interval)
    if len(payload.indicators) > indicators.MAX_INDICATORS_PER_REQUEST:
        raise ApiError(
            422,
            "zu_viele_indikatoren",
            "Maximal "
            f"{indicators.MAX_INDICATORS_PER_REQUEST} Indikatoren pro Anfrage.",
        )
    # Erste Runde: Typ/Parameter jeder Spec validieren + effektive Parameter
    # (Defaults aufgefüllt) für ID und Berechnung übernehmen → 422 vor 404.
    specs: list[tuple[str, dict[str, float], str]] = []
    for wunsch in payload.indicators:
        try:
            effektiv = indicators.effektive_parameter(wunsch.type, wunsch.params)
        except (ValueError, TypeError) as exc:
            # TypeError: nicht-zahlartige Params (z. B. ``null``) würden ohne
            # ihn als 500 durchrutschen — der Vertrag kennt hier nur 422.
            raise ApiError(422, "ungueltiger_indikator", str(exc)) from exc
        specs.append((wunsch.type, effektiv, indicators.indikator_id(wunsch.type, effektiv)))

    balken = data_service.read_ohlcv(payload.symbol, payload.interval)
    geprueft = validate_symbol(payload.symbol)
    close = balken["close"].reset_index(drop=True)
    # T25: OHLCV-Spalten für die neuen Typen (close-basierte Bestandstypen
    # ignorieren sie — deren Responses bleiben unverändert).
    high = balken["high"].reset_index(drop=True)
    low = balken["low"].reset_index(drop=True)
    volume = balken["volume"].reset_index(drop=True)
    ergebnisse: list[dict[str, Any]] = []
    for wunsch, (typ, effektiv, identitaet) in zip(payload.indicators, specs, strict=True):
        serien = indicators.berechne_indikator(
            close, typ, effektiv, high=high, low=low, volume=volume
        )
        ergebnisse.append(
            {
                "type": wunsch.type,
                "params": wunsch.params,
                "id": identitaet,
                "series": [
                    {"key": key, "values": _werte_als_json(serie)}
                    for key, serie in serien.items()
                ],
            }
        )
    ts = [int(wert) for wert in balken["ts"]]
    return {
        "symbol": geprueft,
        "interval": payload.interval,
        "count": len(ts),
        "ts": ts,
        "indicators": ergebnisse,
    }


# ---------------------------------------------------------------------------
# Download-Jobs (§C.1/C.2)
# ---------------------------------------------------------------------------


@router.post("/data/downloads")
@_ohne_provider_fehler
async def create_download(payload: DownloadCreateRequest) -> dict[str, str]:
    """Legt einen Download-Job an → ``{job_id}``; Fortschritt via ``GET /api/events``."""
    zeile = await downloads.submit_job(
        payload.symbols,
        payload.interval,
        period=payload.period,
        start=payload.start,
        end=payload.end,
        repair=payload.repair,
        force_full=payload.force_full,
    )
    return {"job_id": zeile.job_id}


@router.get("/data/gaps")
@_ohne_provider_fehler
def check_gaps(symbols: str, interval: str = "1d") -> dict[str, Any]:
    """Lücken-Report (Welle F, T-F11): fehlende Handelstage im Bestandsfenster.

    Umschließt :func:`quantlab.services.downloads.pruefe_lücken` (Heuristik
    T19: Mo–Fr ohne Bar zwischen ``first_ts`` und ``last_ts``; US-Feiertage
    melden sich als Lücke — dokumentierte Übermeldung, der Repair-Fetch ist
    idempotent). **Nur ``1d``** liefert messbare Lücken; andere gültige
    Intervalle laufen durch und liefern leere Listen (ehrlicher Pass-through).

    Query: ``symbols`` als kommaseparierte Liste (Duplikate entfallen,
    Reihenfolge bleibt), ``interval`` optional (Default ``1d``). Antwort:
    ``{"interval", "count", "entries"}`` mit je Eintrag
    ``{"symbol", "interval", "hat_bestand", "count", "luecken"}`` —
    ``luecken`` sind ``[{"von": "YYYY-MM-DD", "bis": "YYYY-MM-DD",
    "fehltage": n}, …]``; ``count`` zählt die Lücken **des Eintrags**
    (Top-Level ``count`` die Summe). Leerer/fehlender Bestand →
    ``hat_bestand: false`` und ``luecken: []`` (kein 404 — „kein Bestand“
    ist hier ein report-fähiger Zustand, anders als beim OHLC-Read). Ungültiges
    Symbol/Intervall → 422 (deutsch, Fehlervertrag C.1).
    """
    validate_interval(interval)
    kandidaten: list[str] = []
    for roh in symbols.split(","):
        if not roh.strip():
            continue  # leere Segmente („a,,b“) sind kein Symbolfehler
        geprueft = validate_symbol(roh)
        if geprueft not in kandidaten:
            kandidaten.append(geprueft)
    if not kandidaten:
        raise ProviderError(
            "Mindestens ein Symbol angeben.", 422, "leere_symbol_liste"
        )
    eintraege: list[dict[str, Any]] = []
    for symbol in kandidaten:
        luecken = downloads.pruefe_lücken(symbol, interval)
        meta = parquet_store.get_bar_meta(symbol, interval)
        hat_bestand = meta is not None and meta.first_ts is not None
        eintraege.append(
            {
                "symbol": symbol,
                "interval": interval,
                "hat_bestand": hat_bestand,
                "count": len(luecken),
                "luecken": luecken,
            }
        )
    return {
        "interval": interval,
        "count": sum(eintrag["count"] for eintrag in eintraege),
        "entries": eintraege,
    }


@router.get("/data/downloads")
def list_downloads(limit: int = 50) -> dict[str, Any]:
    """Download-Jobs, neueste zuerst (Job-Buchhaltung inkl. Status/Fehler)."""
    return {
        "downloads": [
            _download_als_dict(job) for job in downloads_repo.list_downloads(limit)
        ]
    }


@router.get("/data/downloads/{job_id}")
def get_download(job_id: str) -> dict[str, Any]:
    """Status eines einzelnen Jobs (inkl. Fehler-Zusammenfassung)."""
    job = downloads_repo.get_download(job_id)
    if job is None:
        raise ApiError(404, "job_nicht_gefunden", f"Kein Download-Job „{job_id}“ gefunden.")
    return _download_als_dict(job)


@router.post("/data/downloads/{job_id}/cancel")
@_ohne_provider_fehler
async def cancel_download(job_id: str) -> dict[str, str]:
    """Kooperativer Abbruch: Status vorwärts auf ``cancelled``, dann Job wecken."""
    job = await asyncio.to_thread(downloads.cancel_job_anfordern, job_id)
    if job is None:
        raise ApiError(404, "job_nicht_gefunden", f"Kein Download-Job „{job_id}“ gefunden.")
    downloads.abbruch_signalieren(job_id)
    return {"job_id": job_id, "status": job.status}
