"""Backtest-Endpunkte (C.1, T10; v2 T29/T31): synchron ausgeführte Backtest-Läufe.

``POST /api/backtest/runs`` führt den Backtest **synchron** aus (Backtests
sind schnell — kein Job-System) und persistiert das Ergebnis in SQLite
(``backtest_results``). ``GET`` liefert Liste/Detail (Metriken + Equity
resampled ≤ 2000 Punkte + Trades), ``DELETE`` löscht. Fehler 404/422 mit
deutscher Meldung (Vertrag C.1).

v2 (T29): ``symbols`` (Liste) aktiviert den Multi-Asset-Pfad — Auswertung auf
der **Schnittmenge** der Handelskalender (Start/Ende im Response unter
``period``/``symbols`` dokumentiert), Short-Positionen über negative
Zielgewichte (Σ|w| ≤ 1, keine Leverage). Single-Symbol-Gewichte übersteigen
1 (Σ|w| > 1) werden mit 422 abgewiesen; für Multi-Symbol-Katalog-Strategien
werden je-Symbol-Signale normiert (Ø bei vollen Signalen). Der Response ist
additiv angereichert: ``period``, ``symbols``, ``drawdown``, ``monthly_returns``,
``fills`` (Marker-Daten mit symbol/direction) und ``benchmark`` (Buy&Hold des
Primärsymbols) — die Anreicherung wird im Lauf gespeichert, POST = GET.

v2 (T31): ``strategy: "user:<id>@<version>"`` löst eine User-Strategie (T30)
über :func:`quantlab.services.sandbox.lade_user_strategie` auf und führt den
Code je Symbol in der Sandbox aus (E030). Die Gewichte fließen unverändert in
die Engine — **keine stille Normalisierung** (T29-Übergabe): User-Gewichte mit
Σ|w| > 1 werden von der Engine mit 422 ``gewichte_summe`` abgewiesen, damit
die API Leverage-Fehler nicht versteckt. Katalog-Strategien normieren
weiterhin. Zusätzlich: Grid- und Walk-Forward-Optimierung als Jobs
(E032) — ``POST /api/backtest/grid`` und ``POST /api/backtest/walkforward``
legen einen ``backtest_jobs``-Eintrag an und liefern sofort ``{job_id}``;
Ausführung serialisiert im Hintergrund (Muster Download-Schleuse) mit
SSE-Fortschritt (``backtest_job_progress``/``backtest_job_finished``) und
kooperativem Abbruch (``POST /api/backtest/jobs/{id}/cancel``).

B-Fix-4 (E051–E056): ``POST /api/backtest/runs`` ist ``async def`` und schaltet
eine **Daten-Vorbereitungsphase** vor den Lauf — fehlende Marktdaten werden
über :mod:`quantlab.services.daten_vorbereitung` automatisch geladen
(``auto_download: bool = True``; das Opt-out erhält den Bestands-404). Der Bericht
landet additiv im Response-Block ``data_prep`` (im Enrichment persistiert —
POST == GET bleibt) und der Fortschritt als SSE
``backtest_prepare_progress {token, done, total, symbol}`` mit dem optionalen
Request-Feld ``prepare_token`` (ohne Token: leerer Token; Vertrag:
``quantlab.services.daten_vorbereitung``). Die Optimierungs-Jobs führen
dieselbe Vorbereitung mit ``phase: "preparing_data"`` im Job-Fortschritt aus.

Die Endpunkte sind überwiegend normale ``def`` (nicht ``async``): SQLite- und
pandas-Aufrufe laufen synchron im FastAPI-Threadpool (Muster: ``quantlab.db``).
Die Job-Endpunkte sind ``async def`` (Muster Downloads): sie legen den Job an
und reihn ihn im Event-Loop ein; schwere Arbeit läuft im Threadpool — seit
B-Fix-4/E052 auch ``POST /api/backtest/runs`` (Prep ist async, der Engine-Kern
läuft via ``asyncio.to_thread`` weiter im Threadpool).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import (
    MAX_EQUITY_POINTS,
    SIZING_TARGET_WEIGHTS,
    WEIGHT_SUM_TOL,
    align_frames,
    benchmark_buy_hold,
    build_enrichment,
    resample_equity,
    run_backtest,
    run_portfolio_backtest,
)
from quantlab.backtest.strategies import STRATEGIES, StrategySpec, strategy_catalog
from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.services import backtest_jobs, data_service, daten_vorbereitung, sandbox
from quantlab.services.planspiel import regeln as psb_regeln
from quantlab.storage.repos import backtest_repo

router = APIRouter(tags=["backtest"])

#: Mindestanzahl Balken für einen sinnvollen Lauf (Signal + Ausführung).
_MIN_BARS = 2


class CostsIn(BaseModel):
    """Transaktionskosten in Basispunkten auf Trade-Volumen (§F).

    Welle H (E100, additiv): ``gebuehren_modell='psb'`` schaltet auf das
    Kostenmodell der Planspiel-Börse um (0,3 % min 15 € je Order —
    ``regeln.psb_kostenmodell()``); ``commission_bps``/``slippage_bps`` werden
    in diesem Modus ignoriert. Default „standard“ ⇒ Verhalten unverändert.
    """

    commission_bps: float = Field(default=10.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)
    gebuehren_modell: Literal["standard", "psb"] = "standard"


class BacktestRunRequest(BaseModel):
    """Request für ``POST /api/backtest/runs`` (C.1; v2 T29).

    ``start``/``end``: ISO-Datum/-Zeit (z. B. ``2023-01-01``) oder int64
    Mikrosekunden UTC; ``start`` inklusive, ``end`` exklusiv, ``None`` =
    unbegrenzt. ``fill``/``sizing``/``sizing_fraction`` sind additive Optionen
    der Engine (§F: „Fill t+1 close“ bzw. „fraction_of_equity“ konfigurierbar).

    v2: ``symbols`` (Liste) = Multi-Asset-Lauf; ``rebalance_every`` > 0 setzt
    alle n Balken auf die Zielgewichte zurück (Basket-Rebalancing) und schaltet
    auch für ein Symbol auf den Multi-Asset-Pfad um. ``sizing=fraction_of_equity``
    bleibt dem v1-Ein-Symbol-Pfad (ohne ``rebalance_every``) vorbehalten.
    T31: ``strategy`` darf auch eine User-Strategie-Referenz
    ``"user:<id>@<version>"`` sein (T30) — ``params`` wird dann unverändert an
    den Strategie-Code übergeben (keine Schema-Validierung).

    B-Fix-4 (E053/E056): ``auto_download`` (Default an) schaltet die Daten-
    Vorbereitungsphase ein — fehlende Marktdaten werden vor dem Lauf automatisch
    geladen; ``false`` erhält den Bestands-404 (deterministische Re-Runs).
    ``prepare_token`` ist optional und dient nur der Zuordnung der SSE-Events
    ``backtest_prepare_progress`` zum auslösenden Submit (ohne Token: leeres
    Token — Vertrag ``services.daten_vorbereitung``).
    """

    strategy: str
    symbol: str | None = None
    symbols: list[str] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    interval: str = "1d"
    start: str | int | None = None
    end: str | int | None = None
    costs: CostsIn = Field(default_factory=CostsIn)
    initial_capital: float = Field(default=10_000.0, gt=0)
    fill: Literal["open", "close"] = "open"
    sizing: Literal["target_weights", "fraction_of_equity"] = "target_weights"
    sizing_fraction: float = Field(default=1.0, gt=0, le=1)
    rebalance_every: int = Field(default=0, ge=0)
    project_id: int | None = None
    auto_download: bool = True
    prepare_token: str | None = None


class GridOptimierungRequest(BacktestRunRequest):
    """Request für ``POST /api/backtest/grid`` (T31, E032).

    ``grid`` mappt Parameter-Namen auf Wert-Listen; ausgeführt wird das
    kartesische Produkt (Obergrenze 200 Kombinationen → 422, sonst beim Submit
    synchron geprüft). Basis sind ``params``; Grid-Werte übersteuern je
    Kombination den Basis-Parameter.
    """

    grid: dict[str, list[Any]] = Field(default_factory=dict)


class WalkForwardRequest(GridOptimierungRequest):
    """Request für ``POST /api/backtest/walkforward`` (T31, E032).

    Rollierendes Walk-Forward: Train = ``train_bars`` (N), Test =
    ``test_bars`` (M), Schrittweite = M; Fenster beginnen am Datenanfang.
    Weniger als 2 vollständige Fenster → Job-Fehler mit deutscher Meldung
    (die Fensterzahl hängt von der geladenen Historie ab und wird im Job
    geprüft — der Submit prüft N/M ≥ 1 per Schema).
    """

    train_bars: int = Field(gt=0)
    test_bars: int = Field(gt=0)


# `_to_us` lebt als :func:`quantlab.services.backtest_jobs.zeitstempel_zu_us`
# im Service-Modul (T31-QA-FUND 1) — Walk-Forward-Jobs brauchen denselben
# Vertrag, und der Router delegiert, statt Duplikat-Logik zu pflegen.


def _resolve_params(spec_params: list[dict[str, Any]], given: dict[str, Any]) -> dict[str, Any]:
    """Prüft Parameter gegen das Strategie-Schema und ergänzt Defaults (422 sonst)."""
    known = {str(param["name"]): param for param in spec_params}
    resolved: dict[str, Any] = {}
    for key, value in given.items():
        param = known.get(key)
        if param is None:
            raise ApiError(
                422,
                "unbekannter_parameter",
                f"Unbekannter Parameter „{key}“ für diese Strategie.",
                hint=f"Erlaubt: {', '.join(sorted(known)) or '—'}.",
            )
        ptype = param["type"]
        if ptype == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise _type_error(key, "int")
        elif ptype == "float":
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise _type_error(key, "float")
            value = float(value)
        elif ptype == "bool":
            if not isinstance(value, bool):
                raise _type_error(key, "bool")
        elif not isinstance(value, str):
            raise _type_error(key, "str")
        minimum = param.get("minimum")
        maximum = param.get("maximum")
        if (minimum is not None and value < minimum) or (
            maximum is not None and value > maximum
        ):
            raise ApiError(
                422,
                "parameterbereich",
                f"Parameter „{key}“ liegt außerhalb des erlaubten Bereichs "
                f"[{minimum}, {maximum}].",
            )
        resolved[key] = value
    for param in spec_params:
        resolved.setdefault(str(param["name"]), param["default"])
    return resolved


def _type_error(key: str, expected: str) -> ApiError:
    return ApiError(
        422,
        "parametertyp",
        f"Parameter „{key}“ hat den falschen Typ (erwartet: {expected}).",
    )


def _run_detail(run: backtest_repo.BacktestRun) -> dict[str, Any]:
    """Persistierter Lauf → Antwort-Darstellung (POST und GET identisch).

    T29: ``period``/``symbols``/``drawdown``/``monthly_returns``/``fills``/
    ``benchmark`` kommen aus dem gespeicherten Anreicherungs-Block —
    Bestandszeilen ohne Block liefern die leeren Strukturen (additiver Vertrag).
    B-Fix-4: ``data_prep`` ist der Bericht der Daten-Vorbereitungsphase
    (``None`` bei ``auto_download=false`` bzw. Bestands-Läufen).
    """
    anreicherung = run.enrichment if isinstance(run.enrichment, dict) else {}
    return {
        "run_id": run.run_id,
        "project_id": run.project_id,
        "strategy": run.strategy,
        "symbol": run.symbol,
        "interval": run.interval,
        "params": run.params,
        "created_at": run.created_at,
        "metrics": run.metrics,
        "equity": run.equity,
        "trades": run.trades,
        "period": anreicherung.get("period", {}),
        "symbols": anreicherung.get("symbols", []),
        "drawdown": anreicherung.get("drawdown", []),
        "monthly_returns": anreicherung.get("monthly_returns", {}),
        "fills": anreicherung.get("fills", []),
        "benchmark": anreicherung.get("benchmark"),
        "data_prep": anreicherung.get("data_prep"),
    }


@router.get("/backtest/strategies")
def get_strategies() -> dict[str, Any]:
    """Katalog der Referenz-Strategien inkl. Parameter-Schema (T11/T14)."""
    return strategy_catalog()


def _effektive_symbole(request: BacktestRunRequest) -> list[str]:
    """v1 ``symbol`` oder v2 ``symbols`` → Symbol-Liste (422 bei Widersprüchen)."""
    if request.symbols is None:
        if request.symbol is None or not request.symbol.strip():
            raise ApiError(
                422,
                "symbole_fehlen",
                "Weder „symbol“ noch „symbols“ angegeben.",
                hint="Einzel-Lauf: symbol; Multi-Asset: symbols als Liste.",
            )
        return [request.symbol]
    if not request.symbols:
        raise ApiError(
            422,
            "symbole_leer",
            "Die Symbolliste „symbols“ ist leer.",
            hint="Mindestens ein Symbol angeben oder das Feld weglassen.",
        )
    if len(set(request.symbols)) != len(request.symbols):
        raise ApiError(
            422,
            "doppelte_symbole",
            "Die Symbolliste „symbols“ enthält Duplikate.",
            hint="Jedes Symbol darf nur einmal vorkommen.",
        )
    return list(request.symbols)


def _multi_gewichte(
    spec: StrategySpec,
    params: dict[str, Any],
    daten: dict[str, pd.DataFrame],
    common_ts: np.ndarray,
) -> pd.DataFrame:
    """Strategie-Gewichte für den Multi-Asset-Pfad (T29).

    Single-Symbol-Strategien werden je Symbol auf dessen **eigener** Historie
    ausgewertet (Look-ahead-frei, alle Signale ≤ t) und auf das gemeinsame
    Zeitgrid geschnitten. Übersteigt die Zeilensumme Σ|w| 1, werden die
    Gewichte auf Σ|w| = 1 normiert (volles Kapital, relative Gewichte
    erhalten, keine Leverage); Warm-up-NaN zählt als 0. Gilt ausschließlich
    für **Katalog**-Strategien — User-Strategien (T31) werden nicht normiert
    (Σ|w| > 1 → 422 durch die Engine).
    """
    symbole = list(daten.keys())
    if spec.multi_symbol:
        _, aligned, _ = align_frames(daten)
        weights = spec.func(aligned, **params)
    else:
        grid = pd.Index(common_ts)
        spalten: dict[str, pd.Series] = {}
        for sym in symbole:
            frame = daten[sym]
            eigene = spec.func(frame, **params).iloc[:, 0]
            eigene = eigene.set_axis(pd.Index(np.asarray(frame["ts"], dtype="int64")))
            spalten[sym] = eigene.reindex(grid)
        weights = pd.DataFrame(spalten)[symbole]
    abs_summe = weights.abs().sum(axis=1)
    skala = abs_summe.where(abs_summe > 1.0 + WEIGHT_SUM_TOL, 1.0)
    return weights.div(skala, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# User-Strategien (T31): Sandbox-Verdrahtung „user:<id>@<version>“
# ---------------------------------------------------------------------------


def _user_strategie_code(ref: str) -> str:
    """Löst ``user:<id>@<version>`` → gespeicherten Code auf (T30-Vertrag).

    Unbekannte Strategie/Version → 404, ungültiges Format → 422 (Resolver:
    :func:`quantlab.services.sandbox.lade_user_strategie`).
    """
    _, version = sandbox.lade_user_strategie(ref)
    return version.code


def _sandbox_gewichte(
    code: str, daten: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Führt User-Strategie-Code in der Sandbox aus → Gewichte-Frame (T31).

    Fehlausgänge (Verbotsliste, Timeout, Speicher, Laufzeit, Vertrag) → 422
    ``sandbox_<art>`` mit deutscher Meldung und Zeilen-Hinweis (Muster
    ``routers/strategies.py``). Die Ausgabe (eine Gewichtsspalte, ``NaN`` =
    Warm-up/flat) wird unverändert weitergereicht — Wertepolitik (Σ|w| ≤ 1,
    keine Negativgewichte im v1-Pfad) prüft die Engine mit 422.
    """
    ergebnis = sandbox.fuehre_strategie_aus(code, daten, params)
    if not ergebnis.ok:
        hinweis: str | None = None
        if ergebnis.traceback_zeilen:
            erste = ergebnis.traceback_zeilen[0]
            hinweis = f"Zeile {erste.get('zeile')}: {erste.get('text')}"
        elif ergebnis.verstoesse:
            erste = ergebnis.verstoesse[0]
            hinweis = f"Zeile {erste.get('zeile')}: {erste.get('meldung')}"
        raise ApiError(
            422,
            f"sandbox_{ergebnis.error_kind}",
            ergebnis.meldung or "Die Strategie-Ausführung ist fehlgeschlagen.",
            hint=hinweis,
        )
    werte = [np.nan if w is None else float(w) for w in ergebnis.gewichte]
    return pd.DataFrame({"weight": pd.Series(werte, index=daten.index)})


def _lade_daten(
    symbole: list[str],
    interval: str,
    start_us: int | None,
    end_us: int | None,
) -> dict[str, pd.DataFrame]:
    """Lädt die OHLCV-Frames aller Symbole (404/422 über den Provider-Vertrag)."""
    daten: dict[str, pd.DataFrame] = {}
    for sym in symbole:
        try:
            daten[sym] = data_service.read_ohlcv(sym, interval, start=start_us, end=end_us)
        except ProviderError as exc:
            raise exc.to_api_error() from exc
    return daten


def _user_multi_gewichte(
    code: str,
    params: dict[str, Any],
    daten: dict[str, pd.DataFrame],
    common_ts: np.ndarray,
) -> pd.DataFrame:
    """User-Strategie-Gewichte für den Multi-Asset-Pfad (T31) — **ohne** Normalisierung.

    Die Sandbox liefert je Symbol eine Gewichtsspalte (ausgewertet auf der
    eigenen Historie, auf das gemeinsame Grid geschnitten wie bei Katalog-
    Strategien). Bewusst **keine** Σ|w|-Normalisierung (T29-Übergabe): Die
    Engine weist Σ|w| > 1 + tol mit 422 ``gewichte_summe`` ab, damit
    Leverage-Fehler nicht still versteckt werden; negative Gewichte (Short)
    sind erlaubt.
    """
    grid = pd.Index(common_ts)
    spalten: dict[str, pd.Series] = {}
    for sym in daten:
        eigene = _sandbox_gewichte(code, daten[sym], params).iloc[:, 0]
        eigene = eigene.set_axis(pd.Index(np.asarray(daten[sym]["ts"], dtype="int64")))
        spalten[sym] = eigene.reindex(grid)
    return pd.DataFrame(spalten)[list(daten.keys())].fillna(0.0)


def _zeitfenster_us(request: BacktestRunRequest) -> tuple[int | None, int | None]:
    """``start``/``end`` → (µs inklusiv, µs exklusiv) mit Kreuzungs-Prüfung (422).

    Ein Vertrag an **einer** Stelle: der Sync-Endpunkt (B-Fix-4) normalisiert
    den Zeitraum schon vor der Daten-Vorbereitungsphase, der Kern nutzt
    dieselbe Funktion (T31-QA-FUND 1 — keine Duplikat-Logik).
    """
    start_us = backtest_jobs.zeitstempel_zu_us(request.start, "start")
    end_us = backtest_jobs.zeitstempel_zu_us(request.end, "end")
    if start_us is not None and end_us is not None and start_us > end_us:
        raise ApiError(
            422,
            "ungueltiger_zeitraum",
            "„start“ liegt nach „end“.",
            hint="start ist inklusive, end exklusiv (Mikrosekunden UTC).",
        )
    return start_us, end_us


def _validiere_sizing(request: BacktestRunRequest, symbole: list[str]) -> None:
    """Multi-/Sizing-Kombination prüfen (422) — auch vor der Vorbereitungsphase."""
    multi = len(symbole) > 1 or request.rebalance_every > 0
    if multi and request.sizing != SIZING_TARGET_WEIGHTS:
        raise ApiError(
            422,
            "sizing_nicht_unterstuetzt",
            "„sizing=fraction_of_equity“ ist nur für Ein-Symbol-Läufe ohne "
            "„rebalance_every“ verfügbar (Multi-Asset: target_weights).",
        )


def _fuehre_request_aus(
    request: BacktestRunRequest,
    persistieren: bool = True,
    daten_bericht: dict[str, Any] | None = None,
) -> tuple[backtest_repo.BacktestRun | None, dict[str, Any]]:
    """Kern eines Backtest-Laufs (POST /runs und Grid-Jobs, T31).

    Lädt Daten, wählt den Pfad (v1-Einzelsymbol / Multi-Asset / Katalog /
    User-Strategie), führt die Engine aus und liefert
    ``(persistierter Lauf | None, Antwort-Detail)``. Mit
    ``persistieren=False`` (Grid-Kombinationen) wird nichts geschrieben;
    das Detail trägt ``run_id=None``. Deterministisch — die beste Grid-
    Kombination wird zur Persistierung einfach erneut ausgeführt.

    ``daten_bericht`` (B-Fix-4): Bericht der Daten-Vorbereitungsphase, additiv
    ins Enrichment des Laufs gemergt (POST == GET bleibt erhalten);
    ``None`` (Grid-Kombinationen, ``auto_download=false``) schreibt nichts.
    """
    strategie_ref = request.strategy
    ist_user = strategie_ref.startswith("user:")
    user_code: str | None = None
    spec: StrategySpec | None = None
    if ist_user:
        user_code = _user_strategie_code(strategie_ref)
        params = dict(request.params)  # Übergabe an den Code — keine Schema-Prüfung
        strategie_name = strategie_ref
    else:
        spec = STRATEGIES.get(strategie_ref)
        if spec is None:
            raise ApiError(
                422,
                "unbekannte_strategie",
                f"Unbekannte Strategie „{strategie_ref}“.",
                hint=f"Verfügbar: {', '.join(sorted(STRATEGIES))} oder user:<id>@<version>.",
            )
        params = _resolve_params(spec.params, request.params)
        strategie_name = spec.name

    start_us, end_us = _zeitfenster_us(request)

    symbole = _effektive_symbole(request)
    _validiere_sizing(request, symbole)
    multi = len(symbole) > 1 or request.rebalance_every > 0

    daten = _lade_daten(symbole, request.interval, start_us, end_us)

    costs = (
        psb_regeln.psb_kostenmodell()
        if request.costs.gebuehren_modell == "psb"
        else CostModel(
            commission_bps=request.costs.commission_bps,
            slippage_bps=request.costs.slippage_bps,
        )
    )

    if multi:
        common_ts, aligned, symbols_info = align_frames(daten)
        if len(common_ts) < _MIN_BARS:
            geladen = ", ".join(
                f"{item['symbol']}: {item['bars_loaded']}" for item in symbols_info
            )
            raise ApiError(
                422,
                "zu_wenige_balken",
                f"Die gemeinsame Historie der Symbole umfasst nur {len(common_ts)} "
                f"Balken (mindestens {_MIN_BARS} erforderlich). Geladen: {geladen}.",
                hint=(
                    "Multi-Asset wird auf die Schnittmenge der Handelskalender "
                    "ausgerichtet — Zeiträume/Historie der Symbole erweitern."
                ),
            )
        if user_code is not None:
            weights = _user_multi_gewichte(user_code, params, daten, common_ts)
        else:
            weights = _multi_gewichte(spec, params, daten, common_ts)
        result = run_portfolio_backtest(
            aligned,
            weights,
            costs,
            initial_capital=request.initial_capital,
            fill=request.fill,
            rebalance_every=request.rebalance_every,
        )
        enrichment = build_enrichment(
            result.equity,
            result.fills,
            symbols_info,
            benchmark_symbol=symbole[0],
            benchmark=result.benchmark,
        )
        return _persistiere(
            persistieren,
            request,
            strategie_name,
            symbole,
            params,
            result.metrics,
            resample_equity(result.equity, MAX_EQUITY_POINTS),
            result.trades,
            enrichment,
            daten_bericht,
        )

    data = daten[symbole[0]]
    if len(data) < _MIN_BARS:
        raise ApiError(
            422,
            "zu_wenige_balken",
            f"Der gewählte Zeitraum enthält zu wenige Balken "
            f"({len(data)}; mindestens {_MIN_BARS} erforderlich).",
            hint="Zeitraum erweitern oder anderes Intervall wählen.",
        )

    if user_code is not None:
        weights = _sandbox_gewichte(user_code, data, params)
    else:
        weights = spec.func(data, **params)
    result = run_backtest(
        data,
        weights,
        costs,
        symbol=symbole[0],
        initial_capital=request.initial_capital,
        fill=request.fill,
        sizing=request.sizing,
        sizing_fraction=request.sizing_fraction,
    )
    enrichment = build_enrichment(
        result.equity,
        result.fills,
        [
            {
                "symbol": symbole[0],
                "bars_loaded": len(data),
                "bars_used": len(data),
            }
        ],
        benchmark_symbol=symbole[0],
        benchmark=benchmark_buy_hold(data, request.initial_capital),
    )
    return _persistiere(
        persistieren,
        request,
        strategie_name,
        symbole,
        params,
        result.metrics,
        resample_equity(result.equity, MAX_EQUITY_POINTS),
        result.trades,
        enrichment,
        daten_bericht,
    )


def _persistiere(
    persistieren: bool,
    request: BacktestRunRequest,
    strategie_name: str,
    symbole: list[str],
    params: dict[str, Any],
    metrics: dict[str, Any],
    equity: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    enrichment: dict[str, Any],
    daten_bericht: dict[str, Any] | None = None,
) -> tuple[backtest_repo.BacktestRun | None, dict[str, Any]]:
    """Lauf-Ergebnis → (optional persistierter Run, Antwort-Detail).

    ``daten_bericht`` (B-Fix-4) wird additiv in den Enrichment-Block gemergt
    (``data_prep``), damit POST == GET auch für Auto-Download-Läufe gilt.
    """
    if daten_bericht is not None:
        enrichment = {**enrichment, "data_prep": daten_bericht}
    if persistieren:
        run = backtest_repo.create_run(
            uuid.uuid4().hex,
            project_id=request.project_id,
            strategy=strategie_name,
            symbol=",".join(symbole),
            interval=request.interval,
            params=params,
            metrics=metrics,
            equity=equity,
            trades=trades,
            enrichment=enrichment,
        )
        return run, _run_detail(run)
    detail = {
        "run_id": None,
        "project_id": request.project_id,
        "strategy": strategie_name,
        "symbol": ",".join(symbole),
        "interval": request.interval,
        "params": params,
        "created_at": "",
        "metrics": metrics,
        "equity": equity,
        "trades": trades,
        "period": enrichment.get("period", {}),
        "symbols": enrichment.get("symbols", []),
        "drawdown": enrichment.get("drawdown", []),
        "monthly_returns": enrichment.get("monthly_returns", {}),
        "fills": enrichment.get("fills", []),
        "benchmark": enrichment.get("benchmark"),
        "data_prep": daten_bericht,
    }
    return None, detail


@router.post("/backtest/runs")
async def post_backtest_run(request: BacktestRunRequest) -> dict[str, Any]:
    """Führt einen Backtest synchron aus und persistiert das Ergebnis → ``{run_id}``.

    Zwei-Phasen (B-Fix-4, E051–E056): bei ``auto_download`` (Default an) läuft
    vor dem Lauf die Daten-Vorbereitungsphase — fehlende Marktdaten werden
    automatisch geladen (SSE ``backtest_prepare_progress`` mit
    ``prepare_token``); bleiben Symbole danach leer → 422 mit deutscher
    Sammelmeldung je Symbol. Der Bericht liegt additiv im Response-Block
    ``data_prep`` (persistiert, POST == GET). Mit ``auto_download: false``
    entfällt die Phase (Bestandsverhalten inkl. 404 für fehlende Symbole).
    Der Engine-Kern läuft weiterhin synchron im Threadpool
    (:func:`asyncio.to_thread`, E052).

    Ein Symbol (``symbol``) läuft über die v1-Engine (long/flat, inkl.
    fraction_of_equity-Sizing); mehrere Symbole (``symbols``) bzw.
    ``rebalance_every > 0`` über die Multi-Asset-Engine (Schnittmengen-
    Alignment, Short-fähig, target_weights). ``strategy="user:<id>@<version>"``
    führt die User-Strategie in der Sandbox aus (T30/T31).
    """
    # Vor der Vorbereitungsphase validieren, was ohne Datenlast entscheidbar
    # ist (422 vor Download); Strategie-/Parameter-Details prüft der Kern.
    symbole = _effektive_symbole(request)
    start_us, end_us = _zeitfenster_us(request)
    _validiere_sizing(request, symbole)

    bericht: dict[str, Any] | None = None
    if request.auto_download:
        try:
            bericht = await daten_vorbereitung.bereite_daten_vor(
                symbole,
                request.interval,
                start_us=start_us,
                end_us=end_us,
                token=request.prepare_token or "",
            )
        except ProviderError as exc:
            raise exc.to_api_error() from exc
        if not bericht["geladen"]:
            # Alles auf Bestand — kein Download, also kein data_prep-Block:
            # der persistierte Enrichment-Vertrag der Bestands-Läufe bleibt
            # exakt unverändert (additiv nur bei echtem Auto-Download).
            bericht = None

    _, detail = await asyncio.to_thread(
        _fuehre_request_aus, request, True, daten_bericht=bericht
    )
    return detail


@router.get("/backtest/runs")
def get_backtest_runs(
    limit: int = 50, project_id: int | None = None
) -> list[dict[str, Any]]:
    """Backtest-Historie (Kopfdaten + Metriken, ohne Equity/Trades)."""
    if limit < 1:
        raise ApiError(
            422, "ungueltiges_limit", "„limit“ muss eine positive ganze Zahl sein."
        )
    runs = backtest_repo.list_runs(limit=limit, project_id=project_id)
    return [
        {
            "run_id": run.run_id,
            "project_id": run.project_id,
            "strategy": run.strategy,
            "symbol": run.symbol,
            "interval": run.interval,
            "params": run.params,
            "created_at": run.created_at,
            "metrics": run.metrics,
        }
        for run in runs
    ]


@router.get("/backtest/runs/{run_id}")
def get_backtest_run(run_id: str) -> dict[str, Any]:
    """Detail eines Laufs: Metriken + Equity (≤ 2000 Punkte) + Trades."""
    run = backtest_repo.get_run(run_id)
    if run is None:
        raise ApiError(
            404,
            "backtest_nicht_gefunden",
            f"Backtest-Lauf „{run_id}“ wurde nicht gefunden.",
        )
    return _run_detail(run)


@router.delete("/backtest/runs/{run_id}")
def delete_backtest_run(run_id: str) -> dict[str, Any]:
    """Löscht einen Backtest-Lauf (404, wenn unbekannt)."""
    if not backtest_repo.delete_run(run_id):
        raise ApiError(
            404,
            "backtest_nicht_gefunden",
            f"Backtest-Lauf „{run_id}“ wurde nicht gefunden.",
        )
    return {"deleted": True, "run_id": run_id}


# ---------------------------------------------------------------------------
# Optimierungs-Jobs (T31, E032): Grid / Walk-Forward
# ---------------------------------------------------------------------------


def _validiere_job_anfrage(
    strategie_ref: str, grid: dict[str, list[Any]]
) -> None:
    """Submit-Prüfung: Strategie existiert, Grid innerhalb des Caps (422 vor Anlage)."""
    if strategie_ref.startswith("user:"):
        _user_strategie_code(strategie_ref)
    elif STRATEGIES.get(strategie_ref) is None:
        raise ApiError(
            422,
            "unbekannte_strategie",
            f"Unbekannte Strategie „{strategie_ref}“.",
            hint=f"Verfügbar: {', '.join(sorted(STRATEGIES))} oder user:<id>@<version>.",
        )
    backtest_jobs.grid_kombinationen(grid)


@router.post("/backtest/grid")
async def post_grid(payload: GridOptimierungRequest) -> dict[str, Any]:
    """Startet eine Grid-Optimierung als Job → ``{job_id}`` (E032).

    Kartesisches Produkt der Parameterbereiche (Cap 200 Kombinationen → 422
    beim Submit); Ausführung als ``backtest_jobs``-Eintrag mit SSE-Fortschritt
    (``backtest_job_progress``/``backtest_job_finished``) und kooperativem
    Abbruch. Ergebnis: beste Kombination (persistierter Lauf, ``run_id``) +
    sortiertes Ranking via ``GET /api/backtest/jobs/{job_id}``.
    """
    await asyncio.to_thread(_validiere_job_anfrage, payload.strategy, payload.grid)
    job = await backtest_jobs.submit_job("grid", payload.model_dump(mode="json"))
    return backtest_jobs.job_als_dict(job)


@router.post("/backtest/walkforward")
async def post_walkforward(payload: WalkForwardRequest) -> dict[str, Any]:
    """Startet ein rollierendes Walk-Forward als Job → ``{job_id}`` (E032).

    Train = N Balken, Test = M Balken, Schrittweite = M; Fenster beginnen am
    Datenanfang. Zielfunktion der Parameterwahl = Ø Out-of-Sample-Sharpe
    (Fallback Ø OOS-Total-Return bei ``None``) — nie In-Sample. Weniger als
    2 vollständige Fenster → Job-Fehler mit deutscher Meldung.
    """
    await asyncio.to_thread(_validiere_job_anfrage, payload.strategy, payload.grid)
    job = await backtest_jobs.submit_job("walkforward", payload.model_dump(mode="json"))
    return backtest_jobs.job_als_dict(job)


@router.get("/backtest/jobs")
def get_backtest_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Optimierungs-Jobs, neueste zuerst (Kopfdaten + Ergebnis)."""
    if limit < 1:
        raise ApiError(
            422, "ungueltiges_limit", "„limit“ muss eine positive ganze Zahl sein."
        )
    return [backtest_jobs.job_als_dict(job) for job in backtest_repo.list_jobs(limit=limit)]


@router.get("/backtest/jobs/{job_id}")
def get_backtest_job(job_id: str) -> dict[str, Any]:
    """Status/Ergebnis eines Optimierungs-Jobs (404, wenn unbekannt)."""
    job = backtest_repo.get_job(job_id)
    if job is None:
        raise ApiError(
            404,
            "backtest_job_nicht_gefunden",
            f"Backtest-Job „{job_id}“ wurde nicht gefunden.",
        )
    return backtest_jobs.job_als_dict(job)


@router.post("/backtest/jobs/{job_id}/cancel")
async def cancel_backtest_job(job_id: str) -> dict[str, Any]:
    """Bricht einen Optimierungs-Job kooperativ ab (Muster Downloads).

    Der Status wird nur vorwärts auf ``cancelled`` gesetzt (atomarer Guard im
    Repo); der laufende Job prüft das Abbruch-Event zwischen Kombinationen
    bzw. Fenstern und beendet sich ohne Weiteren Teillauf.
    """
    job = await asyncio.to_thread(backtest_jobs.cancel_job_anfordern, job_id)
    if job is None:
        raise ApiError(
            404,
            "backtest_job_nicht_gefunden",
            f"Backtest-Job „{job_id}“ wurde nicht gefunden.",
        )
    backtest_jobs.abbruch_signalieren(job_id)  # im Loop-Thread (asyncio.Event)
    return backtest_jobs.job_als_dict(job)
