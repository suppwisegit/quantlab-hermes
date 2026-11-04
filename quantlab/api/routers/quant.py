"""Quant-Endpunkte (C.1, T8): ``POST /api/quant/{metrics,covariance,pca,var,portfolio}``.

**Schichtenregel (Task §T8):** Der Daten-Zugriff (Parquet über
:mod:`quantlab.services.data_service`) passiert ausschließlich hier — die
Berechnungen in :mod:`quantlab.quant` sind reine Funktionen.

**Alignment-Vertrag:** Alle Symbole (inkl. optionaler Benchmark) werden auf
die **Index-Schnittmenge** ihrer Handelszeitpunkte geschnitten (inner join,
siehe :func:`quantlab.quant.returns.align_close`); der tatsächlich
verwendete Zeitraum (``start_ts``/``end_ts`` inkl. ISO-Ausgabe) und die
Beobachtungszahl ``n_obs`` sind in jedem Response dokumentiert.

**Fehlervertrag (C.1):** fehlende Symbole → 404 (``symbol_nicht_gefunden``,
deutsch, aus dem Service); zu wenig Rendite-Beobachtungen → 422
(``zu_wenig_daten``); < 2 Symbole bei Kovarianz/PCA → 422
(``zu_wenig_symbole``); ungültige Parameter → 422 (``ungueltige_parameter``/
``ungueltige_gewichte`` bzw. Pydantic-Validierung). Alle Endpunkte sind
normale ``def`` — pandas-/sklearn-Rechenlast läuft im Threadpool (Muster
``quantlab.db``/``settings.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError, validate_interval
from quantlab.quant import covariance as covariance_mod
from quantlab.quant import metrics as metrics_mod
from quantlab.quant import pca as pca_mod
from quantlab.quant import portfolio as portfolio_mod
from quantlab.quant import risk as risk_mod
from quantlab.quant.returns import (
    PERIODS_PER_YEAR,
    align_close,
    close_to_close_returns,
    equity_curve,
    require_observations,
)
from quantlab.services import data_service

router = APIRouter(tags=["quant"])

#: Fixe Punkte der Alignment-Dokumentation in jedem Quant-Response (DoD f).
ALIGNMENT_NOTE = (
    "Renditen auf der Index-Schnittmenge (inner join) aller Symbole inkl. "
    "Benchmark; Beobachtungen außerhalb der gemeinsamen Historie fließen "
    "nicht ein — Start/Ende zeigen die tatsächlich verwendete Abdeckung."
)

#: Maximale Punkte der Portfolio-Equity-Kurve im Response (gleichmäßige Stichprobe).
_MAX_EQUITY_POINTS = 2000

#: Konfidenzniveau der festen VaR-Spalte in der Kennzahlen-Tabelle (§F-Default).
_METRICS_VAR_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Request-Schemas
# ---------------------------------------------------------------------------


class QuantBaseRequest(BaseModel):
    """Gemeinsame Eingabe aller Quant-Endpunkte (C.1)."""

    symbols: list[str] = Field(min_length=1, description="Symbol-Kürzel, z. B. [\"AAPL\"].")
    interval: str = Field(default="1d", description="yfinance-Intervall, Default 1d.")
    start: int | None = Field(default=None, description="Start (int64 µs UTC, inklusive).")
    end: int | None = Field(default=None, description="Ende (int64 µs UTC, exklusiv).")
    periods_per_year: int = Field(
        default=PERIODS_PER_YEAR, ge=1, description="Annualisierung, Default 252 (§F)."
    )


class WeightsMixin(BaseModel):
    """Optionale Gewichte: Liste (Position = Reihenfolge von ``symbols``) oder Mapping."""

    weights: list[float] | dict[str, float] | None = Field(
        default=None,
        description=(
            "Portfolio-Gewichte; Liste in Symbol-Reihenfolge oder {symbol: anteil}. "
            "Ohne Angabe: Gleichgewichtung. Wird auf Summe 1 normalisiert."
        ),
    )


class MetricsRequest(QuantBaseRequest, WeightsMixin):
    """``POST /api/quant/metrics`` — Kennzahlen je Symbol (+ Portfolio bei Gewichten)."""

    benchmark: str | None = Field(default=None, description="Benchmark-Symbol für Beta/Alpha/IR.")
    risk_free: float = Field(
        default=0.0, description="Risikofreier Satz (annualisiert), Default 0."
    )


class CovarianceRequest(QuantBaseRequest):
    """``POST /api/quant/covariance`` — Kovarianz-/Korrelationsmatrix (annualisiert)."""

    method: Literal["sample", "ledoit_wolf"] = Field(
        default="ledoit_wolf", description="Schätzer, Default LedoitWolf (§F)."
    )
    annualize: bool = Field(default=True, description="Kovarianz × periods_per_year.")


class PcaRequest(QuantBaseRequest):
    """``POST /api/quant/pca`` — PCA auf standardisierten Renditen."""

    n_components: int = Field(
        default=3, ge=1, description="Anzahl Hauptkomponenten, Default 3."
    )


class VarRequest(QuantBaseRequest, WeightsMixin):
    """``POST /api/quant/var`` — Value-at-Risk (historisch/parametrisch)."""

    confidence: float = Field(
        default=risk_mod.DEFAULT_CONFIDENCE,
        gt=0.0,
        lt=1.0,
        description="Konfidenzniveau α, Default 0.95 (§F).",
    )
    method: Literal["historical", "parametric"] = Field(
        default=risk_mod.DEFAULT_VAR_METHOD, description="VaR-Methode, Default historisch (§F)."
    )
    horizon_days: int = Field(
        default=1, ge=1, description="Horizont in Tagen; Skalierung via √h (i.i.d.-Näherung)."
    )


class PortfolioRequest(QuantBaseRequest, WeightsMixin):
    """``POST /api/quant/portfolio`` — kombinierte Portfolio-Analyse."""

    benchmark: str | None = Field(default=None, description="Benchmark-Symbol für Beta/Alpha/IR.")
    risk_free: float = Field(
        default=0.0, description="Risikofreier Satz (annualisiert), Default 0."
    )


# ---------------------------------------------------------------------------
# Response-Schemas
# ---------------------------------------------------------------------------


class QuantMeta(BaseModel):
    """Alignment-/Zeitraum-Dokumentation in jedem Quant-Response (DoD f)."""

    interval: str
    n_obs: int
    start_ts: int
    end_ts: int
    start: str
    end: str
    alignment: str


class MetricRow(BaseModel):
    """Kennzahlen-Zeile je Symbol/Portfolio; ``None`` = fachlich nicht definierbar."""

    total_return: float | None
    cagr: float | None
    ann_return: float | None
    ann_vol: float | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    max_dd: float | None
    var_95: float | None = Field(default=None, description="VaR historisch 95 %, täglich.")
    beta: float | None = None
    alpha: float | None = None
    ir: float | None = None


class MatrixBlock(BaseModel):
    """Quadratische Matrix (Symbole in Reihenfolge ``symbols``)."""

    symbols: list[str]
    matrix: list[list[float]]


class MetricsResponse(QuantMeta):
    """Kennzahlen je Symbol + optionale Portfolio-Zeile (``results["Portfolio"]``)."""

    benchmark: str | None
    risk_free: float
    weights_used: dict[str, float] | None
    results: dict[str, MetricRow]


class CovarianceResponse(QuantMeta):
    """Kovarianz- und Korrelationsmatrix aus derselben Rendite-Matrix (§F)."""

    method: str
    shrinkage: float | None = Field(description="LedoitWolf-α ∈ [0,1]; None bei sample.")
    annualized: bool
    periods_per_year: int
    covariance: MatrixBlock
    correlation: MatrixBlock


class PcaResponse(QuantMeta):
    """Erklärte Varianz + Loadings (Index ``PC1..PCk`` × Symbole)."""

    n_components: int
    explained_variance: list[float]
    explained_variance_ratio: list[float]
    cumulative_variance_ratio: list[float]
    loadings: MatrixBlock


class VarRow(BaseModel):
    """VaR (positiver Verlustbetrag); CVaR nur historisch (Bonus, dokumentiert)."""

    var: float
    cvar: float | None


class VarResponse(QuantMeta):
    """VaR/CVaR je Symbol + Portfolio-Zeile (Gewichte normalisiert)."""

    method: str
    confidence: float
    horizon_days: int
    weights_used: dict[str, float]
    results: dict[str, VarRow]
    portfolio: VarRow


class EquityPoint(BaseModel):
    """Ein Punkt der Portfolio-Equity-Kurve (gleichmäßig gesampelt, ≤ 2000 Punkte)."""

    ts: int
    value: float


class PortfolioResponse(QuantMeta):
    """Kombinierte Analyse: Gewichte, Kennzahlen je Symbol + Portfolio, Equity-Kurve."""

    weights_used: dict[str, float]
    risk_free: float
    benchmark: str | None
    results: dict[str, MetricRow]
    equity: list[EquityPoint]


# ---------------------------------------------------------------------------
# Daten-Zugriff + Fehlerübersetzung (nur hier, nicht im quant-Paket)
# ---------------------------------------------------------------------------


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    """Duplikate entfernen (Reihenfolge bleibt; Doppel-Auswahl wäre störend still)."""
    unique: list[str] = []
    for symbol in symbols:
        if symbol not in unique:
            unique.append(symbol)
    return unique


def _load_close_frame(
    symbols: list[str], interval: str, start: int | None, end: int | None
) -> pd.DataFrame:
    """Lädt Close-Serien je Symbol und schneidet sie auf die Schnittmenge (DoD f).

    404 (``symbol_nicht_gefunden``) / 422 (``ungueltiges_symbol``/``intervall``)
    kommen aus dem Service bzw. der Intervall-Prüfung (Vertrag C.1).
    """
    try:
        validate_interval(interval)
        series: dict[str, pd.Series] = {}
        for symbol in symbols:
            bars = data_service.read_ohlcv(symbol, interval, start=start, end=end)
            if bars.empty:
                series[symbol] = pd.Series(dtype=float)
            else:
                indexed = bars.set_index("ts")["close"]
                series[symbol] = pd.Series(
                    indexed.to_numpy(dtype=float), index=indexed.index, dtype=float
                )
    except ProviderError as exc:
        raise exc.to_api_error() from exc
    return align_close(pd.DataFrame(series))


def _aligned_returns(
    symbols: list[str], interval: str, start: int | None, end: int | None
) -> pd.DataFrame:
    """Close-Matrix → simple Close-to-close-Renditen (Default, siehe ``returns.py``)."""
    frame = _load_close_frame(symbols, interval, start, end)
    returns = close_to_close_returns(frame, method="simple")
    return returns.astype(float)


def _meta(returns: pd.DataFrame, interval: str) -> QuantMeta:
    start_ts, end_ts = int(returns.index[0]), int(returns.index[-1])
    return QuantMeta(
        interval=interval,
        n_obs=int(len(returns)),
        start_ts=start_ts,
        end_ts=end_ts,
        start=_iso_utc(start_ts),
        end=_iso_utc(end_ts),
        alignment=ALIGNMENT_NOTE,
    )


def _iso_utc(ts_us: int) -> str:
    return datetime.fromtimestamp(ts_us / 1_000_000, tz=UTC).isoformat()


def _require_data(returns: pd.DataFrame) -> None:
    """Zu wenig gemeinsame Historie → 422 ``zu_wenig_daten`` (deutsch, DoD)."""
    try:
        require_observations(returns)
    except ValueError as exc:
        raise ApiError(422, "zu_wenig_daten", str(exc)) from exc


def _parse_weights(
    weights: list[float] | dict[str, float] | None, symbols: list[str]
) -> dict[str, float]:
    """Normalisiert die Gewichts-Eingabe auf Summe 1; Fehler → 422 (deutsch)."""
    if weights is None:
        return portfolio_mod.equal_weights(symbols)
    try:
        if isinstance(weights, dict):
            return portfolio_mod.normalize_weights(
                {str(key): float(value) for key, value in weights.items()}, symbols=symbols
            )
        if len(weights) != len(symbols):
            raise ApiError(
                422,
                "ungueltige_gewichte",
                f"Anzahl der Gewichte ({len(weights)}) passt nicht zur "
                f"Symbolanzahl ({len(symbols)}).",
            )
        return portfolio_mod.normalize_weights(
            dict(zip(symbols, (float(w) for w in weights), strict=True)), symbols=symbols
        )
    except ValueError as exc:
        raise ApiError(422, "ungueltige_gewichte", str(exc)) from exc


def _metric_row(
    series: pd.Series,
    *,
    benchmark: pd.Series | None,
    rf: float,
    periods_per_year: int,
) -> MetricRow:
    """Kennzahlen-Zeile (Basis + VaR 95 % + optional Beta/Alpha/IR, §F)."""
    row: dict[str, Any] = metrics_mod.performance_metrics(
        series, rf=rf, periods_per_year=periods_per_year
    )
    row["var_95"] = risk_mod.historical_var(series, confidence=_METRICS_VAR_CONFIDENCE)
    if benchmark is None:
        row.update(beta=None, alpha=None, ir=None)
    else:
        beta, alpha = risk_mod.beta_alpha(series, benchmark, periods_per_year)
        row.update(
            beta=beta,
            alpha=alpha,
            ir=risk_mod.information_ratio(series, benchmark, periods_per_year),
        )
    return MetricRow(**row)


def _sampled_equity(series: pd.Series, max_points: int = _MAX_EQUITY_POINTS) -> list[EquityPoint]:
    """Equity-Kurve gleichmäßig auf ≤ ``max_points`` verdichten (Endpunkt bleibt)."""
    values = series.to_numpy(dtype=float)
    if len(values) > max_points:
        indices = np.unique(np.linspace(0, len(values) - 1, max_points).round().astype(int))
    else:
        indices = np.arange(len(values))
    timestamps = series.index.to_numpy()[indices]
    return [
        EquityPoint(ts=int(timestamps[position]), value=float(values[position]))
        for position in indices
    ]


# ---------------------------------------------------------------------------
# Endpunkte
# ---------------------------------------------------------------------------


@router.post("/quant/metrics", response_model=MetricsResponse)
def post_metrics(request: MetricsRequest) -> MetricsResponse:
    """Kennzahlen je Symbol (+ Portfolio-Zeile bei Gewichten); optional Benchmark (C.1)."""
    symbols = _dedupe_symbols(request.symbols)
    benchmark = request.benchmark
    load_symbols = symbols + ([benchmark] if benchmark else [])
    returns = _aligned_returns(load_symbols, request.interval, request.start, request.end)
    _require_data(returns)
    meta = _meta(returns, request.interval)
    benchmark_series = returns[benchmark] if benchmark else None
    weights_used = (
        _parse_weights(request.weights, symbols) if request.weights is not None else None
    )

    results: dict[str, MetricRow] = {}
    for symbol in symbols:
        results[symbol] = _metric_row(
            returns[symbol],
            benchmark=benchmark_series,
            rf=request.risk_free,
            periods_per_year=request.periods_per_year,
        )
    if weights_used is not None:
        portfolio_series = portfolio_mod.portfolio_returns(returns[symbols], weights_used)
        results[portfolio_mod.PORTFOLIO_KEY] = _metric_row(
            portfolio_series,
            benchmark=benchmark_series,
            rf=request.risk_free,
            periods_per_year=request.periods_per_year,
        )
    return MetricsResponse(
        **meta.model_dump(),
        benchmark=benchmark,
        risk_free=request.risk_free,
        weights_used=weights_used,
        results=results,
    )


@router.post("/quant/covariance", response_model=CovarianceResponse)
def post_covariance(request: CovarianceRequest) -> CovarianceResponse:
    """Kovarianz (sample/LedoitWolf, annualisiert × ppy) + Korrelation (C.1, §F)."""
    symbols = _dedupe_symbols(request.symbols)
    if len(symbols) < 2:
        raise ApiError(
            422,
            "zu_wenig_symbole",
            "Für die Kovarianz-Schätzung werden mindestens 2 Symbole benötigt.",
            hint="Metriken für ein einzelnes Symbol: POST /api/quant/metrics.",
        )
    returns = _aligned_returns(symbols, request.interval, request.start, request.end)
    _require_data(returns)
    meta = _meta(returns, request.interval)
    matrix, shrinkage = covariance_mod.covariance_matrix(
        returns,
        request.method,
        annualize=request.annualize,
        periods_per_year=request.periods_per_year,
    )
    correlation = covariance_mod.correlation_matrix(returns)
    order = [str(column) for column in returns.columns]
    return CovarianceResponse(
        **meta.model_dump(),
        method=request.method,
        shrinkage=shrinkage,
        annualized=request.annualize,
        periods_per_year=request.periods_per_year,
        covariance=MatrixBlock(symbols=order, matrix=matrix.to_numpy().tolist()),
        correlation=MatrixBlock(symbols=order, matrix=correlation.to_numpy().tolist()),
    )


@router.post("/quant/pca", response_model=PcaResponse)
def post_pca(request: PcaRequest) -> PcaResponse:
    """PCA auf standardisierten Renditen: erklärte Varianz + Loadings (C.1, §F)."""
    symbols = _dedupe_symbols(request.symbols)
    if len(symbols) < 2:
        raise ApiError(
            422,
            "zu_wenig_symbole",
            "Für die Hauptkomponenten-Analyse werden mindestens 2 Symbole benötigt.",
            hint="Metriken für ein einzelnes Symbol: POST /api/quant/metrics.",
        )
    returns = _aligned_returns(symbols, request.interval, request.start, request.end)
    _require_data(returns)
    upper = min(returns.shape)
    if request.n_components > upper:
        raise ApiError(
            422,
            "ungueltige_parameter",
            f"n_components muss zwischen 1 und {upper} liegen (erhalten: "
            f"{request.n_components}).",
        )
    try:
        result = pca_mod.run_pca(returns, n_components=request.n_components)
    except ValueError as exc:
        # Konstante Rendite-Serien (Standardabweichung 0) sind Daten-Problem (422).
        raise ApiError(422, "zu_wenig_daten", str(exc)) from exc
    order = [str(column) for column in result.loadings.columns]
    return PcaResponse(
        **_meta(returns, request.interval).model_dump(),
        n_components=result.n_components,
        explained_variance=result.explained_variance,
        explained_variance_ratio=result.explained_variance_ratio,
        cumulative_variance_ratio=result.cumulative_variance_ratio,
        loadings=MatrixBlock(symbols=order, matrix=result.loadings.to_numpy().tolist()),
    )


@router.post("/quant/var", response_model=VarResponse)
def post_var(request: VarRequest) -> VarResponse:
    """VaR je Symbol + Portfolio (historisch/parametrisch), optional CVaR (C.1, Bonus)."""
    symbols = _dedupe_symbols(request.symbols)
    returns = _aligned_returns(symbols, request.interval, request.start, request.end)
    _require_data(returns)
    meta = _meta(returns, request.interval)
    weights_used = _parse_weights(request.weights, symbols)
    portfolio_series = portfolio_mod.portfolio_returns(returns[symbols], weights_used)

    def var_row(series: pd.Series) -> VarRow:
        if request.method == "parametric":
            value = risk_mod.parametric_var(series, request.confidence)
            return VarRow(var=risk_mod.scale_horizon(value, request.horizon_days), cvar=None)
        value = risk_mod.historical_var(series, request.confidence)
        shortfall = risk_mod.historical_cvar(series, request.confidence)
        return VarRow(
            var=risk_mod.scale_horizon(value, request.horizon_days),
            cvar=risk_mod.scale_horizon(shortfall, request.horizon_days),
        )

    results: dict[str, VarRow] = {symbol: var_row(returns[symbol]) for symbol in symbols}
    return VarResponse(
        **meta.model_dump(),
        method=request.method,
        confidence=request.confidence,
        horizon_days=request.horizon_days,
        weights_used=weights_used,
        results=results,
        portfolio=var_row(portfolio_series),
    )


@router.post("/quant/portfolio", response_model=PortfolioResponse)
def post_portfolio(request: PortfolioRequest) -> PortfolioResponse:
    """Kombinierte Portfolio-Analyse: Gewichte, Kennzahlen je Symbol + Portfolio (C.1)."""
    symbols = _dedupe_symbols(request.symbols)
    benchmark = request.benchmark
    load_symbols = symbols + ([benchmark] if benchmark else [])
    returns = _aligned_returns(load_symbols, request.interval, request.start, request.end)
    _require_data(returns)
    meta = _meta(returns, request.interval)
    benchmark_series = returns[benchmark] if benchmark else None
    weights_used = _parse_weights(request.weights, symbols)

    analysis = portfolio_mod.analyze(
        returns[symbols],
        weights_used,
        benchmark=benchmark_series,
        rf=request.risk_free,
        periods_per_year=request.periods_per_year,
    )
    results: dict[str, MetricRow] = {}
    for name, row in analysis.items():
        enriched = dict(row)
        series = (
            portfolio_mod.portfolio_returns(returns[symbols], weights_used)
            if name == portfolio_mod.PORTFOLIO_KEY
            else returns[name]
        )
        enriched["var_95"] = risk_mod.historical_var(
            series, confidence=_METRICS_VAR_CONFIDENCE
        )
        results[name] = MetricRow(**enriched)

    portfolio_series = portfolio_mod.portfolio_returns(returns[symbols], weights_used)
    equity = _sampled_equity(equity_curve(portfolio_series))
    return PortfolioResponse(
        **meta.model_dump(),
        weights_used=weights_used,
        risk_free=request.risk_free,
        benchmark=benchmark,
        results=results,
        equity=equity,
    )
