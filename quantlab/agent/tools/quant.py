"""Quant-Tools (T14, PLAN §E 5–8): Kennzahlen, Kovarianz, PCA, VaR.

Die Berechnungen laufen direkt über die I/O-freien Module aus
:mod:`quantlab.quant` (Schichtenregel T8 — der Datenzugriff auf
:mod:`quantlab.services.data_service` passiert hier im Tool, **nicht** über
die REST-Router); alle pandas-/sklearn-Anteile laufen im Threadpool.
Konventionen §F/E007: 252 Perioden, LedoitWolf-Default, VaR historisch.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, daten_fehler, pruefe_args, schema_von, zahl
from quantlab.providers.base import ProviderError, validate_interval
from quantlab.quant import covariance as covariance_mod
from quantlab.quant import metrics as metrics_mod
from quantlab.quant import pca as pca_mod
from quantlab.quant import portfolio as portfolio_mod
from quantlab.quant import risk as risk_mod
from quantlab.quant.returns import align_close, close_to_close_returns
from quantlab.services import data_service

# ---------------------------------------------------------------------------
# Gemeinsamer Datenzugriff (Rendite-Matrix auf der Index-Schnittmenge)
# ---------------------------------------------------------------------------


def _renditen_matrix(symbole: list[str], interval: str) -> pd.DataFrame:
    """Lädt Close-Serien und liefert (simple Renditen, tatsächlich genutzte Spalten).

    Alignment wie im Quant-Router: inner join der Handelszeitpunkte
    (:func:`quantlab.quant.returns.align_close`). Fachliche Fehler →
    :class:`ToolFehler` (deutsch, aus dem Daten-Service).
    """
    serien: dict[str, pd.Series] = {}
    try:
        validate_interval(interval)
        for symbol in symbole:
            balken = data_service.read_ohlcv(symbol, interval)
            if balken.empty:
                serien[symbol] = pd.Series(dtype=float)
            else:
                indiziert = balken.set_index("ts")["close"]
                serien[symbol] = pd.Series(
                    indiziert.to_numpy(dtype=float), index=indiziert.index, dtype=float
                )
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    frame = align_close(pd.DataFrame(serien))
    return close_to_close_returns(frame, method="simple").astype(float)


def _normiere_gewichte(
    gewichte: list[float] | dict[str, float] | None, symbole: list[str]
) -> dict[str, float]:
    """Gewichts-Eingabe (Liste in Symbol-Reihenfolge oder Mapping) → normalisiert."""
    if gewichte is None:
        return portfolio_mod.equal_weights(symbole)
    try:
        if isinstance(gewichte, dict):
            mapping = {str(k): float(v) for k, v in gewichte.items()}
        else:
            if len(gewichte) != len(symbole):
                raise ValueError(
                    f"Anzahl der Gewichte ({len(gewichte)}) passt nicht zur "
                    f"Symbolanzahl ({len(symbole)})."
                )
            mapping = dict(zip(symbole, (float(w) for w in gewichte), strict=True))
        return portfolio_mod.normalize_weights(mapping, symbols=symbole)
    except ValueError as exc:
        raise ToolFehler(f"Ungültige Gewichte: {exc}") from exc


def _zeitraum(returns: pd.DataFrame) -> dict[str, str | int]:
    """Dokumentation des tatsächlich verwendeten Zeitraums (Muster Quant-Router)."""
    return {
        "n_obs": int(len(returns)),
        "von": datetime.fromtimestamp(int(returns.index[0]) / 1_000_000, tz=UTC).isoformat(),
        "bis": datetime.fromtimestamp(int(returns.index[-1]) / 1_000_000, tz=UTC).isoformat(),
    }


def _matrix_block(matrix: pd.DataFrame) -> dict[str, Any]:
    return {
        "symbols": [str(s) for s in matrix.columns],
        "matrix": [[zahl(w) for w in zeile] for zeile in matrix.to_numpy().tolist()],
    }


# ---------------------------------------------------------------------------
# 5) compute_metrics
# ---------------------------------------------------------------------------


class ComputeMetricsArgs(BaseModel):
    """Argumente von ``compute_metrics``."""

    symbols: list[str] = Field(min_length=1, description='Symbol-Kürzel, z. B. ["AAPL", "MSFT"].')
    weights: list[float] | dict[str, float] | None = Field(
        default=None,
        description=(
            "Portfolio-Gewichte: Liste in Symbol-Reihenfolge (z. B. [0.6, 0.4]) oder "
            'Mapping (z. B. {"AAPL": 0.6}); ohne Angabe entfällt die Portfolio-Zeile.'
        ),
    )
    interval: str = Field(default="1d", description='yfinance-Intervall, Default "1d".')
    benchmark: str | None = Field(
        default=None, description='Benchmark-Symbol für Beta/Alpha/IR, z. B. "^GSPC".'
    )
    risk_free: float = Field(
        default=0.0, description="Risikofreier Satz (annualisiert, Dezimalzahl), Default 0."
    )


async def compute_metrics(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Kennzahlen (Total Return, CAGR, Vol, Sharpe, Sortino, Calmar, MaxDD, VaR).

    Je Symbol; bei Gewichten zusätzlich eine Portfolio-Zeile, mit Benchmark
    Beta/Alpha/IR.
    """

    def _berechnen() -> dict[str, Any]:
        args = pruefe_args(ComputeMetricsArgs, roh_args, "compute_metrics")
        symbole = list(dict.fromkeys(args.symbols))
        lade = symbole + ([args.benchmark] if args.benchmark else [])
        returns = _renditen_matrix(lade, args.interval)
        if returns.empty:
            raise ToolFehler(
                "Zu wenig gemeinsame Historie für die Kennzahlen-Berechnung "
                "(mindestens 2 Beobachtungen nötig)."
            )
        benchmark_serie = returns[args.benchmark] if args.benchmark else None
        ergebnisse: dict[str, Any] = {}
        for symbol in symbole:
            zeile: dict[str, Any] = metrics_mod.performance_metrics(
                returns[symbol], rf=args.risk_free
            )
            zeile["var_95"] = risk_mod.historical_var(returns[symbol], 0.95)
            if benchmark_serie is not None:
                beta, alpha = risk_mod.beta_alpha(returns[symbol], benchmark_serie)
                zeile.update(
                    beta=beta,
                    alpha=alpha,
                    ir=risk_mod.information_ratio(returns[symbol], benchmark_serie),
                )
            else:
                zeile.update(beta=None, alpha=None, ir=None)
            ergebnisse[symbol] = {k: zahl(v) for k, v in zeile.items()}
        antwort: dict[str, Any] = {
            "interval": args.interval,
            **_zeitraum(returns),
            "benchmark": args.benchmark,
            "risk_free": args.risk_free,
            "results": ergebnisse,
        }
        if args.weights is not None:
            gewichte = _normiere_gewichte(args.weights, symbole)
            portfolio = portfolio_mod.portfolio_returns(returns[symbole], gewichte)
            zeile = metrics_mod.performance_metrics(portfolio, rf=args.risk_free)
            zeile["var_95"] = risk_mod.historical_var(portfolio, 0.95)
            if benchmark_serie is not None:
                beta, alpha = risk_mod.beta_alpha(portfolio, benchmark_serie)
                zeile.update(
                    beta=beta,
                    alpha=alpha,
                    ir=risk_mod.information_ratio(portfolio, benchmark_serie),
                )
            else:
                zeile.update(beta=None, alpha=None, ir=None)
            antwort["weights_used"] = gewichte
            antwort["results"]["Portfolio"] = {k: zahl(v) for k, v in zeile.items()}
        return antwort

    return await asyncio.to_thread(_berechnen)


# ---------------------------------------------------------------------------
# 6) compute_covariance
# ---------------------------------------------------------------------------


class ComputeCovarianceArgs(BaseModel):
    """Argumente von ``compute_covariance``."""

    symbols: list[str] = Field(
        min_length=2, description="Mindestens 2 Symbol-Kürzel, z. B. [\"AAPL\", \"MSFT\"]."
    )
    interval: str = Field(default="1d", description='yfinance-Intervall, Default "1d".')
    method: Literal["ledoit_wolf", "sample"] = Field(
        default="ledoit_wolf",
        description='Schätzer: "ledoit_wolf" (Default, shrinkage) oder "sample".',
    )
    annualize: bool = Field(default=True, description="Kovarianz × 252 annualisieren.")


async def compute_covariance(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Kovarianz- und Korrelationsmatrix (annualisiert) auf gemeinsamer Historie."""

    def _berechnen() -> dict[str, Any]:
        args = pruefe_args(ComputeCovarianceArgs, roh_args, "compute_covariance")
        symbole = list(dict.fromkeys(args.symbols))
        returns = _renditen_matrix(symbole, args.interval)
        if returns.empty:
            raise ToolFehler(
                "Zu wenig gemeinsame Historie für die Kovarianz-Schätzung "
                "(mindestens 2 Beobachtungen nötig)."
            )
        matrix, shrinkage = covariance_mod.covariance_matrix(
            returns, args.method, annualize=args.annualize
        )
        korrelation = covariance_mod.correlation_matrix(returns)
        return {
            "interval": args.interval,
            **_zeitraum(returns),
            "method": args.method,
            "shrinkage": zahl(shrinkage),
            "annualized": args.annualize,
            "covariance": _matrix_block(matrix),
            "correlation": _matrix_block(korrelation),
        }

    return await asyncio.to_thread(_berechnen)


# ---------------------------------------------------------------------------
# 7) run_pca
# ---------------------------------------------------------------------------


class RunPcaArgs(BaseModel):
    """Argumente von ``run_pca``."""

    symbols: list[str] = Field(
        min_length=2,
        description='Mindestens 2 Symbole, z. B. ["AAPL", "MSFT", "GOOGL"].' 
    )
    interval: str = Field(default="1d", description='yfinance-Intervall, Default "1d".')
    n_components: int = Field(default=3, ge=1, description="Anzahl Hauptkomponenten, Default 3.")


async def run_pca(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Hauptkomponenten-Analyse auf standardisierten Renditen (erklärte Varianz + Loadings)."""

    def _berechnen() -> dict[str, Any]:
        args = pruefe_args(RunPcaArgs, roh_args, "run_pca")
        symbole = list(dict.fromkeys(args.symbols))
        returns = _renditen_matrix(symbole, args.interval)
        if returns.empty:
            raise ToolFehler(
                "Zu wenig gemeinsame Historie für die PCA (mindestens "
                "2 Beobachtungen nötig)."
            )
        try:
            ergebnis = pca_mod.run_pca(returns, n_components=args.n_components)
        except ValueError as exc:
            raise ToolFehler(str(exc)) from exc
        return {
            "interval": args.interval,
            **_zeitraum(returns),
            "n_components": ergebnis.n_components,
            "explained_variance_ratio": [zahl(v) for v in ergebnis.explained_variance_ratio],
            "cumulative_variance_ratio": [
                zahl(v) for v in ergebnis.cumulative_variance_ratio
            ],
            "loadings": _matrix_block(ergebnis.loadings),
        }

    return await asyncio.to_thread(_berechnen)


# ---------------------------------------------------------------------------
# 8) estimate_var
# ---------------------------------------------------------------------------


class EstimateVarArgs(BaseModel):
    """Argumente von ``estimate_var``."""

    symbols: list[str] = Field(min_length=1, description='Symbol-Kürzel, z. B. ["AAPL"].')
    interval: str = Field(default="1d", description='yfinance-Intervall, Default "1d".')
    weights: list[float] | dict[str, float] | None = Field(
        default=None,
        description="Portfolio-Gewichte (Liste oder Mapping); ohne Angabe: gleichverteilt.",
    )
    confidence: float = Field(
        default=0.95, gt=0.0, lt=1.0, description="Konfidenzniveau α, Default 0.95."
    )
    method: Literal["historical", "parametric"] = Field(
        default="historical", description='"historical" (Default) oder "parametric".'
    )
    horizon_days: int = Field(default=1, ge=1, description="Horizont in Tagen, Default 1.")


async def estimate_var(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Value-at-Risk (und CVaR bei historisch) je Symbol plus Portfolio-Zeile."""

    def _berechnen() -> dict[str, Any]:
        args = pruefe_args(EstimateVarArgs, roh_args, "estimate_var")
        symbole = list(dict.fromkeys(args.symbols))
        returns = _renditen_matrix(symbole, args.interval)
        if returns.empty:
            raise ToolFehler(
                "Zu wenig gemeinsame Historie für die VaR-Berechnung "
                "(mindestens 2 Beobachtungen nötig)."
            )
        gewichte = _normiere_gewichte(args.weights, symbole)
        portfolio = portfolio_mod.portfolio_returns(returns[symbole], gewichte)

        def var_zeile(serie: pd.Series) -> dict[str, Any]:
            if args.method == "parametric":
                wert = risk_mod.scale_horizon(
                    risk_mod.parametric_var(serie, args.confidence), args.horizon_days
                )
                return {"var": zahl(wert), "cvar": None}
            wert = risk_mod.scale_horizon(
                risk_mod.historical_var(serie, args.confidence), args.horizon_days
            )
            cvar = risk_mod.scale_horizon(
                risk_mod.historical_cvar(serie, args.confidence), args.horizon_days
            )
            return {"var": zahl(wert), "cvar": zahl(cvar)}

        return {
            "interval": args.interval,
            **_zeitraum(returns),
            "method": args.method,
            "confidence": args.confidence,
            "horizon_days": args.horizon_days,
            "weights_used": gewichte,
            "results": {symbol: var_zeile(returns[symbol]) for symbol in symbole},
            "portfolio": var_zeile(portfolio),
        }

    return await asyncio.to_thread(_berechnen)


# ---------------------------------------------------------------------------
# Registrierung (§E-Reihenfolge 5–8)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="compute_metrics",
            description=(
                "Berechnet Kennzahlen (Total Return, CAGR, ann. Vol, Sharpe, Sortino, "
                "Calmar, MaxDD, VaR 95 %; mit Benchmark: Beta/Alpha/IR) je Symbol und — "
                "bei Gewichten — für das Gesamtportfolio. "
                'Beispiel: {"symbols": ["AAPL", "MSFT"], "weights": [0.6, 0.4], '
                '"benchmark": "^GSPC"}.'
            ),
            parameters=schema_von(ComputeMetricsArgs),
            handler=compute_metrics,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="compute_covariance",
            description=(
                "Kovarianz- und Korrelationsmatrix auf gemeinsamer Historie "
                "(Default: LedoitWolf, annualisiert × 252). "
                'Beispiel: {"symbols": ["AAPL", "MSFT", "GOOGL"]}.'
            ),
            parameters=schema_von(ComputeCovarianceArgs),
            handler=compute_covariance,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="run_pca",
            description=(
                "Hauptkomponenten-Analyse (PCA) auf standardisierten Renditen: "
                "erklärte Varianz je Komponente + Loadings je Symbol. "
                'Beispiel: {"symbols": ["AAPL", "MSFT", "GOOGL"], "n_components": 2}.'
            ),
            parameters=schema_von(RunPcaArgs),
            handler=run_pca,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="estimate_var",
            description=(
                "Value-at-Risk je Symbol und Portfolio (historisch inkl. CVaR oder "
                "parametrisch), mit Horizont-Skalierung via √h. "
                'Beispiel: {"symbols": ["AAPL"], "confidence": 0.95, "horizon_days": 10}.'
            ),
            parameters=schema_von(EstimateVarArgs),
            handler=estimate_var,
        )
    )


_registrieren()
