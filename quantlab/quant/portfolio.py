"""Portfolio-Berechnungen: Gewichte, Portfolio-Renditen, kombinierte Analyse (T8).

**Gewichte:** Optional als Mapping Symbol → Anteil. Sie werden auf Summe 1
normalisiert (T9 zeigt einen Normalisierungshinweis); fehlende Symbole gelten
als 0 gewichtet. Negative Summe-0- oder unbekannte Symbole → :class:`ValueError`
(deutsch, Router → 422).

**Portfolio-Renditen:** lineare Gewichtssumme ``Σ w_i · r_i`` auf der
aligned Rendite-Matrix — exakt für simple Renditen (Entscheidungsdoku in
:mod:`quantlab.quant.returns`).

**Kombinierte Analyse** (:func:`analyze`): Kennzahlen je Symbol plus
Portfolio-Zeile — identische Konventionen/Funktionen wie die Einzel-Symbole
(:mod:`quantlab.quant.metrics`), optional mit Beta/Alpha/IR gegen eine
Benchmark-Serie (§F).

Reine Funktionen; Daten-Zugriff nur in der Routerschicht.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from quantlab.quant import metrics as metrics_mod
from quantlab.quant import risk as risk_mod
from quantlab.quant.returns import (
    PERIODS_PER_YEAR,
    equity_curve,
    require_observations,
)

#: Schlüssel der Portfolio-Zeile im Ergebnis-Dictionary von :func:`analyze`.
PORTFOLIO_KEY: str = "Portfolio"


def normalize_weights(
    weights: Mapping[str, float],
    symbols: list[str] | None = None,
) -> dict[str, float]:
    """Normalisiert Gewichte auf Summe 1; liefert ein Mapping nur gesetzter Symbole.

    ``symbols`` (optional) legt die Reihenfolge/Teilmenge fest: Gewichte für
    nicht gelistete Symbole werden verworfen. Fehler (deutsch): leere Eingabe,
    negative Gewichte, Summe ≤ 0.
    """
    if not weights:
        raise ValueError("Gewichte fehlen oder sind leer.")
    cleaned: dict[str, float] = {}
    for symbol, weight in weights.items():
        value = float(weight)
        if value < 0.0:
            raise ValueError(f"Negatives Gewicht für „{symbol}“: {value}.")
        cleaned[symbol] = value
    if symbols is not None:
        known = set(symbols)
        cleaned = {symbol: value for symbol, value in cleaned.items() if symbol in known}
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("Gewichtssumme muss größer als 0 sein.")
    return {symbol: value / total for symbol, value in cleaned.items()}


def equal_weights(symbols: list[str]) -> dict[str, float]:
    """Gleichgewichtung ``1/n`` (Default, wenn kein Gewicht angegeben ist)."""
    if not symbols:
        raise ValueError("Für eine Gleichgewichtung wird mindestens 1 Symbol benötigt.")
    share = 1.0 / float(len(symbols))
    return {symbol: share for symbol in symbols}


def portfolio_returns(
    returns: pd.DataFrame, weights: Mapping[str, float]
) -> pd.Series:
    """Gewichtete Portfolio-Rendite-Serie auf der (aligned) Rendite-Matrix.

    Nicht gewichtete Symbole fließen mit 0 ein (Gewichte werden über
    :func:`normalize_weights` auf die vorhandenen Spalten begrenzt und
    **nach** dem Schnitt neu-normalisiert — ein bewusst wegfallendes Symbol
    verschiebt die verbleibenden Anteile proportional).
    """
    require_observations(returns)
    usable = normalize_weights(dict(weights), symbols=[str(c) for c in returns.columns])
    series = pd.Series(0.0, index=returns.index, dtype=float)
    for symbol, weight in usable.items():
        series = series + returns[symbol] * weight
    return series


def portfolio_equity(
    returns: pd.DataFrame, weights: Mapping[str, float], initial: float = 1.0
) -> pd.Series:
    """Portfolio-Equity-Kurve (inkl. Startwert, Muster :func:`.returns.equity_curve`)."""
    return equity_curve(portfolio_returns(returns, weights), initial=initial)


def analyze(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, dict[str, Any]]:
    """Kombinierte Analyse: Kennzahlen je Symbol + Portfolio-Zeile (Schlüssel ``"Portfolio"``).

    Je Zeile (Symbol **und** Portfolio) ein Dictionary gemäß
    :func:`quantlab.quant.metrics.performance_metrics`; zusätzlich
    ``beta``/``alpha``/``ir`` (oder ``None``), wenn eine Benchmark-Serie
    übergeben wird (§F: OLS, Alpha × ppy). Die Benchmark muss auf demselben
    Index liegen wie ``returns`` (Schnitt: Routerschicht schneidet vorher).
    """
    require_observations(returns)
    results: dict[str, dict[str, Any]] = {}
    for symbol in returns.columns:
        row = metrics_mod.performance_metrics(
            returns[symbol], rf=rf, periods_per_year=periods_per_year
        )
        row.update(_relative_row(returns[symbol], benchmark, periods_per_year))
        results[str(symbol)] = row
    portfolio_series = portfolio_returns(returns, weights)
    portfolio_row = metrics_mod.performance_metrics(
        portfolio_series, rf=rf, periods_per_year=periods_per_year
    )
    portfolio_row.update(_relative_row(portfolio_series, benchmark, periods_per_year))
    results[PORTFOLIO_KEY] = portfolio_row
    return results


def _relative_row(
    series: pd.Series, benchmark: pd.Series | None, periods_per_year: int
) -> dict[str, float | None]:
    """Beta/Alpha/IR einer Serie gegen die Benchmark (oder None ohne Benchmark)."""
    if benchmark is None:
        return {"beta": None, "alpha": None, "ir": None}
    beta, alpha = risk_mod.beta_alpha(series, benchmark, periods_per_year)
    return {
        "beta": beta,
        "alpha": alpha,
        "ir": risk_mod.information_ratio(series, benchmark, periods_per_year),
    }
