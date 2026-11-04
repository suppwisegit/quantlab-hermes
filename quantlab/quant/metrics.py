"""Kennzahlen auf Rendite-Serien/Equity-Kurven (T8) — Konventionen §F, bindend.

- **Ann. Rendite** = Ø Periodenrendite × periods_per_year (arithmetisch, Default 252).
- **CAGR** aus der Equity-Kurve: ``(Ende/Start)^(ppy/n) − 1`` — nie aus Ø-Renditen.
- **Ann. Vol** = std(ddof=1) × √ppy (Stichproben-Standardabweichung).
- **Sharpe** = (ann. Rendite − rf)/ann. Vol, ``rf`` Default 0 (konfigurierbar).
- **Sortino** = (ann. Rendite − Target)/Downside-Deviation; Downside-Deviation =
  ``√(mean(min(r − target, 0)²)) × √ppy`` — Mittelung über die **volle**
  Stichprobe (klassische LPM₂-Definition), nicht nur über die negativen Treffer.
  Konvention **E018**: das Sortino-Target ist fest **0** (MAR = 0);
  ``risk_free`` fließt ausschließlich in den Sharpe-Zähler ein — nie in den
  Sortino.
- **MaxDD** pfadbasiert aus der Equity-Kurve (``equity/cummax − 1``, Minimum),
  nie annualisiert; Rückgabe **negativ** (≤ 0).
- **Calmar** = CAGR/|MaxDD|.

Alle Funktionen sind rein (keine I/O). Werte, die fachlich nicht definierbar
sind (z. B. Sharpe bei Vol = 0), liefern ``None`` — kein ``inf``/``nan``, damit
die JSON-Serialisierung wohldefiniert bleibt. Zu wenig Input (< 2 Beobachtungen)
→ :class:`ValueError` (deutsch; Router → 422).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.quant.returns import (
    PERIODS_PER_YEAR,
    annualization_factor,
    annualize_return,
    annualize_volatility,
    require_observations,
)

#: Schlüssel des Kennzahlen-Dictionarys aus :func:`performance_metrics` (Reihenfolge
#: = Anzeige-Reihenfolge im Frontend; T9/T14 lesen genau diese Schlüssel).
METRIC_KEYS: tuple[str, ...] = (
    "total_return",
    "cagr",
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "calmar",
    "max_dd",
)

#: Absoluter Toleranz-Schwellenwert gegen Float-Artefakte: „theoretisch 0“-Werte
#: (z. B. std einer konstanten Serie ≈ 1e-17 statt exakt 0) zählen als 0 — die
#: abgeleitete Kennzahl ist dann ``None`` statt ±inf (JSON-sicher).
_ZERO_TOLERANCE = 1e-12


def total_return(equity: pd.Series) -> float:
    """Gesamtrendite der Equity-Kurve: ``Ende/Start − 1``."""
    values = equity.to_numpy(dtype=float)
    if len(values) == 0 or values[0] == 0.0:
        raise ValueError("Equity-Kurve leer oder Startwert 0 — Total Return undefiniert.")
    return float(values[-1] / values[0] - 1.0)


def equity_path(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Analyse-Pfad **inkl. Startwert** (n + 1 Punkte) — Basis der Pfadkennzahlen.

    Gegenüber :func:`quantlab.quant.returns.equity_curve` (n Punkte, Anzeige-
    Kurve) zählt dieser Pfad den impliziten Startwert ``initial`` als eigene
    erste Stufe: ``total_return``/``cagr``/``max_dd`` laufen über
    ``len(returns)`` Perioden — Referenzformel ``CAGR = (Ende/initial)^(ppy/n) − 1``.
    """
    cumulative = (1.0 + returns).cumprod() * initial
    start = pd.Series([initial], dtype=float)
    return pd.concat([start, cumulative], ignore_index=True)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Pfadbezogenes Drawdown-Profil: ``equity/cummax(equity) − 1`` (≤ 0)."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximaler Drawdown (pfadbasiert, §F) — als **negativer** Wert (≤ 0).

    Referenz: ``min(equity/cummax − 1)``; die Kurve beginnt mit dem Startwert,
    ein nie fallender Pfad ergibt 0.0.
    """
    values = equity.to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("Equity-Kurve leer — MaxDD undefiniert.")
    running_peak = np.maximum.accumulate(values)
    return float(np.min(values / running_peak - 1.0))


def cagr(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float | None:
    """CAGR aus der Equity-Kurve (§F): ``(Ende/Start)^(ppy/n) − 1`` mit n = Perioden.

    ``equity`` ist der Pfad **inkl. Startwert** (Muster: :func:`equity_path`);
    ``n = len(equity) − 1`` abgelaufene Perioden. Nicht definierbar
    (n < 1, Start ≤ 0) → ``None``. Überlauf der Potenz (Hyper-Growth über
    sehr kurze Laufzeit) → ebenfalls ``None`` statt ``inf`` (§F „kein inf",
    REV-T23 Fix 4a — bewusste Ausnahme vom T8-Besitz dieses Moduls).
    """
    values = equity.to_numpy(dtype=float)
    periods = len(values) - 1
    if periods < 1 or values[0] <= 0.0:
        return None
    growth = float(values[-1]) / float(values[0])
    if growth <= 0.0:
        # Totalverlust: CAGR konventionsgemäß −100 %.
        return -1.0
    try:
        # Python-float-Potenz wirft bei Überlauf OverflowError (numpy-Skalare
        # würden stattdessen inf + RuntimeWarning liefern) — deterministisch
        # fangbar und damit JSON-sicher ``None``.
        return float(growth ** (float(periods_per_year) / float(periods)) - 1.0)
    except OverflowError:
        return None


def annualized_return(
    returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualisierte (arithmetische) Rendite: Ø Tagesrendite × ppy (§F)."""
    return annualize_return(float(returns.mean()), periods_per_year)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualisierte Volatilität: std(ddof=1) × √ppy (§F)."""
    return annualize_volatility(float(returns.std(ddof=1)), periods_per_year)


def downside_deviation(
    returns: pd.Series, target: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Downside-Deviation: ``√(mean(min(r − target, 0)²)) × √ppy`` (volle Stichprobe)."""
    deviations = np.minimum(returns.to_numpy(dtype=float) - float(target), 0.0)
    return float(np.sqrt(np.mean(deviations**2)) * annualization_factor(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float | None:
    """Sharpe = (ann. Rendite − rf)/ann. Vol (§F); ``None`` bei Vol ≈ 0."""
    vol = annualized_volatility(returns, periods_per_year)
    if not np.isfinite(vol) or vol <= _ZERO_TOLERANCE:
        return None
    return (annualized_return(returns, periods_per_year) - float(rf)) / vol


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float | None:
    """Sortino = (ann. Rendite − Target)/Downside-Deviation; ``None`` bei DD ≈ 0.

    Konvention **E018**: Sortino-Target ist 0 (MAR = 0, hier der Default) —
    ``rf`` fließt ausschließlich in den Sharpe (:func:`sharpe_ratio`).
    """
    dd = downside_deviation(returns, target, periods_per_year)
    if dd <= _ZERO_TOLERANCE:
        return None
    return (annualized_return(returns, periods_per_year) - float(target)) / dd


def calmar_ratio(
    equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float | None:
    """Calmar = CAGR/|MaxDD| (§F); ``None``, wenn kein Drawdown (MaxDD = 0) — kein ``inf``."""
    cagr_value = cagr(equity, periods_per_year)
    mdd = max_drawdown(equity)
    if cagr_value is None or mdd >= 0.0:
        return None
    return cagr_value / abs(mdd)


def performance_metrics(
    returns: pd.Series,
    *,
    rf: float = 0.0,
    target: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """Alle Basis-Kennzahlen einer Rendite-Serie in einem Dictionary (Schlüssel: METRIC_KEYS).

    ``rf``/``target`` Default 0 (§F konfigurierbar). ``total_return``/``cagr``/
    ``max_dd``/``calmar`` werden pfadbasiert über :func:`equity_path`
    (inkl. Startwert 1.0) berechnet — identische Basis für Einzel-Symbole und
    Portfolio (T9/T14).
    """
    require_observations(returns)
    equity = equity_path(returns)
    return {
        "total_return": float(total_return(equity)),
        "cagr": cagr(equity, periods_per_year),
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, rf=rf, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, target=target, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(equity, periods_per_year),
        "max_dd": max_drawdown(equity),
    }
