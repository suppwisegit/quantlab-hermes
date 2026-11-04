"""Risikomaße: VaR/CVaR, Beta/Alpha (OLS), Information Ratio (T8) — §F, bindend.

- **VaR historisch** (Default, 95 %): das (1 − α)-Perzentil der Renditen,
  als **positiver Verlustbetrag** berichtet: ``VaR = −Quantil(r, 1 − α)``.
  Aussage: „An α Anteil der Tage fällt die Rendite mindestens so schlecht aus.“
  Ein negatives VaR ist möglich (selbst das Perzentil ist eine Rendite) und
  wird bewusst nicht auf 0 geklemmt.
- **VaR parametrisch** (Normalnäherung): ``VaR = −(μ + σ · z_{1−α})`` mit
  ``z`` aus der Standardnormalverteilung, σ mit ddof=1.
- **CVaR / Expected Shortfall** (Bonus, historisch): mittlere Rendite streng
  unterhalb des historischen VaR-Perzentils, ebenfalls als positiver Verlust.
- **Horizont-Skalierung:** VaR/CVaR für ``horizon_days > 1`` über
  ``√horizon`` (i.i.d.-Näherung, „Square-Root-of-Time“) — dokumentierte
  Näherung, keine exakte Mehr-Tages-Verteilung.
- **Beta/Alpha** via statsmodels-OLS der Symbol- auf Benchmark-Renditen
  (``r = α + β · b``); **Alpha wird × 252 annualisiert** (§F, arithmetisch).
  Beide Serien werden vorher auf die Schnittmenge gebracht (inner join).
- **IR** = (Port − Bench, annualisiert)/Tracking-Error (annualisiert),
  Tracking-Error = std(ddof=1) der Differenz × √ppy.

Reine Funktionen; zu wenig Input (< 2 Beobachtungen) → :class:`ValueError`
(deutsch, Router → 422). Nicht definierbare Werte (σ = 0) → ``None``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.quant.returns import PERIODS_PER_YEAR, align_returns, require_observations

#: Unterstützte VaR-Methoden (Default: ``historical``, §F).
VAR_METHODS: tuple[str, ...] = ("historical", "parametric")
DEFAULT_VAR_METHOD: str = "historical"

#: Default-Konfidenzniveau (§F: 95 %).
DEFAULT_CONFIDENCE: float = 0.95


#: Absoluter Toleranz-Schwellenwert gegen Float-Artefakte: ein Tracking-Error
#: unterhalb dieser Grenze („theoretisch 0“, z. B. Float-Rest bei konstanter
#: Differenz) gilt als 0 → IR ist ``None`` statt ±inf (JSON-sicher).
_ZERO_TOLERANCE = 1e-12


def _validate_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"Konfidenzniveau muss zwischen 0 und 1 liegen (exklusiv), erhalten: {confidence}."
        )
    return float(confidence)


def historical_var(
    returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE
) -> float:
    """Historisches VaR als positiver Verlustbetrag: ``−Quantil(r, 1 − α)`` (§F)."""
    require_observations(returns)
    alpha = _validate_confidence(confidence)
    quantile = float(np.percentile(returns.to_numpy(dtype=float), (1.0 - alpha) * 100.0))
    return -quantile


def parametric_var(
    returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE
) -> float:
    """Parametrisches (Normal-)VaR: ``−(μ + σ · z_{1−α})``, σ mit ddof=1."""
    # M2a/E-M08: scipy lazy in der Funktion — Android/Chaquopy hat kein
    # scipy (Desktop-Verhalten unverändert).
    from scipy.stats import norm

    require_observations(returns)
    alpha = _validate_confidence(confidence)
    values = returns.to_numpy(dtype=float)
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    z = float(norm.ppf(1.0 - alpha))
    return -(mu + sigma * z)


def historical_cvar(
    returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE
) -> float:
    """Historisches CVaR (Bonus): Ø Rendite im Tail ≤ VaR-Perzentil, als positiver Verlust."""
    require_observations(returns)
    alpha = _validate_confidence(confidence)
    values = returns.to_numpy(dtype=float)
    quantile = float(np.percentile(values, (1.0 - alpha) * 100.0))
    tail = values[values <= quantile]
    return -float(np.mean(tail))


def scale_horizon(value: float, horizon_days: int) -> float:
    """Square-Root-of-Time-Skalierung auf ``horizon_days`` (i.i.d.-Näherung, dokumentiert)."""
    if horizon_days < 1:
        raise ValueError("horizon_days muss mindestens 1 sein.")
    return float(value) * float(np.sqrt(float(horizon_days)))


def beta_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> tuple[float, float]:
    """OLS-Regression ``r = α + β · b`` auf der Index-Schnittmenge; Rückgabe (β, α_jährlich).

    Alpha (täglich, Regressionskonstante) wird gemäß §F arithmetisch × ppy
    annualisiert. Ungleich lange Historien werden vorher ge-schnitten
    (:func:`quantlab.quant.returns.align_returns`); Benchmark-Differenz-Tests
    nutzen dieselbe Konvention.
    """
    # M2a/E-M08: statsmodels lazy in der Funktion — Android/Chaquopy hat kein
    # statsmodels (Desktop-Verhalten unverändert).
    import statsmodels.api as sm

    require_observations(returns)
    require_observations(benchmark)
    aligned = align_returns(
        pd.concat({"returns": returns, "benchmark": benchmark}, axis=1, sort=True)
    )
    x = aligned["benchmark"].to_numpy(dtype=float)
    y = aligned["returns"].to_numpy(dtype=float)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    alpha_daily = float(model.params[0])
    beta = float(model.params[1])
    return beta, alpha_daily * float(periods_per_year)


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float | None:
    """IR = annualisierte Überrendite / Tracking-Error; ``None`` bei Tracking-Error ≈ 0."""
    require_observations(returns)
    require_observations(benchmark)
    aligned = align_returns(
        pd.concat({"returns": returns, "benchmark": benchmark}, axis=1, sort=True)
    )
    active = aligned["returns"] - aligned["benchmark"]
    tracking_error = float(active.std(ddof=1)) * float(np.sqrt(float(periods_per_year)))
    if not np.isfinite(tracking_error) or tracking_error <= _ZERO_TOLERANCE:
        return None
    excess_annual = float(active.mean()) * float(periods_per_year)
    return excess_annual / tracking_error
