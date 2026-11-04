"""Kovarianz- und Korrelationsmatrizen (T8) — Konventionen §F, bindend.

- **Default LedoitWolf** (:class:`sklearn.covariance.LedoitWolf`): Kovarianz-Schätzer
  mit automatischer Schrumpfung Richtung diagonaler Zielmatrix — stabilisiert
  Schätzungen bei wenigen Beobachtungen relativ zur Symbolzahl. Das
  Schrumpfungs-α (``shrinkage_``) liegt definitorisch in [0, 1] und wird im
  Response ausgewiesen.
- **``sample``** als Option: unvoreingenommene Stichproben-Kovarianz (ddof=1).
- **Annualisiert × 252** (``periods_per_year`` konfigurierbar, §F): die
  Schätzung läuft auf Tagesrenditen, die Multiplikation mit ppy erfolgt danach
  (Kovarianz ist linear in der Skala — deshalb × 252, Volatilität dagegen √252).
- **Korrelation aus derselben Rendite-Matrix** (§F): ``DataFrame.corr()``
  (Pearson, ddof-unabhängig) — bewusst auf identischer Datenbasis wie die
  Kovarianz, damit Heatmaps (T9) konsistent sind.

Reine Funktionen; zu wenig Input (< 2 Beobachtungen) → :class:`ValueError`
(deutsch, Router → 422).
"""

from __future__ import annotations

import pandas as pd

from quantlab.quant.returns import PERIODS_PER_YEAR, require_observations

#: Unterstützte Schätz-Methoden (Default: ``ledoit_wolf``, §F).
COV_METHODS: tuple[str, ...] = ("sample", "ledoit_wolf")
DEFAULT_COV_METHOD: str = "ledoit_wolf"


def sample_covariance(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Stichproben-Kovarianz (ddof=1), optional annualisiert (× ppy)."""
    require_observations(returns)
    factor = float(periods_per_year) if annualize else 1.0
    return returns.cov(ddof=1) * factor


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> tuple[pd.DataFrame, float]:
    """LedoitWolf-Kovarianz (sklearn) auf Tagesrenditen; Rückgabe (Matrix, Shrinkage-α).

    Identisch zu ``LedoitWolf().fit(X).covariance_`` (auf Tagesbasis) — die
    Annualisierung ist die nachgelagerte Multiplikation × ppy (§F). ``α``
    ist das sklearn-``shrinkage_`` ∈ [0, 1]; die resultierende Matrix ist
    konstruktiv positiv-semidefinit (konvexe Kombination von PSD-Matrizen).
    """
    # M2a/E-M08: sklearn lazy in der Funktion — Android/Chaquopy hat kein
    # scikit-learn; der Import des Moduls bleibt dadurch leicht (Desktop-
    # Verhalten unverändert, nur der erste Aufruf zahlt den Import).
    from sklearn.covariance import LedoitWolf

    require_observations(returns)
    estimator = LedoitWolf()
    estimator.fit(returns.to_numpy(dtype=float))
    factor = float(periods_per_year) if annualize else 1.0
    matrix = pd.DataFrame(
        estimator.covariance_ * factor,
        index=returns.columns,
        columns=returns.columns,
        dtype=float,
    )
    return matrix, float(estimator.shrinkage_)


def covariance_matrix(
    returns: pd.DataFrame,
    method: str = DEFAULT_COV_METHOD,
    *,
    annualize: bool = True,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> tuple[pd.DataFrame, float | None]:
    """Dispatch über :data:`COV_METHODS`; Rückgabe (Matrix, Shrinkage-α oder None).

    Unbekannte Methode → :class:`ValueError` (deutsch).
    """
    if method == "sample":
        return sample_covariance(
            returns, annualize=annualize, periods_per_year=periods_per_year
        ), None
    if method == "ledoit_wolf":
        return ledoit_wolf_covariance(
            returns, annualize=annualize, periods_per_year=periods_per_year
        )
    raise ValueError(
        f"Unbekannte Kovarianz-Methode „{method}“. Gültig: {', '.join(COV_METHODS)}."
    )


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson-Korrelationsmatrix aus **derselben** Rendite-Matrix (§F)."""
    require_observations(returns)
    return returns.corr()
