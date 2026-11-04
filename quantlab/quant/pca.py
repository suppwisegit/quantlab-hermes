"""Hauptkomponenten-Analyse (PCA) auf standardisierten Renditen (T8) — §F, bindend.

**Standardisierung vor der PCA:** Jede Rendite-Spalte wird auf
``(x − mean)/std(ddof=1)`` gebracht. Damit gehen alle Symbole mit gleicher
Varianz in die Zerlegung ein — die Komponenten strukturieren die
**Korrelations**-Struktur des Marktes (Level-/Sektor-Faktoren), nicht die
reine Volatilitäts-Hierarchie.

**Loadings:** ``PCA.components_`` von sklearn — Zeilen = Hauptachsen im
Symbolraum (Einheitsnorm), Spalten = Symbole. ``loadings[PC1, AAPL]`` ist die
Ausprägung von AAPL auf der ersten Hauptkomponente; das Vorzeichen pro
Komponente ist willkürlich (sklearn-SVD), Beträge/Vergleiche sind stabil.

Reine Funktionen; ungültiges ``n_components`` bzw. zu wenig Daten →
:class:`ValueError` (deutsch, Router → 422).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.quant.returns import require_observations

#: Default-Anzahl der Komponenten (Task §T8).
DEFAULT_N_COMPONENTS: int = 3


@dataclass(frozen=True)
class PCAResult:
    """Ergebnis einer PCA: erklärte Varianz + Loadings (Index ``PC1..PCk``)."""

    n_components: int
    n_obs: int
    explained_variance: list[float]
    explained_variance_ratio: list[float]
    cumulative_variance_ratio: list[float]
    loadings: pd.DataFrame  # Index: PC1..PCk, Spalten: Symbole


def standardize(returns: pd.DataFrame) -> pd.DataFrame:
    """Spaltenweise Standardisierung ``(x − mean)/std(ddof=1)`` (PCA-Input, §F)."""
    std = returns.std(ddof=1)
    degenerate = std[std <= 0.0]
    if not degenerate.empty:
        raise ValueError(
            "Symbol(e) mit konstanter Rendite (Standardabweichung 0) können nicht "
            f"standardisiert werden: {', '.join(map(str, degenerate.index))}."
        )
    return (returns - returns.mean()) / std


def run_pca(returns: pd.DataFrame, n_components: int = DEFAULT_N_COMPONENTS) -> PCAResult:
    """PCA auf standardisierten Renditen; ``n_components`` Default 3 (§T8).

    ``n_components`` wird bewusst **nicht** still gekappt: Werte außerhalb von
    ``1 ≤ n_components ≤ min(#Symbole, #Beobachtungen)`` sind ein Fachfehler
    (Router → 422). ``svd_solver="full"`` für deterministische Ergebnisse
    (Tests/Goldwerte).
    """
    # M2a/E-M08: sklearn lazy in der Funktion — Android/Chaquopy hat kein
    # scikit-learn (Desktop-Verhalten unverändert).
    from sklearn.decomposition import PCA

    require_observations(returns)
    upper = min(returns.shape[0], returns.shape[1])
    if n_components < 1 or n_components > upper:
        raise ValueError(
            f"n_components muss zwischen 1 und {upper} liegen "
            f"(erhalten: {n_components})."
        )
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(standardize(returns).to_numpy(dtype=float))
    labels = [f"PC{i + 1}" for i in range(n_components)]
    return PCAResult(
        n_components=n_components,
        n_obs=int(returns.shape[0]),
        explained_variance=[float(v) for v in model.explained_variance_],
        explained_variance_ratio=[float(v) for v in model.explained_variance_ratio_],
        cumulative_variance_ratio=list(np.cumsum(model.explained_variance_ratio_).tolist()),
        loadings=pd.DataFrame(
            model.components_, index=labels, columns=returns.columns, dtype=float
        ),
    )
