"""Backtest-Analyse-Endpunkte (E059, Welle B §B2): Monte-Carlo-Resampling.

``POST /api/backtest/runs/{run_id}/montecarlo`` liest die Trades eines
**persistierten** Laufs (:mod:`quantlab.storage.repos.backtest_repo`) und
resampelt die Renditen der geschlossenen Trades mit Zurücklegen
(:mod:`quantlab.backtest.monte_carlo`). Antwort: Anzahl Pfade, der Equity-
Fan (Perzentile p5/p25/p50/p75/p95 je Trade-Index, Index 0 = Startkapital)
und die Verteilungen von Endvermögen und MaxDD.

Fehler mit deutscher Meldung (Vertrag C.1): 404 für unbekannte ``run_id``,
422 wenn der Lauf weniger als 5 geschlossene Trades mit Rendite trägt.
Der Endpunkt ist ein normales ``def`` — SQLite- und NumPy-Aufrufe laufen
synchron im FastAPI-Threadpool (Muster ``routers/backtest.py``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantlab.backtest.monte_carlo import (
    DEFAULT_N_PFADE,
    MIN_TRADES,
    geschlossene_renditen,
    simuliere,
)
from quantlab.errors import ApiError
from quantlab.storage.repos import backtest_repo

router = APIRouter(tags=["backtest"])


class MonteCarloRequest(BaseModel):
    """Request für ``POST /api/backtest/runs/{run_id}/montecarlo`` (E059).

    ``n_pfade``: Anzahl simulierter Pfade (Default 1000); ``seed``: optionaler
    Reproduzierbarkeits-Anker — gleicher Seed liefert bitidentische Ergebnisse
    (nichtnegativ, ``ge=0``; negative Werte wären 422 statt NumPy-500).
    """

    n_pfade: int = Field(default=DEFAULT_N_PFADE, ge=1, le=100_000)
    seed: int | None = Field(default=None, ge=0)


@router.post("/backtest/runs/{run_id}/montecarlo")
def post_monte_carlo(run_id: str, request: MonteCarloRequest) -> dict[str, Any]:
    """Monte-Carlo-Resampling der geschlossenen Trades eines Laufs (E059).

    Startkapital ist der erste Equity-Punkt des gespeicherten Laufs (das
    Resampling der Persistierung behält den ersten Punkt stets — Muster
    ``resample_equity``); ohne Equity-Fallback das Engine-Default. ``pfade``
    bestätigt die Anzahl simulierter Pfade, ``perzentile`` liefert den Fan
    als Equity-Serien je Trade-Index und ``endvermoegen``/``max_dd`` die
    Verteilungsstatistiken (min/p5/p25/median/p75/p95/max/mean).
    """
    run = backtest_repo.get_run(run_id)
    if run is None:
        raise ApiError(
            404,
            "backtest_nicht_gefunden",
            f"Backtest-Lauf „{run_id}“ wurde nicht gefunden.",
        )
    renditen = geschlossene_renditen(run.trades)
    if len(renditen) < MIN_TRADES:
        raise ApiError(
            422,
            "zu_wenige_trades",
            f"Der Lauf enthält nur {len(renditen)} geschlossene Trades mit "
            f"Rendite — Monte-Carlo benötigt mindestens {MIN_TRADES}.",
            hint="Einen Lauf mit häufigeren Trades wählen oder länger backtesten.",
        )
    startkapital = float(run.equity[0]["equity"]) if run.equity else 10_000.0
    ergebnis = simuliere(
        renditen,
        startkapital=startkapital,
        n_pfade=request.n_pfade,
        seed=request.seed,
    )
    return {
        "run_id": run_id,
        "pfade": ergebnis.n_pfade,
        "n_trades": ergebnis.n_trades,
        "seed": request.seed,
        "startkapital": ergebnis.startkapital,
        "perzentile": ergebnis.perzentile,
        "endvermoegen": ergebnis.endvermoegen,
        "max_dd": ergebnis.max_dd,
    }
