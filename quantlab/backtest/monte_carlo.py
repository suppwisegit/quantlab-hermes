"""Monte-Carlo-Resampling einer Trade-Folge (E059, Welle B §B2).

Konvention (RECH-Block B2): die **Renditen der geschlossenen Trades** werden
MIT Zurücklegen neu gezogen (Bootstrapping) und zu Equity-Pfaden aufmultipliziert
(compounding: ``equity_t = equity_{t-1} · (1 + r_t)``). Daraus entstehen

- der **Equity-Fan** je Trade-Index (Perzentile 5/25/50/75/95 — je Serie der
  Pfadlängen + 1 Startpunkt, Index 0 = Startkapital),
- die **Verteilung des Endvermögens** und
- die **Verteilung des pfadbasierten MaxDD** (≤ 0, Engine-Konvention).

Typische Erkenntnis (Fachliteratur): das resampelte MaxDD liegt typisch über
dem im Backtest beobachteten — die UI zeigt dazu einen festen Hinweis.

Determinismus: ein gleicher ``seed`` liefert bitidentische Ergebnisse (NumPy
``default_rng`` + fester Rechenweg); ``seed=None`` zieht System-Entropie.
Offene Trades und Trades ohne ``pnl_pct`` werden bereits beim Einsammeln der
Renditen ausgeschlossen (:func:`geschlossene_renditen`) — identischer Filter
wie die Kern-Metriken E057 (backtesting.py-Konvention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quantlab.errors import ApiError

#: Mindestanzahl geschlossener Trades für eine belastbare Simulation (E059).
MIN_TRADES: int = 5

#: Default-Pfadanzahl des Endpunkts (E059).
DEFAULT_N_PFADE: int = 1000

#: Perzentil-Fan des Equity-Verlaufs (Schlüssel → Perzentil, RECH-Konvention).
PERZENTILE: dict[str, float] = {
    "p5": 5.0,
    "p25": 25.0,
    "p50": 50.0,
    "p75": 75.0,
    "p95": 95.0,
}

#: Obergrenze der Pfade pro Anfrage (Schutz vor Speicher-Extremen).
MAX_N_PFADE: int = 100_000

#: Obergrenze der Zugriffe gesamt (``n_pfade · n_trades``, REV-B2 Mangel 3):
#: begrenzt die Arrays ``zuege``/``pfade``/``kurven``/``hoch``/``drawdowns``
#: auf zusammen ~100 MB (8-Byte-Floats) statt Multi-GB-Extremen.
MAX_ZUEGE: int = 1_000_000


@dataclass(frozen=True)
class MonteCarloErgebnis:
    """Ergebnis der Simulation (JSON-tauglich aufbereitet im Router)."""

    n_pfade: int
    n_trades: int
    startkapital: float
    perzentile: dict[str, list[float]]  # p5…p95 je Equity-Serie (Länge n_trades+1)
    endvermoegen: dict[str, float]  # Verteilungsstatistik der Pfad-Endwerte
    max_dd: dict[str, float]  # Verteilungsstatistik der Pfad-MaxDD (≤ 0)


def geschlossene_renditen(trades: list[dict[str, Any]]) -> list[float]:
    """``pnl_pct`` der **geschlossenen** Trades (offene/None ausgeschlossen).

    Deckungsgleich mit der Basis der E057-Kernmetriken: offene Episoden
    (``open``) tragen unrealisierte PnL und werden nicht resampelt; Trades
    ohne berechenbare Rendite (``pnl_pct = None``) scheiden ebenfalls aus.
    """
    renditen: list[float] = []
    for trade in trades:
        if trade.get("open") is not False:
            continue
        pct = trade.get("pnl_pct")
        if pct is not None:
            renditen.append(float(pct))
    return renditen


def _verteilung(werte: np.ndarray) -> dict[str, float]:
    """Min/P5/P25/Median/P75/P95/Max/Mean einer Verteilung (floats)."""
    return {
        "min": float(np.min(werte)),
        "p5": float(np.percentile(werte, 5.0)),
        "p25": float(np.percentile(werte, 25.0)),
        "median": float(np.percentile(werte, 50.0)),
        "p75": float(np.percentile(werte, 75.0)),
        "p95": float(np.percentile(werte, 95.0)),
        "max": float(np.max(werte)),
        "mean": float(np.mean(werte)),
    }


def simuliere(
    trade_renditen: list[float],
    *,
    startkapital: float = 10_000.0,
    n_pfade: int = DEFAULT_N_PFADE,
    seed: int | None = None,
) -> MonteCarloErgebnis:
    """Resampelt die Trade-Renditen MIT Zurücklegen zu ``n_pfade`` Equity-Pfaden.

    Ein Pfad zieht ``len(trade_renditen)`` Züge aus den Renditen (mit
    Zurücklegen, ``rng.integers``) und multipliziert sie auf das Startkapital.
    Weniger als :data:`MIN_TRADES` Renditen → ``ApiError`` 422 mit deutscher
    Meldung; ``n_pfade`` außerhalb 1…:data:`MAX_N_PFADE` ebenfalls 422.
    Gleicher ``seed`` → bitidentisches Ergebnis.
    """
    anzahl = len(trade_renditen)
    if anzahl < MIN_TRADES:
        raise ApiError(
            422,
            "zu_wenige_trades",
            f"Monte-Carlo benötigt mindestens {MIN_TRADES} geschlossene Trades "
            f"(vorhanden: {anzahl}).",
            hint="Länger backtesten oder eine Strategie mit häufigeren Trades wählen.",
        )
    if isinstance(n_pfade, bool) or not isinstance(n_pfade, int):
        raise ApiError(
            422, "pfade_typ", "„n_pfade“ muss eine ganze Zahl sein."
        )
    if not 1 <= n_pfade <= MAX_N_PFADE:
        raise ApiError(
            422,
            "pfade_bereich",
            f"„n_pfade“ muss zwischen 1 und {MAX_N_PFADE} liegen (ist: {n_pfade}).",
        )
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ApiError(
            422, "seed_typ", "„seed“ muss eine ganze Zahl sein."
        )
    if seed is not None and seed < 0:
        raise ApiError(
            422,
            "ungueltiger_seed",
            "„seed“ darf nicht negativ sein (NumPy-Zufallsanker ≥ 0).",
            hint="Für Reproduzierbarkeit eine nichtnegative Ganzzahl wählen.",
        )
    if not startkapital > 0:
        raise ApiError(
            422, "startkapital", "„startkapital“ muss positiv sein."
        )
    if n_pfade * anzahl > MAX_ZUEGE:
        raise ApiError(
            422,
            "umfang_zu_gross",
            f"„n_pfade“ × Trade-Anzahl überschreitet das Limit von "
            f"{MAX_ZUEGE} (ist: {n_pfade} × {anzahl}).",
            hint="Pfadanzahl reduzieren — der Equity-Fan bleibt auch mit "
            "weniger Pfaden belastbar.",
        )

    renditen = np.asarray(trade_renditen, dtype=float)
    rng = np.random.default_rng(seed)
    # MIT Zurücklegen: gleichverteilter Indexzugriff auf die Renditen.
    zuege = rng.integers(0, anzahl, size=(n_pfade, anzahl))
    pfade = float(startkapital) * np.cumprod(1.0 + renditen[zuege], axis=1)

    # Fan inklusive Startpunkt (Index 0 = Startkapital) → Länge n_trades + 1.
    kurven = np.concatenate(
        [np.full((n_pfade, 1), float(startkapital)), pfade], axis=1
    )
    fan = {
        key: [float(wert) for wert in np.percentile(kurven, quantil, axis=0)]
        for key, quantil in PERZENTILE.items()
    }

    # Pfadbasiertes MaxDD (≤ 0, Engine-Konvention: eq / laufendes Hoch − 1).
    hoch = np.maximum.accumulate(kurven, axis=1)
    drawdowns = kurven / hoch - 1.0
    max_dds = drawdowns.min(axis=1)
    endwerte = pfade[:, -1]

    return MonteCarloErgebnis(
        n_pfade=n_pfade,
        n_trades=anzahl,
        startkapital=float(startkapital),
        perzentile=fan,
        endvermoegen=_verteilung(endwerte),
        max_dd=_verteilung(max_dds),
    )
