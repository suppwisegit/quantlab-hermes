"""Backtesting-Engine (T10/T29) — Bar-Loop über OHLC-Daten (§F Backtest, E008).

Ausführungsmodell (E008, struktureller Look-Ahead-Schutz):

1. Die Strategie liefert Zielgewichte ``w(t)`` **aus Daten ≤ t** (API:
   ``generate_weights(data) -> DataFrame`` mit genau einer Gewichtsspalte).
2. Die Engine shiftet intern um 1 Bar: als Buch-Zielgewicht von Bar ``i``
   gilt ``w(i-1)`` — der frühestmögliche Fill ist damit Bar ``t+1``, niemals
   Bar ``t``. Bar 0 ist definitionsgemäß flat.
3. Fill-Preis: ``open`` von Bar ``i`` (Default, E008 „Fill t+1 open") oder
   ``close`` von Bar ``i`` (Option „t+1 close").
4. Kosten: :mod:`quantlab.backtest.costs` — bps auf Trade-Volumen, vom Cash
   abgezogen; kann das Cash kurzzeitig leicht negativ machen (Gebührenstaub).
5. Mark-to-Market je Bar auf ``close`` → Equity-Kurve.

Sizing (§F):

- ``target_weights`` (Default): Bei Änderung des (geshifteten) Zielgewichts
  wird die Position exakt auf ``w × Equity`` umgebaut (Zielgewicht-Rebalancing);
  unverändertes Ziel → kein Trade (kein Drift-Churn).
- ``fraction_of_equity`` (nur :func:`run_backtest`, ein Symbol): Einstieg
  size't einmalig mit ``sizing_fraction × Equity``; solange das Signal > 0
  bleibt, wird nicht nachgeschärft, erst Signal → 0 schließt vollständig.

Trades: Fills werden zu Trades (Positionsepisoden) aggregiert — Einstieg/
Ausstieg (Zeit, Preis, Grund), PnL (Einzahlungen aus Verkäufen minus
Auszahlungen für Käufe, Gebühren inklusive; offene Trades unrealisiert zum
letzten Close) und Gebühren. :func:`run_backtest` (v1, ein Symbol) bleibt
long/flat: Gewichte-Summe > 1 + Toleranz bzw. negative Gewichte → 422 (§F).

Multi-Asset v2 (:func:`run_portfolio_backtest`, T29):

- ``data`` ist ein ``dict[symbol, DataFrame]``; die Frames werden auf die
  **Schnittmenge** der Handelszeitpunkte gebracht (inner join, Konvention
  ``quant/returns.py``) — Symbole mit kürzerer Historie kürzen das gemeinsame
  Zeitgrid, Start/Ende werden im Ergebnis ausgewiesen (``period``/``symbols``).
- ``weights`` ist ein DataFrame mit **einer Spalte je Symbol** (Spaltenname =
  Symbol). Pro Zeile gilt: ``Σ|w| ≤ 1 + tol`` (keine Leverage); **negative
  Gewichte sind Short-Positionen** — der Short-Verkauf schreibt den Erlös
  dem Cash gut, der Rückkauf belastet es, Kosten sind symmetrisch in bps auf
  das Trade-Volumen. ``NaN`` zählt als 0 (flat). Ein Short-Verlust kann Cash
  und Equity **negativ** machen — bewusst ohne Margin-Call/Liquidationslogik
  (Kennzahlen bleiben definiert); bei Σ|w| ≤ 1 bleibt das Cash bis zum
  Rückkauf ≥ 0, erst starker Gegenlauf mit Netto-Short treibt es unter null.
- Ein gemeinsamer Cash-Pool über alle Symbole; Zielgewichte beziehen sich auf
  das Portfolio-Equity zu **Bar-Beginn** (vor den Fills des Balkens). Je
  Symbol wird bei abweichendem Ziel auf ``w × Equity`` umgebaut; ein
  Vorzeichenwechsel in einem Fill schließt die alte Positionsepisode und
  eröffnet die neue (Gebühren werden um Umsatzanteile aufgeteilt).
- ``rebalance_every > 0`` erzwingt zusätzlich alle ``n`` Balken die
  Zurücksetzung auf die Zielgewichte (Grund ``rebalance_schedule``) — damit
  ist Basket-Rebalancing trotz Drift ausdrückbar (Grund ``rebalance`` bei
  Signalwechsel bleibt unverändert).
- Trades/Fills führen je Eintrag ``symbol`` und ``direction``
  (``long``/``short``) — Marker-Daten für die Chart-Darstellung (T31).

Ergebnis-Anreicherung (T29): Drawdown-Kurve (``drawdown_frame``), Monats-
Matrix (``monthly_return_matrix``), Buy&Hold-Benchmark des Primärsymbols
(``benchmark_buy_hold``) und der JSON-taugliche Anreicherungs-Block
(``build_enrichment``) für Persistierung und Response.

Kennzahlen (Total/CAGR/AnnReturn/AnnVol/Sharpe/Sortino/Calmar/MaxDD) folgen
§F und werden aus der Equity-Kurve berechnet (Konventionen E007:
252/√252, std(ddof=1), MaxDD pfadbasiert; E018: Sortino-Target = 0 —
``risk_free`` fließt ausschließlich in den Sharpe-Zähler ein). Nicht-finite
Werte (z. B. Sharpe bei Vol = 0) werden als ``None`` serialisiert.

Welle B (E057/E058, additiv): :func:`compute_trade_metrics` erweitert die
Metriken beider Engine-Pfade um das backtesting.py-/TradingView-Kernset
(Exposure, Buy&Hold, Win Rate, Profit Factor, Expectancy, Ø Win/Loss,
Best/Worst, Ø Drawdown, MaxDD-Dauer, Trade-Zählungen) — None-Politik wie
E007 (keine Trades → ``win_rate``/``profit_factor`` = ``None``, nicht
0/1). Je geschlossener Episode führen Trades zusätzlich ``mfe``/``mae``
und ``mfe_pct``/``mae_pct`` (Run-up/Drawdown der Position aus high/low der
gehaltenen Balken; short spiegelverkehrt; fehlen high/low, wird defensive
der Close herangezogen). ``reason_exit``/``bars_held`` bestanden bereits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.errors import ApiError

#: Toleranz für die Gewichte-Prüfung (Summe ≤ 1 + tol, keine Negativgewichte).
WEIGHT_SUM_TOL: float = 1e-6

#: Toleranz, ab der eine Zielgewichts-Änderung einen Fill auslöst.
WEIGHT_CHANGE_TOL: float = 1e-9

#: Staub-Schwelle für Anteile (Float-Rundung nach Voltausgängen).
_SHARE_DUST: float = 1e-9

#: Optionen für Fill-Preis und Sizing (§F).
FILL_OPEN = "open"
FILL_CLOSE = "close"
SIZING_TARGET_WEIGHTS = "target_weights"
SIZING_FRACTION_OF_EQUITY = "fraction_of_equity"

#: Resampling-Obergrenze für die Equity-Kurve (DoD T10: ≤ 2000 Punkte).
MAX_EQUITY_POINTS: int = 2000


@dataclass(frozen=True)
class BacktestResult:
    """Ergebnis eines Backtest-Laufs (vor Persistierung)."""

    equity: pd.DataFrame  # Spalten: ts (int64 µs UTC), equity (float)
    trades: list[dict[str, Any]]
    fills: list[dict[str, Any]]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Ergebnis eines Multi-Asset-Laufs (T29, vor Persistierung).

    ``equity``/``drawdown``/``benchmark`` sind voll aufgelöst (unresampled);
    die Routerschicht resampelt für Persistierung/Response auf
    ``MAX_EQUITY_POINTS`` Punkte (:func:`resample_indices`, gemeinsamer Stride
    für Equity und Drawdown).
    """

    equity: pd.DataFrame  # ts (int64 µs UTC), equity (float)
    drawdown: pd.DataFrame  # ts, drawdown (≤ 0, pfadbasiert)
    monthly_returns: dict[str, dict[str, float | None]]
    trades: list[dict[str, Any]]
    fills: list[dict[str, Any]]
    metrics: dict[str, Any]
    period: dict[str, int]  # start_ts, end_ts, bars (gemeinsames Zeitgrid)
    symbols: list[dict[str, Any]]  # symbol, bars_loaded, bars_used je Symbol
    benchmark: pd.DataFrame | None  # Buy&Hold Primärsymbol: ts, equity


def _invalid(code: str, message: str, hint: str | None = None) -> ApiError:
    return ApiError(422, code, message, hint=hint)


def _weights_array(data: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    """Validiert das Strategie-Gewichte-Frame und liefert die Gewichtsspalte.

    v1 long/flat auf einem Symbol: genau eine Spalte, Zeilenzahl = Balkenzahl,
    Werte in ``[0 − tol, 1 + tol]`` (Summe = Wert, da eine Spalte). ``NaN``
    (Strategie-Warm-up) zählt als 0 (flat).
    """
    if not isinstance(weights, pd.DataFrame):
        raise _invalid(
            "gewichte_form",
            "Strategie-Gewichte müssen ein pandas-DataFrame sein.",
        )
    if weights.shape[1] != 1:
        raise _invalid(
            "gewichte_form",
            f"Strategie-Gewichte müssen genau eine Spalte haben (v1: ein Symbol), "
            f"erhalten: {weights.shape[1]}.",
        )
    if len(weights) != len(data):
        raise _invalid(
            "gewichte_zeilen",
            f"Zeilenzahl der Gewichte ({len(weights)}) passt nicht zur "
            f"Balkenzahl ({len(data)}).",
        )
    values = weights.iloc[:, 0].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    if (values < -WEIGHT_SUM_TOL).any():
        raise _invalid(
            "gewichte_negativ",
            "Negative Zielgewichte sind nicht erlaubt (v1: long/flat).",
        )
    if (values > 1.0 + WEIGHT_SUM_TOL).any():
        raise _invalid(
            "gewichte_summe",
            "Die Summe der Zielgewichte übersteigt 1 (keine Leverage erlaubt).",
            hint="Gewichte müssen im Bereich 0…1 liegen (long/flat).",
        )
    return values


def run_backtest(
    data: pd.DataFrame,
    weights: pd.DataFrame,
    costs: CostModel | None = None,
    *,
    symbol: str | None = None,
    initial_capital: float = 10_000.0,
    fill: str = FILL_OPEN,
    sizing: str = SIZING_TARGET_WEIGHTS,
    sizing_fraction: float = 1.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Simuliert den Bar-Loop und liefert Equity-Kurve, Trades, Fills, Metriken.

    ``data`` im kanonischen Schema (ts/open/high/low/close[/volume], aufsteigend;
    genutzt werden ts/open/close). ``weights``: Strategie-Ausgabe — Zeile i ist
    das Zielgewicht aus Daten ≤ i; die Engine führt es frühestens bei Bar i+1
    aus (E008). Ungültige Konfiguration → ApiError 422.

    ``symbol`` (optional, T29): wird in Fills (``symbol`` + ``direction``) und
    Trades (``symbol``) gestanzt, wenn gesetzt — Marker-Daten für T31. Ohne
    ``symbol`` bleibt die Ausgabe bitidentisch zur v1-Form.
    """
    cost_model = costs if costs is not None else CostModel()
    if isinstance(initial_capital, bool) or not isinstance(initial_capital, int | float):
        raise _invalid("startkapital_typ", "initial_capital muss eine Zahl sein.")
    if initial_capital <= 0:
        raise _invalid("startkapital", "initial_capital muss positiv sein.")
    if fill not in (FILL_OPEN, FILL_CLOSE):
        raise _invalid(
            "fill_modus",
            f"Unbekannter Fill-Modus „{fill}“.",
            hint=f"Erlaubt: {FILL_OPEN}, {FILL_CLOSE}.",
        )
    if sizing not in (SIZING_TARGET_WEIGHTS, SIZING_FRACTION_OF_EQUITY):
        raise _invalid(
            "sizing_modus",
            f"Unbekanntes Sizing „{sizing}“.",
            hint=f"Erlaubt: {SIZING_TARGET_WEIGHTS}, {SIZING_FRACTION_OF_EQUITY}.",
        )
    if not 0.0 < sizing_fraction <= 1.0:
        raise _invalid("sizing_anteil", "sizing_fraction muss in (0, 1] liegen.")

    n = len(data)
    if n == 0:
        raise _invalid("zu_wenige_balken", "Keine Balken zum Backtesten vorhanden.")
    w = _weights_array(data, weights)
    open_ = data["open"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    ts = data["ts"].to_numpy(dtype="int64")
    # E058: MFE/MAE aus high/low der gehaltenen Balken — defensive
    # fillna(close), wenn die Spalten fehlen oder nicht-finite Werte tragen
    # (Engine-Guard prüft nur open/close, §E006).
    high = data["high"].to_numpy(dtype=float) if "high" in data.columns else close
    low = data["low"].to_numpy(dtype=float) if "low" in data.columns else close
    high = np.where(np.isfinite(high), high, close)
    low = np.where(np.isfinite(low), low, close)

    # E008: 1-Bar-Shift — Buch-Ziel von Bar i ist das Signal von Bar i-1.
    target = np.zeros(n, dtype=float)
    if n > 1:
        target[1:] = w[:-1]

    cash = float(initial_capital)
    shares = 0.0
    book_weight = 0.0  # aktuell implementiertes Zielgewicht im Depot
    episode: dict[str, Any] | None = None
    fills: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_values = np.empty(n, dtype=float)
    positions = np.zeros(n, dtype=float)  # E057: je Balken 1.0/0.0 (Position am Tagesmark)

    def _new_episode(i: int, price: float, tgt: float, reason: str) -> dict[str, Any]:
        return {
            "entry_index": i,
            "entry_ts": int(ts[i]),
            "entry_price": float(price),
            "weight": float(tgt),
            "reason_entry": reason,
            "peak_shares": 0.0,
            "max_high": float("-inf"),  # E058: Extrema der gehaltenen Balken
            "min_low": float("inf"),
            "cash_out": 0.0,  # Σ Käufe inkl. Gebühren
            "cash_in": 0.0,  # Σ Verkäufe nach Gebührenabzug
            "fees": 0.0,
        }

    def _close_trade(i: int, price: float, still_open: bool) -> None:
        nonlocal episode
        assert episode is not None
        ep = episode
        if still_open:
            # Unrealisiert: Marktwert + Verkäufe-Einzahlungen − Käufe-Auszahlungen.
            pnl = shares * float(price) + ep["cash_in"] - ep["cash_out"]
        else:
            pnl = ep["cash_in"] - ep["cash_out"]
        # E058: MFE/MAE je geschlossener Episode (long: max(high)−entry bzw.
        # entry−min(low)); Basis ist der Entry-Preis. Offene Episoden bleiben
        # None (Feld existiert für einen stabilen Trade-Vertrag).
        mfe: float | None = None
        mae: float | None = None
        if not still_open and math.isfinite(ep["max_high"]) and ep["entry_price"] > 0:
            mfe = ep["max_high"] - ep["entry_price"]
            mae = ep["entry_price"] - ep["min_low"]
        trade = {
            "direction": "long",
            "entry_index": ep["entry_index"],
            "entry_ts": ep["entry_ts"],
            "entry_price": ep["entry_price"],
            "exit_index": None if still_open else i,
            "exit_ts": None if still_open else int(ts[i]),
            "exit_price": None if still_open else float(price),
            "shares": ep["peak_shares"],
            "weight": ep["weight"],
            "fees": ep["fees"],
            "pnl": float(pnl),
            "pnl_pct": (float(pnl) / ep["cash_out"]) if ep["cash_out"] > 0 else None,
            "reason_entry": ep["reason_entry"],
            "reason_exit": None if still_open else "signal_flat",
            "bars_held": None if still_open else i - ep["entry_index"],
            "mfe": None if mfe is None else float(mfe),
            "mae": None if mae is None else float(mae),
            "mfe_pct": None if mfe is None else float(mfe / ep["entry_price"]),
            "mae_pct": None if mae is None else float(mae / ep["entry_price"]),
            "open": still_open,
        }
        if symbol is not None:
            trade["symbol"] = symbol
        trades.append(trade)
        episode = None

    def _execute(i: int, price: float, tgt: float, reason: str) -> None:
        nonlocal cash, shares, book_weight, episode
        equity_pre = cash + shares * price
        value_now = shares * price
        value_tgt = tgt * equity_pre
        delta_value = value_tgt - value_now
        notional = abs(delta_value)
        fee = cost_model.cost(notional)
        delta_shares = delta_value / price
        buying = delta_value > 0

        if buying and episode is None:
            episode = _new_episode(i, price, tgt, reason)
        if episode is not None:
            if buying:
                episode["cash_out"] += notional + fee
            else:
                episode["cash_in"] += notional - fee
            episode["fees"] += fee

        shares += delta_shares
        cash -= delta_value + fee

        if episode is not None:
            episode["peak_shares"] = max(episode["peak_shares"], shares)
            if shares <= _SHARE_DUST:
                shares = 0.0
                _close_trade(i, price, still_open=False)

        eintrag = {
            "index": i,
            "ts": int(ts[i]),
            "price": float(price),
            "shares": float(delta_shares),
            "notional": float(notional),
            "cost": float(fee),
            "weight_before": float(book_weight),
            "weight_after": float(tgt),
            "reason": reason,
        }
        if symbol is not None:
            eintrag["symbol"] = symbol
            eintrag["direction"] = "long"
        fills.append(eintrag)
        book_weight = tgt

    for i in range(n):
        # Preis-Guard (E006): open/close müssen endlich und > 0 sein — ein
        # 0/NaN/negativer Preis würde Fills (delta_shares = ΔWert/Preis) und
        # die Equity-Kurve lautlos verfälschen. Früh abbrechen, nie raten.
        if not (
            math.isfinite(open_[i])
            and open_[i] > 0.0
            and math.isfinite(close[i])
            and close[i] > 0.0
        ):
            raise _invalid(
                "ungueltige_preisdaten",
                f"Ungültige Preisdaten bei Balken {i} (ts={int(ts[i])}): "
                "open und close müssen endliche, positive Zahlen sein.",
            )
        price = open_[i] if fill == FILL_OPEN else close[i]
        raw_target = target[i]
        if sizing == SIZING_FRACTION_OF_EQUITY:
            # Einmalig size'n, nicht nachschärfen: Einstieg bei Signal > 0 mit
            # fixem Anteil, vollständiger Ausstieg bei Signal → 0.
            if episode is None and raw_target > WEIGHT_CHANGE_TOL:
                _execute(i, price, sizing_fraction, "signal_long")
            elif episode is not None and raw_target <= WEIGHT_CHANGE_TOL:
                _execute(i, price, 0.0, "signal_flat")
        elif abs(raw_target - book_weight) > WEIGHT_CHANGE_TOL:
            if book_weight <= WEIGHT_CHANGE_TOL:
                reason = "signal_long"
            elif raw_target <= WEIGHT_CHANGE_TOL:
                reason = "signal_flat"
            else:
                reason = "rebalance"
            _execute(i, price, raw_target, reason)
        # Mark-to-Market auf Bar-Close (Tagesmark) — Zustand je Bar, nicht Endzustand.
        equity_values[i] = cash + shares * close[i]
        # E057/E058: Positions-Historie (Exposure) + high/low-Extrema der
        # aktuell gehaltenen Episode (Einstiegsbar inklusive, Ausstiegsbar —
        # an dem die Episode bereits geschlossen ist — exklusive).
        if episode is not None:
            positions[i] = 1.0
            if high[i] > episode["max_high"]:
                episode["max_high"] = float(high[i])
            if low[i] < episode["min_low"]:
                episode["min_low"] = float(low[i])

    if episode is not None:
        _close_trade(n - 1, float(close[-1]), still_open=True)

    equity = pd.DataFrame({"ts": ts, "equity": equity_values})
    metrics = compute_equity_metrics(equity_values, periods_per_year=periods_per_year)
    metrics.update(
        compute_trade_metrics(trades, equity_values, positions=positions, closes=close)
    )
    return BacktestResult(equity=equity, trades=trades, fills=fills, metrics=metrics)


# ---------------------------------------------------------------------------
# Multi-Asset-Engine v2 (T29): Schnittmengen-Alignment, Short, Rebalancing
# ---------------------------------------------------------------------------


def align_frames(
    data: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Richtet mehrere Symbol-Frames aufs gemeinsame Zeitgrid aus (Schnittmenge).

    Konvention wie ``quant/returns.py``: inner join auf den ``ts``-Zeitpunkten
    (int64 µs UTC, aufsteigend sortiert) — Beobachtungen außerhalb der
    gemeinsamen Historie fließen bewusst nicht ein. Liefert
    ``(gemeinsame_ts, ausgerichtete_frames, symbol_info)``; die Info enthält je
    Symbol ``bars_loaded``/``bars_used`` (für die Response-Dokumentation).
    """
    if not isinstance(data, dict) or not data:
        raise _invalid(
            "strategie_daten",
            "Keine Symbole für den Backtest angegeben.",
        )
    symbols = list(data.keys())
    common: np.ndarray | None = None
    for sym in symbols:
        frame = data[sym]
        for spalte in ("ts", "open", "close"):
            if spalte not in frame.columns:
                raise _invalid(
                    "strategie_daten",
                    f"Für Symbol „{sym}“ fehlt die benötigte Spalte „{spalte}“.",
                )
        sym_ts = np.asarray(frame["ts"], dtype="int64")
        if len(np.unique(sym_ts)) != len(sym_ts):
            raise _invalid(
                "ungueltige_preisdaten",
                f"Für Symbol „{sym}“ enthält die Historie doppelte Zeitstempel.",
            )
        common = sym_ts if common is None else np.intersect1d(common, sym_ts)
    assert common is not None
    common_ts = np.sort(common)
    grid = pd.Index(common_ts)
    aligned: dict[str, pd.DataFrame] = {}
    info: list[dict[str, Any]] = []
    for sym in symbols:
        frame = data[sym]
        idx = pd.Index(np.asarray(frame["ts"], dtype="int64"))
        # Schnittmenge ⊆ eigene ts — reindex trifft exakt (keine NaNs, kein Fill).
        aligned[sym] = frame.set_axis(idx).reindex(grid)
        info.append(
            {
                "symbol": sym,
                "bars_loaded": int(len(frame)),
                "bars_used": int(len(common_ts)),
            }
        )
    return common_ts, aligned, info


def _weights_matrix(
    symbols: list[str], weights: pd.DataFrame, n: int
) -> np.ndarray:
    """Validiert das Multi-Asset-Gewichte-Frame → Matrix (Zeilen × Symbole).

    Spaltennamen müssen die Symbole abbilden, Zeilenzahl = gemeinsame Balken.
    ``NaN`` → 0; pro Zeile gilt ``Σ|w| ≤ 1 + tol`` (keine Leverage, negative
    Gewichte = Short).
    """
    if not isinstance(weights, pd.DataFrame):
        raise _invalid(
            "gewichte_form", "Strategie-Gewichte müssen ein pandas-DataFrame sein."
        )
    columns = [str(col) for col in weights.columns]
    fehlt = [sym for sym in symbols if sym not in columns]
    zuviel = [col for col in columns if col not in symbols]
    if fehlt or zuviel:
        raise _invalid(
            "gewichte_form",
            "Gewichte-Spalten passen nicht zu den Symbolen "
            f"(fehlend: {', '.join(fehlt) or '—'}; unbekannt: {', '.join(zuviel) or '—'}).",
            hint="Eine Gewichtsspalte je Symbol (Spaltenname = Symbol).",
        )
    if len(weights) != n:
        raise _invalid(
            "gewichte_zeilen",
            f"Zeilenzahl der Gewichte ({len(weights)}) passt nicht zur gemeinsamen "
            f"Balkenzahl ({n}).",
        )
    matrix = np.column_stack(
        [weights[str(sym)].to_numpy(dtype=float) for sym in symbols]
    )
    matrix = np.nan_to_num(matrix, nan=0.0)
    if (np.abs(matrix).sum(axis=1) > 1.0 + WEIGHT_SUM_TOL).any():
        raise _invalid(
            "gewichte_summe",
            "Die Summe der absoluten Zielgewichte übersteigt 1 "
            "(keine Leverage; negative Gewichte sind Short, aber Σ|w| ≤ 1).",
        )
    return matrix


def run_portfolio_backtest(
    data: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    costs: CostModel | None = None,
    *,
    initial_capital: float = 10_000.0,
    fill: str = FILL_OPEN,
    rebalance_every: int = 0,
    periods_per_year: int = 252,
) -> PortfolioBacktestResult:
    """Multi-Asset-Bar-Loop (T29): ein Cash-Pool, n Symbole, Short erlaubt.

    ``data``: ``dict[symbol, Frame]`` (kanonisches Schema); die Frames werden
    hier auf die Schnittmenge der Handelszeitpunkte gebracht. ``weights``:
    eine Gewichtsspalte je Symbol (Spaltenname = Symbol); Zeile i ist das
    Zielgewicht aus Daten ≤ i, ausgeführt frühestens bei Bar i+1 (E008).
    ``Σ|w| ≤ 1 + tol`` je Zeile (keine Leverage), negative Gewichte = Short —
    Buchung: Verkaufserlös dem Cash gutgeschrieben, Rückkauf belastet es,
    Kosten symmetrisch in bps auf das Trade-Volumen. ``rebalance_every > 0``
    setzt zusätzlich alle n Balken auf die Zielgewichte zurück. Fills/Trades
    führen ``symbol`` und ``direction`` (long/short) als Marker-Daten.
    """
    cost_model = costs if costs is not None else CostModel()
    if isinstance(initial_capital, bool) or not isinstance(initial_capital, int | float):
        raise _invalid("startkapital_typ", "initial_capital muss eine Zahl sein.")
    if initial_capital <= 0:
        raise _invalid("startkapital", "initial_capital muss positiv sein.")
    if fill not in (FILL_OPEN, FILL_CLOSE):
        raise _invalid(
            "fill_modus",
            f"Unbekannter Fill-Modus „{fill}“.",
            hint=f"Erlaubt: {FILL_OPEN}, {FILL_CLOSE}.",
        )
    if isinstance(rebalance_every, bool) or not isinstance(rebalance_every, int):
        raise _invalid(
            "rebalance_typ", "rebalance_every muss eine ganze Zahl sein."
        )
    if rebalance_every < 0:
        raise _invalid("rebalance_bereich", "rebalance_every muss ≥ 0 sein.")

    common_ts, aligned, symbols_info = align_frames(data)
    symbols = list(aligned.keys())
    n = len(common_ts)
    if n == 0:
        raise _invalid(
            "zu_wenige_balken",
            "Die Symbole haben keine gemeinsamen Handelszeitpunkte (Schnittmenge leer).",
        )
    k = len(symbols)
    w = _weights_matrix(symbols, weights, n)
    open_m = np.column_stack(
        [aligned[sym]["open"].to_numpy(dtype=float) for sym in symbols]
    )
    close_m = np.column_stack(
        [aligned[sym]["close"].to_numpy(dtype=float) for sym in symbols]
    )
    # E058: high/low-Matrizen je Symbol — defensive fillna(close) je Spalte,
    # wenn high/low fehlen oder nicht-finite Werte tragen (Engine-Guard prüft
    # nur open/close, §E006).
    high_spalten: list[np.ndarray] = []
    low_spalten: list[np.ndarray] = []
    for sym in symbols:
        sym_close = aligned[sym]["close"].to_numpy(dtype=float)
        sym_high = (
            aligned[sym]["high"].to_numpy(dtype=float)
            if "high" in aligned[sym].columns
            else sym_close
        )
        sym_low = (
            aligned[sym]["low"].to_numpy(dtype=float)
            if "low" in aligned[sym].columns
            else sym_close
        )
        high_spalten.append(np.where(np.isfinite(sym_high), sym_high, sym_close))
        low_spalten.append(np.where(np.isfinite(sym_low), sym_low, sym_close))
    high_m = np.column_stack(high_spalten)
    low_m = np.column_stack(low_spalten)

    # E008: 1-Bar-Shift — Buch-Ziel von Bar i ist das Signal von Bar i-1.
    target = np.zeros((n, k), dtype=float)
    if n > 1:
        target[1:] = w[:-1]

    cash = float(initial_capital)
    shares = np.zeros(k, dtype=float)
    book = np.zeros(k, dtype=float)  # je Symbol das zuletzt gebuchte Zielgewicht
    episodes: list[dict[str, Any] | None] = [None] * k
    fills: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_values = np.empty(n, dtype=float)
    positions = np.zeros(n, dtype=float)  # E057: je Balken 1.0/0.0 (irgendeine Position)

    def _new_episode(
        s: int, i: int, price: float, tgt: float, reason: str, direction: str
    ) -> dict[str, Any]:
        return {
            "symbol": symbols[s],
            "direction": direction,
            "entry_index": i,
            "entry_ts": int(common_ts[i]),
            "entry_price": float(price),
            "weight": float(tgt),
            "reason_entry": reason,
            "peak_shares": 0.0,
            "max_high": float("-inf"),  # E058: Extrema der gehaltenen Balken
            "min_low": float("inf"),
            "cash_out": 0.0,  # Σ Käufe inkl. Gebühren
            "cash_in": 0.0,  # Σ Verkäufe nach Gebührenabzug
            "fees": 0.0,
        }

    def _close_episode(
        s: int, i: int, price: float, still_open: bool, reason_exit: str | None
    ) -> None:
        ep = episodes[s]
        assert ep is not None
        if still_open:
            pnl = shares[s] * float(price) + ep["cash_in"] - ep["cash_out"]
        else:
            pnl = ep["cash_in"] - ep["cash_out"]
        base = ep["cash_out"] if ep["cash_out"] > 0 else ep["cash_in"]
        # E058: MFE/MAE je geschlossener Episode — long: max(high)−entry bzw.
        # entry−min(low); short spiegelverkehrt (entry−min(low) als Run-up,
        # max(high)−entry als Drawdown). Basis ist der Entry-Preis; offene
        # Episoden bleiben None (stabile Trade-Verträge).
        mfe: float | None = None
        mae: float | None = None
        if not still_open and math.isfinite(ep["max_high"]) and ep["entry_price"] > 0:
            aufwaerts = ep["max_high"] - ep["entry_price"]
            abwaerts = ep["entry_price"] - ep["min_low"]
            mfe, mae = (
                (aufwaerts, abwaerts) if ep["direction"] == "long" else (abwaerts, aufwaerts)
            )
        trades.append(
            {
                "symbol": ep["symbol"],
                "direction": ep["direction"],
                "entry_index": ep["entry_index"],
                "entry_ts": ep["entry_ts"],
                "entry_price": ep["entry_price"],
                "exit_index": None if still_open else i,
                "exit_ts": None if still_open else int(common_ts[i]),
                "exit_price": None if still_open else float(price),
                "shares": ep["peak_shares"],
                "weight": ep["weight"],
                "fees": ep["fees"],
                "pnl": float(pnl),
                "pnl_pct": (float(pnl) / base) if base > 0 else None,
                "reason_entry": ep["reason_entry"],
                "reason_exit": None if still_open else reason_exit,
                "bars_held": None if still_open else i - ep["entry_index"],
                "mfe": None if mfe is None else float(mfe),
                "mae": None if mae is None else float(mae),
                "mfe_pct": None if mfe is None else float(mfe / ep["entry_price"]),
                "mae_pct": None if mae is None else float(mae / ep["entry_price"]),
                "open": still_open,
            }
        )
        episodes[s] = None

    def _buchen(
        ep: dict[str, Any], value: float, fee: float, *, purchase: bool
    ) -> None:
        """Cash-Fluss-Zuordnung: Kauf belastet (cash_out), Verkauf/Short-Erlös
        schreibt gut (cash_in); Gebühren laufen in der Episode mit."""
        if purchase:
            ep["cash_out"] += value + fee
        else:
            ep["cash_in"] += value - fee
        ep["fees"] += fee

    def _execute_leg(
        s: int, i: int, price: float, equity_pre: float, tgt: float, reason: str
    ) -> None:
        nonlocal cash
        value_now = shares[s] * price
        value_tgt = tgt * equity_pre
        delta_value = value_tgt - value_now
        notional = abs(delta_value)
        if notional == 0.0:
            return
        fee = cost_model.cost(notional)
        delta_shares = delta_value / price
        before = float(shares[s])
        after = before + delta_shares
        cash -= delta_value + fee
        positionssign = after if after != 0.0 else before
        direction = "long" if positionssign > 0 else "short"
        flip = before != 0.0 and abs(after) > _SHARE_DUST and (before > 0) != (after > 0)

        if flip:
            # Ein Fill schließt die alte und eröffnet die neue Positionseite —
            # Umsatz/Gebühr werden um die Umsatzanteile aufgeteilt.
            close_value = abs(before) * price
            open_value = abs(after) * price
            fee_close = fee * close_value / (close_value + open_value)
            fee_open = fee - fee_close
            _buchen(episodes[s], close_value, fee_close, purchase=before < 0)
            _close_episode(s, i, price, still_open=False, reason_exit="signal_flip")
            neue = _new_episode(
                s,
                i,
                price,
                tgt,
                reason,
                "long" if after > 0 else "short",
            )
            _buchen(neue, open_value, fee_open, purchase=after > 0)
            neue["peak_shares"] = abs(after)
            episodes[s] = neue
        else:
            ep = episodes[s]
            if ep is None and abs(after) > _SHARE_DUST:
                ep = _new_episode(
                    s, i, price, tgt, reason, "long" if delta_shares > 0 else "short"
                )
                episodes[s] = ep
            if ep is not None:
                _buchen(ep, notional, fee, purchase=delta_shares > 0)
                ep["peak_shares"] = max(ep["peak_shares"], abs(after))

        shares[s] = 0.0 if abs(after) <= _SHARE_DUST else after
        if episodes[s] is not None and abs(shares[s]) == 0.0:
            _close_episode(s, i, price, still_open=False, reason_exit="signal_flat")

        fills.append(
            {
                "index": i,
                "ts": int(common_ts[i]),
                "symbol": symbols[s],
                "direction": direction,
                "price": float(price),
                "shares": float(delta_shares),
                "notional": float(notional),
                "cost": float(fee),
                "weight_before": float(book[s]),
                "weight_after": float(tgt),
                "reason": reason,
            }
        )
        book[s] = tgt

    for i in range(n):
        # Preis-Guard (E006) je Symbol — ein 0/NaN-Preis würde Fills verfälschen.
        for s, sym in enumerate(symbols):
            open_i = open_m[i, s]
            close_i = close_m[i, s]
            if not (
                math.isfinite(open_i)
                and open_i > 0.0
                and math.isfinite(close_i)
                and close_i > 0.0
            ):
                raise _invalid(
                    "ungueltige_preisdaten",
                    f"Ungültige Preisdaten für „{sym}“ bei Balken {i} "
                    f"(ts={int(common_ts[i])}): open und close müssen endliche, "
                    "positive Zahlen sein.",
                )
        prices = open_m[i] if fill == FILL_OPEN else close_m[i]
        equity_pre = cash + float(np.dot(shares, prices))
        forced = rebalance_every > 0 and i > 0 and i % rebalance_every == 0
        for s in range(k):
            tgt = float(target[i, s])
            unverändert = abs(tgt - book[s]) <= WEIGHT_CHANGE_TOL
            if not forced and unverändert:
                continue
            if forced and unverändert:
                if abs(tgt) <= WEIGHT_CHANGE_TOL:
                    continue  # flat + Ziel 0 → nichts zu rebalancen
                reason = "rebalance_schedule"
            elif abs(book[s]) <= WEIGHT_CHANGE_TOL:
                reason = (
                    "signal_long" if tgt > WEIGHT_CHANGE_TOL else "signal_short"
                )
            elif abs(tgt) <= WEIGHT_CHANGE_TOL:
                reason = "signal_flat"
            else:
                reason = "rebalance"
            _execute_leg(s, i, prices[s], equity_pre, tgt, reason)
        # Mark-to-Market auf Bar-Close (Tagesmark) über alle Symbole.
        equity_values[i] = cash + float(np.dot(shares, close_m[i]))
        # E057/E058: Positions-Historie (irgendeine offene Episode) + high/low-
        # Extrema je gehaltener Episode (Einstiegsbar inklusive, Ausstiegsbar —
        # an dem die Episode bereits geschlossen ist — exklusive).
        for s in range(k):
            ep = episodes[s]
            if ep is not None:
                positions[i] = 1.0
                if high_m[i, s] > ep["max_high"]:
                    ep["max_high"] = float(high_m[i, s])
                if low_m[i, s] < ep["min_low"]:
                    ep["min_low"] = float(low_m[i, s])

    for s in range(k):
        if episodes[s] is not None:
            _close_episode(
                s, n - 1, float(close_m[n - 1, s]), still_open=True, reason_exit=None
            )

    equity = pd.DataFrame({"ts": common_ts, "equity": equity_values})
    metrics = compute_equity_metrics(equity_values, periods_per_year=periods_per_year)
    metrics.update(
        compute_trade_metrics(
            trades,
            equity_values,
            positions=positions,
            # Buy&Hold des Primärsymbols auf dem gemeinsamen Grid (Konvention
            # benchmark_buy_hold; Schnittmenge → keine NaNs in close).
            closes=aligned[symbols[0]]["close"].to_numpy(dtype=float),
        )
    )
    benchmark = benchmark_buy_hold(aligned[symbols[0]], initial_capital)
    return PortfolioBacktestResult(
        equity=equity,
        drawdown=drawdown_frame(equity),
        monthly_returns=monthly_return_matrix(equity),
        trades=trades,
        fills=fills,
        metrics=metrics,
        period={
            "start_ts": int(common_ts[0]),
            "end_ts": int(common_ts[-1]),
            "bars": n,
        },
        symbols=symbols_info,
        benchmark=benchmark,
    )


# ---------------------------------------------------------------------------
# Kennzahlen (§F Quant-Konventionen, E007) — aus der Equity-Kurve selbst
# berechnet (T8 entsteht parallel; bewusster DRY-Verzicht, T11/T14 gluen).
# ---------------------------------------------------------------------------


def _num(value: float | None) -> float | None:
    """float oder ``None`` für fehlende/nicht-finite Werte (JSON-verträglich)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def compute_equity_metrics(
    values: np.ndarray,
    *,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict[str, Any]:
    """Total/CAGR/AnnReturn/AnnVol/Sharpe/Sortino/Calmar/MaxDD nach §F.

    - CAGR aus der Equity-Kurve: ``(end/start)^(ppy/(n−1)) − 1`` (Bar-Zählung).
    - AnnReturn = Ø Bar-Rendite × ppy; AnnVol = std(ddof=1) × √ppy.
    - Sharpe = (AnnReturn − rf)/AnnVol — ``rf`` wirkt **ausschließlich** hier
      (E018). Sortino mit Target 0 (MAR = 0): Downside-Deviation unter 0,
      ``√(mean(min(r, 0)²)) × √ppy`` — unabhängig von ``rf``.
    - MaxDD pfadbasiert (≤ 0, nie annualisiert); Calmar = CAGR/|MaxDD|.
    Nicht definierbare Werte → ``None`` (z. B. Sharpe bei Vol = 0).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    first = float(values[0])
    last = float(values[-1])
    total_return = last / first - 1.0 if first > 0 else None

    years = (n - 1) / periods_per_year if periods_per_year > 0 else 0.0
    if years > 0:
        ratio = last / first if first > 0 else 0.0
        cagr = ratio ** (1.0 / years) - 1.0 if ratio > 0 else -1.0
    else:
        cagr = 0.0

    if n >= 2 and first > 0:
        rets = np.diff(values) / values[:-1]
        ann_return = float(rets.mean()) * periods_per_year
        ann_vol = float(rets.std(ddof=1)) * math.sqrt(periods_per_year)
        # E018: Downside-Deviation um das Target 0 (MAR = 0) — ``risk_free``
        # fließt ausschließlich in den Sharpe-Zähler ein, nie in den Sortino.
        downside_dev = (
            float(np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)))
            * math.sqrt(periods_per_year)
        )
    else:
        ann_return = 0.0
        ann_vol = 0.0
        downside_dev = 0.0

    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else None
    sortino = ann_return / downside_dev if downside_dev > 0 else None

    if n >= 2:
        peaks = np.maximum.accumulate(values)
        drawdowns = values / peaks - 1.0
        max_dd = float(drawdowns.min())
    else:
        max_dd = 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else None

    return {
        "total_return": _num(total_return),
        "cagr": _num(cagr),
        "ann_return": _num(ann_return),
        "ann_vol": _num(ann_vol),
        "sharpe": _num(sharpe),
        "sortino": _num(sortino),
        "calmar": _num(calmar),
        "max_dd": _num(max_dd),
    }


def compute_trade_metrics(
    trades: list[dict[str, Any]],
    equity_values: np.ndarray,
    *,
    positions: np.ndarray | None = None,
    closes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Trade-/Exposure-Kennzahlen nach backtesting.py-/TradingView-Kernset (E057).

    Reine Funktion über die Engine-Ergebnisse — alles ist aus der Equity-
    Kurve, der optionalen Positions-Historie und der Trade-Liste berechenbar
    (keine Engine-Parameter). None-Politik wie E007 (``_num``): nicht
    definierbare Werte sind ``None``, **nicht** 0/1 — insbesondere führen
    null geschlossene Trades zu ``win_rate = None`` und
    ``profit_factor = None``.

    - ``exposure``: Anteil der Balken mit Position (Basis ``positions`` —
      je Balken 1.0/0.0; ohne Historie ``None``).
    - ``buy_hold_return``: ``close_ende / close_0 − 1`` des Primärsymbols
      (Basis ``closes``; ohne Kursreihe ``None``) — bestehende Benchmark-
      Konvention ``benchmark_buy_hold``.
    - ``win_rate``/``profit_factor``/``expectancy``/``avg_win``/``avg_loss``/
      ``best_trade``/``worst_trade``: je Trade über ``pnl`` bzw. ``pnl_pct``
      (Gebühren inklusive, §F) **geschlossener** Episoden — offene Trades
      (unrealisiert) fließen bewusst nicht ein (backtesting.py-Konvention,
      Deckung mit dem Monte-Carlo-Filter E059). ``profit_factor`` ist ohne
      Verlusttranche nicht definierbar → ``None``.
    - ``avg_drawdown``: Ø Tiefpunkt der einzelnen Drawdown-Phasen (≤ 0;
      ohne Phase 0.0); ``max_dd_duration_bars``: längste Folge von Balken
      unter dem bisherigen Hoch (≥ 0, monoton steigende Kurve → 0).
    - ``trades_count``/``trades_long``/``trades_short``: Anzahl geschlossener
      Episoden gesamt/je Richtung (int).
    """
    values = np.asarray(equity_values, dtype=float)

    exposure: float | None = None
    if positions is not None:
        positions_arr = np.asarray(positions, dtype=float)
        if len(positions_arr) > 0:
            exposure = float(positions_arr.mean())

    buy_hold_return: float | None = None
    if closes is not None:
        closes_arr = np.asarray(closes, dtype=float)
        if len(closes_arr) > 0 and closes_arr[0] > 0:
            buy_hold_return = float(closes_arr[-1]) / float(closes_arr[0]) - 1.0

    geschlossene = [trade for trade in trades if trade.get("open") is False]
    pcts = [
        float(trade["pnl_pct"])
        for trade in geschlossene
        if trade.get("pnl_pct") is not None
    ]
    gewinne = [pct for pct in pcts if pct > 0.0]
    verluste = [pct for pct in pcts if pct < 0.0]

    win_rate = (len(gewinne) / len(geschlossene)) if geschlossene else None
    if verluste:
        profit_factor = sum(gewinne) / abs(sum(verluste))
    else:
        profit_factor = None  # ohne Verlusttranche nicht definierbar (auch 0 Trades)
    expectancy = (sum(pcts) / len(pcts)) if pcts else None
    avg_win = (sum(gewinne) / len(gewinne)) if gewinne else None
    avg_loss = (sum(verluste) / len(verluste)) if verluste else None
    best_trade = max(pcts) if pcts else None
    worst_trade = min(pcts) if pcts else None

    # Drawdown-Phasen (Ø Tiefpunkt + längste Unter-Hoch-Folge) aus der Equity:
    # eine Phase = Folge von Balken unter dem bisherigen Hoch (dd < 0); je
    # Phase zählt ihr Tiefpunkt in den Durchschnitt, die Dauer in die Max-Dauer.
    avg_drawdown = 0.0
    max_dd_duration_bars = 0
    if len(values) > 0:
        peaks = np.maximum.accumulate(values)
        drawdowns = values / peaks - 1.0
        phasen: list[float] = []
        aktuelle_dauer = 0
        phasen_min: float | None = None
        for dd in drawdowns:
            if dd < 0.0:
                aktuelle_dauer += 1
                phasen_min = dd if phasen_min is None else min(phasen_min, float(dd))
            else:
                if aktuelle_dauer > 0:
                    phasen.append(float(phasen_min) if phasen_min is not None else 0.0)
                    max_dd_duration_bars = max(max_dd_duration_bars, aktuelle_dauer)
                aktuelle_dauer = 0
                phasen_min = None
        if aktuelle_dauer > 0:
            phasen.append(float(phasen_min) if phasen_min is not None else 0.0)
            max_dd_duration_bars = max(max_dd_duration_bars, aktuelle_dauer)
        avg_drawdown = sum(phasen) / len(phasen) if phasen else 0.0

    return {
        "exposure": _num(exposure),
        "buy_hold_return": _num(buy_hold_return),
        "win_rate": _num(win_rate),
        "profit_factor": _num(profit_factor),
        "expectancy": _num(expectancy),
        "avg_win": _num(avg_win),
        "avg_loss": _num(avg_loss),
        "best_trade": _num(best_trade),
        "worst_trade": _num(worst_trade),
        "avg_drawdown": _num(avg_drawdown),
        "max_dd_duration_bars": int(max_dd_duration_bars),
        "trades_count": len(geschlossene),
        "trades_long": sum(
            1 for trade in geschlossene if trade.get("direction") == "long"
        ),
        "trades_short": sum(
            1 for trade in geschlossene if trade.get("direction") == "short"
        ),
    }


# ---------------------------------------------------------------------------
# Equity-Resampling + Ergebnis-Anreicherung (T29): Drawdown-Kurve, Monats-
# Matrix, Buy&Hold-Benchmark — deterministisch, JSON-tauglich.
# ---------------------------------------------------------------------------


def resample_indices(n: int, max_points: int = MAX_EQUITY_POINTS) -> list[int]:
    """Gleichmäßiger Stride über ``n`` Punkte (≤ ``max_points``), T29-Helfer.

    Erste/letzte Position bleiben immer; deterministisch ohne Off-by-one
    (Muster/M7-Fix aus :func:`resample_equity`). Gleich aufgerufen für Equity
    und Drawdown liefern beide Serien punktgleiche Zeitachsen.
    """
    limit = max(2, int(max_points))
    if n <= 0:
        return []
    if n <= limit:
        return list(range(n))
    stride = math.ceil((n - 1) / (limit - 1))
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return indices


def resample_equity(
    equity: pd.DataFrame, max_points: int = MAX_EQUITY_POINTS
) -> list[dict[str, Any]]:
    """Dünnt die Equity-Kurve auf ≤ ``max_points`` Punkte ein (erste/letzte bleiben).

    Gleichmäßiger Stride, deterministisch; bei n ≤ max_points unverändert. Der
    Stride teilt ``(n − 1)`` durch ``(limit − 1)`` Slots auf — der letzte
    Balken wird dadurch immer erreicht, ohne das Limit zu überschreiten (kein
    Off-by-one, M7-Fix: n = 2·max_points liefert ≤ max_points Punkte).
    """
    n = len(equity)
    if n == 0:
        return []
    indices = resample_indices(n, max_points)
    ts = equity["ts"].to_numpy(dtype="int64")
    values = equity["equity"].to_numpy(dtype=float)
    return [{"ts": int(ts[i]), "equity": float(values[i])} for i in indices]


def drawdown_frame(equity: pd.DataFrame) -> pd.DataFrame:
    """Pfadbasierte Drawdown-Kurve ``eq_t / max(eq_0…eq_t) − 1`` (≤ 0)."""
    values = equity["equity"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(values)
    drawdown = values / peaks - 1.0
    return pd.DataFrame(
        {"ts": equity["ts"].to_numpy(dtype="int64"), "drawdown": drawdown}
    )


def monthly_return_matrix(
    equity: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    """Monatsrenditen als Jahr×Monat-Matrix (T29).

    Basis ist das **letzte Equity-Ende je Kalendermonat** (Zeitzone UTC);
    Rendite = Monatsende/Vormonatsende − 1. Der erste beobachtete Monat misst
    gegen ``equity[0]`` (= ``initial_capital`` am Bar 0) — Teilmonate sind
    damit ausgewiesen. Alle 12 Monate je Jahr liegen vor, Monate ohne Daten
    sind ``null`` (Matrix-Form für die UI). Jahre/monate als String-Schlüssel.
    """
    n = len(equity)
    if n == 0:
        return {}
    ts = equity["ts"].to_numpy(dtype="int64")
    values = equity["equity"].to_numpy(dtype=float)
    periods = pd.to_datetime(ts, unit="us", utc=True).tz_localize(None).to_period("M")
    last_index: dict[Any, int] = {}
    for i, periode in enumerate(periods):
        last_index[periode] = i  # ts aufsteigend → letzter Index gewinnt
    matrix: dict[str, dict[str, float | None]] = {}
    previous_end: float | None = None
    initial = float(values[0])
    for periode in sorted(last_index):
        year_key = str(periode.year)
        if year_key not in matrix:
            matrix[year_key] = {str(m): None for m in range(1, 13)}
        end = float(values[last_index[periode]])
        base = initial if previous_end is None else previous_end
        rendite = (end / base - 1.0) if base > 0 else None
        matrix[year_key][str(periode.month)] = _num(rendite)
        previous_end = end
    return matrix


def benchmark_buy_hold(data: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Buy&Hold-Referenz des Primärsymbols: ``initial · close_t / close_0``.

    Reine Referenzkurve auf den Balken des Symbols (keine Gebühren, kein
    Engine-Loop); Basis ist der erste Schlusskurs des geladenen Fensters.
    """
    ts = data["ts"].to_numpy(dtype="int64")
    close = data["close"].to_numpy(dtype=float)
    base = float(close[0])
    values = float(initial_capital) * close / base
    return pd.DataFrame({"ts": ts, "equity": values})


def build_enrichment(
    equity: pd.DataFrame,
    fills: list[dict[str, Any]],
    symbols_info: list[dict[str, Any]],
    *,
    benchmark_symbol: str | None = None,
    benchmark: pd.DataFrame | None = None,
    max_points: int = MAX_EQUITY_POINTS,
) -> dict[str, Any]:
    """JSON-tauglicher Anreicherungs-Block (T29) für Persistierung + Response.

    Enthält Periode (Start/Ende/Balken des gemeinsamen Grids), Symbol-Info
    (geladen/genutzt — Alignment-Dokumentation), Drawdown-Kurve (resampled,
    punktgleich mit der Equity), Monatsrenditen-Matrix, die Fills als
    Marker-Daten (symbol/direction) und optional die Buy&Hold-Benchmark des
    Primärsymbols (resampled). Alles deterministisch aus dem Lauf abgeleitet —
    POST- und GET-Antwort bleiben dadurch identisch.
    """
    ts = equity["ts"].to_numpy(dtype="int64")
    drawdown = drawdown_frame(equity)["drawdown"].to_numpy(dtype=float)
    indices = resample_indices(len(equity), max_points)
    benchmark_out: dict[str, Any] | None = None
    if benchmark is not None and benchmark_symbol is not None and len(benchmark) > 0:
        bh_ts = benchmark["ts"].to_numpy(dtype="int64")
        bh_values = benchmark["equity"].to_numpy(dtype=float)
        bh_indices = resample_indices(len(benchmark), max_points)
        benchmark_out = {
            "symbol": benchmark_symbol,
            "equity": [
                {"ts": int(bh_ts[j]), "equity": float(bh_values[j])} for j in bh_indices
            ],
        }
    return {
        "period": {"start_ts": int(ts[0]), "end_ts": int(ts[-1]), "bars": len(equity)},
        "symbols": [dict(item) for item in symbols_info],
        "drawdown": [
            {"ts": int(ts[j]), "drawdown": float(drawdown[j])} for j in indices
        ],
        "monthly_returns": monthly_return_matrix(equity),
        "fills": [dict(fill) for fill in fills],
        "benchmark": benchmark_out,
    }
