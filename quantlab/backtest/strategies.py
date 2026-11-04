"""Referenz-Strategien (T10/T29) mit Parameter-Schema für UI (T31) und Agent (T14).

Vertrag (§F Backtest): Eine Strategie ist eine Funktion
``(data, **params) -> DataFrame[Gewichte]`` — Zeile ``i`` ist das Zielgewicht
**aus Daten ≤ i** (``NaN`` im Warm-up = flat). Der strukturelle
Look-Ahead-Schutz liegt in der Engine (1-Bar-Shift, E008); eine Strategie kann
ihn per Bauart nicht verletzen.

Vertrag v2 (T29):

- Single-Symbol-Strategien erhalten das kanonische OHLCV-Frame eines Symbols
  und liefern **eine** Gewichtsspalte (long/flat, Werte ∈ {0, 1}).
- Multi-Symbol-Strategien (``multi_symbol=True``) erhalten
  ``dict[symbol, DataFrame]`` (bereits aufs gemeinsame Zeitgrid ausgerichtet)
  und liefern **eine Gewichtsspalte je Symbol** (Spaltenname = Symbol).
- Negative Gewichte (Short) sind im Engine-Vertrag zulässig (Σ|w| ≤ 1); die
  hier gelisteten Referenz-Strategien bleiben long/flat — Short entsteht über
  den Gewichte-Vertrag (User-Strategien, Rotation), nicht über den Katalog.

Die Registry :data:`STRATEGIES` mappt Namen auf :class:`StrategySpec` — das
Parameter-Schema (Feldname/Typ/Default/Beschreibung + optionale Grenzen) ist
die Grundlage für dynamische Formulare (T31) und Tool-Schemas (T14).

Indikator-Werte (EMA/RSI/MACD/Bollinger) kommen aus
:mod:`quantlab.services.indicators` — genau ein Ort für Werte (E011), damit
Backtest-Signale dieselben Konventionen nutzen wie die Charts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.errors import ApiError
from quantlab.services import indicators as _ind

#: Typliterale des Parameter-Schemas (für UI-Formulare/T14-Tool-Schemas).
PARAM_TYPES = ("int", "float", "bool", "str")


@dataclass(frozen=True)
class StrategySpec:
    """Metadaten + Funktion einer Referenz-Strategie."""

    name: str
    label: str  # deutsche Anzeige-Bezeichnung (E001)
    description: str  # deutsche Kurzbeschreibung
    func: Callable[..., pd.DataFrame]
    params: list[dict[str, Any]] = field(default_factory=list)
    #: ANNAHME: Gewichte-Ausgabe ist v1 long/flat (0/1) auf einem Symbol.
    long_flat: bool = True
    #: T29: True → func erhält ``dict[symbol, DataFrame]`` und liefert eine
    #: Gewichtsspalte je Symbol (Multi-Asset: Rotation, Basket).
    multi_symbol: bool = False


def _close_column(data: pd.DataFrame) -> pd.Series:
    close = data.get("close")
    if not isinstance(close, pd.Series) or close.empty:
        raise ApiError(
            422,
            "strategie_daten",
            "Strategie benötigt eine „close“-Spalte mit mindestens einem Balken.",
        )
    return close.astype(float)


def _synth_dict(data: Any, name: str) -> dict[str, pd.DataFrame]:
    """Prüft die Multi-Symbol-Eingabe (``dict[symbol, Frame]``) und liefert sie."""
    if not isinstance(data, dict) or not data:
        raise ApiError(
            422,
            "strategie_daten",
            f"„{name}“ ist eine Multi-Symbol-Strategie und benötigt Kursdaten "
            "für mindestens ein Symbol.",
        )
    for sym, frame in data.items():
        close = frame.get("close") if isinstance(frame, pd.DataFrame) else None
        if not isinstance(close, pd.Series) or close.empty:
            raise ApiError(
                422,
                "strategie_daten",
                f"Symbol „{sym}“ hat keine „close“-Daten für „{name}“.",
            )
    return data


def _zustandsmaschine(
    eingang: np.ndarray, einzieg: np.ndarray, aussteig: np.ndarray
) -> np.ndarray:
    """Breakout-Signal-Logik: 1, sobald der Eingang **strikt über** der
    Einzugsschwelle liegt (Durchbruch nach oben); 0, sobald er **strikt unter**
    der Auszugsschwelle liegt; dazwischen vorheriger Zustand (Hold). Strikte
    Vergleiche, damit flache Fenster (σ = 0 → Band = Kurs) keinen Breakout
    vortäuschen. ``NaN`` (Warm-up) lässt den Zustand unverändert (Start: flat).
    """
    zustand = 0.0
    out = np.empty(len(eingang), dtype=float)
    for i, wert in enumerate(eingang):
        if np.isfinite(wert):
            if zustand == 0.0 and wert > einzieg[i]:
                zustand = 1.0
            elif zustand == 1.0 and wert < aussteig[i]:
                zustand = 0.0
        out[i] = zustand
    return out


def _zustandsmaschine_untere(
    eingang: np.ndarray, einzieg: np.ndarray, aussteig: np.ndarray
) -> np.ndarray:
    """Mean-Reversion-Signal-Logik (invertiert): 1, sobald der Eingang <
    Einzugsschwelle (überverkauft); 0, sobald er > Auszugsschwelle steigt;
    dazwischen vorheriger Zustand (Hold). ``NaN`` → Zustand unverändert."""
    zustand = 0.0
    out = np.empty(len(eingang), dtype=float)
    for i, wert in enumerate(eingang):
        if np.isfinite(wert):
            if zustand == 0.0 and wert < einzieg[i]:
                zustand = 1.0
            elif zustand == 1.0 and wert > aussteig[i]:
                zustand = 0.0
        out[i] = zustand
    return out


def sma_crossover(data: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Long, solange der schnelle SMA über dem langsamen liegt — sonst flat.

    Klassisches Crossover (Golden/Death Cross): ``w = 1`` falls
    ``SMA(fast) > SMA(slow)``, sonst ``0``; im Warm-up (fehlende Fenster)
    flat. ``fast`` muss kleiner als ``slow`` sein (422 sonst).
    """
    if fast >= slow:
        raise ApiError(
            422,
            "strategie_parameter",
            f"„fast“ ({fast}) muss kleiner als „slow“ ({slow}) sein.",
            hint="Beispiel: fast=50, slow=200 (Default).",
        )
    close = _close_column(data)
    sma_fast = _ind.sma(close, int(fast))
    sma_slow = _ind.sma(close, int(slow))
    weight = (sma_fast > sma_slow).astype(float)  # NaN-Vergleiche → False → flat
    return pd.DataFrame({"weight": weight}, index=data.index)


def ema_crossover(data: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Long, solange der schnelle EMA über dem langsamen liegt — sonst flat.

    Wie :func:`sma_crossover`, aber mit exponentiell geglätteten Durchschnitten
    (``span=period``, ``adjust=False`` — Konvention E011/Indikatoren); reagiert
    schneller, seedet am ersten Schlusskurs. ``fast`` < ``slow`` (422 sonst).
    """
    if int(fast) < 1 or int(slow) < 1:
        raise ApiError(
            422,
            "strategie_parameter",
            "„fast“ und „slow“ müssen mindestens 1 sein.",
        )
    if fast >= slow:
        raise ApiError(
            422,
            "strategie_parameter",
            f"„fast“ ({fast}) muss kleiner als „slow“ ({slow}) sein.",
            hint="Beispiel: fast=50, slow=200 (Default).",
        )
    close = _close_column(data)
    ema_fast = _ind.ema(close, int(fast))
    ema_slow = _ind.ema(close, int(slow))
    weight = (ema_fast > ema_slow).astype(float)
    return pd.DataFrame({"weight": weight}, index=data.index)


def macd_crossover(
    data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Long, solange die MACD-Linie über ihrer Signal-Linie liegt — sonst flat.

    MACD = EMA(fast) − EMA(slow), Signal = EMA der MACD-Linie (Konvention
    E011); Histogramm > 0 ⇒ Momentum bleibt long. ``fast`` < ``slow`` (422).
    """
    if int(fast) < 1 or int(slow) < 2 or int(signal) < 1:
        raise ApiError(
            422,
            "strategie_parameter",
            "„fast“ ≥ 1, „slow“ ≥ 2 und „signal“ ≥ 1 erforderlich.",
        )
    if fast >= slow:
        raise ApiError(
            422,
            "strategie_parameter",
            f"„fast“ ({fast}) muss kleiner als „slow“ ({slow}) sein.",
            hint="Beispiel: fast=12, slow=26, signal=9 (Default).",
        )
    close = _close_column(data)
    linien = _ind.macd(close, int(fast), int(slow), int(signal))
    weight = (linien["macd"] > linien["signal"]).astype(float)
    return pd.DataFrame({"weight": weight}, index=data.index)


def momentum(
    data: pd.DataFrame, lookback: int = 126, threshold: float = 0.0
) -> pd.DataFrame:
    """Long bei positivem Momentum über ``lookback`` (Schwellen-Variante).

    ``w = 1`` falls ``(close / close.shift(lookback) − 1) > threshold``,
    sonst ``0``; im Warm-up flat. Die Top-k-Variante greift erst mit mehreren
    Instrumenten; auf einem Symbol ist k = 1 identisch mit der Schwelle.
    """
    if int(lookback) < 1:
        raise ApiError(
            422,
            "strategie_parameter",
            "„lookback“ muss mindestens 1 sein.",
        )
    close = _close_column(data)
    mom = close / close.shift(int(lookback)) - 1.0
    weight = (mom > float(threshold)).astype(float)  # NaN-Warm-up → flat
    return pd.DataFrame({"weight": weight}, index=data.index)


def rsi_reversion(
    data: pd.DataFrame,
    period: int = 14,
    buy_below: float = 30.0,
    sell_above: float = 50.0,
) -> pd.DataFrame:
    """Mean-Reversion: Long bei überverkauftem RSI, Ausstieg über der Schwelle.

    Einstieg, wenn RSI (Wilder, E011) unter ``buy_below`` fällt; Ausstieg,
    wenn RSI über ``sell_above`` steigt; dazwischen wird gehalten. In starken
    Aufwärtstrends bleibt das Signal flat (nie überverkauft). Bedingungen:
    ``0 ≤ buy_below < sell_above ≤ 100``.
    """
    if int(period) < 2:
        raise ApiError(
            422,
            "strategie_parameter",
            "„period“ muss mindestens 2 sein (RSI-Seed über ``period`` Deltas).",
        )
    if not (0.0 <= float(buy_below) < float(sell_above) <= 100.0):
        raise ApiError(
            422,
            "strategie_parameter",
            f"„buy_below“ ({buy_below}) muss unter „sell_above“ ({sell_above}) "
            "liegen; beide in 0…100.",
            hint="Beispiel: buy_below=30, sell_above=50 (Default).",
        )
    close = _close_column(data)
    rsi = _ind.rsi(close, int(period)).to_numpy(dtype=float)
    weight = _zustandsmaschine_untere(
        rsi, np.full(len(rsi), float(buy_below)), np.full(len(rsi), float(sell_above))
    )
    return pd.DataFrame({"weight": weight}, index=data.index)


def bollinger_breakout(
    data: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger-Bruch: Long oberhalb des oberen Bandes, Ausstieg am Mittelband.

    Einstieg, wenn der Schluss über ``SMA(period) + num_std·σ`` schließt
    (Momentum-Breakout); Ausstieg zurück am Mittelband. Bänder nach E011
    (σ mit ddof=1). ``period ≥ 2``, ``num_std > 0``.
    """
    if int(period) < 2:
        raise ApiError(
            422,
            "strategie_parameter",
            "„period“ muss mindestens 2 sein.",
        )
    if not (np.isfinite(float(num_std)) and float(num_std) > 0.0):
        raise ApiError(
            422,
            "strategie_parameter",
            f"„num_std“ muss eine endliche Zahl > 0 sein (erhalten: {num_std}).",
        )
    close = _close_column(data)
    baender = _ind.bollinger(close, int(period), float(num_std))
    weight = _zustandsmaschine(
        close.to_numpy(dtype=float),
        baender["upper"].to_numpy(dtype=float),
        baender["middle"].to_numpy(dtype=float),
    )
    return pd.DataFrame({"weight": weight}, index=data.index)


def donchian_breakout(
    data: pd.DataFrame, entry_period: int = 20, exit_period: int = 10
) -> pd.DataFrame:
    """Donchian-Kanal (Turtle-Stil): Long auf neue Hochs, Ausstieg auf neue Tiefs.

    Einstieg, wenn der Schluss über das Maximum der Hochs der vorherigen
    ``entry_period`` Balken steigt; Ausstieg unter das Minimum der Tiefs der
    vorherigen ``exit_period`` Balken (je exklusive, 1 Bar verschoben — kein
    Look-ahead). Dazwischen wird gehalten.
    """
    if int(entry_period) < 1 or int(exit_period) < 1:
        raise ApiError(
            422,
            "strategie_parameter",
            "„entry_period“ und „exit_period“ müssen mindestens 1 sein.",
        )
    high = data.get("high")
    low = data.get("low")
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
        raise ApiError(
            422,
            "strategie_daten",
            "Donchian-Breakout benötigt „high“- und „low“-Spalten.",
        )
    close = _close_column(data)
    upper = (
        high.astype(float)
        .rolling(int(entry_period), min_periods=int(entry_period))
        .max()
        .shift(1)
    )
    lower = (
        low.astype(float)
        .rolling(int(exit_period), min_periods=int(exit_period))
        .min()
        .shift(1)
    )
    weight = _zustandsmaschine(
        close.to_numpy(dtype=float),
        upper.to_numpy(dtype=float),
        lower.to_numpy(dtype=float),
    )
    return pd.DataFrame({"weight": weight}, index=data.index)


def buy_hold(data: pd.DataFrame) -> pd.DataFrame:
    """Benchmark-Referenz: immer long (Gewicht 1 ab Bar 0, kein Warm-up)."""
    close = _close_column(data)
    weight = pd.Series(1.0, index=close.index)
    return pd.DataFrame({"weight": weight})


def momentum_rotation(
    data: dict[str, pd.DataFrame], lookback: int = 126, top_n: int = 1
) -> pd.DataFrame:
    """Momentum-Rotation (Multi-Asset): gleichgewichtet in die Top-n nach Rendite.

    Je Balken erhält jedes der ``top_n`` Symbole mit dem höchsten Momentum
    (``close/close.shift(lookback) − 1``) das Gewicht ``1/top_n``, alle
    übrigen 0; im Warm-up (weniger als ``lookback`` Balken) bleibt das
    Portfolio flat. Bei Gleichstand entscheidet die Symbol-Reihenfolge der
    Eingabe (deterministisch, ``rank(method="first")``). Sind weniger
    Symbole auswählbar als ``top_n``, bleibt die Restquote in Cash.
    """
    kurse = _synth_dict(data, "Momentum-Rotation")
    if int(lookback) < 1:
        raise ApiError(
            422,
            "strategie_parameter",
            "„lookback“ muss mindestens 1 sein.",
        )
    if int(top_n) < 1:
        raise ApiError(
            422,
            "strategie_parameter",
            "„top_n“ muss mindestens 1 sein.",
        )
    close_frame = pd.concat(
        {sym: frame["close"].astype(float) for sym, frame in kurse.items()},
        axis=1,
        join="inner",
    )
    mom = close_frame / close_frame.shift(int(lookback)) - 1.0
    ranks = mom.rank(axis=1, ascending=False, method="first", na_option="bottom")
    # Warm-up-Guard: Zeilen mit NaN-Momentum (alle Kandidaten zu jung) bleiben
    # flat — na_option="bottom" würde sonst NaNs durchranken lassen.
    selected = ranks.le(float(top_n)) & mom.notna()
    weight = selected.astype(float) / float(top_n)
    return weight[list(kurse.keys())]


def basket(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Basket (Multi-Asset): konstant gleiche Gewichte ``1/n`` über alle Symbole.

    Gleichgewichteter Korb ab Bar 1 (Engine-Shift); mit ``rebalance_every > 0``
    im Request wird periodisch auf die Zielgewichte zurückgesetzt
    (Basket-Rebalancing gegen Drift). Bei einem Symbol identisch mit buy_hold.
    """
    kurse = _synth_dict(data, "Basket")
    anteil = 1.0 / len(kurse)
    close_frame = pd.concat(
        {sym: frame["close"].astype(float) for sym, frame in kurse.items()},
        axis=1,
        join="inner",
    )
    weight = pd.DataFrame(anteil, index=close_frame.index, columns=list(kurse.keys()))
    return weight


#: Registry-Export (T11 „STRATEGIES“): Name → Spec mit Parameter-Schema.
#: Schema-Felder: name, type, default, description (de), minimum/maximum (optional).
STRATEGIES: dict[str, StrategySpec] = {
    "sma_crossover": StrategySpec(
        name="sma_crossover",
        label="SMA Crossover",
        description=(
            "Long, solange der schnelle Durchschnitt über dem langsamen liegt "
            "(Golden Cross); flat beim Death Cross."
        ),
        func=sma_crossover,
        params=[
            {
                "name": "fast",
                "type": "int",
                "default": 50,
                "minimum": 1,
                "description": "Periode des schnellen SMA.",
            },
            {
                "name": "slow",
                "type": "int",
                "default": 200,
                "minimum": 2,
                "description": "Periode des langsamen SMA (muss > fast sein).",
            },
        ],
    ),
    "momentum": StrategySpec(
        name="momentum",
        label="Momentum",
        description=(
            "Long, wenn die Rendite über den Lookback-Perioden die Schwelle "
            "übersteigt; sonst flat (Time-Series-Momentum)."
        ),
        func=momentum,
        params=[
            {
                "name": "lookback",
                "type": "int",
                "default": 126,
                "minimum": 1,
                "description": "Rückblick in Balken für die Momentum-Berechnung.",
            },
            {
                "name": "threshold",
                "type": "float",
                "default": 0.0,
                "description": (
                    "Mindest-Return (Dezimalzahl, z. B. 0.05 = 5 %), "
                    "damit das Signal long geht."
                ),
            },
        ],
    ),
    "ema_crossover": StrategySpec(
        name="ema_crossover",
        label="EMA Crossover",
        description=(
            "Wie SMA Crossover, aber mit exponentiell geglätteten Durchschnitten — "
            "reagiert schneller auf Kursänderungen."
        ),
        func=ema_crossover,
        params=[
            {
                "name": "fast",
                "type": "int",
                "default": 50,
                "minimum": 1,
                "description": "Periode des schnellen EMA.",
            },
            {
                "name": "slow",
                "type": "int",
                "default": 200,
                "minimum": 2,
                "description": "Periode des langsamen EMA (muss > fast sein).",
            },
        ],
    ),
    "macd_crossover": StrategySpec(
        name="macd_crossover",
        label="MACD Crossover",
        description=(
            "Long, solange die MACD-Linie über ihrer Signal-Linie liegt "
            "(Histogramm positiv); flat bei negativem Momentum."
        ),
        func=macd_crossover,
        params=[
            {
                "name": "fast",
                "type": "int",
                "default": 12,
                "minimum": 1,
                "description": "Schnelle EMA-Periode der MACD-Linie.",
            },
            {
                "name": "slow",
                "type": "int",
                "default": 26,
                "minimum": 2,
                "description": "Langsame EMA-Periode der MACD-Linie (muss > fast sein).",
            },
            {
                "name": "signal",
                "type": "int",
                "default": 9,
                "minimum": 1,
                "description": "EMA-Periode der Signal-Linie.",
            },
        ],
    ),
    "rsi_reversion": StrategySpec(
        name="rsi_reversion",
        label="RSI Mean-Reversion",
        description=(
            "Kauft überverkaufte Märkte (RSI unter der Einstiegsschwelle) und "
            "verkauft, wenn der RSI die Ausstiegsschwelle überschreitet."
        ),
        func=rsi_reversion,
        params=[
            {
                "name": "period",
                "type": "int",
                "default": 14,
                "minimum": 2,
                "description": "RSI-Periode (Wilder-Smoothing).",
            },
            {
                "name": "buy_below",
                "type": "float",
                "default": 30.0,
                "minimum": 0.0,
                "description": "Einstieg, wenn RSI unter diesem Wert liegt.",
            },
            {
                "name": "sell_above",
                "type": "float",
                "default": 50.0,
                "minimum": 0.0,
                "maximum": 100.0,
                "description": "Ausstieg, wenn RSI über diesem Wert liegt.",
            },
        ],
    ),
    "bollinger_breakout": StrategySpec(
        name="bollinger_breakout",
        label="Bollinger Breakout",
        description=(
            "Long, wenn der Schlusskurs über das obere Bollinger-Band bricht; "
            "Ausstieg zurück am Mittelband."
        ),
        func=bollinger_breakout,
        params=[
            {
                "name": "period",
                "type": "int",
                "default": 20,
                "minimum": 2,
                "description": "Fenster des Mittelbands (SMA) und der Std-Abweichung.",
            },
            {
                "name": "num_std",
                "type": "float",
                "default": 2.0,
                "description": "Abstand der Bänder in Standardabweichungen.",
            },
        ],
    ),
    "donchian_breakout": StrategySpec(
        name="donchian_breakout",
        label="Donchian Breakout",
        description=(
            "Turtle-Stil: Long auf neue Hochs (Einstiegskanal), Ausstieg auf "
            "neue Tiefs (Ausstiegskanal)."
        ),
        func=donchian_breakout,
        params=[
            {
                "name": "entry_period",
                "type": "int",
                "default": 20,
                "minimum": 1,
                "description": "Kanallänge für den Einstieg (Hochs, vorherige Balken).",
            },
            {
                "name": "exit_period",
                "type": "int",
                "default": 10,
                "minimum": 1,
                "description": "Kanallänge für den Ausstieg (Tiefs, vorherige Balken).",
            },
        ],
    ),
    "momentum_rotation": StrategySpec(
        name="momentum_rotation",
        label="Momentum-Rotation (Multi-Asset)",
        description=(
            "Rotiert gleichgewichtet in die Top-n Symbole mit dem höchsten "
            "Momentum (Multi-Asset: braucht mehrere Symbole)."
        ),
        func=momentum_rotation,
        params=[
            {
                "name": "lookback",
                "type": "int",
                "default": 126,
                "minimum": 1,
                "description": "Momentum-Rückblick in Balken.",
            },
            {
                "name": "top_n",
                "type": "int",
                "default": 1,
                "minimum": 1,
                "description": "Anzahl gehaltener Symbole (je 1/top_n gewichtet).",
            },
        ],
        multi_symbol=True,
    ),
    "basket": StrategySpec(
        name="basket",
        label="Basket (Gleichgewicht, Multi-Asset)",
        description=(
            "Gleichgewichteter Korb über alle Symbole (1/n je Symbol); mit "
            "rebalance_every im Request wird periodisch nachgeschärft."
        ),
        func=basket,
        params=[],
        multi_symbol=True,
    ),
    "buy_hold": StrategySpec(
        name="buy_hold",
        label="Buy & Hold",
        description=(
            "Benchmark-Referenz ohne Signal: kauft einmal und hält (Gewicht 1)."
        ),
        func=buy_hold,
        params=[],
    ),
}


def strategy_catalog() -> dict[str, dict[str, Any]]:
    """Serialisierbarer Katalog (GET /api/backtest/strategies, UI/Agent)."""
    return {
        spec.name: {
            "name": spec.name,
            "label": spec.label,
            "description": spec.description,
            "long_flat": spec.long_flat,
            "multi_symbol": spec.multi_symbol,
            "params": [dict(param) for param in spec.params],
        }
        for spec in STRATEGIES.values()
    }
