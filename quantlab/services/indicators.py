"""Indikator-Berechnung auf OHLC-Schlüssen (T6, E011 — genau ein Ort für Werte).

Dashboard, Agent-Tool ``create_chart`` und künftige Backtest-Signale nutzen
dieselben Werte; Goldwert-Tests liegen nur hier (``tests/test_indicators.py``).

**Konventionen (eingefroren, in den Tests gegen Referenzen geprüft):**

- **SMA** — rollierender Mittelwert über ``period`` Schlüsse; die ersten
  ``period - 1`` Werte sind NaN (führende Lücke).
- **EMA** — exponentiell geglättet mit ``span = period`` und ``adjust=False``
  (rekursiv ``ema[i] = α·c[i] + (1-α)·ema[i-1]`` mit ``α = 2/(period+1)``),
  startend am ersten Schlusskurs (Standard-Chartkonvention, z. B. TradingView);
  daher **keine** führende Lücke.
- **RSI** — **Wilder-Smoothing (Wilder 1978, dokumentiert):** Seed =
  SMA über die ersten ``period`` Kursänderungen, danach rekursiv
  ``avg = (prev·(period-1) + x) / period`` — äquivalent zu einer EMA mit
  ``α = 1/period``, aber mit SMA-Startwert. Deshalb bewusst **nicht**
  ``pandas .ewm()`` (das am ersten Wert seedet) und deshalb führende Lücke von
  ``period`` Werten (Seed braucht ``period`` Deltas). Randfälle: kein
  Durchschnittsverlust (nur Gewinne) → RSI 100; kein Durchschnittsgewinn
  (nur Verluste) → RSI 0; beide 0 (durchgehend flach) → RSI 50.
- **MACD** — ``EMA(fast) - EMA(slow)``; Signal = ``EMA(signal)`` über der
  MACD-Linie (gleiche EMA-Konvention); Histogramm = MACD − Signal.
- **Bollinger** — Mittelerte = SMA(period); obere/untere Linie = Mittelwert
  ± ``std`` · Standardabweichung des rollierenden Fensters mit ``ddof=1``
  (Stichprobe — pandas-Default); führende Lücke wie SMA.

T25-Erweiterung (OHLCV-Indikatoren — Konventionen ebenso eingefroren):

- **ATR (Wilder)** — True Range ``TR[i] = max(high−low, |high−close[i−1]|,
  |low−close[i−1]|)`` mit ``TR[0] = high[0] − low[0]`` (ohne Vortages-
  Schluss definiert). Seed = SMA über die ersten ``period`` TRs → erster
  Wert bei Index ``period − 1``; Rekursion ``atr = (prev·(period−1) + TR)/period``
  (Wilder-Muster wie RSI). Führende Lücke: ``period − 1`` Werte (TR[0] ist
  definiert, anders als die RSI-Deltas).
- **OBV** — kumulativ: ``obv[0] = 0``, danach ``± volume`` je Vorzeichen der
  Schlussänderung (gleichbleibender Schluss → unverändert). Keine Lücke.
- **VWAP (kumulativ, §T25)** — typischer Preis ``tp = (high+low+close)/3``;
  ``vwap[i] = Σ(tp·volume) / Σ(volume)`` kumulativ ab dem **ersten Balken
  der geladenen Serie** (Tagesdaten: eine Session pro Bar — die Session-
  Grenze ist der Serienanfang). ``Σ volume == 0`` → NaN. Keine Lücke.
- **Stochastic (14/3/3)** — roh ``%K = 100·(close − LL(k)) / (HH(k) − LL(k))``
  über ``k`` Balken; ``%K`` = SMA(roh, ``smooth``); ``%D`` = SMA(%K, ``d``).
  ``HH == LL`` (fensterbreite 0) → NaN. Lücken: ``k−1`` / ``k+smooth−2`` /
  ``k+smooth+d−3`` Werte.
- **ADX (Wilder)** — ``+DM/−DM`` nur der jeweils größere, positive High-/Low-
  Schritt; TR und DM Wilder-glättet (SMA-Seed, erster Wert bei Index
  ``period − 1``); ``±DI = 100 · DM_smooth / TR_smooth`` (``TR_smooth == 0``
  → beide DI 0); ``DX = 100·|+DI − −DI| / (+DI + −DI)`` (Summe 0 → DX 0,
  „kein richtungsgebender Trend“); ``ADX`` = Wilder-Glättung der DX → erster
  Wert bei Index ``2·(period − 1)``.
- **CCI (20)** — ``tp = (high+low+close)/3``; ``CCI = (tp − SMA(tp, period)) /
  (0.015 · MAD)`` mit **mittlerer absoluter Abweichung** MAD des Fensters
  vom Fenster-SMA (nicht Std-Abweichung, Lambert-Original); Konstante 0.015.
  MAD 0 (perfekt flaches Fenster) → NaN; Lücke ``period − 1``.
- **Williams %R (14)** — ``%R = −100 · (HH − close) / (HH − LL)`` über
  ``period`` Balken (Werte in [−100, 0]); ``HH == LL`` → NaN; Lücke
  ``period − 1``.
- **Ichimoku (9/26/52)** — ``tenkan = (HH(t)+LL(t))/2``, ``kijun =
  (HH(k)+LL(k))/2``; ``senkou_a = (tenkan + kijun)/2`` und ``senkou_b =
  (HH(s)+LL(s))/2``, **vorverschoben um ``kijun``** Balken (Wolke in der
  Zukunft); ``chikou = close``, um ``kijun`` **zurück**verschoben.
  Displacement alignmentstreu als Lücken: Senkou A/B laufen bis zum letzten
  Balken, ihre **führende** Lücke wächst um ``kijun`` (die ``kijun`` Balken
  Wolke jenseits des letzten Balkens entfallen — dokumentierter Kompromiss,
  eine alignmentstreue Serie kennt keine Zukunftsts); Chikou trägt seine
  ``kijun`` NaN-Lücke **am Ende** der Serie.

Alle Funktionen sind **alignmentstreu**: Rückgabe hat exakt die Länge und den
Index des Eingabe-``pd.Series``; nicht definierte Werte sind NaN. Der Router
(serialisiert in ``routers/data.py``) übersetzt NaN → ``null`` im JSON — das
Frontend startet die Linie ab dem ersten gültigen Wert.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Gültige Indikator-Typen des ``POST /api/data/indicators``-Vertrags (C.1).
#: Reihenfolge eingefroren: die ersten fünf sind die close-basierten T6-Typen
#: (Bestands-Fehlermeldung bleibt Präfix-stabil), danach die T25-OHLCV-Typen.
INDICATOR_TYPES: tuple[str, ...] = (
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger",
    "atr",
    "obv",
    "vwap",
    "stoch",
    "adx",
    "cci",
    "williamsr",
    "ichimoku",
)

#: Obergrenze je Anfrage (sinnvolle Charts brauchen < 5; Schutz vor Missbrauch).
MAX_INDICATORS_PER_REQUEST = 8


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average über ``period`` Schlüsse; führende NaNs (siehe Modul)."""
    _validate_period(period, minimum=1)
    return close.astype("float64").rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponentieller Durchschnitt (``span=period``, ``adjust=False``); seedet am ersten Wert."""
    _validate_period(period, minimum=1)
    return close.astype("float64").ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI mit Wilder-Smoothing (Konvention siehe Moduldocstring)."""
    _validate_period(period, minimum=2)
    close = close.astype("float64")
    n = len(close)
    result = np.full(n, np.nan)
    if n <= period:
        return pd.Series(result, index=close.index, dtype="float64")

    deltas = close.diff().to_numpy(dtype="float64")  # deltas[0] ist NaN
    gains = np.where(np.isnan(deltas), 0.0, np.clip(deltas, 0.0, None))
    losses = np.where(np.isnan(deltas), 0.0, np.clip(-deltas, 0.0, None))

    # Wilder-Seed: SMA über die ersten `period` Änderungen (deltas[1..period]).
    avg_gain = gains[1 : period + 1].mean()
    avg_loss = losses[1 : period + 1].mean()
    result[period] = _rsi_value(avg_gain, avg_loss)

    # Wilder-Rekursion: avg = (prev·(period-1) + x) / period.
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        result[i] = _rsi_value(avg_gain, avg_loss)

    return pd.Series(result, index=close.index, dtype="float64")


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    """RSI aus geglätteten Durchschnitten inkl. der Randfälle aus dem Moduldocstring."""
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0  # durchgehend flach → neutral
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, pd.Series]:
    """MACD-Linie, Signal-Linie und Histogramm (Konventionen siehe Moduldocstring).

    Liefert ein dict mit den Schlüsseln ``"macd"``, ``"signal"`` und
    ``"histogram"``; alle drei alignmentstreu zum Eingabe-Index.
    """
    _validate_period(fast, minimum=1, name="fast")
    _validate_period(slow, minimum=2, name="slow")
    _validate_period(signal, minimum=1, name="signal")
    if fast >= slow:
        raise ValueError(
            f"MACD braucht fast < slow (erhalten: fast={fast}, slow={slow})."
        )
    close = close.astype("float64")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger(close: pd.Series, period: int = 20, std: float = 2.0) -> dict[str, pd.Series]:
    """Bollinger-Bänder: obere/mittlere/untere Linie (Konventionen siehe Moduldocstring)."""
    _validate_period(period, minimum=2)
    if not (math.isfinite(std) and std > 0.0):
        raise ValueError(f"„std“ muss eine positive Zahl sein (erhalten: {std}).")
    close = close.astype("float64")
    middle = sma(close, period)
    rolling_std = close.rolling(window=period, min_periods=period).std(ddof=1)
    deviation = rolling_std * float(std)
    return {
        "upper": middle + deviation,
        "middle": middle,
        "lower": middle - deviation,
    }


# ---------------------------------------------------------------------------
# T25: OHLCV-Indikatoren (ATR, OBV, VWAP, Stochastic, ADX, CCI, W%R, Ichimoku)
# ---------------------------------------------------------------------------


def _wilder_glaettung(werte: np.ndarray, period: int) -> np.ndarray:
    """Wilder-Smoothing (Muster RSI): SMA-Seed, Rekursion ``α = 1/period``.

    Verträgt führende NaNs der Eingabe (z. B. DX erst ab Index ``period−1``):
    Der Seed ist die SMA über die ersten ``period`` **definierten** Werte.
    Innere NaNs sind nicht vorgesehen (alle Caller konstruieren die Eingabe
    ohne solche — Guards setzen stattdessen definierte Werte, siehe ADX).
    """
    n = len(werte)
    out = np.full(n, np.nan)
    definiert = ~np.isnan(werte)
    if not definiert.any():
        return out
    start = int(np.argmax(definiert))
    if n - start < period:
        return out
    avg = float(werte[start : start + period].mean())
    out[start + period - 1] = avg
    for i in range(start + period, n):
        avg = (avg * (period - 1) + float(werte[i])) / period
        out[i] = avg
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> np.ndarray:
    """True Range (Konvention siehe Moduldocstring ATR): ``TR[0] = high−low``."""
    h = high.to_numpy(dtype="float64")
    lo = low.to_numpy(dtype="float64")
    c = close.to_numpy(dtype="float64")
    n = len(c)
    tr = np.full(n, np.nan)
    if n == 0:
        return tr
    tr[0] = h[0] - lo[0]
    if n > 1:
        hc = np.abs(h[1:] - c[:-1])
        lc = np.abs(lo[1:] - c[:-1])
        tr[1:] = np.maximum(np.maximum(h[1:] - lo[1:], hc), lc)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range mit Wilder-Glättung (Konvention siehe Moduldocstring)."""
    _validate_period(period, minimum=1)
    return pd.Series(
        _wilder_glaettung(true_range(high, low, close), period),
        index=close.index,
        dtype="float64",
    )


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume, kumulativ (Konvention siehe Moduldocstring)."""
    c = close.to_numpy(dtype="float64")
    v = volume.to_numpy(dtype="float64")
    if len(c) != len(v):
        raise ValueError(
            "„obv“ braucht gleich lange close-/volume-Serien "
            f"(erhalten: {len(c)} vs. {len(v)})."
        )
    out = np.zeros(len(c), dtype="float64")
    for i in range(1, len(c)):
        out[i] = out[i - 1] + float(np.sign(c[i] - c[i - 1])) * v[i]
    return pd.Series(out, index=close.index, dtype="float64")


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Kumulativer VWAP ab Serienanfang (Session = Bar bei Tagesdaten, §T25)."""
    h = high.to_numpy(dtype="float64")
    lo = low.to_numpy(dtype="float64")
    c = close.to_numpy(dtype="float64")
    v = volume.to_numpy(dtype="float64")
    if not (len(h) == len(lo) == len(c) == len(v)):
        raise ValueError("„vwap“ braucht gleich lange high-/low-/close-/volume-Serien.")
    tp = (h + lo + c) / 3.0
    cum_pv = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    out = np.full(len(c), np.nan)
    positiv = cum_v > 0
    out[positiv] = cum_pv[positiv] / cum_v[positiv]
    return pd.Series(out, index=close.index, dtype="float64")


def stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, smooth: int = 3, d: int = 3
) -> dict[str, pd.Series]:
    """Stochastic %K/%D mit %K-Glättung (Konvention siehe Moduldocstring)."""
    _validate_period(k, minimum=1, name="k")
    _validate_period(smooth, minimum=1, name="smooth")
    _validate_period(d, minimum=1, name="d")
    h = high.astype("float64")
    lo = low.astype("float64")
    c = close.astype("float64")
    hh = h.rolling(window=k, min_periods=k).max()
    ll = lo.rolling(window=k, min_periods=k).min()
    spannweite = hh - ll
    mit_spannweite = spannweite > 0  # False bei führenden NaNs und flachem Fenster
    with np.errstate(invalid="ignore", divide="ignore"):
        roh = (c - ll) / spannweite * 100.0
    roh = roh.where(mit_spannweite)
    k_linie = roh.rolling(window=smooth, min_periods=smooth).mean()
    d_linie = k_linie.rolling(window=d, min_periods=d).mean()
    return {"k": k_linie, "d": d_linie}


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> dict[str, pd.Series]:
    """ADX mit +DI/−DI, Wilder-Glättung (Konventionen siehe Moduldocstring)."""
    _validate_period(period, minimum=2)
    h = high.to_numpy(dtype="float64")
    lo = low.to_numpy(dtype="float64")
    c = close.to_numpy(dtype="float64")
    n = len(c)
    plus_dm = np.zeros(n, dtype="float64")
    minus_dm = np.zeros(n, dtype="float64")
    for i in range(1, n):
        aufwaerts = h[i] - h[i - 1]
        abwaerts = lo[i - 1] - lo[i]
        plus_dm[i] = aufwaerts if (aufwaerts > abwaerts and aufwaerts > 0.0) else 0.0
        minus_dm[i] = abwaerts if (abwaerts > aufwaerts and abwaerts > 0.0) else 0.0

    sm_tr = _wilder_glaettung(true_range(high, low, close), period)
    sm_plus = _wilder_glaettung(plus_dm, period)
    sm_minus = _wilder_glaettung(minus_dm, period)

    # DI: vor dem Seed (Index < period−1) NaN — dort ist keine Aussage möglich;
    # TR-Summe 0 (durchgehend flach) → beide DI 0 (Konvention Moduldocstring).
    di_definiert = ~np.isnan(sm_tr)
    mit_bewegung = di_definiert & (sm_tr > 0)
    ohne_bewegung = di_definiert & ~(sm_tr > 0)
    plus_di = np.full(n, np.nan, dtype="float64")
    minus_di = np.full(n, np.nan, dtype="float64")
    plus_di[mit_bewegung] = 100.0 * sm_plus[mit_bewegung] / sm_tr[mit_bewegung]
    minus_di[mit_bewegung] = 100.0 * sm_minus[mit_bewegung] / sm_tr[mit_bewegung]
    plus_di[ohne_bewegung] = 0.0
    minus_di[ohne_bewegung] = 0.0

    # DX: |+DI − −DI| / Summe; Summe 0 → DX 0 (kein richtungsgebender Trend),
    # vor dem DI-Seed → NaN (der ADX-Seed startet erst danach).
    di_summe = plus_di + minus_di
    dx = np.full(n, np.nan, dtype="float64")
    gueltig = ~np.isnan(di_summe)
    summen = di_summe[gueltig]
    dx[gueltig] = np.where(
        summen > 0.0,
        100.0 * np.abs(plus_di[gueltig] - minus_di[gueltig]) / np.where(summen > 0.0, summen, 1.0),
        0.0,
    )

    index = close.index
    return {
        "adx": pd.Series(_wilder_glaettung(dx, period), index=index, dtype="float64"),
        "plus_di": pd.Series(plus_di, index=index, dtype="float64"),
        "minus_di": pd.Series(minus_di, index=index, dtype="float64"),
    }


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index (0.015-Konstante, MAD — siehe Moduldocstring)."""
    _validate_period(period, minimum=2)
    tp = (high.astype("float64") + low.astype("float64") + close.astype("float64")) / 3.0
    mittel = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda fenster: float(np.mean(np.abs(fenster - fenster.mean()))), raw=True
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        ergebnis = (tp - mittel) / (0.015 * mad)
    return ergebnis.where(mad > 0)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R in [−100, 0] (Konvention siehe Moduldocstring)."""
    _validate_period(period, minimum=2)
    hh = high.astype("float64").rolling(window=period, min_periods=period).max()
    ll = low.astype("float64").rolling(window=period, min_periods=period).min()
    spannweite = hh - ll
    mit_spannweite = spannweite > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ergebnis = -100.0 * (hh - close.astype("float64")) / spannweite
    return ergebnis.where(mit_spannweite)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
) -> dict[str, pd.Series]:
    """Ichimoku Hyōsen: Tenkan/Kijun/Senkou A/B/Chikou inkl. Verschiebung.

    **Displacement (dokumentiert, §T25):** Senkou A/B sind um ``kijun`` Balken
    **vor**verschoben (Wolke in der Zukunft) — ``senkou[i]`` zeigt den Wert,
    der aus Balken ``i − kijun`` berechnet wurde; die Serie läuft deshalb bis
    zum letzten Balken und ihre **führende** Lücke wächst um ``kijun``
    (Senkou A: ``kijun`` Verschiebung + ``kijun − 1`` Kijun-Basis ≈ Index
    ``2·kijun − 1``, Senkou B: ``senkou − 1 + kijun``). Die ``kijun`` Balken
    Wolke **jenseits** des letzten Balkens kann eine alignmentstreue Serie
    nicht abbilden — sie entfällt (dokumentierter Kompromiss). Chikou ist um
    ``kijun`` **zurück**verschoben (``chikou[i] = close[i + kijun]``) und hat
    seine ``kijun`` NaN-Lücke deshalb **am Ende** der Serie.
    """
    _validate_period(tenkan, minimum=1, name="tenkan")
    _validate_period(kijun, minimum=1, name="kijun")
    _validate_period(senkou, minimum=1, name="senkou")
    h = high.astype("float64")
    lo = low.astype("float64")

    def mitte(fenster: int) -> pd.Series:
        return (
            h.rolling(window=fenster, min_periods=fenster).max()
            + lo.rolling(window=fenster, min_periods=fenster).min()
        ) / 2.0

    tenkan_linie = mitte(tenkan)
    kijun_linie = mitte(kijun)
    senkou_a = ((tenkan_linie + kijun_linie) / 2.0).shift(kijun)
    senkou_b = mitte(senkou).shift(kijun)
    chikou = close.astype("float64").shift(-kijun)
    return {
        "tenkan": tenkan_linie,
        "kijun": kijun_linie,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }


# ---------------------------------------------------------------------------
# Validierung der Indikator-Anfragen (Router-Vertrag, deutsche Fehler)
# ---------------------------------------------------------------------------


def _validate_period(period: int, *, minimum: int, name: str = "period") -> None:
    """Perioden-Plug: ganze Zahl im Bereich [minimum, 500]."""
    if not isinstance(period, (int, np.integer)) or isinstance(period, bool):
        raise ValueError(f"„{name}“ muss eine ganze Zahl sein (erhalten: {period!r}).")
    if period < minimum or period > 500:
        raise ValueError(
            f"„{name}“ muss zwischen {minimum} und 500 liegen (erhalten: {period})."
        )


#: Defaults je Typ (C.1/T6: SMA/EMA 20, RSI 14, MACD 12/26/9, Bollinger 20/2;
#: T25: ATR 14, OBV/VWAP parameterlos, Stoch 14/3/3, ADX 14, CCI 20,
#: Williams %R 14, Ichimoku 9/26/52).
_DEFAULTS: dict[str, dict[str, float]] = {
    "sma": {"period": 20},
    "ema": {"period": 20},
    "rsi": {"period": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "std": 2.0},
    "atr": {"period": 14},
    "obv": {},
    "vwap": {},
    "stoch": {"k": 14, "smooth": 3, "d": 3},
    "adx": {"period": 14},
    "cci": {"period": 20},
    "williamsr": {"period": 14},
    "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52},
}

#: Erlaubte Parameter-Schlüssel je Typ.
_ERLAUBT: dict[str, tuple[str, ...]] = {
    "sma": ("period",),
    "ema": ("period",),
    "rsi": ("period",),
    "macd": ("fast", "slow", "signal"),
    "bollinger": ("period", "std"),
    "atr": ("period",),
    "obv": (),
    "vwap": (),
    "stoch": ("k", "smooth", "d"),
    "adx": ("period",),
    "cci": ("period",),
    "williamsr": ("period",),
    "ichimoku": ("tenkan", "kijun", "senkou"),
}


#: Parameter-Reihenfolge in der Indikator-ID je Typ (kanonisch, nicht
#: alphabetisch — MACD liest sich als ``fast_slow_signal``: ``macd:12_26_9``).
_ID_REIHENFOLGE: dict[str, tuple[str, ...]] = {
    "sma": ("period",),
    "ema": ("period",),
    "rsi": ("period",),
    "macd": ("fast", "slow", "signal"),
    "bollinger": ("period", "std"),
    "atr": ("period",),
    "obv": (),
    "vwap": (),
    "stoch": ("k", "smooth", "d"),
    "adx": ("period",),
    "cci": ("period",),
    "williamsr": ("period",),
    "ichimoku": ("tenkan", "kijun", "senkou"),
}

#: Perioden-Parameter der T25-Typen mit je kleinstem sinnvollem Wert.
_T25_PERIODEN: dict[str, tuple[tuple[str, int], ...]] = {
    "atr": (("period", 1),),
    "stoch": (("k", 1), ("smooth", 1), ("d", 1)),
    "adx": (("period", 2),),
    "cci": (("period", 2),),
    "williamsr": (("period", 2),),
    "ichimoku": (("tenkan", 1), ("kijun", 1), ("senkou", 1)),
}

#: OHLCV-Spalten, die ein T25-Typ braucht (close liefert der Bestands-Parameter).
_BENOETIGTE_SPALTEN: dict[str, tuple[str, ...]] = {
    "atr": ("high", "low"),
    "obv": ("volume",),
    "vwap": ("high", "low", "volume"),
    "stoch": ("high", "low"),
    "adx": ("high", "low"),
    "cci": ("high", "low"),
    "williamsr": ("high", "low"),
    "ichimoku": ("high", "low"),
}


def effektive_parameter(indikator_typ: str, params: dict[str, object]) -> dict[str, float]:
    """Prüft Typ/Parameter und übernimmt Defaults → effektive Parametersammlung.

    Löst :class:`ValueError` (Router → 422, deutsch) bei unbekanntem Typ,
    unbekannten Parametern oder sinnwidrigen Werten (Perioden außerhalb
    1–500 bzw. 2–500, ``fast ≥ slow``, ``std ≤ 0``). Wird auch für die
    stabile Indikator-ID (``indikator_id``) genutzt, damit die ID immer die
    **effektiven** Parameter spiegelt — unabhängig davon, was der Caller
    explizit geschickt hat.
    """
    if indikator_typ not in INDICATOR_TYPES:
        raise ValueError(
            f"Unbekannter Indikator-Typ „{indikator_typ}“. "
            f"Gültig: {', '.join(INDICATOR_TYPES)}."
        )
    unbekannt = set(params) - set(_ERLAUBT[indikator_typ])
    if unbekannt:
        raise ValueError(
            f"Unbekannte Parameter {sorted(unbekannt)} für „{indikator_typ}“. "
            f"Erlaubt: {', '.join(_ERLAUBT[indikator_typ])}."
        )
    uebernommen = {**_DEFAULTS[indikator_typ], **params}
    if indikator_typ == "macd":
        _validate_period(int(uebernommen["fast"]), minimum=1, name="fast")
        _validate_period(int(uebernommen["slow"]), minimum=2, name="slow")
        _validate_period(int(uebernommen["signal"]), minimum=1, name="signal")
        if int(uebernommen["fast"]) >= int(uebernommen["slow"]):
            raise ValueError(
                "MACD braucht fast < slow "
                f"(erhalten: fast={uebernommen['fast']}, slow={uebernommen['slow']})."
            )
        return uebernommen
    if indikator_typ == "bollinger":
        _validate_period(int(uebernommen["period"]), minimum=2)
        if not (math.isfinite(float(uebernommen["std"])) and float(uebernommen["std"]) > 0.0):
            raise ValueError(
                f"„std“ muss eine positive Zahl sein (erhalten: {uebernommen['std']})."
            )
        return uebernommen
    if indikator_typ in _T25_PERIODEN:
        for name, minimum in _T25_PERIODEN[indikator_typ]:
            wert = uebernommen[name]
            # Typprüfung VOR ``int()``: ohne sie gab ``None`` einen TypeError
            # (Router → 500), ``"abc"`` eine englische int()-Meldung und
            # ``True`` schlüpfte als ``1`` durch die Reichweiten-Prüfung.
            numerisch = isinstance(
                wert, (int, float, np.integer, np.floating)
            ) and not isinstance(wert, bool)
            unendlich = isinstance(wert, (float, np.floating)) and not math.isfinite(
                float(wert)
            )
            if not numerisch or unendlich:
                raise ValueError(
                    f"„{name}“ muss eine ganze Zahl sein (erhalten: {wert!r})."
                )
            _validate_period(int(wert), minimum=minimum, name=name)
            # int-Konvertierung zurückschreiben, damit die ID (``indikator_id``)
            # „14“ statt „14.9“/„14.0“ spiegelt — nur im T25-Zweig (bollinger
            # „20_2.0“ ist eingefroren).
            uebernommen[name] = int(wert)
        return uebernommen
    if indikator_typ in ("obv", "vwap"):
        return uebernommen  # parameterlos — Defaults (leer) sind vollständig
    _validate_period(int(uebernommen["period"]), minimum=1 if indikator_typ != "rsi" else 2)
    return uebernommen


def berechne_indikator(
    close: pd.Series,
    indikator_typ: str,
    params: dict[str, object],
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Berechnet einen Indikator-Typ mit Parametern → ``{key: Serie}``.

    ``params`` ist für ``sma``/``ema``/``rsi`` ``{"period": int}`` (Defaults
    20/20/14), für ``macd`` ``{"fast","slow","signal"}`` (12/26/9) und für
    ``bollinger`` ``{"period","std"}`` (20/2). Die T25-Typen (ATR, OBV, VWAP,
    Stoch, ADX, CCI, Williams %R, Ichimoku) rechnen auf OHLCV-Basis und
    brauchen die benannten Spalten (siehe :data:`_BENOETIGTE_SPALTEN`) als
    gleichlange Series zum Index von ``close`` — der Router reicht die
    OHLC-Balken durch. Fehlt eine Spalte, gibt es einen deutschen
    :class:`ValueError` (Router → 422). Unbekannte Typen/Parameter und
    sinnwidrige Werte → :class:`ValueError` (Router → 422, deutsch);
    close-basierte Bestandstypen verhalten sich unverändert.
    """
    uebernommen = effektive_parameter(indikator_typ, params)
    if indikator_typ == "sma":
        return {"sma": sma(close, int(uebernommen["period"]))}
    if indikator_typ == "ema":
        return {"ema": ema(close, int(uebernommen["period"]))}
    if indikator_typ == "rsi":
        return {"rsi": rsi(close, int(uebernommen["period"]))}
    if indikator_typ == "macd":
        return macd(
            close,
            fast=int(uebernommen["fast"]),
            slow=int(uebernommen["slow"]),
            signal=int(uebernommen["signal"]),
        )
    if indikator_typ == "bollinger":
        return bollinger(
            close, period=int(uebernommen["period"]), std=float(uebernommen["std"])
        )

    vorhanden = {"high": high, "low": low, "volume": volume}
    fehlt = [name for name in _BENOETIGTE_SPALTEN[indikator_typ] if vorhanden[name] is None]
    if fehlt:
        raise ValueError(
            f"„{indikator_typ}“ benötigt die Spalten {', '.join(fehlt)} "
            "der OHLC-Balken (in dieser Anfrage nicht übergeben)."
        )
    if indikator_typ == "atr":
        return {"atr": atr(high, low, close, int(uebernommen["period"]))}  # type: ignore[arg-type]
    if indikator_typ == "obv":
        return {"obv": obv(close, volume)}  # type: ignore[arg-type]
    if indikator_typ == "vwap":
        return {"vwap": vwap(high, low, close, volume)}  # type: ignore[arg-type]
    if indikator_typ == "stoch":
        return stoch(
            high,  # type: ignore[arg-type]
            low,  # type: ignore[arg-type]
            close,
            k=int(uebernommen["k"]),
            smooth=int(uebernommen["smooth"]),
            d=int(uebernommen["d"]),
        )
    if indikator_typ == "adx":
        return adx(high, low, close, int(uebernommen["period"]))  # type: ignore[arg-type]
    if indikator_typ == "cci":
        return {"cci": cci(high, low, close, int(uebernommen["period"]))}  # type: ignore[arg-type]
    if indikator_typ == "williamsr":
        return {
            "williams_r": williams_r(high, low, close, int(uebernommen["period"]))  # type: ignore[arg-type]
        }
    return ichimoku(
        high,  # type: ignore[arg-type]
        low,  # type: ignore[arg-type]
        close,
        tenkan=int(uebernommen["tenkan"]),
        kijun=int(uebernommen["kijun"]),
        senkou=int(uebernommen["senkou"]),
    )


def indikator_id(indikator_typ: str, params: dict[str, object]) -> str:
    """Stabile ID ``{typ}:{werte}`` (kanonische Parameter-Reihenfolge) — Identität fürs Frontend.

    Beispiele: ``sma:20``, ``rsi:14``, ``macd:12_26_9``, ``bollinger:20_2.0``,
    ``atr:14``, ``stoch:14_3_3``, ``ichimoku:9_26_52``; die parameterlosen
    Typen ``obv``/``vwap`` bleiben ohne Suffix. Der Router übergibt die
    **effektiven** Parameter (Defaults aufgefüllt,
    :func:`effektive_parameter`), damit die ID unabhängig davon, welche
    Parameter der Caller explizit schickte, dieselbe Konvention spiegelt wie
    das Frontend (``paneManager.ts``).
    """
    if params:
        reihenfolge = _ID_REIHENFOLGE.get(indikator_typ, tuple(sorted(params)))
        join = "_".join(str(params[key]) for key in reihenfolge if key in params)
        return f"{indikator_typ}:{join}"
    return indikator_typ
