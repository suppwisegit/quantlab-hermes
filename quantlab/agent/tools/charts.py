"""Chart-Tools (T14, PLAN §E 11–12): Chart-Artefakte erzeugen und „lesen“.

**Entscheidung (dokumentiert):** ``artifacts.project_id`` ist laut §D.1
``NOT NULL`` — es gibt bewusst **keine** globalen Artefakte. ``create_chart``
ohne Projekt-Kontext (globaler Chat) liefert daher als Tool-Ergebnis einen
deutschen, handlungsorientierten Fehler („Für Charts muss ein Projekt gewählt
sein“); der Run läuft damit weiter und der Agent kann den Nutzer um Projekt-
wahl bitten. ``read_chart`` funktioniert dagegen projekt-unabhängig, wenn ein
``symbol`` (statt ``artifact_id``) gegeben ist.

Indikator-Werte berechnet — gemäß E011 — ausschließlich das Backend
(:mod:`quantlab.services.indicators`); das Tool validiert die Indikator-Wünsche
mit ``effektive_parameter`` und speichert die **effektiven** Parameter im
Artefakt-Payload (ID-Konvention ``sma:20``, ``macd:12_26_9``).

Welle I (E103) hebt zwei alte Bewusst-Grenzen: ``get_indicators`` liefert
kompakte Indikator-Zusammenfassungen (letzte Werte, Lage-/Kreuzungs-Befund,
Extrema — bewusst keine Vollserien, Kontext-Budget) und
``manage_chart_drawings`` delegiert Zeichnungen auf
:mod:`quantlab.storage.repos.charts_repo` (T18-CRUD).
``save_analyse`` schreibt Analyse-Artefakte (``kind="analysis"``).
"""

from __future__ import annotations

import asyncio
import json
import re
from bisect import bisect_right
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, daten_fehler, pruefe_args, schema_von
from quantlab.agent.tools.kontext import aktives_projekt
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.services import data_service
from quantlab.services import indicators as indicators_mod
from quantlab.storage import parquet_store
from quantlab.storage.repos import charts_repo, projects_repo

_US_PER_TAG = parquet_store.US_PER_DAY


# ---------------------------------------------------------------------------
# 11) create_chart
# ---------------------------------------------------------------------------


class IndicatorSpecArgs(BaseModel):
    """Ein Indikator-Wunsch (Typ + optionale Parameter, Defaults laut Modul)."""

    type: Literal[
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
    ] = Field(
        description=(
            "Indikator-Typ: " + ", ".join(indicators_mod.INDICATOR_TYPES) + "."
        )
    )
    params: dict[str, int | float] = Field(
        default_factory=dict,
        description=(
            "Parameter je Typ (unvollständige Angaben werden mit Defaults "
            "aufgefüllt): sma/ema/rsi {\"period\": 20|20|14}, macd {\"fast\": 12, "
            "\"slow\": 26, \"signal\": 9}, bollinger {\"period\": 20, \"std\": 2}, "
            "atr/adx/cci/williamsr {\"period\": 14|14|20|14}, stoch {\"k\": 14, "
            "\"smooth\": 3, \"d\": 3}, ichimoku {\"tenkan\": 9, \"kijun\": 26, "
            "\"senkou\": 52}; obv/vwap sind parameterlos."
        ),
    )


class CreateChartArgs(BaseModel):
    """Argumente von ``create_chart`` (T34: Vergleich, Log, Adjust als Konfiguration)."""

    symbol: str = Field(min_length=1, description='Symbol-Kürzel, z. B. "SAP".')
    interval: str = Field(default="1d", description='yfinance-Intervall, z. B. "1d".')
    indicators: list[IndicatorSpecArgs] = Field(
        default_factory=list,
        description=(
            'Indikator-Wünsche, z. B. [{"type": "rsi"}] oder '
            '[{"type": "sma", "params": {"period": 50}}].'
        ),
    )
    compare_symbols: list[str] = Field(
        default_factory=list,
        description=(
            "Vergleichs-Symbole (max. 8, als zusätzliche Linien im Chart), z. B. "
            '["MSFT", "GOOGL"] — Performance-Vergleich mit festem Referenzpunkt '
            "(E034)."
        ),
    )
    log_scale: bool = Field(
        default=False,
        description="true = logarithmische Preisachse (sinnvoll bei langen Zeiträumen).",
    )
    adjust: Literal["raw", "adjusted"] = Field(
        default="raw",
        description='Kursbasis: "raw" (Rohkurse, Default) oder "adjusted" '
        "(Splits/Dividenden bereinigt, E022).",
    )
    title: str | None = Field(
        default=None, description='Optionaler Chart-Titel, z. B. "SAP mit RSI".'
    )


async def create_chart(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Erzeugt ein Chart-Artefakt im aktiven Projekt (live im Artefakte-Panel)."""
    args = pruefe_args(CreateChartArgs, roh_args, "create_chart")
    projekt_id = aktives_projekt()
    if projekt_id is None:
        # Entscheidung (siehe Modul-Docstring): Artefakte sind immer projekt-
        # gebunden (§D.1 NOT NULL) — ohne Projekt ist ein Chart-Artefakt nicht
        # adressierbar. Deutscher Fachfehler statt stiller Verweigerung.
        raise ToolFehler(
            "Für Charts muss ein Projekt gewählt sein — Artefakte werden immer "
            "in einem Projekt gespeichert. Bitte im Chat ein Projekt auswählen "
            "oder anlegen und den Wunsch anschließend wiederholen."
        )

    def _ausfuehren() -> dict[str, Any]:
        if not parquet_store.has_data(args.symbol, args.interval):
            raise ToolFehler(
                f"Für „{args.symbol}“ ({args.interval}) sind keine Daten gespeichert. "
                "(Hinweis: download_data kann die Daten zuerst laden.)"
            )
        # Vergleichs-Symbole: bereinigen (Duplikate, Haupt-Symbol), Cap 8 wie im
        # chartStore des Dashboards (T24, E034).
        vergleich: list[str] = []
        for kandidat in args.compare_symbols:
            kandidat = (kandidat or "").strip()
            if not kandidat or kandidat == args.symbol or kandidat in vergleich:
                continue
            if len(vergleich) >= 8:
                raise ToolFehler(
                    "Maximal 8 Vergleichs-Symbole pro Chart (Dashboard-Konvention "
                    "E034) — bitte die Liste kürzen."
                )
            vergleich.append(kandidat)
        specs: list[dict[str, Any]] = []
        for wunsch in args.indicators:
            try:
                effektiv = indicators_mod.effektive_parameter(
                    wunsch.type, dict(wunsch.params)
                )
            except ValueError as exc:
                raise ToolFehler(str(exc)) from exc
            # Effektive Parameter unverändert in den Payload (ID-Konvention
            # ``sma:20``/``macd:12_26_9`` — int bleibt int, wie im Dashboard).
            specs.append({"type": wunsch.type, "params": dict(effektiv)})
        titel = (
            args.title.strip()
            if args.title and args.title.strip()
            else f"{args.symbol} ({args.interval})"
        )
        artefakt = projects_repo.create_artifact(
            projekt_id,
            "chart",
            titel,
            {
                "symbol": args.symbol,
                "interval": args.interval,
                "indicators": specs,
                "compare_symbols": vergleich,
                "log_scale": args.log_scale,
                "adjust": args.adjust,
                "title": titel,
            },
        )
        return {
            "status": "erstellt",
            "artefakt_id": artefakt.id,
            "project_id": projekt_id,
            "titel": titel,
            "payload": artefakt.payload,
            "vergleichs_symbole": vergleich,
            "hinweis": (
                "Das Chart erscheint sofort im Artefakte-Panel des Projekts "
                "(Vergleichs-Symbole/Log-Skala/Adjust sind als Konfiguration "
                "hinterlegt). Zeichnungen (Trendlinie, Fibs, Text) legst du im "
                "Chat über manage_chart_drawings an oder direkt im Chart-UI."
            ),
        }

    return await asyncio.to_thread(_ausfuehren)


# ---------------------------------------------------------------------------
# 12) read_chart
# ---------------------------------------------------------------------------


class ReadChartArgs(BaseModel):
    """Argumente von ``read_chart`` — ``symbol`` **oder** ``artifact_id``."""

    symbol: str | None = Field(
        default=None, description='Symbol-Kürzel, z. B. "AAPL" (oder artifact_id angeben).'
    )
    artifact_id: int | None = Field(
        default=None, ge=1, description="ID eines Chart-Artefakts im aktiven Projekt."
    )
    interval: str | None = Field(
        default=None, description='Intervall, z. B. "1d" (Default des Artefakts bzw. "1d").'
    )


def _trend_und_vol(balken: pd.DataFrame) -> dict[str, Any]:
    """Trend-Statistik (Ø-Rendite, lineare Steigung) + annualisierte Volatilität."""
    closes = balken["close"].astype(float)
    renditen = closes.pct_change().dropna()
    if renditen.empty:
        return {
            "avg_rendite_pro_balken_prozent": None,
            "trend_linear_slope_prozent_pro_balken": None,
            "volatilitaet_annualisiert_prozent": None,
        }
    std = float(renditen.std(ddof=1))
    vol = std * float(np.sqrt(252.0)) * 100.0 if std > 0 else None
    x = np.arange(len(closes), dtype=float)
    y = closes.to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    mittel = float(closes.mean())
    return {
        "avg_rendite_pro_balken_prozent": round(float(renditen.mean()) * 100.0, 4),
        "trend_linear_slope_prozent_pro_balken": (
            round(slope / mittel * 100.0, 4) if mittel != 0 else None
        ),
        # ANNAHME: Annualisierung mit 252 Tagesperioden — bei Intraday-Intervallen
        # ist die Zahl als Größenordnung (per-bar-Hochrechnung) zu lesen.
        "volatilitaet_annualisiert_prozent": round(vol, 2) if vol is not None else None,
    }


async def read_chart(roh_args: dict[str, Any]) -> dict[str, Any]:
    """„Liest" einen Chart: Konfiguration + Daten-Zusammenfassung (Kurs, 52W, Trend, Vol)."""
    args = pruefe_args(ReadChartArgs, roh_args, "read_chart")
    projekt_id = aktives_projekt()

    def _konfiguration() -> tuple[str, str, str | None]:
        if args.artifact_id is not None:
            if projekt_id is None:
                raise ToolFehler(
                    "artifact_id benötigt ein gewähltes Projekt (Artefakte sind projektgebunden)."
                )
            artefakt = projects_repo.get_artifact(projekt_id, args.artifact_id)
            if artefakt is None:
                raise ToolFehler(
                    f"Artefakt {args.artifact_id} wurde im aktiven Projekt nicht gefunden."
                )
            if artefakt.kind != "chart":
                raise ToolFehler(
                    f"Artefakt {args.artifact_id} ist vom Typ „{artefakt.kind}“, nicht „chart“."
                )
            symbol = str(artefakt.payload.get("symbol") or "")
            if not symbol:
                raise ToolFehler(
                    f"Chart-Artefakt {args.artifact_id} trägt kein Symbol im Payload."
                )
            interval = str(artefakt.payload.get("interval") or "1d")
            titel = artefakt.title
            return symbol, interval, titel
        if args.symbol is None or not args.symbol.strip():
            raise ToolFehler(
                "Bitte „symbol“ (z. B. \"AAPL\") oder „artifact_id“ angeben."
            )
        return args.symbol.strip(), args.interval or "1d", None

    try:
        symbol, interval, titel = await asyncio.to_thread(_konfiguration)
        balken = await asyncio.to_thread(
            data_service.read_ohlcv, symbol, interval, None, None, 400
        )
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    if balken.empty:
        raise ToolFehler(f"Für „{symbol}“ ({interval}) liegen keine Balken vor.")

    ts_liste = balken["ts"].astype("int64").tolist()
    letzter_ts = int(ts_liste[-1])
    jahresgrenze = letzter_ts - 365 * _US_PER_TAG
    start_index = max(0, bisect_right(ts_liste, jahresgrenze) - 1)
    fenster = balken.iloc[start_index:]
    letzter = balken.iloc[-1]
    konfiguration: dict[str, Any] = {"symbol": symbol, "interval": interval}
    if titel is not None:
        konfiguration["titel"] = titel

    return {
        "konfiguration": konfiguration,
        "zusammenfassung": {
            "letzter_kurs": round(float(letzter["close"]), 4),
            "zeitpunkt": datetime.fromtimestamp(letzter_ts / 1_000_000, tz=UTC).isoformat(),
            "hoch_52w": round(float(fenster["high"].max()), 4),
            "tief_52w": round(float(fenster["low"].min()), 4),
            "anzahl_balken": int(len(balken)),
            **_trend_und_vol(balken),
        },
    }


# ---------------------------------------------------------------------------
# Neu (Welle I, E103): get_indicators — kompakte Indikator-Zusammenfassung
# ---------------------------------------------------------------------------

#: Ergebnis-Budget in Zeichen (DoD Welle I: „GESAMT unter ~6.000“ vor dem
#: 8.000-Zeichen-Kapping der Dispatch-Schicht).
_ZIEL_ZEICHEN = 6_000


class GetIndicatorsArgs(BaseModel):
    """Argumente von ``get_indicators`` (gleiche Indikator-Specs wie ``create_chart``)."""

    symbol: str = Field(min_length=1, description='Symbol-Kürzel, z. B. "SAP".')
    interval: str = Field(default="1d", description='yfinance-Intervall, z. B. "1d".')
    indicators: list[IndicatorSpecArgs] = Field(
        min_length=1,
        description=(
            'Indikator-Wünsche wie bei create_chart, z. B. '
            '[{"type": "sma", "params": {"period": 50}}, {"type": "rsi"}] '
            "(max. 8 pro Anfrage)."
        ),
    )


def _datum(ts_us: int) -> str:
    """int64 µs UTC → ISO-Datum (kompakt, Tagesauflösung)."""
    return datetime.fromtimestamp(ts_us / 1_000_000, tz=UTC).date().isoformat()


def _letzte(serie: pd.Series) -> float | None:
    """Letzter definierter Wert einer Indikator-Serie (``None`` bei nur-NaN)."""
    definiert = serie.dropna()
    if definiert.empty:
        return None
    return float(definiert.iloc[-1])


def _extrema(serie: pd.Series, ts_liste: list[int]) -> dict[str, Any] | None:
    """Höchstes/Tiefstes der definierten Serie mit Balken-Datum (``None`` bei nur-NaN)."""
    definiert = serie.dropna()
    if definiert.empty:
        return None
    hoch_pos = int(definiert.index[int(definiert.values.argmax())])
    tief_pos = int(definiert.index[int(definiert.values.argmin())])
    return {
        "hoch": {
            "wert": round(float(definiert.max()), 4),
            "datum": _datum(ts_liste[hoch_pos]),
        },
        "tief": {
            "wert": round(float(definiert.min()), 4),
            "datum": _datum(ts_liste[tief_pos]),
        },
    }


def _zu_wenig(typ: str, balken: int) -> str:
    """Deutscher Hinweis, wenn die Historie kürzer als die Indikator-Lücke ist."""
    return (
        f"Zu wenig Balken für {typ}: die vorhandenen {balken} Balken reichen "
        "nicht aus — die Serie enthält noch keinen gültigen Wert (mehr Historie "
        "laden, z. B. mit download_data und einem früheren Startdatum)."
    )


def _zahl_text(wert: float) -> str:
    """Kompakte Zahl im Befund-Text (4 signifikante Stellen)."""
    return f"{wert:.4g}"


def _letzter_cross(hist: pd.Series) -> int | None:
    """Balken seit der letzten Vorzeichenänderung des Histogramms (``None`` = nie)."""
    definiert = hist.dropna()
    if len(definiert) < 2:
        return None
    vorzeichen = np.sign(definiert.to_numpy(dtype="float64"))
    letzter = len(vorzeichen) - 1
    for i in range(letzter, 0, -1):
        if vorzeichen[i] != 0 and vorzeichen[i - 1] != 0 and vorzeichen[i] != vorzeichen[i - 1]:
            return letzter - i
    return None


def _befund_text(
    typ: str,
    effektiv: dict[str, float],
    serien: dict[str, pd.Series],
    kurs: float,
    balken: int,
) -> str:
    """Pragmatischer deutscher Lage-/Kreuzungs-Befund je Indikator-Typ."""
    if typ in ("sma", "ema"):
        wert = _letzte(serien[typ])
        if wert is None:
            return _zu_wenig(typ, balken)
        abstand = (kurs - wert) / wert * 100.0 if wert != 0 else 0.0
        seite = "über" if kurs > wert else "unter"
        return (
            f"Kurs {seite} {typ.upper()}{int(effektiv['period'])} "
            f"({_zahl_text(kurs)} vs. {_zahl_text(wert)}, {abstand:+.1f} %)."
        )
    if typ == "rsi":
        wert = _letzte(serien["rsi"])
        if wert is None:
            return _zu_wenig(typ, balken)
        if wert >= 70:
            zone = "überkauft (≥ 70)"
        elif wert <= 30:
            zone = "überverkauft (≤ 30)"
        else:
            zone = "neutraler Bereich"
        return f"RSI{int(effektiv['period'])} bei {_zahl_text(wert)} — {zone}."
    if typ == "macd":
        histogramm = _letzte(serien["histogram"])
        if histogramm is None:
            return _zu_wenig(typ, balken)
        vorzeichen = "positiv (MACD über Signal)" if histogramm > 0 else (
            "negativ (MACD unter Signal)" if histogramm < 0 else "flach (auf der Signal-Linie)"
        )
        abstand = _letzter_cross(serien["histogram"])
        kreuz = (
            f", letzter Cross vor {abstand} Balken" if abstand is not None and abstand > 0
            else ", frischer Cross im letzten Balken" if abstand is not None
            else ""
        )
        return f"MACD-Histogramm {vorzeichen}{kreuz}."
    if typ == "bollinger":
        oberes = _letzte(serien["upper"])
        unteres = _letzte(serien["lower"])
        if oberes is None or unteres is None:
            return _zu_wenig(typ, balken)
        spanne = oberes - unteres
        if spanne <= 0:
            lage = "Bänder extrem eng (keine Lageaussage)"
        else:
            prozent_b = (kurs - unteres) / spanne
            if prozent_b > 1.0:
                lage = "Kurs über dem oberen Band"
            elif prozent_b < 0.0:
                lage = "Kurs unter dem unteren Band"
            elif prozent_b >= 2.0 / 3.0:
                lage = "Kurs im oberen Drittel der Bänder"
            elif prozent_b <= 1.0 / 3.0:
                lage = "Kurs im unteren Drittel der Bänder"
            else:
                lage = "Kurs im mittleren Drittel der Bänder"
        return f"{lage} (oberes Band {_zahl_text(oberes)}, unteres {_zahl_text(unteres)})."
    if typ == "atr":
        wert = _letzte(serien["atr"])
        if wert is None:
            return _zu_wenig(typ, balken)
        anteil = wert / kurs * 100.0 if kurs != 0 else 0.0
        return (
            f"ATR{int(effektiv['period'])} {_zahl_text(wert)} — {anteil:.1f} % des "
            "Kurses (durchschnittliche Balken-Spannweite)."
        )
    if typ == "obv":
        serie = serien["obv"]
        wert = _letzte(serie)
        if wert is None:
            return _zu_wenig(typ, balken)
        definiert = serie.dropna()
        referenz = float(definiert.iloc[max(0, len(definiert) - 1 - 20)])
        trend = (
            "steigend" if wert > referenz else "fallend" if wert < referenz else "seitlich"
        )
        return f"OBV über die letzten 20 Balken {trend} ({_zahl_text(wert)})."
    if typ == "vwap":
        wert = _letzte(serien["vwap"])
        if wert is None:
            return _zu_wenig(typ, balken)
        seite = "über" if kurs >= wert else "unter"
        return f"Kurs {seite} VWAP ({_zahl_text(kurs)} vs. {_zahl_text(wert)})."
    if typ == "stoch":
        k_wert = _letzte(serien["k"])
        d_wert = _letzte(serien["d"])
        if k_wert is None:
            return _zu_wenig(typ, balken)
        if k_wert >= 80:
            zone = "überkauft (≥ 80)"
        elif k_wert <= 20:
            zone = "überverkauft (≤ 20)"
        else:
            zone = "neutraler Bereich"
        if d_wert is None:
            verhaeltnis = ""
        else:
            verhaeltnis = (
                ", %K über %D" if k_wert > d_wert
                else ", %K unter %D" if k_wert < d_wert
                else ", %K auf %D"
            )
        return (
            f"Stochastik %K {_zahl_text(k_wert)} / %D "
            f"{'–' if d_wert is None else _zahl_text(d_wert)} — {zone}{verhaeltnis}."
        )
    if typ == "adx":
        wert = _letzte(serien["adx"])
        if wert is None:
            return _zu_wenig(typ, balken)
        if wert >= 25:
            staerke = "starker Trend"
        elif wert >= 20:
            staerke = "beginnender Trend"
        else:
            staerke = "kein richtungsgebender Trend (ADX < 20)"
        richtung = ""
        plus = _letzte(serien["plus_di"])
        minus = _letzte(serien["minus_di"])
        if plus is not None and minus is not None and plus != minus:
            vorzeichen = ">" if plus > minus else "<"
            folge = "Aufwärtsbewegung dominiert" if plus > minus else "Abwärtsbewegung dominiert"
            richtung = f", +DI {vorzeichen} −DI ({folge})"
        return f"ADX{_int_text(effektiv['period'])} {_zahl_text(wert)} — {staerke}{richtung}."
    if typ == "cci":
        wert = _letzte(serien["cci"])
        if wert is None:
            return _zu_wenig(typ, balken)
        if wert > 100:
            zone = "Überlinie +100 überschritten (starke Aufwärtsbewegung)"
        elif wert < -100:
            zone = "Unterlinie −100 unterschritten (starke Abwärtsbewegung)"
        else:
            zone = "neutraler Bereich (−100 bis +100)"
        return f"CCI{int(effektiv['period'])} bei {_zahl_text(wert)} — {zone}."
    if typ == "williamsr":
        wert = _letzte(serien["williams_r"])
        if wert is None:
            return _zu_wenig(typ, balken)
        if wert > -20:
            zone = "überkauft (> −20)"
        elif wert < -80:
            zone = "überverkauft (< −80)"
        else:
            zone = "neutraler Bereich"
        return f"Williams %R bei {_zahl_text(wert)} — {zone}."
    # verbleibender Typ: ichimoku
    senkou_a = _letzte(serien["senkou_a"])
    senkou_b = _letzte(serien["senkou_b"])
    if senkou_a is None or senkou_b is None:
        return _zu_wenig(typ, balken)
    oben, unten = max(senkou_a, senkou_b), min(senkou_a, senkou_b)
    if kurs > oben:
        lage = "Kurs über der Wolke (bullisch)"
    elif kurs < unten:
        lage = "Kurs unter der Wolke (bärisch)"
    else:
        lage = "Kurs in der Wolke"
    return (
        f"{lage} (Senkou A {_zahl_text(senkou_a)}, Senkou B {_zahl_text(senkou_b)})."
    )


def _int_text(wert: float) -> str:
    """Periode als ganze Zahl im Text (effektive Parameter tragen int-Werte)."""
    return str(int(wert))


def _indikator_zusammenfassung(
    typ: str,
    effektiv: dict[str, float],
    identitaet: str,
    serien: dict[str, pd.Series],
    schluss: pd.Series,
    ts_liste: list[int],
) -> dict[str, Any]:
    """Eine Serie → kompakte Zusammenfassung (letzte Werte, Befund, Extrema)."""
    letzte_werte = {
        schluessel: (
            None if (wert := _letzte(serie)) is None else round(wert, 4)
        )
        for schluessel, serie in serien.items()
    }
    return {
        "type": typ,
        "id": identitaet,
        "letzte_werte": letzte_werte,
        "befund": _befund_text(typ, effektiv, serien, float(schluss.iloc[-1]), len(ts_liste)),
        "extrema": {
            schluessel: extremum
            for schluessel, serie in serien.items()
            if (extremum := _extrema(serie, ts_liste)) is not None
        },
    }


async def get_indicators(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Berechnet Indikatoren und liefert eine kompakte Zusammenfassung (E103)."""
    args = pruefe_args(GetIndicatorsArgs, roh_args, "get_indicators")
    if len(args.indicators) > indicators_mod.MAX_INDICATORS_PER_REQUEST:
        raise ToolFehler(
            f"Maximal {indicators_mod.MAX_INDICATORS_PER_REQUEST} Indikatoren pro "
            "Anfrage (Kontext-Budget) — bitte die Liste kürzen."
        )
    specs: list[tuple[str, dict[str, float], str]] = []
    for wunsch in args.indicators:
        try:
            effektiv = indicators_mod.effektive_parameter(wunsch.type, dict(wunsch.params))
        except (ValueError, TypeError) as exc:
            raise ToolFehler(str(exc)) from exc
        specs.append(
            (wunsch.type, effektiv, indicators_mod.indikator_id(wunsch.type, effektiv))
        )

    def _ausfuehren() -> dict[str, Any]:
        # Gleicher Read-Pfad wie read_chart (letzte 400 Balken reichen für die
        # Zusammenfassung — Vollserien bleiben beim REST-Endpoint /data/indicators).
        balken = data_service.read_ohlcv(args.symbol, args.interval, None, None, 400)
        if balken.empty:
            raise ToolFehler(
                f"Für „{args.symbol}“ ({args.interval}) liegen keine Balken vor."
            )
        ts_liste = [int(wert) for wert in balken["ts"]]
        schluss = balken["close"].reset_index(drop=True)
        hoch = balken["high"].reset_index(drop=True)
        tief = balken["low"].reset_index(drop=True)
        volumen = balken["volume"].reset_index(drop=True)
        indikatoren = [
            _indikator_zusammenfassung(
                typ,
                effektiv,
                identitaet,
                indicators_mod.berechne_indikator(
                    schluss, typ, effektiv, high=hoch, low=tief, volume=volumen
                ),
                schluss,
                ts_liste,
            )
            for typ, effektiv, identitaet in specs
        ]
        ergebnis: dict[str, Any] = {
            "symbol": args.symbol.strip(),
            "interval": args.interval,
            "anzahl_balken": len(balken),
            "zeitraum": {"von": _datum(ts_liste[0]), "bis": _datum(ts_liste[-1])},
            "indikatoren": indikatoren,
        }
        # Budget-Schutz (DoD: gesamt < ~6.000 Zeichen): notfalls fallen die
        # Extrema weg, die Befunde/letzten Werte bleiben immer erhalten.
        if len(json.dumps(ergebnis, ensure_ascii=False)) > _ZIEL_ZEICHEN:
            for zeile in indikatoren:
                zeile.pop("extrema", None)
            ergebnis["hinweis"] = (
                "Extrema wurden der Kompaktheit wegen weggelassen (Ergebnis-Budget); "
                "letzte Werte und Befunde sind vollständig."
            )
        return ergebnis

    try:
        return await asyncio.to_thread(_ausfuehren)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc


# ---------------------------------------------------------------------------
# Neu (Welle I, E103): manage_chart_drawings — Zeichnungs-CRUD über das Repo
# ---------------------------------------------------------------------------

#: Spiegel von ``api/routers/charts.CHART_KEY_MUSTER`` (E034): Format
#: ``"<SYMBOL>|<Intervall>"`` mit der validate_symbol-Grammatik (inkl. ^-Indizes,
#: =F/=X, max. 24 Zeichen). Bewusst lokal definiert — die Tool-Schicht importiert
#: keine API-Router.
CHART_KEY_MUSTER = r"^[A-Za-z0-9_.^=-]{1,24}\|(1d|1h|15m|5m|1m)$"

#: Obergrenze Ankerpunkte je Zeichnung — Spiegel von ``api/routers/charts.MAX_PUNKTE``
#: (ANNAHME wie dort: 2000 sind großzügig über alle Tools).
_MAX_PUNKTE = 2000

#: Obergrenze für ``time`` (wie Router): 2^53 − 1 = Number.MAX_SAFE_INTEGER —
#: oberhalb verliert float64 (JS-Client) Präzision.
_TIME_MAX = 2**53 - 1


class AnkerArgs(BaseModel):
    """Ein Ankerpunkt (E006): ``time`` = int64 µs UTC, ``price`` = Rohkurs."""

    time: int = Field(ge=0, le=_TIME_MAX, description="Zeitstempel in µs UTC (E006).")
    price: float = Field(allow_inf_nan=False, description="Preis (Rohkurs-Skala).")


class ManageChartDrawingsArgs(BaseModel):
    """Argumente von ``manage_chart_drawings`` (aktion=liste|erstellen|loeschen)."""

    aktion: Literal["liste", "erstellen", "loeschen"] = Field(
        description=(
            '"liste" (Zeichnungen je chart_key), "erstellen" (braucht symbol, '
            "typ, punkte) oder \"loeschen\" (braucht drawing_id — oder chart_key "
            'plus "alle": true — und in beiden Fällen bestaetigung=true).'
        ),
    )
    chart_key: str | None = Field(
        default=None,
        max_length=100,
        description='Chart-Schlüssel "<SYMBOL>|<Intervall>" (z. B. "AAPL|1d") — Pflicht bei liste.',
    )
    symbol: str | None = Field(
        default=None,
        min_length=1,
        description=(
            'Symbol (Pflicht bei erstellen), z. B. "AAPL" — der chart_key wird daraus gebaut.'
        ),
    )
    interval: Literal["1d", "1h", "15m", "5m", "1m"] = Field(
        default="1d", description="Intervall (erstellen), z. B. \"1d\"."
    )
    typ: Literal[
        "trendline",
        "ray",
        "hline",
        "vline",
        "fib_retracement",
        "rectangle",
        "ellipse",
        "brush",
        "text",
    ] | None = Field(
        default=None,
        description=(
            'Zeichentool (Pflicht bei erstellen), z. B. "trendline" oder "fib_retracement".'
        ),
    )
    punkte: list[AnkerArgs] | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_PUNKTE,
        description=(
            "Ankerpunkte (Pflicht bei erstellen), z. B. "
            '[{"time": 1704067200000000, "price": 182.5}] — Zeitstempel in µs UTC.'
        ),
    )
    style: dict[str, Any] | None = Field(
        default=None,
        description='Optionales Styling, z. B. {"color": "#d97757", "width": 2}.',
    )
    drawing_id: int | None = Field(
        default=None,
        ge=1,
        description="ID der Zeichnung (loeschen einzeln), z. B. 7.",
    )
    alle: bool = Field(
        default=False,
        description=(
            'Nur loeschen: true = alle Zeichnungen des chart_key löschen (statt drawing_id).'
        ),
    )
    bestaetigung: bool = Field(
        default=False,
        description='Pflicht true bei loeschen (Sicherheitsabfrage — wie depot-loeschen).',
    )


def _valider_chart_key(chart_key: str | None, zweck: str) -> str:
    """chart_key-Format prüfen (Spiegel Router-Muster) — deutscher Fehler sonst."""
    if chart_key is None or not chart_key.strip():
        raise ToolFehler(
            f"„{zweck}“ benötigt einen chart_key im Format "
            '"<SYMBOL>|<Intervall>" (z. B. "AAPL|1d") — bitte zuerst '
            "aktion=liste bzw. das Chart im Dashboard wählen."
        )
    key = chart_key.strip()
    if re.fullmatch(CHART_KEY_MUSTER, key) is None:
        raise ToolFehler(
            f"Ungültiger chart_key „{key}“ — erwartet wird "
            '"<SYMBOL>|<Intervall>" mit Intervall 1d|1h|15m|5m|1m '
            '(z. B. "AAPL|1d").'
        )
    return key


def _drawing_als_dict(zeichnung: charts_repo.ChartDrawing) -> dict[str, Any]:
    """Zeichnung als kompakt-lesbares Ergebnis-Dokument.

    MF-I (REV): Die Punkte-Liste wird bewusst NICHT komplett gespiegelt —
    eine erlaubte 2.000-Punkte-Zeichnung sprengt sonst das 8.000-Z-
    Dispatch-Kapping und der Agent erhielte mitten im JSON abgeschnittenen
    Müll. Stattdessen Anzahl + erster/letzter Anker; die vollständigen
    Punkte liegen im Repo/Chart-UI.
    """
    punkte = zeichnung.points
    anker = (
        {"erster": punkte[0], "letzter": punkte[-1]} if punkte else None
    )
    return {
        "id": zeichnung.id,
        "chart_key": zeichnung.chart_key,
        "tool": zeichnung.tool,
        "anzahl_punkte": len(punkte),
        "anker": anker,
        "style": zeichnung.style,
        "erstellt": zeichnung.created_at,
        "aktualisiert": zeichnung.updated_at,
    }


def _bestands_warnung(
    symbol: str, interval: str, punkte: list[dict[str, Any]]
) -> str | None:
    """Anker-ts gegen das Bestandsfenster plausibilisieren — Warnung, kein Hartfehler.

    Kein Bestand (oder Bestand lesbar leer) ist ebenfalls nur eine Warnung:
    Zeichnungen sind chart_key-gebunden, nicht Bestand-gebunden.
    """
    try:
        balken = data_service.read_ohlcv(symbol, interval)
    except ProviderError:
        return (
            "Es existiert kein Datenbestand für dieses Symbol/Intervall — die "
            "Zeitstempel konnten nicht gegen Kursdaten plausibilisiert werden."
        )
    if balken.empty:
        return (
            "Der Datenbestand für dieses Symbol/Intervall ist leer — die "
            "Zeitstempel konnten nicht gegen Kursdaten plausibilisiert werden."
        )
    erste, letzte = int(balken["ts"].iloc[0]), int(balken["ts"].iloc[-1])
    draussen = sum(1 for punkt in punkte if not (erste <= punkt["time"] <= letzte))
    if draussen == 0:
        return None
    return (
        f"{draussen} von {len(punkte)} Ankerpunkten liegen außerhalb des "
        f"Datenbestands-Fensters ({_datum(erste)} bis {_datum(letzte)}) — die "
        "Zeichnung wurde trotzdem angelegt."
    )


async def manage_chart_drawings(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Listet, erstellt oder löscht Chart-Zeichnungen (Delegation auf charts_repo)."""
    args = pruefe_args(ManageChartDrawingsArgs, roh_args, "manage_chart_drawings")

    def _ausfuehren() -> dict[str, Any]:
        if args.aktion == "liste":
            key = _valider_chart_key(args.chart_key, "aktion=liste")
            zeichnungen = charts_repo.list_drawings(key)
            return {
                "aktion": "liste",
                "chart_key": key,
                "anzahl": len(zeichnungen),
                "zeichnungen": [_drawing_als_dict(z) for z in zeichnungen],
                "hinweis": "Zeichnungen erscheinen im Chart-UI beim nächsten Laden des Charts.",
            }

        if args.aktion == "erstellen":
            if not (args.symbol or "").strip():
                raise ToolFehler(
                    '„erstellen“ benötigt ein "symbol" (z. B. "AAPL") — daraus '
                    "wird der chart_key gebaut."
                )
            if args.typ is None:
                raise ToolFehler(
                    "„erstellen“ benötigt einen \"typ\" — gültige Zeichentools: "
                    f"{', '.join(sorted(charts_repo.DRAWING_TOOLS))}."
                )
            if not args.punkte:
                raise ToolFehler(
                    '„erstellen“ benötigt mindestens einen Ankerpunkt in "punkte" '
                    '([{"time": <µs UTC>, "price": <Kurs>}, …]).'
                )
            symbol = validate_symbol(args.symbol)
            key = f"{symbol}|{args.interval}"
            if re.fullmatch(CHART_KEY_MUSTER, key) is None:
                raise ToolFehler(
                    f"Ungültiger chart_key „{key}“ — das Symbol passt nicht in "
                    'die Chart-Key-Grammatik ("<SYMBOL>|<Intervall>").'
                )
            punkte = [{"time": punkt.time, "price": punkt.price} for punkt in args.punkte]
            warnung = _bestands_warnung(symbol, args.interval, punkte)
            zeichnung = charts_repo.create_drawing(key, args.typ, punkte, style=args.style)
            ergebnis: dict[str, Any] = {
                "aktion": "erstellen",
                "zeichnung": _drawing_als_dict(zeichnung),
                "hinweis": "Die Zeichnung erscheint im Chart-UI beim nächsten Laden des Charts.",
            }
            if warnung is not None:
                ergebnis["warnung"] = warnung
            return ergebnis

        # aktion == "loeschen"
        if not args.bestaetigung:
            raise ToolFehler(
                "Löschen erfordert die Sicherheitsabfrage — bitte "
                '"bestaetigung": true mitsenden.'
            )
        if args.drawing_id is not None:
            existierend = charts_repo.get_drawing(args.drawing_id)
            if existierend is None:
                raise ToolFehler(
                    f"Zeichnung mit ID {args.drawing_id} nicht gefunden — bitte "
                    "zuerst aktion=liste mit dem passenden chart_key ausführen."
                )
            charts_repo.delete_drawing(args.drawing_id)
            return {
                "aktion": "loeschen",
                "geloescht": 1,
                "zeichnung": _drawing_als_dict(existierend),
                "hinweis": "Die Zeichnung wurde gelöscht.",
            }
        if args.alle:
            key = _valider_chart_key(args.chart_key, 'aktion=loeschen mit "alle": true')
            anzahl = charts_repo.delete_drawings(key)
            return {
                "aktion": "loeschen",
                "chart_key": key,
                "geloescht": anzahl,
                "hinweis": f"{anzahl} Zeichnung(en) von {key} gelöscht.",
            }
        raise ToolFehler(
            '„loeschen“ benötigt eine "drawing_id" — oder "chart_key" plus '
            '"alle": true, um alle Zeichnungen eines Charts zu löschen.'
        )

    try:
        return await asyncio.to_thread(_ausfuehren)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc


# ---------------------------------------------------------------------------
# Neu (Welle I, E103): save_analyse — Analyse-Artefakt (kind="analysis")
# ---------------------------------------------------------------------------


class SaveAnalyseArgs(BaseModel):
    """Argumente von ``save_analyse`` (Muster save_report, aber ohne Empfehlung)."""

    project_id: int | None = Field(
        default=None,
        ge=1,
        description="Ziel-Projekt (im Chat gewähltes Projekt wird automatisch verwendet).",
    )
    title: str = Field(
        min_length=1,
        max_length=200,
        description='Titel der Analyse, z. B. "SAP Chart-Analyse KW 39".',
    )
    content_md: str = Field(
        min_length=1,
        max_length=100_000,
        description="Vollständige Analyse als Markdown.",
    )
    symbols: list[str] | None = Field(
        default=None,
        description='Betreffene Symbole, z. B. ["SAP"] oder ["AAPL", "MSFT"].',
    )
    rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Optionale Bewertung 1–5 (1 = sehr schlecht, 5 = sehr attraktiv).",
    )


def _validierte_symbole(symbole: list[str] | None) -> list[str]:
    """Symbole bereinigen (Leerraum, Duplikate) und gegen validate_symbol prüfen."""
    if not symbole:
        return []
    gereinigt: list[str] = []
    for roh in symbole:
        kandidat = (roh or "").strip()
        if not kandidat:
            continue
        try:
            geprueft = validate_symbol(kandidat)
        except ProviderError as exc:
            raise ToolFehler(
                f'Ungültiges Symbol „{kandidat}“ in "symbols": {exc.message}'
            ) from exc
        if geprueft not in gereinigt:
            gereinigt.append(geprueft)
    return gereinigt


async def save_analyse(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Speichert eine Analyse (Markdown) als Artefakt kind=analysis im Projekt."""
    args = pruefe_args(SaveAnalyseArgs, roh_args, "save_analyse")
    projekt_id = args.project_id if args.project_id is not None else aktives_projekt()
    if projekt_id is None:
        raise ToolFehler(
            "Für Analysen muss ein Projekt gewählt sein — Artefakte werden immer "
            "in einem Projekt gespeichert. Bitte eine project_id angeben oder im "
            "Chat ein Projekt auswählen und den Wunsch erneut senden."
        )
    symbole = _validierte_symbole(args.symbols)

    def _speichern() -> dict[str, Any]:
        if projects_repo.get_project(projekt_id) is None:
            raise ToolFehler(f"Projekt mit ID {projekt_id} nicht gefunden.")
        payload: dict[str, Any] = {"content_md": args.content_md}
        if symbole:
            payload["symbols"] = symbole
        if args.rating is not None:
            payload["rating"] = args.rating
        artefakt = projects_repo.create_artifact(
            projekt_id, "analysis", args.title.strip(), payload
        )
        return {
            "status": "gespeichert",
            "artefakt_id": artefakt.id,
            "project_id": projekt_id,
            "titel": artefakt.title,
            "hinweis": "Die Analyse erscheint im Artefakte-Panel des Projekts.",
        }

    return await asyncio.to_thread(_speichern)


# ---------------------------------------------------------------------------
# Registrierung (§E-Reihenfolge 11–12 + Welle I/E103)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="create_chart",
            description=(
                "Erzeugt ein Chart-Artefakt im aktiven Projekt — es erscheint sofort "
                "live im Artefakte-Panel (Kerzen + Volumen + Indikatoren; optional "
                "Vergleichs-Symbole, Log-Skala, adjustierte Kurse als Konfiguration; "
                "Zeichnungen über manage_chart_drawings bzw. Chart-UI). Benötigt ein "
                "gewähltes Projekt. "
                'Beispiel: {"symbol": "SAP", "interval": "1d", '
                '"indicators": [{"type": "rsi"}], "compare_symbols": ["AAPL"], '
                '"log_scale": true, "title": "SAP vs. AAPL"}.'
            ),
            parameters=schema_von(CreateChartArgs),
            handler=create_chart,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="read_chart",
            description=(
                "Liest einen Chart aus: Konfiguration plus Daten-Zusammenfassung "
                "(letzter Kurs, 52-Wochen-Hoch/Tief, Trend-Statistik, annualisierte "
                "Volatilität). Beispiel: {\"symbol\": \"AAPL\"} oder "
                '{"artifact_id": 3}.'
            ),
            parameters=schema_von(ReadChartArgs),
            handler=read_chart,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="get_indicators",
            description=(
                "Berechnet technische Indikatoren für ein gespeichertes Symbol und "
                "liefert eine kompakte Zusammenfassung je Indikator (letzte Werte, "
                "deutscher Lage-/Kreuzungs-Befund, Extrema mit Datum) statt "
                "Vollserien. Beispiel: {\"symbol\": \"SAP\", \"interval\": \"1d\", "
                '"indicators": [{"type": "sma", "params": {"period": 50}}, '
                '{"type": "rsi"}]}.'
            ),
            parameters=schema_von(GetIndicatorsArgs),
            handler=get_indicators,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="manage_chart_drawings",
            description=(
                "Verwaltet Chart-Zeichnungen (Trendlinie, Ray, horizontale/vertikale "
                "Linie, Fib-Retracement, Rechteck, Ellipse, Brush, Text): auflisten "
                "je chart_key, anlegen (Symbol + Intervall + Ankerpunkte in µs UTC) "
                "oder löschen (einzeln per drawing_id oder alle je chart_key — "
                "jeweils mit bestaetigung=true). "
                'Beispiel: {"aktion": "erstellen", "symbol": "SAP", "interval": '
                '"1d", "typ": "trendline", "punkte": [{"time": 1704067200000000, '
                '"price": 118.0}, {"time": 1706745600000000, "price": 125.5}]}.'
            ),
            parameters=schema_von(ManageChartDrawingsArgs),
            handler=manage_chart_drawings,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="save_analyse",
            description=(
                "Speichert eine ausformulierte Analyse (Markdown) als Artefakt im "
                "Projekt — sie erscheint sofort im Artefakte-Panel; Symbole und "
                "eine Bewertung 1–5 sind optional. "
                'Beispiel: {"title": "SAP Chart-Analyse", "content_md": "# SAP …", '
                '"symbols": ["SAP"], "rating": 4}.'
            ),
            parameters=schema_von(SaveAnalyseArgs),
            handler=save_analyse,
        )
    )


_registrieren()
