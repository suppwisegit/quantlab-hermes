"""Screener-Service (T32, E027/E033): Regel-Engine über das lokale Universum.

Der Screener läuft **ausschließlich lokal** (Parquet/SQLite, E027) — nie
on-demand auf Netzdaten. Universum = alle Registry-fremd gelisteten Symbole
mit 1d-Bestand (``bar_meta.row_count > 0``).

**Regel-Modell (§T32, Persistenzvertrag screener_repo/E027):** eine Regel ist
``{"feld": str, "op": str, "wert": Zahl | [min, max]}``; Regeln sind
AND-verknüpft, Operatoren sind ``gt`` (größer), ``lt`` (kleiner) und
``between`` (Intervall inklusive). Unbekanntes Feld/Op bzw. sinnwidrige Werte
→ :class:`~quantlab.providers.base.ProviderError` 422 (deutsch, Router mappt
1:1 in den Fehlervertrag).

**Feld-Konventionen (eingefroren, Goldwerte in tests/test_screener.py):**

- ``perf_1w`` / ``perf_1m`` / ``perf_3m`` / ``perf_1y`` — Performance über
  5/21/63/252 Tages-Balken als **Bruchzahl** (0.05 = +5 %), Basis
  ``close``/``close[-1-n] - 1`` (Preisberechnung ``adjust="adjusted"``, E022 —
  Splits/Dividenden dürfen Performance nicht verzerren). Symbol jünger als das
  Fenster → Wert ``None`` → die Zeile fällt bei einer solchen Regel raus
  (dokumentierte Semantik, DoD e).
- ``rsi_14`` — Wilder-RSI (Periode 14) über :mod:`quantlab.services.indicators`
  (genau ein Ort für Indikator-Werte, E011), letzter Wert.
- ``ann_vol`` — annualisierte Volatilität der Tagesrenditen der letzten
  höchstens 252 Balken: ``std(ddof=1) · √252`` (Bruchzahl, 0.4 = 40 %).
- ``sma50_ratio`` — ``close / SMA(50)`` (Verhältnis, 1.03 = 3 % über dem
  Durchschnitt); ``None`` solange die SMA-Lücke läuft.
- ``last_price`` — letzter (adjustierter) Schlusskurs.
- ``change_pct`` — Tagesveränderung in **Prozent** (Konvention T22-Quotes:
  ``(last/prev - 1) · 100``).
- Fundamentale Felder (nur berechenbar, wenn Fundamentals-Bestand existiert —
  sonst ``None`` und die Zeile fällt bei einer solchen Regel raus):
  ``pe`` (KGV = letzter Kurs / verwässertes EPS; EPS ≤ 0 → ``None``),
  ``market_cap`` (Kurs × ausgegebene Aktien), ``div_yield`` (Summe der
  Dividenden der letzten 365 Tage / Kurs, Bruchzahl) sowie ``netto_marge``,
  ``roe`` und ``umsatzwachstum_yoy`` aus
  :func:`quantlab.services.fundamentals.berechne_ratios`.
- Planspiel-Felder (Welle H3/E095, nur berechenbar mit ``psb_kennzahlen``-
  Bestand für die ISIN des Symbols — sonst ``None`` wie oben): ``kgv``
  (KGV aus der VWD-Sektion ``getKennzahlen``, Verhältniszahl),
  ``div_rendite`` (VWD-Dividendenrendite, Bruchzahl — der Snippet-%-Wert
  wird durch 100 geteilt, Konvention wie ``div_yield``) und
  ``analysten_upside`` (``(kursziel − letzter Kurs) / letzter Kurs`` als
  Bruchzahl; Kursziel aus ``getAnalystenempfehlungen``, letzter Kurs best
  effort aus dem Parquet-Bestand **der ISIN** — ``1d``, egal welche Quelle,
  ohne Bestand nicht berechenbar). Die ISIN eines Symbols ergibt sich aus
  (1) der ISIN-Form des Symbols selbst (E093: ISIN als QuantLab-Symbol),
  (2) der Registry-Brücke ``symbols.isin`` (E091) oder (3) dem
  ``psb_instrumente``-Eintrag mit ``yahoo_kuerzel == symbol``. Der Screener
  liest ausschließlich lokale Bestände (E027) — es gibt hier keinen
  frisch-Abruf; der Cache wird über ``GET /api/planspiel/kennzahlen`` bzw.
  das Tool ``planspiel_kennzahlen`` gefüllt.

**S&P-500-Bulk-Import (E027, opt-in):** :func:`lade_sp500_liste` lädt die
Wikipedia-Liste (httpx, Transport injizierbar für httpx.MockTransport-Tests)
und :func:`parse_sp500_html` friert den Tabellen-Pars (Spalte „Symbol“ +
„Security“) gegen die Fixture ein. Der Abruf ist ein **Live-Fixpunkt**: ändert
Wikipedia die Struktur, liefert der Parser zu wenig Zeilen → 502 (deutsch),
nie ein stiller Teilimport.
"""

from __future__ import annotations

import html as html_mod
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pandas as pd

from quantlab.providers.base import ProviderError
from quantlab.services import data_service, indicators
from quantlab.services import fundamentals as fundamentals_service
from quantlab.storage import parquet_store
from quantlab.storage.repos import psb_repo, symbols_repo

# ---------------------------------------------------------------------------
# Feld- und Operator-Katalog
# ---------------------------------------------------------------------------

#: Performance-Felder → Rückblick in Tages-Balken (Handels Tage, §T32).
PERF_FENSTER: dict[str, int] = {
    "perf_1w": 5,
    "perf_1m": 21,
    "perf_3m": 63,
    "perf_1y": 252,
}

#: Fundamentale Felder (nur mit Fundamentals-Bestand berechenbar).
FUNDAMENTAL_FELDE: frozenset[str] = frozenset(
    {"pe", "market_cap", "div_yield", "netto_marge", "roe", "umsatzwachstum_yoy"}
)

#: Planspiel-Felder (H3/E095 — nur mit psb_kennzahlen-Bestand der ISIN).
PSB_FELDE: frozenset[str] = frozenset(
    {"kgv", "div_rendite", "analysten_upside"}
)

#: Alle gültigen Screener-Felder (technisch aus dem 1d-Bestand + fundamental
#: + Planspiel-Kennzahlen).
SCREENER_FELDE: tuple[str, ...] = (
    *PERF_FENSTER,
    "rsi_14",
    "ann_vol",
    "sma50_ratio",
    "last_price",
    "change_pct",
    *FUNDAMENTAL_FELDE,
    *PSB_FELDE,
)

#: Gültige Operatoren (§T32).
SCREENER_OPS: tuple[str, ...] = ("gt", "lt", "between")

#: Sortierrichtungen der Ergebnisliste (deutsche Vertragswerte).
RICHTUNGEN: tuple[str, ...] = ("auf", "ab")

#: Wie viele Balken je Symbol gelesen werden (deckt perf_1y = 253 Balken,
#: SMA-50-Lücke und 252 Renditen für die ann_vol ab; RSI ist bei 400 Balken
#: längst konvergiert — Wilder-Glättung mit Seed-SMA).
_BALKEN_LIMIT = 400

#: Handelstage pro Jahr (Annualisierung der Tagesvolatilität).
_HANDELSTAGE = 252.0

#: Rendite-Fenster der annualisierten Volatilität (1 Jahr Tagesrenditen).
_ANN_VOL_FENSTER = 252


# ---------------------------------------------------------------------------
# Regeln: Validierung + Prüfung
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regel:
    """Eine validierte Screener-Regel (``wert`` bei ``between`` = (min, max))."""

    feld: str
    op: str
    wert: float | tuple[float, float]


def _regel_fehler(meldung: str, hint: str | None = None) -> ProviderError:
    return ProviderError(meldung, 422, "ungueltige_regel", hint=hint)


def _endlich(wert: float | None) -> float | None:
    """Nicht-endliche Rechenergebnisse (0-Close-Division, pct_change) → ``None``.

    Muster :func:`_als_zahl` in der Gegenrichtung: eine Serie mit ``0``-Closes
    (Datenfehler) erzeugt ``inf``/``ZeroDivisionError`` — der Screener soll
    solche Symbole nicht crashen, sondern wie fehlende Daten behandeln (DoD e).
    """
    if wert is None or not np.isfinite(wert):
        return None
    return float(wert)


def _als_zahl(wert: Any, feld: str) -> float:
    """Regelwert → float; bool/NaN/unendlich → 422 (deutsch)."""
    if isinstance(wert, bool) or not isinstance(wert, (int, float)):
        raise _regel_fehler(
            f"Der Wert der Regel „{feld}“ muss eine Zahl sein "
            f"(erhalten: {wert!r})."
        )
    zahl = float(wert)
    if not np.isfinite(zahl):
        raise _regel_fehler(
            f"Der Wert der Regel „{feld}“ muss eine endliche Zahl sein."
        )
    return zahl


def pruefe_regeln(regeln: Any) -> list[Regel]:
    """Validiert den Regel-AST → Liste von :class:`Regel` (AND-verknüpft).

    ``regeln`` muss eine nicht-leere Liste ``{feld, op, wert}`` sein;
    ``between`` verlangt ``[min, max]`` mit ``min <= max``. Verstoß →
    :class:`ProviderError` 422 mit deutscher Meldung.
    """
    if not isinstance(regeln, list) or not regeln:
        raise _regel_fehler(
            "Mindestens eine Screener-Regel angeben.",
            hint="Eine Regel besteht aus {feld, op, wert} — z. B. "
            '{"feld": "rsi_14", "op": "lt", "wert": 30}.',
        )
    geprueft: list[Regel] = []
    for roh in regeln:
        if not isinstance(roh, dict):
            raise _regel_fehler(
                f"Regeln müssen Objekte {{feld, op, wert}} sein (erhalten: {roh!r})."
            )
        feld = roh.get("feld")
        op = roh.get("op")
        if feld not in SCREENER_FELDE:
            raise _regel_fehler(
                f"Unbekanntes Screener-Feld „{feld}“.",
                hint="Gültige Felder: " + ", ".join(SCREENER_FELDE) + ".",
            )
        if op not in SCREENER_OPS:
            raise _regel_fehler(
                f"Unbekannter Operator „{op}“ für Feld „{feld}“.",
                hint="Gültige Operatoren: " + ", ".join(SCREENER_OPS) + ".",
            )
        wert = roh.get("wert")
        if op == "between":
            if not isinstance(wert, (list, tuple)) or len(wert) != 2:
                raise _regel_fehler(
                    f"„between“ braucht einen Wert als [min, max] "
                    f"(erhalten: {wert!r})."
                )
            min_, max_ = (_als_zahl(w, feld) for w in wert)
            if min_ > max_:
                raise _regel_fehler(
                    f"Der Minimalwert darf nicht über dem Maximalwert liegen "
                    f"({min_} > {max_})."
                )
            geprueft.append(Regel(feld, op, (min_, max_)))
        else:
            geprueft.append(Regel(feld, op, _als_zahl(wert, feld)))
    return geprueft


def regel_erfuellt(wert: float | None, regel: Regel) -> bool:
    """Wert gegen eine Regel; ``None`` (fehlende Daten) erfüllt nie (DoD e)."""
    if wert is None:
        return False
    if regel.op == "gt":
        return wert > regel.wert  # type: ignore[return-value]
    if regel.op == "lt":
        return wert < regel.wert  # type: ignore[return-value]
    min_, max_ = regel.wert  # type: ignore[misc]
    return bool(min_ <= wert <= max_)


# ---------------------------------------------------------------------------
# Technische Feldwerte aus dem 1d-Bestand (Goldwerte: tests/test_screener.py)
# ---------------------------------------------------------------------------


def berechne_feldwerte(balken: pd.DataFrame) -> dict[str, float | None]:
    """Technische Screener-Felder aus einem Balken-DataFrame (aufsteigend).

    Erwartet die Spalten ``close`` (mindestens); Goldwerte gegen eine
    handgerechnete Serie liegen in ``tests/test_screener.py``. ``None`` =
    Fenster/Historie reicht nicht **oder** nicht-endlicher Wert (0-Close-
    Serie, M3-Guard ``_endlich``) — dokumentierte Semantik (DoD e).
    """
    closes = balken["close"].astype("float64").reset_index(drop=True)
    n = len(closes)
    if n == 0:
        return {"last_price": None, "change_pct": None, **{f: None for f in PERF_FENSTER},
                "rsi_14": None, "ann_vol": None, "sma50_ratio": None}
    letzter = float(closes.iloc[-1])
    werte: dict[str, float | None] = {"last_price": _endlich(letzter)}
    # Tagesveränderung in Prozent (Konvention T22-Quotes); 0-Close-Vortag
    # (Division) → None statt ZeroDivisionError/inf (M3-Guard).
    vortag = float(closes.iloc[-2]) if n >= 2 else None
    werte["change_pct"] = (
        _endlich((letzter / vortag - 1.0) * 100.0)
        if vortag is not None and vortag > 0
        else None
    )
    # Performance-Fenster als Bruchzahl; zu junge Serie → None (DoD e).
    for feld, rueckblick in PERF_FENSTER.items():
        basis = float(closes.iloc[-1 - rueckblick]) if n >= rueckblick + 1 else None
        werte[feld] = (
            _endlich(letzter / basis - 1.0) if basis is not None and basis > 0 else None
        )
    # RSI (Wilder) über indicators — genau ein Ort für Indikator-Werte (E011).
    rsi_serie = indicators.rsi(closes, 14)
    werte["rsi_14"] = (
        None if pd.isna(rsi_serie.iloc[-1]) else _endlich(float(rsi_serie.iloc[-1]))
    )
    # Annualisierte Volatilität: std(ddof=1) der letzten ≤ 252 Tagesrenditen;
    # nicht-endliche Renditen (0-Close) fliegen vor der Streuung raus (M3).
    renditen = closes.pct_change().to_numpy(dtype="float64")
    renditen = renditen[np.isfinite(renditen)]
    if len(renditen) >= 2:
        stichprobe = renditen[-_ANN_VOL_FENSTER:]
        werte["ann_vol"] = _endlich(
            float(np.std(stichprobe, ddof=1) * np.sqrt(_HANDELSTAGE))
        )
    else:
        werte["ann_vol"] = None
    # SMA-Verhältnis close/SMA(50) — None während der führenden Lücke.
    sma_serie = indicators.sma(closes, 50)
    if pd.isna(sma_serie.iloc[-1]) or float(sma_serie.iloc[-1]) <= 0:
        werte["sma50_ratio"] = None
    else:
        werte["sma50_ratio"] = _endlich(letzter / float(sma_serie.iloc[-1]))
    return werte


# ---------------------------------------------------------------------------
# Fundamentale Feldwerte (nur mit Fundamentals-Bestand, §T32)
# ---------------------------------------------------------------------------

#: Alias-Kandidaten (yfinance-„pretty“-Namen, case-insensitiv; ANNAHME:
#: Zeilennamen folgen dem yfinance-Statement-Vokabular wie in services/
#: fundamentals._ALIASE — EPS/Aktienzahl ergänzen die dortigen Kataloge).
_EPS_ALIASE: tuple[str, ...] = ("Diluted EPS", "Basic EPS")
_AKTIENZAHL_ALIASE: tuple[str, ...] = (
    "Ordinary Shares Number",
    "Share Issued",
    "Common Stock, Shares Outstanding",
)

#: Ratios aus ``berechne_ratios``, die der Screener direkt durchreicht.
_RATIO_FELDE: tuple[str, ...] = ("netto_marge", "roe", "umsatzwachstum_yoy")


def _neuester_alias_wert(
    symbol: str, statement: str, aliase: tuple[str, ...]
) -> float | None:
    """Neuester Wert eines Statement-Zeilen-Alias (annual → quarterly-Fallback).

    Muster :func:`fundamentals.berechne_ratios`: fehlt das annual-Statement
    komplett, wird quarterly gelesen. Fehlt der Alias/der Wert → ``None``.
    """
    zeilen: list[dict[str, Any]] = []
    for freq in ("annual", "quarterly"):
        zeilen = [
            zeile
            for zeile in fundamentals_service.lies_zeilen(symbol, statement, freq)
            if zeile["value"] is not None
        ]
        if zeilen:
            break
    if not zeilen:
        return None
    neueste = max(zeile["as_of"] for zeile in zeilen)
    klein = {alias.casefold() for alias in aliase}
    for zeile in zeilen:
        if zeile["as_of"] == neueste and zeile["line_item"].casefold() in klein:
            return float(zeile["value"])  # type: ignore[arg-type]
    return None


def _dividendenrendite(symbol: str, letzter_kurs: float | None) -> float | None:
    """Summe der Dividenden der letzten 365 Tage / letzter Kurs (Bruchzahl).

    Basis ist der lokale ``corporate_actions``-Bestand (E022) — kein Netzcall.
    Keine Events/Kurs → ``None``.
    """
    if letzter_kurs is None or letzter_kurs <= 0:
        return None
    stichtag = (datetime.now(UTC) - timedelta(days=365)).timestamp() * 1_000_000
    summe = sum(
        aktion.value
        for aktion in data_service.lese_actions(symbol)
        if aktion.action == "dividend" and aktion.ts_us >= int(stichtag)
    )
    if summe <= 0:
        return None
    return summe / letzter_kurs


def berechne_fundamentale_werte(
    symbol: str, letzter_kurs: float | None
) -> dict[str, float | None]:
    """Fundamentale Screener-Felder (§T32) — alles lokale Bestände.

    ``pe``/``market_cap`` brauchen EPS/Aktienzahl aus dem Fundamentals-Bestand
    und den Kurs; fehlt eine Eingabe (oder EPS ≤ 0) bleibt das Feld ``None``
    (keine Fehler — der Screener filtert dann schlicht nicht danach). Die
    Ratios kommen aus :func:`fundamentals.berechne_ratios`.
    """
    werte: dict[str, float | None] = {f: None for f in FUNDAMENTAL_FELDE}
    ratios = fundamentals_service.berechne_ratios(symbol)
    for eintrag in ratios.get("ratios", []):
        if eintrag.get("key") in _RATIO_FELDE:
            werte[eintrag["key"]] = eintrag.get("wert")  # type: ignore[index]
    eps = _neuester_alias_wert(symbol, "income", _EPS_ALIASE)
    aktien = _neuester_alias_wert(symbol, "balance", _AKTIENZAHL_ALIASE)
    if letzter_kurs is not None and eps is not None and eps > 0:
        werte["pe"] = letzter_kurs / eps
    if letzter_kurs is not None and aktien is not None and aktien > 0:
        werte["market_cap"] = letzter_kurs * aktien
    werte["div_yield"] = _dividendenrendite(symbol, letzter_kurs)
    return werte


# ---------------------------------------------------------------------------
# Screening über das Universum (E027: synchron, nur lokal)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Planspiel-Feldwerte (H3/E095 — nur lokale psb_kennzahlen-/Parquet-Bestände)
# ---------------------------------------------------------------------------

#: ISIN-Form (ISO 6166, ohne Checksummen-Prüfung) — Muster
#: ``agent/tools/daten.py``; hier bewusst EIGEN, damit der Screener nicht vom
#: Tool-Layer abhängt (Schichtungsvertrag: Screener liest nur Storage/Service).
_PSB_ISIN_MUSTER = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

#: Alias-Teilstrings der VWD-Kennzahlen-Labels (case-insensitive; ANNAHME:
#: die Snippet-Labels sind deutsch und stabil, Teilstring-Match toleriert
#: Klammer-Zusätze wie „Kurs-Gewinn-Verhältnis (KGV)“).
_KGV_ALIAS = "kurs-gewinn"
_DIV_RENDITE_ALIAS = "dividendenrendite"


def _als_float(wert: Any) -> float | None:
    """Parser-Wert (float | str — „226,68 Mrd.“ bleibt String) → float | None."""
    if isinstance(wert, bool) or not isinstance(wert, (int, float)):
        return None
    return _endlich(float(wert))


def _kennzahl_nach_alias(
    kennzahlen: dict[str, Any], alias: str
) -> float | None:
    """Erster float-Wert aus ``getKennzahlen``, dessen Label den Alias enthält."""
    for label, wert in kennzahlen.items():
        if isinstance(label, str) and alias in label.casefold():
            zahl = _als_float(wert)
            if zahl is not None:
                return zahl
    return None


def _isin_fuer_symbol(
    symbol: str, psb_nach_kuerzel: dict[str, str]
) -> str | None:
    """Symbol → ISIN (Kette: ISIN-Form → ``symbols.isin`` → ``yahoo_kuerzel``)."""
    geprueft = symbol.strip().upper()
    if _PSB_ISIN_MUSTER.match(geprueft):
        return geprueft
    eintrag = symbols_repo.get_symbol(symbol)
    if eintrag is not None and eintrag.isin:
        return eintrag.isin.strip().upper()
    return psb_nach_kuerzel.get(symbol)


def _letzter_kurs(isin: str) -> float | None:
    """Letzter 1d-Schlusskurs der ISIN (Parquet, egal welche Quelle) — sonst None."""
    try:
        balken = parquet_store.read_bars(isin, "1d", limit=1)
    except Exception:  # noqa: BLE001 — defekter/fehlender Bestand ist kein Fachfehler
        return None
    if balken.empty:
        return None
    return _endlich(float(balken["close"].iloc[-1]))


def berechne_psb_werte(symbol: str, psb_nach_kuerzel: dict[str, str]) -> dict[str, float | None]:
    """Planspiel-Screener-Felder (H3/E095) — ausschließlich lokale Bestände.

    Cache-Miss (``psb_kennzahlen`` leer für die ISIN) oder fehlende Eingabe →
    ``None`` (dokumentierte Semantik wie FUNDAMENTAL_FELDE, DoD e: die Zeile
    fällt bei einer solchen Regel raus). Kein Netz (E027) — der Cache wird
    über REST ``frisch=true``/Tool ``planspiel_kennzahlen`` gefüllt.
    """
    werte: dict[str, float | None] = {f: None for f in PSB_FELDE}
    isin = _isin_fuer_symbol(symbol, psb_nach_kuerzel)
    if isin is None:
        return werte
    gecacht = psb_repo.kennzahlen_lesen(isin)
    kennzahlen = gecacht.get("getKennzahlen") or {}
    if isinstance(kennzahlen, dict):
        werte["kgv"] = _kennzahl_nach_alias(kennzahlen, _KGV_ALIAS)
        rendite = _kennzahl_nach_alias(kennzahlen, _DIV_RENDITE_ALIAS)
        # Snippet liefert Prozent (1,34 % → 1.34) — Bruchzahl-Konvention wie div_yield.
        werte["div_rendite"] = (
            _endlich(rendite / 100.0) if rendite is not None else None
        )
    analysten = gecacht.get("getAnalystenempfehlungen") or {}
    kursziel = analysten.get("kursziel") if isinstance(analysten, dict) else None
    kursziel_zahl = _als_float(kursziel)
    letzter = _letzter_kurs(isin)
    if kursziel_zahl is not None and letzter is not None and letzter > 0:
        werte["analysten_upside"] = _endlich((kursziel_zahl - letzter) / letzter)
    return werte


def universum_symbole() -> list[str]:
    """Alle Symbole mit 1d-Bestand (``bar_meta.row_count > 0``), sortiert (E027)."""
    return sorted(
        {
            meta.symbol
            for meta in parquet_store.list_bar_meta()
            if meta.interval == "1d" and meta.row_count > 0
        }
    )


def _benoetigte_felde(
    regeln: list[Regel], sort_feld: str | None
) -> tuple[bool, bool, set[str]]:
    """``(brauche_fundamentals, brauche_psb, technische_felde)`` für die Ergebniswerte.

    Es werden nur die geforderten Felder berechnet (Regeln + Sortierung) —
    Fundamentals-/PSB-Lookups je Symbol bleiben so auf den Bedarf begrenzt.
    """
    felder = {regel.feld for regel in regeln}
    if sort_feld is not None:
        felder.add(sort_feld)
    fundamental = bool(felder & FUNDAMENTAL_FELDE)
    psb = bool(felder & PSB_FELDE)
    technisch = felder - FUNDAMENTAL_FELDE - PSB_FELDE
    return fundamental, psb, technisch


def screen(
    regeln: Any,
    *,
    sort_feld: str | None = None,
    richtung: str = "ab",
    limit: int | None = None,
) -> dict[str, Any]:
    """Führt die Regeln über das lokale Universum aus (§T32, synchron).

    Antwort: ``{"universum": geprüfte Symbole, "count": Treffer, "ergebnisse":
    [{symbol, name, werte: {feld: wert}}, …]}``. AND-Verknüpfung; ein Regel-
    feld ohne berechenbaren Wert lässt die Zeile herausfallen (DoD e).
    Sortierung über ``sort_feld`` (``richtung`` ``auf``/``ab``, ``None``-Werte
    immer am Ende), ``limit`` kürzt die Trefferliste.

    Fachliche Fehler (:class:`ProviderError` 422): unbekanntes Feld/Op,
    sinnwidrige Werte, unbekanntes ``sort_feld``/``richtung``.
    """
    geprueft = pruefe_regeln(regeln)
    if sort_feld is not None and sort_feld not in SCREENER_FELDE:
        raise ProviderError(
            f"Unbekanntes Sortierfeld „{sort_feld}“.",
            422,
            "ungueltiges_sortierfeld",
            hint="Gültige Felder: " + ", ".join(SCREENER_FELDE) + ".",
        )
    if richtung not in RICHTUNGEN:
        raise ProviderError(
            f"Unbekannte Sortierrichtung „{richtung}“.",
            422,
            "ungueltige_richtung",
            hint="Gültige Werte: " + ", ".join(RICHTUNGEN) + ".",
        )
    if limit is not None and limit < 1:
        raise ProviderError(
            "„limit“ muss mindestens 1 sein.", 422, "ungueltiges_limit"
        )

    brauche_fundamentals, brauche_psb, _ = _benoetigte_felde(geprueft, sort_feld)
    namen = symbols_repo_namen()
    # PSB-Brücke (H3/E095) nur bei Bedarf: yahoo_kuerzel → ISIN einmal je Lauf.
    psb_nach_kuerzel: dict[str, str] = (
        {
            i.yahoo_kuerzel: i.isin
            for i in psb_repo.liste()
            if i.yahoo_kuerzel
        }
        if brauche_psb
        else {}
    )
    treffer: list[dict[str, Any]] = []
    universum = 0
    for symbol in universum_symbole():
        universum += 1
        try:
            balken = data_service.read_ohlcv(
                symbol, "1d", limit=_BALKEN_LIMIT, adjust="adjusted"
            )
        except ProviderError:
            continue  # defekter/leerer Bestand — Symbol überspringen (Muster movers)
        werte: dict[str, float | None] = berechne_feldwerte(balken)
        if brauche_fundamentals:
            werte.update(berechne_fundamentale_werte(symbol, werte.get("last_price")))
        if brauche_psb:
            werte.update(berechne_psb_werte(symbol, psb_nach_kuerzel))
        # Vertrag: die Antwort enthält alle berechneten Felder (technisch
        # immer, fundamental sobald der Block benötigt/berechnet wurde) —
        # das Frontend kann beliebige Spalten anzeigen/sortieren.
        relevant = dict(werte)
        if not all(regel_erfuellt(relevant.get(regel.feld), regel) for regel in geprueft):
            continue
        treffer.append(
            {
                "symbol": symbol,
                "name": namen.get(symbol),
                "werte": relevant,
            }
        )

    if sort_feld is not None:
        ohne_wert = [zeile for zeile in treffer if zeile["werte"].get(sort_feld) is None]
        mit_wert = [zeile for zeile in treffer if zeile["werte"].get(sort_feld) is not None]
        mit_wert.sort(
            key=lambda zeile: float(zeile["werte"][sort_feld]),  # type: ignore[arg-type]
            reverse=(richtung == "ab"),
        )
        treffer = mit_wert + ohne_wert  # fehlende Werte stets am Ende
    gekuerzt = len(treffer) if limit is None else min(limit, len(treffer))
    return {
        "universum": universum,
        "count": len(treffer),
        "ergebnisse": treffer[:gekuerzt],
    }


def symbols_repo_namen() -> dict[str, str]:
    """Registry-Namen (``symbol → name``) für die Ergebnisliste."""
    return {
        eintrag.symbol: eintrag.name
        for eintrag in symbols_repo.list_symbols()
        if eintrag.name
    }


# ---------------------------------------------------------------------------
# S&P-500-Liste (E027): Wikipedia laden + parsen (Live-Fixpunkt, Fixture)
# ---------------------------------------------------------------------------

#: Wikipedia-Liste der S&P-500-Gesellschaften (E027).
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

#: Injektionspunkt für Tests (httpx.MockTransport, E013: Suite bleibt netzfrei) —
#: Muster StooqProvider-``transport``; ``None`` = echter Netzabruf.
_TRANSPORT: httpx.BaseTransport | None = None

#: Mindestanzahl erwarteter Zeilen — darunter gilt die Struktur als geändert
#: (502 statt stiller Teilimport; DoD: Fixture-Assert ≥ 450, Format ``TICKER``).
SP500_MIN_ERWARTET = 450

#: Einträge der geparsten Liste (``symbol`` = Wikipedia-Schreibweise).
_TIMEOUT_S = 20.0

_ZELLE_MUSTER = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL)
_TAG_MUSTER = re.compile(r"<[^>]+>")
#: Gültiges Börsenkürzel (Wikipedia-Spalte „Symbol“): Großbuchstaben/Ziffern
#: plus Punkt/Strich (Klassen-Aktien) — alles andere ist keine Ticker-Zeile.
_TICKER_MUSTER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def _zellen_text(roh: str) -> str:
    """HTML-Zelle → Klartext (Tags weg, Entities dekodiert, Whitespace normal)."""
    text = _TAG_MUSTER.sub("", roh)
    text = html_mod.unescape(text)
    return " ".join(text.split())


def parse_sp500_html(html: str) -> list[dict[str, str]]:
    """Wikipedia-HTML → ``[{symbol, name}, …]`` (Parser gegen Fixture eingefroren).

    Es wird die Konstituenten-Tabelle (``id="constituents"``) gelesen; je
    Zeile liefern die erste Zelle das Kürzel (Spalte „Symbol“) und die zweite
    den Firmennamen („Security“). Zeilen ohne gültiges Kürzel (Kopfzeile,
    Fußnoten-Zeilen) entfallen; Duplikate bleiben beim ersten Vorkommen.
    """
    table_beginn = html.find('id="constituents"')
    if table_beginn < 0:
        raise ProviderError(
            "Die Wikipedia-Liste der S&P-500-Gesellschaften hat eine unerwartete "
            "Struktur (Tabelle „constituents“ fehlt).",
            502,
            "sp500_struktur",
        )
    table_ende = html.find("</table>", table_beginn)
    tabelle = html[table_beginn:table_ende] if table_ende >= 0 else html[table_beginn:]
    ergebnis: list[dict[str, str]] = []
    gesehen: set[str] = set()
    for zeile in tabelle.split("<tr"):
        zellen = [_zellen_text(zelle) for zelle in _ZELLE_MUSTER.findall(zeile)]
        if len(zellen) < 2:
            continue  # Kopf-/Strukturzeilen ohne Datenzellen
        symbol = zellen[0]
        name = zellen[1]
        if not _TICKER_MUSTER.match(symbol):
            continue  # Kopfzeile („Symbol“) oder abweichende Zeile
        if symbol in gesehen:
            continue
        gesehen.add(symbol)
        ergebnis.append({"symbol": symbol, "name": name or ""})
    if len(ergebnis) < SP500_MIN_ERWARTET:
        raise ProviderError(
            "Die Wikipedia-Liste lieferte nur "
            f"{len(ergebnis)} gültige Symbole (erwartet ≥ {SP500_MIN_ERWARTET}) — "
            "die Seitenstruktur hat sich vermutlich geändert.",
            502,
            "sp500_unvollstaendig",
        )
    return ergebnis


def lade_sp500_liste(
    *, transport: httpx.BaseTransport | None = None, url: str = SP500_URL
) -> list[dict[str, str]]:
    """Lädt die Wikipedia-Liste (httpx, Transport injizierbar) und parst sie.

    Netz-/HTTP-Pannen → 502 ``sp500_abruf``; Struktur-/Mengenabweichungen →
    502 über :func:`parse_sp500_html` (Live-Fixpunkt, E027).
    """
    nutze_transport = transport if transport is not None else _TRANSPORT
    try:
        with httpx.Client(
            transport=nutze_transport,
            timeout=_TIMEOUT_S,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            },
            follow_redirects=True,
        ) as client:
            antwort = client.get(url)
            antwort.raise_for_status()
            inhalt = antwort.text
    except ProviderError:
        raise
    except httpx.HTTPError as exc:
        raise ProviderError(
            f"Die S&P-500-Liste konnte nicht geladen werden: {exc}",
            502,
            "sp500_abruf",
            hint="Netzverbindung prüfen — der Import ist optional und kann "
            "später erneut gestartet werden.",
        ) from exc
    return parse_sp500_html(inhalt)


def normalisiere_yahoo_kuerzel(symbol: str) -> str:
    """Wikipedia-Schreibweise → Yahoo-Kürzel (Klassen-Aktien Punkt → Strich).

    ANNAHME (dokumentiert): Wikipedia listet Klassen-Aktien mit Punkt
    („BRK.B“, „BF.B“), Yahoo kennt sie mit Bindestrich („BRK-B“, „BF-B“) —
    alle anderen Kürzel bleiben unverändert. Unbekannte Randfälle scheitern
    spätestens an der Suche-/Job-Fehlerisolation.
    """
    return symbol.replace(".", "-")
