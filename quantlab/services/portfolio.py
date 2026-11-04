"""Portfolio-Service (T23): Ledger → Positionen, Bewertung, Performance.

Verbindet das Transaktions-Ledger (``portfolios_repo``, T18) mit dem
Rechenkern :mod:`quantlab.quant.portfolio_perf` und dem Marktdaten-Read-Pfad
(:func:`quantlab.services.data_service.read_ohlcv`, Intervall ``1d``):

- **Buchungen laden:** eigenes Ledger plus **eingehende** Umbuchungen (bei
  anderen Depots als ``transfer`` auf dieses Depot gebucht) — der Engine-
  Typ ``transfer`` wird dazu in ``transfer_out``/``transfer_in`` aufgelöst
  (E025: TWR-neutral, XIRR-wirksam als Paar). Gebühren einer Umbuchung zahlt
  allein die sendende Seite; der Empfänger bucht den Netto-Betrag.
  ANNAHME (T23): Das T18-Repo bietet keine Abfrage „eingehende Transfers“ —
  gelöst durch Scannen der anderen Depots (Depotanzahl ist klein; reine
  Lesezugriffe).
- **Bewertung:** Daily-Closes **raw** (E022/E031) je gehaltenem Symbol;
  fehlende Kurse → Carry-Forward, gezählt und im Response markiert; Symbole
  ohne jegliche Kurse tragen 0 bei und werden gemeldet (kein Fehler).
- **Benchmark:** optionales lokales Symbol — Beta/Alpha via OLS aus dem
  v1-Risikomodul (:func:`quantlab.quant.risk.beta_alpha`), Überrendite =
  TWR − Benchmark-Gesamtrendite über der Schnittmenge. Fehlende lokale
  Daten lassen den Router 404 antworten (SymbolNotFoundError durchreichen).

Fehler: fachliche numerische Randfälle heben ``ValueError`` (deutsch, Router
→ 422); ``ProviderError`` aus dem Datenpfad wird vom Router übersetzt.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quantlab.providers.base import SymbolNotFoundError
from quantlab.quant import metrics as metrics_mod
from quantlab.quant import portfolio_perf as perf
from quantlab.quant import risk as risk_mod
from quantlab.quant.returns import PERIODS_PER_YEAR
from quantlab.services import data_service
from quantlab.storage.repos import portfolios_repo, symbols_repo

#: Bewertungsintervall des Portfolios (E031: Daily-Close).
BEWERTUNGS_INTERVALL = "1d"

#: Zulässige ``method``-Werte des Performance-Endpunkts (§C.4).
PERFORMANCE_METHODEN: tuple[str, ...] = ("twr", "xirr", "all")


# ---------------------------------------------------------------------------
# Ledger → Engine-Buchungen
# ---------------------------------------------------------------------------


def _zu_buchung(tx: portfolios_repo.Transaction, typ: str) -> perf.Buchung:
    """Repo-Transaktion → Engine-Buchung (``transfer`` bereits in out/in aufgelöst)."""
    netto = typ == "transfer_in"  # Gebühren zahlt die Sendeseite (E025)
    return perf.Buchung(
        ts=tx.ts,
        typ=typ,
        symbol=tx.symbol,
        quantity=tx.quantity,
        price=tx.price,
        amount=tx.amount,
        fees=0.0 if netto else tx.fees,
        taxes=0.0 if netto else tx.taxes,
    )


def buchungen_laden(portfolio_id: int) -> list[perf.Buchung]:
    """Alle für das Depot wirksamen Buchungen, chronologisch (stabil sortiert)."""
    buchungen = [
        _zu_buchung(tx, "transfer_out" if tx.type == "transfer" else tx.type)
        for tx in portfolios_repo.list_transactions(portfolio_id)
    ]
    for depot in portfolios_repo.list_portfolios():
        if depot.id == portfolio_id:
            continue
        for tx in portfolios_repo.list_transactions(depot.id):
            if tx.type == "transfer" and tx.transfer_portfolio_id == portfolio_id:
                buchungen.append(_zu_buchung(tx, "transfer_in"))
    buchungen.sort(key=lambda b: b.ts_us)  # stabil: Reihenfolge im Ledger bleibt
    return buchungen


def _gehaltene_symbole(buchungen: list[perf.Buchung]) -> list[str]:
    """Symbole mit Kauf-Historie (Bewertungsrelevant; Dividenden-only zählt nicht)."""
    return sorted({b.symbol for b in buchungen if b.typ == "buy" and b.symbol})


def kurse_laden(symbole: list[str]) -> dict[str, pd.Series | None]:
    """Daily-Close-Serien (raw, E031) je Symbol; ``None`` ohne lokale Daten."""
    kurse: dict[str, pd.Series | None] = {}
    for symbol in symbole:
        try:
            bars = data_service.read_ohlcv(symbol, BEWERTUNGS_INTERVALL)
        except SymbolNotFoundError:
            kurse[symbol] = None  # bewusst: Bewertung 0 + Meldung, kein Fehler
            continue
        if bars.empty:
            kurse[symbol] = None
            continue
        kurse[symbol] = pd.Series(
            bars["close"].to_numpy(dtype=float),
            index=bars["ts"].to_numpy(dtype="int64"),
            dtype=float,
        )
    return kurse


def _waehrung_je_symbol(
    buchungen: list[perf.Buchung], base_currency: str
) -> dict[str, str | None]:
    """Quellwährung je Symbol: Symbol-Registry (``symbols.currency``) › ``None``.

    ``None`` = unbekannt; die UI warnt bei Währungsmix ohne FX (E031-ANNAHME,
    R19 — keine Umrechnung in v2). ``base_currency`` fließt nur in den
    Warnungsvergleich ein.
    """
    waehrungen: dict[str, str | None] = {
        b.symbol: None for b in buchungen if b.symbol is not None
    }
    for symbol in waehrungen:
        eintrag = symbols_repo.get_symbol(symbol)
        if eintrag is not None and eintrag.currency:
            waehrungen[symbol] = eintrag.currency
    return waehrungen


def _waehrungs_warnungen(waehrungen: dict[str, str | None], base: str) -> list[str]:
    """Deutsche Hinweise zum Währungsmix (E031: geführt + gewarnt, kein FX)."""
    vorhandene = sorted({w for w in waehrungen.values() if w})
    if len(vorhandene) > 1 or (vorhandene and vorhandene != [base]):
        return [
            "Währungsmix ohne Umrechnung (v2): Positionen führen ihre "
            f"Quellwährung ({', '.join(vorhandene)}), das Depot rechnet in {base}."
        ]
    return []


# ---------------------------------------------------------------------------
# Positionen (Ø-EK Default, FIFO Option — E031)
# ---------------------------------------------------------------------------


def positionen_report(
    depot: portfolios_repo.Portfolio, method: str = "avg_cost"
) -> dict[str, Any]:
    """Response-Payload des Positions-Endpunkts (§C.4, E031).

    Bewertung zum letzten verfügbaren Daily-Close je Symbol (``as_of`` =
    jüngster Bar aller Symbole); ein Symbol ohne Bar an ``as_of`` ist
    Carry-Forward und markiert. Symbole ohne jegliche Kurse liefern
    ``last_price: null`` (Marktwert 0) plus Warnung — kein Fehler.
    """
    if method not in perf.METHODEN:
        raise ValueError(
            f"Unbekanntes Einstandsverfahren „{method}“. "
            f"Gültig: {', '.join(perf.METHODEN)}."
        )
    buchungen = buchungen_laden(depot.id)
    ergebnis = perf.positionen_ableiten(buchungen, method)
    kurse = kurse_laden(_gehaltene_symbole(buchungen))
    waehrungen = _waehrung_je_symbol(buchungen, depot.base_currency)

    as_of: int | None = None
    letzte: dict[str, tuple[float, int]] = {}
    for symbol, series in kurse.items():
        if series is None or series.empty:
            continue
        ts = int(series.index[-1])
        letzte[symbol] = (float(series.iloc[-1]), ts)
        as_of = ts if as_of is None else max(as_of, ts)

    positionen: list[dict[str, Any]] = []
    marktwert_gesamt = 0.0
    unrealisiert_gesamt = 0.0
    realisiert_gesamt = 0.0
    warnungen: list[str] = []
    for symbol, pos in sorted(ergebnis.positionen.items()):
        kurs, kurs_ts = letzte.get(symbol, (None, None))
        marktwert = None
        unrealisiert = None
        carry = False
        if kurs is not None and kurs_ts is not None:
            marktwert = pos.menge * kurs
            einstand = pos.einstand_stueck
            unrealisiert = (
                marktwert - pos.menge * einstand if einstand is not None else marktwert
            )
            carry = as_of is not None and kurs_ts < as_of
            marktwert_gesamt += marktwert
            unrealisiert_gesamt += unrealisiert
        elif pos.menge != 0.0:
            warnungen.append(
                f"kein_kurs:{symbol} — keine gespeicherten Kurse; Position "
                "trägt 0 zum Marktwert bei."
            )
        realisiert_gesamt += pos.realisiert_netto
        positionen.append(
            {
                "symbol": symbol,
                "quantity": pos.menge,
                "cost_per_share": pos.einstand_stueck,
                "cost_basis": pos.einstand_wert,
                "last_price": kurs,
                "price_ts": kurs_ts,
                "price_carry_forward": carry,
                "market_value": marktwert,
                "unrealized_pnl": unrealisiert,
                "realized_pnl": pos.realisiert_netto,
                "realized_pnl_price_only": pos.realisiert_preis_pnl,
                "fees_taxes": pos.gebuehren_steuern,
                "dividends_total": pos.dividenden_total,
                "dividends_count": pos.dividenden_count,
                "dividends_last_ts": pos.dividenden_letzte_ts,
                "currency": waehrungen.get(symbol),
                "first_ts": pos.erste_ts,
                "last_ts": pos.letzte_ts,
            }
        )
    dividenden = [
        {
            "symbol": zeile["symbol"],
            "total": zeile["dividends_total"],
            "count": zeile["dividends_count"],
            "last_ts": zeile["dividends_last_ts"],
            "currency": zeile["currency"],
        }
        for zeile in positionen
        if zeile["dividends_count"] > 0
    ]
    warnungen.extend(_waehrungs_warnungen(waehrungen, depot.base_currency))
    return {
        "portfolio_id": depot.id,
        "method": method,
        "base_currency": depot.base_currency,
        "as_of": as_of,
        "cash": ergebnis.cash,
        "positions": positionen,
        "totals": {
            "market_value_positions": marktwert_gesamt,
            "cash": ergebnis.cash,
            "total_value": marktwert_gesamt + ergebnis.cash,
            "unrealized_pnl": unrealisiert_gesamt,
            "realized_pnl": realisiert_gesamt,
        },
        "dividenden": dividenden,
        "warnungen": warnungen,
    }


# ---------------------------------------------------------------------------
# Performance: TWR + XIRR + Equity + Kennzahlen + Benchmark
# ---------------------------------------------------------------------------


def _benchmark_block(
    benchmark: str,
    taeglich: pd.Series,
    twr_total: float | None,
    start_us: int | None,
    end_us: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Beta/Alpha/Überrendite gegen ein lokales Symbol (OLS-Reuse, §F).

    Rückgabe ``(block, warnung)`` — ``block None`` + Warnung, wenn die
    Schnittmenge zu klein für OLS ist. Fehlende lokale Daten werfen
    ``SymbolNotFoundError`` (Router → 404, §T23: „nur wenn lokal vorhanden“).
    """
    bars = data_service.read_ohlcv(
        benchmark, BEWERTUNGS_INTERVALL, start=start_us, end=end_us
    )
    if bars.empty:
        return None, f"Benchmark „{benchmark}“ liefert im Zeitraum keine Kurse."
    bench_close = pd.Series(
        bars["close"].to_numpy(dtype=float),
        index=bars["ts"].to_numpy(dtype="int64"),
        dtype=float,
    )
    bench_returns = bench_close.pct_change(fill_method=None).dropna()
    verbunden = pd.concat({"portfolio": taeglich, "bench": bench_returns}, axis=1).dropna()
    if len(verbunden) < 2:
        return None, (
            f"Benchmark „{benchmark}“ überlappt zu wenig mit den Bewertungs"
            "tagen — Beta/Alpha nicht berechenbar."
        )
    beta, alpha = risk_mod.beta_alpha(
        verbunden["portfolio"], verbunden["bench"], PERIODS_PER_YEAR
    )
    # Gesamtrendite der Benchmark über dasselbe Bewertungsfenster (die Balken
    # wurden mit start/end des Portfoliogrids geladen) — vergleichbar mit TWR.
    bench_total = (
        float(bench_close.iloc[-1] / bench_close.iloc[0] - 1.0)
        if len(bench_close) >= 2
        else None
    )
    excess = None if (twr_total is None or bench_total is None) else twr_total - bench_total
    return (
        {
            "symbol": benchmark,
            "total_return": bench_total,
            "beta": beta,
            "alpha": alpha,
            "excess_return": excess,
            "n_obs": int(len(verbunden)),
        },
        None,
    )


def performance_report(
    depot: portfolios_repo.Portfolio,
    start_us: int | None = None,
    end_us: int | None = None,
    benchmark: str | None = None,
    method: str = "all",
) -> dict[str, Any]:
    """Response-Payload des Performance-Endpunkts (§C.4, E025/E031).

    ``method`` steuert die Berechnung: ``twr`` (TWR + Equity + Kennzahlen),
    ``xirr`` (nur XIRR), ``all`` (alles; XIRR-Randfälle werden als ``null``
    + Hinweisfeld gemeldet statt 422). Numerische Fehler ohne definierbares
    Ergebnis (z. B. XIRR ohne Vorzeichenwechsel bei ``method=xirr``) heben
    ``ValueError`` → Router 422 (R14).

    XIRR-Semantik (REV-T23 Fix 1): XIRR misst **ab dem ersten Cashflow** —
    es gibt keinen unteren Fensterschnitt; Buchungen vor ``start`` gehören
    zur Geldfluss-Historie. Buchungen **nach dem Fensterende** (``end``,
    bzw. heute als Default) fließen dagegen nicht ein: der Endwert wird zum
    effektiven Fensterende bewertet und spätere Buchungen würden das Ergebnis
    verzerren.
    """
    if method not in PERFORMANCE_METHODEN:
        raise ValueError(
            f"Unbekannte Performance-Methode „{method}“. "
            f"Gültig: {', '.join(PERFORMANCE_METHODEN)}."
        )
    buchungen = buchungen_laden(depot.id)
    leer_hinweis = "Das Depot hat noch keine Buchungen." if not buchungen else None
    kurse = kurse_laden(_gehaltene_symbole(buchungen))
    start_tag = None if start_us is None else start_us // perf.US_PRO_TAG
    end_tag = None if end_us is None else (end_us - 1) // perf.US_PRO_TAG
    bewertung = perf.bewertungsreihe(buchungen, kurse, start_tag, end_tag)
    twr_ergebnis = perf.twr(bewertung, buchungen)

    xirr_wert: float | None = None
    xirr_hinweis: str | None = None
    if method in ("xirr", "all"):
        endwert = bewertung.werte[-1] if bewertung.werte else None
        if endwert is None:
            cash_standalone = perf.positionen_ableiten(buchungen).cash
            endwert = cash_standalone
        end_tag_effektiv = (
            bewertung.grid_us[-1] // perf.US_PRO_TAG
            if bewertung.grid_us
            else (end_tag if end_tag is not None else perf.heute_tag())
        )
        # REV-T23 Fix 1: XIRR-Fensterfilter — nur Buchungen bis zum effektiven
        # Fensterende; spätere Flows würden das windowierte Ergebnis verzerren.
        # Kein unterer Schnitt: XIRR misst ab dem ersten Flow (siehe Docstring).
        fenster_buchungen = [b for b in buchungen if b.tag <= end_tag_effektiv]
        cashflows = perf.xirr_cashflows(fenster_buchungen, endwert, end_tag_effektiv)
        try:
            xirr_wert = perf.xirr(cashflows)
            if xirr_wert is None:
                xirr_hinweis = (
                    "XIRR konnte im Suchbereich (−99,99 % bis +900 %) keine "
                    "Lösung finden — Rendite extrem oder Cashflows degeneriert."
                )
        except ValueError as exc:
            if method == "xirr":
                raise  # definiertes 422 (E025/DoD b)
            xirr_hinweis = str(exc)  # method=all: null + Hinweisfeld (R14)

    kennzahlen: dict[str, Any] | None = None
    if method in ("twr", "all") and len(twr_ergebnis.taeglich) >= 2:
        serie = perf.taegliche_renditen(twr_ergebnis, bewertung.grid_us)
        kennzahlen = metrics_mod.performance_metrics(serie)

    benchmark_block: dict[str, Any] | None = None
    warnungen: list[str] = []
    if benchmark and method in ("twr", "all") and twr_ergebnis.taeglich:
        serie = perf.taegliche_renditen(twr_ergebnis, bewertung.grid_us)
        fenster_start = bewertung.grid_us[0]
        fenster_ende = bewertung.grid_us[-1] + perf.US_PRO_TAG
        benchmark_block, warnung = _benchmark_block(
            benchmark, serie, twr_ergebnis.total, fenster_start, fenster_ende
        )
        if warnung:
            warnungen.append(warnung)
    elif benchmark and method == "xirr":
        warnungen.append("Benchmark-Vergleich benötigt method=twr oder all.")

    if leer_hinweis:
        warnungen.append(leer_hinweis)
    # REV-T23 Fix 4b: dünnes Bewertungsgrid → Annualisierung der Kennzahlen
    # (CAGR/ann. Rendite extrapolieren Perioden auf ppy) verzerrt — warnen.
    if kennzahlen is not None and bewertung.grid_us:
        kalendertage = (
            bewertung.grid_us[-1] - bewertung.grid_us[0]
        ) // perf.US_PRO_TAG + 1
        if len(bewertung.grid_us) < 0.5 * kalendertage:
            warnungen.append(
                f"Kennzahlen bei dünnem Bewertungsgrid annualisierungsverzerrt: "
                f"{len(bewertung.grid_us)} Bewertungstage auf {kalendertage} "
                "Kalendertage des Fensters."
            )
    if bewertung.carry_tage:
        warnungen.append(
            f"Carry-Forward: {bewertung.carry_tage} Symbol-Tage ohne Kurs wurden "
            "mit dem jeweils letzten bekannten Kurs bewertet (E031)."
        )
    for symbol in bewertung.symbole_ohne_kurs:
        warnungen.append(
            f"kein_kurs:{symbol} — keine gespeicherten Kurse; das Symbol trägt 0 "
            "zum Depotwert bei."
        )
    waehrungen = _waehrung_je_symbol(buchungen, depot.base_currency)
    warnungen.extend(_waehrungs_warnungen(waehrungen, depot.base_currency))

    return {
        "portfolio_id": depot.id,
        "method": method,
        "base_currency": depot.base_currency,
        "start": (
            bewertung.grid_us[0]
            if bewertung.grid_us
            else (start_us if start_us is not None else None)
        ),
        "end": bewertung.grid_us[-1] if bewertung.grid_us else None,
        "bewertungstage": len(bewertung.grid_us),
        "twr": twr_ergebnis.total if method in ("twr", "all") else None,
        "neutrale_tage": twr_ergebnis.neutrale_tage,
        "xirr": xirr_wert,
        "xirr_hinweis": xirr_hinweis,
        "equity": perf.sample_equity(bewertung.grid_us, bewertung.werte),
        "kennzahlen": kennzahlen,
        "benchmark": benchmark_block,
        "cash": perf.positionen_ableiten(buchungen).cash,
        "warnungen": warnungen,
    }
