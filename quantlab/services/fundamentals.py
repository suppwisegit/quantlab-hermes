"""Fundamentals-Service (T21): Long-Format-Persistenz, Pivot-on-read, Ratios.

Übernimmt die Fachlogik zwischen Provider und Router:

- **Speichern (Long-Format):** :func:`speichere_statement` überführt ein
  yfinance-Statement-DataFrame (Index = Kennzahl, Spalten = asOfDate) in
  Zeilen ``(symbol, statement, freq, line_item, as_of, value)`` und **ersetzt**
  den Bestand je ``(symbol, statement, freq)`` vollständig (Upsert-Vertrag
  §T21 — veraltete Perioden entfallen). Yahoo liefert maximal 4 Jahre
  (annual) bzw. 5 Quartale (quarterly) — mehr Historie existiert Quelle-seitig
  nicht; ein anderer ``freq``/anderes Statement bleibt vom Ersetzen unberührt.
  Zeilen ohne Wert (NaN) werden **nicht** gespeichert (R17: „Long-Format ohne
  Nullzeilen“ — fehlende Werte sind abwesende Zeilen, keine NULL-Zeilen).
- **Lesen (Pivot on-read):** :func:`lies_statement_pivot` dreht das Long-Format
  beim Lesen in ``{line_items, periods, values}`` zurück (``values[zeile][spalte]``,
  ``None`` für fehlende Kombinationen; Perioden aufsteigend sortiert).
- **Aktualisieren:** :func:`aktualisiere_symbol` holt alle Statements (annual +
  quarterly) über den Provider und speichert sie; leere Statements werden
  übersprungen (kein Fehler, R17) und löschen vorhandene Daten **nicht**
  (ANNAHME: ein leeres Statement heißt „Quelle liefert nichts“, nicht
  „Daten widerrufen“).
- **Key-Ratios:** :func:`berechne_ratios` berechnet Kennzahlen **aus dem
  gespeicherten Bestand** (Netto-Marge, EBIT-/EBITDA-Marge, ROE, ROA,
  Eigenkapitalquote, Liquiditätsgrad, Umsatzwachstum YoY, FCF-Marge) — je
  neuester Periode je Statement. Fehlende Eingangswerte (Yahoo liefert je
  Symbol unterschiedliche Zeilen) führen zu **abwesenden** Ratios, nicht zu
  Fehlern. Quartalswerte sind Periodenwerte (nicht annualisiert; dokumentiert).

Persistenz-Muster: :mod:`quantlab.db` — eine Verbindung pro Operation,
``BEGIN IMMEDIATE`` für den Ersetzungs-Upsert. Bewusster SQL-Zugriff hier im
Service statt eines eigenen Repos: T18 liefert (Stand dieses Tasks) kein
fundamentals-Repo, und ``storage/repos/`` ist während der Welle B Sperrgebiet
des T18-Fix — das Muster folgt ``downloads.bar_meta_source`` (SQL-Zeilen am
Service, DDL bleibt unangetastet).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quantlab import db
from quantlab.providers import rate_limit
from quantlab.providers.fundamentals import (
    FREQUENZEN,
    FundamentalsProvider,
    validate_frequenz,
    validate_statement,
)
from quantlab.providers.symbols import validate_symbol
from quantlab.storage.repos import freshness_repo

#: Freshness-Domain für Fundamentals-Marker (``data_freshness``, §D.2).
DOMAIN_FUNDAMENTALS = "fundamentals"

# ---------------------------------------------------------------------------
# Provider-Bezug (Injektionspunkt für Tests — Muster services.downloads)
# ---------------------------------------------------------------------------

_PROVIDER: FundamentalsProvider | None = None


def get_provider() -> FundamentalsProvider:
    """Aktiver Fundamentals-Provider (lazy, Prozess-Singleton über den Limiter).

    Seit E087 auf dem **LEICHT**-Circuit (:data:`rate_limit.RATE_LIMITER_LEICHT`):
    der Fundamentals-Refresh teilt sich den Circuit von Quotes-Poll und
    Symbolsuche — nie einen Download-Circuit.
    """
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = FundamentalsProvider(rate_limiter=rate_limit.RATE_LIMITER_LEICHT)
    return _PROVIDER


# ---------------------------------------------------------------------------
# Long-Format: Speichern (vollständiges Ersetzen je statement+freq)
# ---------------------------------------------------------------------------


def dataframe_zu_zeilen(df: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Statement-DataFrame → ``(line_item, as_of, value)``-Zeilen (ohne NaN).

    ``as_of`` wird aus den asOfDate-Spalten (``pd.Timestamp``) als
    ``YYYY-MM-DD`` abgeleitet. Zeilenname und Periode werden als ``str``
    normalisiert — das Long-Format ist damit unabhängig von Pandas-Typen
    speicherbar (SQLite-Vertrag §D.2).
    """
    zeilen: list[tuple[str, str, float]] = []
    if df is None or len(df) == 0:
        return zeilen
    for zeilenname in df.index:
        for spalte in df.columns:
            wert = df.loc[zeilenname, spalte]
            if pd.isna(wert):
                continue  # R17: kein Wert → keine Zeile
            try:
                zahl = float(wert)
            except (TypeError, ValueError):
                continue  # nicht-numerische Zellen (Rare) überspringen
            as_of = pd.Timestamp(spalte).date().isoformat()
            zeilen.append((str(zeilenname).strip(), as_of, zahl))
    return zeilen


def speichere_statement(
    symbol: str,
    statement: str,
    freq: str,
    df: pd.DataFrame,
    *,
    currency: str | None = None,
) -> int:
    """Ersetzt den Bestand je ``(symbol, statement, freq)`` vollständig.

    Liefert die Anzahl gespeicherter Zeilen. Ein **leeres** DataFrame leert
    den Bestand der Kombination bewusst **nicht** (Aufrufer überspringt leere
    Fetches — ANNAHME s. Modul-Docstring); hier bedeutet „aufrufen“:
    „diese Daten sind der neue Stand“. Alles läuft in **einer**
    ``BEGIN IMMEDIATE``-Transaktion ( Lesern nie ein halbes Statement).
    """
    geprueft = validate_symbol(symbol)
    validate_statement(statement)
    validate_frequenz(freq)
    zeilen = dataframe_zu_zeilen(df)
    jetzt = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM fundamentals WHERE symbol = ? AND statement = ? AND freq = ?",
            (geprueft, statement, freq),
        )
        conn.executemany(
            "INSERT INTO fundamentals(symbol, statement, freq, line_item, as_of, "
            "value, currency, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (geprueft, statement, freq, line_item, as_of, wert, currency, jetzt)
                for line_item, as_of, wert in zeilen
            ],
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return len(zeilen)


# ---------------------------------------------------------------------------
# Lesen (Long + Pivot on-read)
# ---------------------------------------------------------------------------


def lies_zeilen(
    symbol: str, statement: str, freq: str
) -> list[dict[str, Any]]:
    """Alle Long-Format-Zeilen einer Kombination (Perioden/Zeilen aufsteigend)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT line_item, as_of, value, currency, updated_at FROM fundamentals "
            "WHERE symbol = ? AND statement = ? AND freq = ? ORDER BY as_of, line_item",
            (validate_symbol(symbol), validate_statement(statement), validate_frequenz(freq)),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "line_item": str(row["line_item"]),
            "as_of": str(row["as_of"]),
            "value": None if row["value"] is None else float(row["value"]),
            "currency": None if row["currency"] is None else str(row["currency"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def lies_statement_pivot(
    symbol: str, statement: str, freq: str
) -> dict[str, Any]:
    """Long-Format → Pivot-Vertrag ``{line_items, periods, values}`` (§C.4).

    ``periods`` aufsteigend (älteste zuerst), ``line_items`` alphabetisch
    (deterministisch — die Yahoo-Zeilenreihenfolge wird im Long-Format nicht
    persistiert), ``values[zeile][spalte]`` mit ``None`` für fehlende
    Kombinationen. ``leer`` zeigt der Router als ``leer:true`` (200 statt
    Fehler, R17).
    """
    zeilen = lies_zeilen(symbol, statement, freq)
    perioden = sorted({zeile["as_of"] for zeile in zeilen})
    line_items = sorted({zeile["line_item"] for zeile in zeilen})
    zellen = {
        (zeile["line_item"], zeile["as_of"]): zeile["value"] for zeile in zeilen
    }
    waehrungen = sorted(
        {zeile["currency"] for zeile in zeilen if zeile["currency"] is not None}
    )
    aktuellst = max((zeile["updated_at"] for zeile in zeilen), default=None)
    return {
        "line_items": line_items,
        "periods": perioden,
        "values": [
            [zellen.get((line_item, periode)) for periode in perioden]
            for line_item in line_items
        ],
        "currency": waehrungen[0] if len(waehrungen) == 1 else None,
        "updated_at": aktuellst,
        "zeilen": len(zeilen),
        "leer": len(zeilen) == 0,
    }


def hat_daten(symbol: str) -> bool:
    """Gibt es zu diesem Symbol irgendeine Fundamentals-Zeile?"""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM fundamentals WHERE symbol = ? LIMIT 1",
            (validate_symbol(symbol),),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def loesche_symbol(symbol: str) -> int:
    """Entfernt alle Fundamentals-Zeilen eines Symbols (Pfad für Aufräumarbeiten)."""
    geprueft = validate_symbol(symbol)
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM fundamentals WHERE symbol = ?", (geprueft,)
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Aktualisieren (Fetch über den Provider, inkl. Freshness-Marker)
# ---------------------------------------------------------------------------


def aktualisiere_symbol(
    symbol: str,
    *,
    provider: FundamentalsProvider | None = None,
    freqs: tuple[str, ...] = FREQUENZEN,
) -> dict[str, Any]:
    """Holt alle Statements (je Frequenz) und speichert sie; Summary als dict.

    Ein Durchgang = **ein** Ticker für alle Statements (netzschonend, R1
    Block 1.2; kein Cross-Refresh-Caching — Details Provider-Docstring).
    Leere Statements werden übersprungen („Quelle liefert nichts“) und zählen
    als ``null`` in der Summary — kein Fehler (R17), vorhandene Daten bleiben
    stehen. Nach dem Lauf wird der Freshness-Marker ``fundamentals`` gesetzt
    (Staleness, E035). Fachliche Fehler des Providers (429/502/422) werden
    deutsch weitergeworfen (Aufrufer — Scheduler — isoliert je Symbol).
    """
    geprueft = validate_symbol(symbol)
    aktiver = provider if provider is not None else get_provider()
    statements: dict[str, int | None] = {}
    zeilen_gesamt = 0
    abrufe = aktiver.fetch_alle_statements(geprueft, freqs=freqs)
    for (statement, freq), df in abrufe.items():
        if df.empty:
            statements[f"{statement}:{freq}"] = None
            continue
        gespeichert = speichere_statement(geprueft, statement, freq, df)
        statements[f"{statement}:{freq}"] = gespeichert
        zeilen_gesamt += gespeichert
    freshness_repo.mark_fresh(DOMAIN_FUNDAMENTALS, geprueft)
    return {
        "symbol": geprueft,
        "statements": statements,
        "zeilen": zeilen_gesamt,
    }


# ---------------------------------------------------------------------------
# Key-Ratios (berechnet aus dem gespeicherten Bestand, §C.4)
# ---------------------------------------------------------------------------

#: Alias-Kandidaten je Eingangswert (yfinance-„pretty“-Namen, klein toleriert).
_ALIASE: dict[str, tuple[str, ...]] = {
    "umsatz": ("Total Revenue", "Operating Revenue"),
    "nettoergebnis": (
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ),
    "ebit": ("EBIT",),
    "ebitda": ("EBITDA",),
    "total_assets": ("Total Assets",),
    "eigenkapital": (
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
    ),
    "umlaufvermoegen": ("Current Assets", "Total Current Assets"),
    "kurzfristige_schulden": ("Current Liabilities", "Total Current Liabilities"),
    "freier_cashflow": ("Free Cash Flow",),
}


def _neueste_werte(symbol: str, statement: str, freq: str) -> tuple[str | None, dict[str, float]]:
    """``(neueste Periode, {line_item: value})`` eines Statements (oder ``({},)``)."""
    zeilen = lies_zeilen(symbol, statement, freq)
    if not zeilen:
        return None, {}
    neueste = max(zeile["as_of"] for zeile in zeilen)
    werte = {
        zeile["line_item"]: zeile["value"]
        for zeile in zeilen
        if zeile["as_of"] == neueste and zeile["value"] is not None
    }
    return neueste, werte


def _alias_wert(werte: dict[str, float], schluessel: str) -> float | None:
    """Erster vorhandener Alias-Wert (case-insensitiv) oder ``None``."""
    klein = {name.casefold(): wert for name, wert in werte.items()}
    for alias in _ALIASE[schluessel]:
        wert = klein.get(alias.casefold())
        if wert is not None:
            return wert
    return None


def _quotient(
    zaehler: float | None, nenner: float | None
) -> float | None:
    """Division mit 0/NONE-Verteidigung — ``None`` statt Fehler/Weltenzahl."""
    if zaehler is None or nenner is None or nenner == 0:
        return None
    return zaehler / nenner


def _umsatzwachstum_yoy(symbol: str, freq: str) -> tuple[float | None, str | None, str | None]:
    """Umsatzwachstum gegen die Vorjahresperiode aus der gespeicherten Reihe.

    Annual: neueste gegen vorletzte Periode (~Jahresabstand). Quarterly:
    neueste gegen die Periode vier Davorliegende (Vorjahresquartal; Yahoo
    liefert genau 5 Quartale, der Vergleich passt ins Fenster). Liefert
    ``(wachstum, aktuelle_periode, vergleichs_periode)``.
    """
    zeilen = [
        zeile
        for zeile in lies_zeilen(symbol, "income", freq)
        if zeile["value"] is not None
        and zeile["line_item"].casefold() in {
            alias.casefold() for alias in _ALIASE["umsatz"]
        }
    ]
    if not zeilen:
        return None, None, None
    perioden = sorted({zeile["as_of"] for zeile in zeilen})
    abstand = 1 if freq == "annual" else 4
    if len(perioden) <= abstand:
        return None, None, None
    def umsatz(periode: str) -> float | None:
        for zeile in zeilen:
            if zeile["as_of"] == periode:
                return zeile["value"]
        return None
    aktuell = umsatz(perioden[-1])
    vergleich = umsatz(perioden[-1 - abstand])
    if aktuell is None or not vergleich:
        return None, perioden[-1], perioden[-1 - abstand]
    return aktuell / vergleich - 1.0, perioden[-1], perioden[-1 - abstand]


def berechne_ratios(symbol: str, *, freq: str = "annual") -> dict[str, Any]:
    """Key-Ratios aus dem gespeicherten Bestand (§C.4: „berechnete Key-Ratios“).

    Basis ist die neueste Periode je Statement der gewünschten Frequenz;
    fehlt die annual vollständig, fällt die Berechnung auf ``quarterly``
    zurück (``basis.freq`` meldet die tatsächlich genutzte Frequenz — viele
    Symbole haben nur Quartalsdaten). Fehlende Eingangswerte lassen die
    jeweilige Kennzahl **entfallen** (R17), Division durch 0 ebenfalls.
    Quartalswerte sind Periodenwerte — **nicht** annualisiert (ANNAHME:
    transparenter als stilisiertes ×4).
    """
    geprueft = validate_symbol(symbol)
    validate_frequenz(freq)
    genutzt = freq
    pivot = lies_statement_pivot(geprueft, "income", freq)
    if pivot["leer"] and freq == "annual":
        genutzt = "quarterly"

    income_periode, income = _neueste_werte(geprueft, "income", genutzt)
    balance_periode, balance = _neueste_werte(geprueft, "balance", genutzt)
    cashflow_periode, cashflow = _neueste_werte(geprueft, "cashflow", genutzt)

    umsatz = _alias_wert(income, "umsatz")
    netto = _alias_wert(income, "nettoergebnis")
    ebit = _alias_wert(income, "ebit")
    ebitda = _alias_wert(income, "ebitda")
    total_assets = _alias_wert(balance, "total_assets")
    eigenkapital = _alias_wert(balance, "eigenkapital")
    umlaufvermoegen = _alias_wert(balance, "umlaufvermoegen")
    kurzfristige_schulden = _alias_wert(balance, "kurzfristige_schulden")
    fcf = _alias_wert(cashflow, "freier_cashflow")

    ratios: list[dict[str, Any]] = []

    def hinzu(
        key: str,
        label: str,
        wert: float | None,
        einheit: str,
        formel: str,
    ) -> None:
        if wert is None:
            return
        ratios.append(
            {
                "key": key,
                "label": label,
                "wert": wert,
                "einheit": einheit,
                "formel": formel,
            }
        )

    # Formeln folgen §C.4 („Netto-Marge = Net Income / Total Revenue …“).
    hinzu(
        "netto_marge", "Netto-Marge", _quotient(netto, umsatz), "anteil",
        "Net Income / Total Revenue",
    )
    hinzu(
        "ebit_marge", "EBIT-Marge", _quotient(ebit, umsatz), "anteil",
        "EBIT / Total Revenue",
    )
    hinzu(
        "ebitda_marge", "EBITDA-Marge", _quotient(ebitda, umsatz), "anteil",
        "EBITDA / Total Revenue",
    )
    hinzu(
        "roe", "Eigenkapitalrendite (ROE)", _quotient(netto, eigenkapital), "anteil",
        "Net Income / Eigenkapital",
    )
    hinzu(
        "roa", "Gesamtkapitalrendite (ROA)", _quotient(netto, total_assets), "anteil",
        "Net Income / Gesamtvermögen",
    )
    hinzu(
        "eigenkapitalquote", "Eigenkapitalquote",
        _quotient(eigenkapital, total_assets), "anteil",
        "Eigenkapital / Gesamtvermögen",
    )
    hinzu(
        "liquiditaetsgrad", "Liquiditätsgrad (Current Ratio)",
        _quotient(umlaufvermoegen, kurzfristige_schulden), "faktor",
        "Umlaufvermögen / kurzfristige Schulden",
    )
    hinzu(
        "fcf_marge", "Free-Cashflow-Marge", _quotient(fcf, umsatz), "anteil",
        "Free Cash Flow / Total Revenue",
    )

    wachstum, wachstum_periode, vergleichs_periode = _umsatzwachstum_yoy(
        geprueft, genutzt
    )
    hinzu(
        "umsatzwachstum_yoy", "Umsatzwachstum (YoY)", wachstum, "wachstum",
        "Total Revenue ggü. Vorjahresperiode",
    )

    return {
        "symbol": geprueft,
        "freq": genutzt,
        "basis": {
            "income": income_periode,
            "balance": balance_periode,
            "cashflow": cashflow_periode,
            "vergleichsperiode": vergleichs_periode,
        },
        "ratios": ratios,
        "leer": not ratios,
    }


__all__ = [
    "DOMAIN_FUNDAMENTALS",
    "aktualisiere_symbol",
    "berechne_ratios",
    "dataframe_zu_zeilen",
    "get_provider",
    "hat_daten",
    "lies_statement_pivot",
    "lies_zeilen",
    "loesche_symbol",
    "speichere_statement",
]
