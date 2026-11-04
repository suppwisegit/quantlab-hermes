"""Fundamentals-Provider (T21, R1 Block 1.3): Abschlussberichte via yfinance.

Holt ``income_stmt`` / ``balance_sheet`` / ``cash_flow`` — jeweils annual
(``yearly``) und quarterly (``quarterly``) — über ``yf.Ticker``. Der DataFrame-
Vertrag von yfinance 1.7.0 ([LOKAL] ``scrapers/fundamentals.py:116``):
**Index = Kennzahlennamen** (pretty, z. B. ``"Total Revenue"``; Präfix
``annual|quarterly|trailing`` ist entfernt), **Spalten = ``pd.Timestamp`` der
``asOfDate``** (absteigend sortiert — neueste Periode zuerst), Werte float.
Yahoo liefert maximal **4 Jahre (annual) bzw. 5 Quartale (quarterly)**,
unabhängig vom Zeitraum — die begrenzte Historie ist Dokumentationsvertrag
dieses Moduls, nicht konfigurierbar. (Live beobachtet, AAPL 2026-09: bis zu
5 Jahresperioden inklusive des laufenden Geschäftsjahres — die Kappung ist
Quellen-Verhalten, kein harter Codepfad; gespeichert wird alles Gelieferte.)

Netzschonung über die T19-Fetch-Schicht (E035):

- **Pacing + Circuit-Breaker:** vor **jedem** Statement-Abruf läuft
  :meth:`quantlab.providers.rate_limit.YahooRateLimiter.vor_yahoo_aufruf`
  — der prozessweite Mindestabstand (0,75 s) und ein offener Cool-down
  greifen damit automatisch auch für Fundamentals. ``YFRateLimitError`` wird
  (wie im yfinance-Provider) gezielt gefangen: kein Retry, direkt
  :class:`ProviderError` 429 ``yahoo_gedrosselt``.
- **Ticker je Durchgang (R1 Block 1.2):** :meth:`fetch_alle_statements` baut
  **ein** ``yf.Ticker``-Objekt für alle Statements eines Refresh-Durchgangs
  (geteilte Session/Crumb; yfinance cachet Financials je Statement+Frequenz
  auf dem Ticker). Der Ticker wird **nicht** über Durchgänge hinweg
  gehalten — yfinances internes Statement-Caching würde sonst einen Refresh
  Tage später einfrieren (gestecktes Statement statt neuem Netzabruf).
- **Backoff + Jitter** für transient scheiternde Abrufe (injizierbar für
  Tests; Muster :mod:`quantlab.providers.yfinance_provider`).

Fehlende Statements sind **kein Fehler**: Yahoo liefert für viele Symbole
(z. B. ETFs, Indizes) einzelne Statements nicht — yfinance gibt dann ein
leeres DataFrame zurück, der Provider liefert es unverändert (``leer``)
weiter; die Bewertung (``leer:true``-Antwort statt Fehler, R17) macht der
Service.

yfinance wird lazy importiert (Muster ``_roh_download``) — Import und
Testsuite bleiben netzfrei; Tests patchen ``yfinance.Ticker`` direkt.
"""

from __future__ import annotations

import random
import time
from typing import Any

import pandas as pd

from quantlab.providers import rate_limit
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol

#: Gültige Statement-Arten (Spalte ``fundamentals.statement``, §D.2).
STATEMENTS: tuple[str, ...] = ("income", "balance", "cashflow")

#: Gültige Frequenzen (Spalte ``fundamentals.freq``, §D.2).
FREQUENZEN: tuple[str, ...] = ("annual", "quarterly")

#: yfinance-Attribut je (statement, freq) — die Properties liefern bereits
#: „pretty“ Zeilennamen (``camel2title``) und die asOfDate-Spaltentimestamps.
_STATEMENT_ATTRS: dict[tuple[str, str], str] = {
    ("income", "annual"): "income_stmt",
    ("income", "quarterly"): "quarterly_income_stmt",
    ("balance", "annual"): "balance_sheet",
    ("balance", "quarterly"): "quarterly_balance_sheet",
    ("cashflow", "annual"): "cash_flow",
    ("cashflow", "quarterly"): "quarterly_cash_flow",
}

#: Dokumentierte Maximale Historie je Frequenz (R1 Block 1.3 — Yahoo-Kappung;
#: Grundgrößenordnung, live können es u. U. +1 laufende Periode sein).
MAX_JAHRE_ANNUAL = 4
MAX_QUARTALE_QUARTERLY = 5


def validate_statement(statement: str) -> str:
    """Prüft die Statement-Art gegen die Schema-Whitelist (§D.2) — 422 deutsch."""
    if statement not in STATEMENTS:
        raise ProviderError(
            f"Unbekannte Statement-Art „{statement}“.",
            422,
            "ungueltiges_statement",
            hint=f"Gültige Arten: {', '.join(STATEMENTS)}.",
        )
    return statement


def validate_frequenz(freq: str) -> str:
    """Prüft die Frequenz gegen die Schema-Whitelist (§D.2) — 422 deutsch."""
    if freq not in FREQUENZEN:
        raise ProviderError(
            f"Unbekannte Frequenz „{freq}“.",
            422,
            "ungueltige_frequenz",
            hint=f"Gültige Frequenzen: {', '.join(FREQUENZEN)}.",
        )
    return freq


class FundamentalsProvider:
    """Abruf der Abschlussberichte über ``yf.Ticker`` (guard pro Statement).

    Alle Konstruktionsparameter sind injizierbar (Muster
    :class:`~quantlab.providers.yfinance_provider.YfinanceProvider`) — Tests
    stellen Versuche/Backoff ab und injizieren einen Limiter mit eigener
    Clock bzw. ``pacing_s=0``.
    """

    def __init__(
        self,
        *,
        max_versuche: int = 3,
        backoff_start_s: float = 2.0,
        jitter_faktor: float = 0.5,
        rate_limiter: rate_limit.YahooRateLimiter | None = None,
    ) -> None:
        self._max_versuche = max(1, int(max_versuche))
        self._backoff_start_s = backoff_start_s
        self._jitter_faktor = jitter_faktor
        self._limiter = (
            rate_limiter if rate_limiter is not None else rate_limit.RATE_LIMITER
        )

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def fetch_alle_statements(
        self, symbol: str, *, freqs: tuple[str, ...] = FREQUENZEN
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """Holt alle Statements der Frequenzen mit **einem** Ticker je Durchgang.

        Rückgabe: ``{(statement, freq): DataFrame}`` — fehlende Statements
        stehen als leeres DataFrame drin (kein Fehler, R17). Ein Durchgang =
        ein Ticker-Objekt (s. Modul-Docstring); wer einzelne Statements
        braucht, nutzt :meth:`fetch_statement`.
        """
        geprueft = validate_symbol(symbol)
        for freq in freqs:
            validate_frequenz(freq)
        ticker = self._neuer_ticker(geprueft)
        ergebnis: dict[tuple[str, str], pd.DataFrame] = {}
        for (statement, freq), attr in _STATEMENT_ATTRS.items():
            if freq not in freqs:
                continue
            roh = self._mit_wiederholung(self._statement_am_ticker, ticker, attr)
            ergebnis[(statement, freq)] = self._normalisiere(roh)
        return ergebnis

    def fetch_statement(self, symbol: str, statement: str, freq: str) -> pd.DataFrame:
        """Lädt **ein** Statement (``leeres`` DataFrame, wenn Yahoo nichts liefert).

        Wirft :class:`ProviderError` (deutsch) bei Drosselung (429), unerreichbarer
        Quelle (502) oder ungültigen Argumenten (422). Das zurückgegebene
        DataFrame hat den yfinance-Rohvertrag: Index = Kennzahl, Spalten =
        asOfDate-``pd.Timestamp`` **absteigend**, Werte float.
        """
        geprueft = validate_symbol(symbol)
        validate_statement(statement)
        validate_frequenz(freq)
        attr = _STATEMENT_ATTRS[(statement, freq)]
        ticker = self._neuer_ticker(geprueft)
        roh = self._mit_wiederholung(self._statement_am_ticker, ticker, attr)
        return self._normalisiere(roh)

    # ------------------------------------------------------------------
    # Innere Mechanik
    # ------------------------------------------------------------------

    def _neuer_ticker(self, symbol: str) -> Any:
        """Frisches ``yf.Ticker``-Objekt (lazy importiert — Tests patchbar)."""
        import yfinance as yf  # lazy: Import/Testsuite bleibt netzfrei

        return yf.Ticker(symbol)

    def _statement_am_ticker(self, ticker: Any, attr: str) -> pd.DataFrame:
        """Ein Statement-Attribut am Ticker — der eigentliche Netzcall.

        Vor dem Attribut-Zugriff: Pacing/Circuit-Guard des prozessweiten
        Limiters (T19/E035) — bei offenem Circuit fliegt hier sofort
        :class:`~quantlab.providers.rate_limit.YahooGedrosseltError`,
        ohne dass Yahoo erreicht wird.
        """
        self._limiter.vor_yahoo_aufruf()
        ergebnis = getattr(ticker, attr)
        if ergebnis is None:
            return pd.DataFrame()
        return ergebnis

    @staticmethod
    def _normalisiere(roh: pd.DataFrame | None) -> pd.DataFrame:
        """Leere/None-Ergebnisse → leeres DataFrame im Vertrag."""
        if roh is None or len(roh) == 0 or len(roh.columns) == 0:
            return pd.DataFrame()
        return roh

    def _mit_wiederholung(self, func, *args, **kwargs):
        """Exponentielles Backoff + Jitter (Start 2 s); Muster yfinance-Provider.

        **Ausnahme Drosselung (E035):** ``YFRateLimitError`` wird von yfinance
        bewusst ohne internes Retry geworfen (R1 Block 1.8) — auch hier kein
        Retry: Circuit öffnet sofort, Fehler als :class:`ProviderError` 429
        ``yahoo_gedrosselt``. Erschöpfte Versuche → 502
        ``quelle_nicht_erreichbar``.
        """
        pause = self._backoff_start_s
        letzter_fehler: BaseException | None = None
        for versuch in range(self._max_versuche):
            try:
                ergebnis = func(*args, **kwargs)
                self._limiter.verzeichne_erfolg()
                return ergebnis
            except rate_limit.YahooGedrosseltError:
                raise  # Guard-Block: Call fand nicht statt — kein Backoff
            except Exception as exc:  # noqa: BLE001 — jede Netz-/Formatpanne zählt
                letzter_fehler = exc
                if rate_limit.ist_rate_limit_fehler(exc):
                    dauer = self._limiter.verzeichne_rate_limit()
                    raise ProviderError(
                        "Yahoo drosselt die Anfragen (429 Too Many Requests). "
                        f"Weitere Yahoo-Abrufe werden für {dauer:.0f} s ausgesetzt.",
                        429,
                        "yahoo_gedrosselt",
                        hint="Etwas seltener anfragen — der Cool-down verdoppelt "
                        "sich bei jeder weiteren Drosselung (max. 900 s).",
                    ) from exc
                if versuch < self._max_versuche - 1:
                    time.sleep(pause + random.uniform(0, pause * self._jitter_faktor))
                    pause *= 2
        raise ProviderError(
            f"Yahoo-Abruf nach {self._max_versuche} Versuchen fehlgeschlagen: {letzter_fehler}",
            502,
            "quelle_nicht_erreichbar",
            hint="Yahoo drosselt ggf. — bitte wenig später erneut versuchen.",
        ) from letzter_fehler


__all__ = [
    "FREQUENZEN",
    "FundamentalsProvider",
    "MAX_JAHRE_ANNUAL",
    "MAX_QUARTALE_QUARTERLY",
    "STATEMENTS",
    "validate_frequenz",
    "validate_statement",
]
