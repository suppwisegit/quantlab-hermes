"""Live-Quotes-Provider (T22, E024): ``Ticker.fast_info`` über die T19-Schicht.

Holt den aktuellen Kurs je Symbol über ``yf.Ticker(symbol).fast_info`` —
**nicht** über ``.info`` (RECHERCHE v2 Block 1.1/1.2: fast_info ist der
beabsichtigte Weg für Live-Quotes; ``.info`` bündelt einen großen
quoteSummary-Call, den Live-Ticker nicht brauchen). ``FastInfo`` imitiert ein
dict; die offiziellen Keys sind camelCase (``lastPrice``), snake_case wird
„secretly supported“ — gelesen wird camelCase mit snake_case-Fallback.

Felder (Kern, E024-Quote-Objekt): ``price`` (``lastPrice``), ``previous_close``,
``change_pct`` (berechnet: ``(last − previousClose)/previousClose · 100``),
``currency`` und ``quote_ts``. Ergänzend die einfachen fast_info-Kennzahlen
(``day_high/day_low, year_high/year_low, market_cap,
fifty_day_average/two_hundred_day_average``) — nur gesetzt, wenn die Quelle sie
liefert.

**quote_ts = Abrufzeitpunkt** (int64 µs UTC, E006): fast_info liefert keine
Kurszeit — die Abrufzeit ist die bestverfügbare Näherung und macht
Marktgenschlossenheit sichtbar (Risiko §T22, Staleness).

Netzschonung über die T19-Fetch-Schicht (E035), Muster
:mod:`quantlab.providers.fundamentals`:

- **Pacing + Circuit-Breaker:** vor jedem fast_info-Zugriff läuft
  :meth:`quantlab.providers.rate_limit.YahooRateLimiter.vor_yahoo_aufruf`.
  ``YFRateLimitError`` wird gezielt gefangen (yfinance retryt intern nicht):
  kein Retry, direkt :class:`ProviderError` 429 ``yahoo_gedrosselt``.
- **Ticker je Abruf:** ein frisches ``yf.Ticker``-Objekt pro Quote (fast_info
  cachet auf dem Ticker — über Durchgänge gehaltene Ticker würden alte Kurse
  liefern).
- **Backoff + Jitter** für transient scheiternde Abrufe (injizierbar, Tests).

**Delisted (T20-Vertrag):** ist ein Symbol in der Registry als delistet
vermerkt (``symbols.delisted=1``), antwortet der Provider sofort mit
404 ``symbol_delisted`` — **ohne** Yahoo zu erreichen. Liefert fast_info
keinen Kurs für ein bekanntes Symbol, wird der Delisted-Vermerk best-effort
gesetzt (Muster ``yfinance_provider`` bei leerer History) und ebenfalls
404 ``symbol_delisted`` gemeldet; unbekannte Symbole erhalten 404
``keine_daten``.

yfinance wird lazy importiert (Muster ``_roh_download``) — Import und
Testsuite bleiben netzfrei; Tests patchen ``yfinance.Ticker`` direkt.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from quantlab.providers import rate_limit
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.providers.yfinance_provider import jetzt_us
from quantlab.services import data_service

#: Kern-Felder des E024-Quote-Objekts (Response-/SSE-Vertrag §C.4/§C.5).
KERNFELDER: tuple[str, ...] = (
    "symbol",
    "price",
    "previous_close",
    "change_pct",
    "currency",
    "quote_ts",
)

#: fast_info-Zusatzfelder (RECHERCHE v2 Block 1.1) — Quote-Attribut → camelCase-
#: Key der Quelle (mit snake_case-Fallback); im JSON nur wenn geliefert.
_ZUSATZFELDER: tuple[tuple[str, str], ...] = (
    ("day_high", "dayHigh"),
    ("day_low", "dayLow"),
    ("year_high", "yearHigh"),
    ("year_low", "yearLow"),
    ("market_cap", "marketCap"),
    ("fifty_day_average", "fiftyDayAverage"),
    ("two_hundred_day_average", "twoHundredDayAverage"),
)

#: Quellen-Keys für den Vortagesschluss („regularMarketPreviousClose“ ist die
#: explizite Handelstags-Variante — Fallback, falls „previousClose“ fehlt).
_KEY_PREVIOUS_CLOSE = ("previousClose", "previous_close")
_KEY_PREVIOUS_CLOSE_FALLBACK = (
    "regularMarketPreviousClose",
    "regular_market_previous_close",
)


@dataclass(frozen=True)
class Quote:
    """Ein Live-Quote im E024-Vertrag; ``quote_ts`` int64 µs UTC (Abrufzeit)."""

    symbol: str
    price: float | None
    previous_close: float | None
    change_pct: float | None
    currency: str | None
    quote_ts: int
    day_high: float | None = None
    day_low: float | None = None
    year_high: float | None = None
    year_low: float | None = None
    market_cap: float | None = None
    fifty_day_average: float | None = None
    two_hundred_day_average: float | None = None

    def zu_dict(self) -> dict[str, Any]:
        """Kernfelder immer, Zusatzfelder nur wenn geliefert (kompaktes JSON)."""
        eintrag: dict[str, Any] = {
            "symbol": self.symbol,
            "price": self.price,
            "previous_close": self.previous_close,
            "change_pct": self.change_pct,
            "currency": self.currency,
            "quote_ts": self.quote_ts,
        }
        for feld, _quelle in _ZUSATZFELDER:
            wert = getattr(self, feld)
            if wert is not None:
                eintrag[feld] = wert
        return eintrag


def _zahl(roh: Any) -> float | None:
    """Best-effort float (``None`` bei None/NaN/nicht-numerisch)."""
    if roh is None or isinstance(roh, bool):
        return None
    try:
        wert = float(roh)
    except (TypeError, ValueError):
        return None
    if wert != wert:  # NaN
        return None
    return wert


def _text(roh: Any) -> str | None:
    wert = roh.strip() if isinstance(roh, str) else None
    return wert or None


def _lese(roh: dict[str, Any], fast_info: Any, *keys: str) -> Any:
    """Erster vorhandener Wert aus dem dict-Bild oder direkt am Objekt."""
    for key in keys:
        if key in roh:
            return roh[key]
    for key in keys:
        wert = getattr(fast_info, key, None)
        if wert is not None:
            return wert
    return None


def quote_aus_fast_info(
    symbol: str, fast_info: Any, *, quote_ts: int | None = None
) -> Quote:
    """Baut das E024-Quote aus einem fast_info-Objekt (dict-artig).

    Reine Abbildung ohne Netz — separat testbar. ``change_pct`` wird nur bei
    vorhandenem, positivem ``previous_close`` berechnet, sonst ``None``
    (z. B. Indizes ohne Vortagesschluss).
    """
    try:
        roh: dict[str, Any] = dict(fast_info)
    except Exception:  # noqa: BLE001 — FastInfo kann beim dict() Netz nachladen
        try:
            roh = {str(schluessel): wert for schluessel, wert in fast_info.items()}
        except Exception:  # noqa: BLE001 — auch items() kann Netz nachladen
            roh = {}
    preis = _zahl(_lese(roh, fast_info, "lastPrice", "last_price"))
    vortag = _zahl(_lese(roh, fast_info, *_KEY_PREVIOUS_CLOSE))
    if vortag is None:
        vortag = _zahl(_lese(roh, fast_info, *_KEY_PREVIOUS_CLOSE_FALLBACK))
    change_pct: float | None = None
    if preis is not None and vortag is not None and vortag > 0:
        change_pct = (preis - vortag) / vortag * 100.0
    zusaetze = {
        feld: _zahl(_lese(roh, fast_info, quelle, feld))
        for feld, quelle in _ZUSATZFELDER
    }
    return Quote(
        symbol=symbol,
        price=preis,
        previous_close=vortag,
        change_pct=change_pct,
        currency=_text(_lese(roh, fast_info, "currency")),
        quote_ts=jetzt_us() if quote_ts is None else quote_ts,
        **zusaetze,
    )


class QuotesProvider:
    """Live-Quote-Abruf über ``yf.Ticker.fast_info`` (Guard je Abruf).

    Alle Konstruktionsparameter sind injizierbar (Muster
    :class:`~quantlab.providers.fundamentals.FundamentalsProvider`) — Tests
    stellen Versuche/Backoff ab und injizieren einen Limiter mit
    ``pacing_s=0``.
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

    def fetch_quote(self, symbol: str) -> Quote:
        """Holt den Live-Quote eines Symbols (Guard vor dem Netzcall).

        Wirft :class:`ProviderError` (deutsch) bei Delisted (404, ohne Netz),
        fehlendem Kurs (404), Drosselung (429) oder unerreichbarer Quelle (502).
        """
        geprueft = validate_symbol(symbol)
        if data_service.ist_delisted(geprueft):
            raise ProviderError(
                f"„{geprueft}“ ist als delistet vermerkt — Live-Kurse sind "
                "nicht mehr verfügbar (Bestand bleibt lesbar).",
                404,
                "symbol_delisted",
                hint="Der gespeicherte Bestand bleibt lesbar; neue Abrufe sind "
                "für delistierte Symbole nicht mehr möglich.",
            )
        fast_info = self._mit_wiederholung(self._fast_info_lesen, geprueft)
        quote = quote_aus_fast_info(geprueft, fast_info)
        if quote.price is None:
            # T20-Vertrag: bekanntes Symbol ohne Kurs → Delisted-Vermerk
            # best-effort setzen; unbekannte → generisches „keine Daten“.
            if data_service.vermerke_delisted_wenn_bekannt(geprueft):
                raise ProviderError(
                    f"„{geprueft}“ liefert keinen Kurs mehr — das Symbol wurde "
                    "als delistet vermerkt (Bestand bleibt erhalten).",
                    404,
                    "symbol_delisted",
                )
            raise ProviderError(
                f"„{geprueft}“ liefert keinen aktuellen Kurs.",
                404,
                "keine_daten",
                hint="Prüfe das Yahoo-Kürzel (z. B. „AAPL“, „^GSPC“).",
            )
        return quote

    # ------------------------------------------------------------------
    # Innere Mechanik
    # ------------------------------------------------------------------

    def _fast_info_lesen(self, symbol: str) -> Any:
        """Ein fast_info-Abruf — der eigentliche Netzcall.

        Vor dem Zugriff: Pacing/Circuit-Guard des prozessweiten Limiters
        (T19/E035) — bei offenem Circuit fliegt hier sofort
        :class:`~quantlab.providers.rate_limit.YahooGedrosseltError`, ohne
        dass Yahoo erreicht wird. Der Ticker ist bewusst frisch je Abruf
        (fast_info cachet auf dem Ticker-Objekt).
        """
        import yfinance as yf  # lazy: Import/Testsuite bleibt netzfrei

        self._limiter.vor_yahoo_aufruf()
        return yf.Ticker(symbol).fast_info

    def _mit_wiederholung(self, func, *args, **kwargs):
        """Exponentielles Backoff + Jitter (Start 2 s); Muster Fundamentals.

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
            f"Yahoo-Abruf nach {self._max_versuchen} Versuchen fehlgeschlagen: {letzter_fehler}",
            502,
            "quelle_nicht_erreichbar",
            hint="Yahoo drosselt ggf. — bitte wenig später erneut versuchen.",
        ) from letzter_fehler


__all__ = [
    "KERNFELDER",
    "Quote",
    "QuotesProvider",
    "quote_aus_fast_info",
]
