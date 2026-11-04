"""Live-Quotes-Service (T22, E024): TTL-Cache, Single-Flight, SSE-Push.

Fachlogik zwischen Quotes-Provider und Router/Scheduler:

- **TTL-Cache je Symbol** (:class:`QuotesService`): Default-TTL = Poll-
  Intervall (``live.quotes_poll_s``, geklemmt auf 15–300 s — E024 „TTL-Cache
  je Symbol (Default = Poll-Intervall)“); für Tests injizierbar (``ttl_s`` +
  ``clock``). Fehlgeschlagene Fetches werden **nicht** negativ gecacht — der
  nächste Abruf versucht wieder (Fehler sind je-Symbol-Felder, kein Batch-
  Kipper).
- **Single-Flight:** parallele Abrufe desselben Symbols führen zu **genau
  einem** Provider-Call — ein :class:`threading.Lock` je Symbol (Threadpool-
  Router-Handler und Poll-Tick teilen dieselben Locks); eine Cache-Prüfung
  unter dem Symbol-Lock (striktes Single-Flight). Der Lock-Zustand wächst
  mit der Symbolmenge und wird nie geschrumpft (Watchlists sind klein —
  bewusst simpel).
- **SSE nur geänderte Werte:** :meth:`QuotesService.geaenderte_seit_publikation`
  vergleicht ``(price, previous_close, currency)`` mit dem zuletzt per SSE
  publizierten Stand — unveränderte Quotes erzeugen **kein** Event (kein
  Spam, E024). Der GET-Endpunkt aktualisiert bewusst **nicht** die Publikations-
  Marker: Er liefert Werte, der Push-Takt gehört dem Poll.
- **Poll-Tick:** :func:`tick_quotes` holt die Default-Watchlist (Liste 1;
  Depot-Symbole kommen mit T28), liefert fehlende Quotes batched nach und
  publiziert das ``quotes``-Event (§C.5) — Fehler je Symbol isolieren.

Einstellungen (Sektion ``live``, E024): ``quotes_enabled`` (Default true) und
``quotes_poll_s`` (15–300, Default 60) — gelesen über
:func:`live_einstellungen`, defensiv geklemmt (Hintergrundjobs sterben nie an
einer verstümmelten Einstellung).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from quantlab import events
from quantlab.providers import rate_limit
from quantlab.providers.base import ProviderError
from quantlab.providers.quotes import Quote, QuotesProvider
from quantlab.providers.yfinance_provider import jetzt_us
from quantlab.storage.repos import settings_repo, watchlists_repo

logger = logging.getLogger(__name__)

#: E024-Vertrag: Poll-Intervall-Grenzen (15–300 s) und Default 60 s.
MIN_QUOTES_POLL_S = 15.0
MAX_QUOTES_POLL_S = 300.0
DEFAULT_QUOTES_POLL_S = 60.0

#: Fallback-Defaults, falls die Settings-Sektion fehlt (Additivität schützt).
_FALLBACK_EINSTELLUNGEN: dict[str, Any] = {
    "quotes_enabled": True,
    "quotes_poll_s": DEFAULT_QUOTES_POLL_S,
}


def live_einstellungen() -> dict[str, Any]:
    """Settings-Sektion ``live`` (E024): quotes_enabled/quotes_poll_s, geklemmt.

    Ungültige Werte (falscher Typ/negativ) werden auf Defaults bzw. den
    E024-Bereich geklemmt statt den Poll zu kippen (Muster
    ``scheduler.scheduler_einstellungen``).
    """
    try:
        sektion = settings_repo.get_section("live")
    except ValueError:  # pragma: no cover — Defaults sind ab T22 vorhanden
        sektion = dict(_FALLBACK_EINSTELLUNGEN)

    roh_poll = sektion.get("quotes_poll_s", DEFAULT_QUOTES_POLL_S)
    try:
        poll = float(roh_poll)
    except (TypeError, ValueError):
        poll = DEFAULT_QUOTES_POLL_S
    poll = min(max(poll, MIN_QUOTES_POLL_S), MAX_QUOTES_POLL_S)

    roh_enabled = sektion.get("quotes_enabled", True)
    enabled = roh_enabled if isinstance(roh_enabled, bool) else True

    return {"quotes_enabled": enabled, "quotes_poll_s": poll}


def watchlist_symbole() -> list[str]:
    """Symbole der Default-Watchlist (Liste 1) — der Poll-/Batch-Default-Umfang."""
    return [
        eintrag.symbol
        for eintrag in watchlists_repo.list_symbols(watchlists_repo.DEFAULT_WATCHLIST_ID)
    ]


# ---------------------------------------------------------------------------
# Provider-Bezug (Injektionspunkt für Tests — Muster services.fundamentals)
# ---------------------------------------------------------------------------

_PROVIDER: QuotesProvider | None = None


def get_provider() -> QuotesProvider:
    """Aktiver Quotes-Provider (lazy, Prozess-Singleton über den Limiter).

    Seit E087 auf dem **LEICHT**-Circuit (:data:`rate_limit.RATE_LIMITER_LEICHT`):
    ein 429 im Quotes-Poll eskaliert nie einen Download-Cooldon mehr.
    """
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = QuotesProvider(rate_limiter=rate_limit.RATE_LIMITER_LEICHT)
    return _PROVIDER


# ---------------------------------------------------------------------------
# Service: TTL-Cache + Single-Flight + Änderungserkennung
# ---------------------------------------------------------------------------


class QuotesService:
    """Batch-Quotes mit TTL-Cache und Single-Flight (E024).

    ``ttl_s=None`` heißt „Poll-Intervall aus der ``live``-Sektion je Abruf
    gelesen“ (E024-Default); ``clock`` ist injizierbar für deterministische
    TTL-Tests. Fehler je Symbol werden als ``{"symbol": …, "error": {code,
    message}}``-Eintrag geliefert — die anderen Symbole bleiben unberührt.
    """

    def __init__(
        self,
        *,
        provider: QuotesProvider | None = None,
        ttl_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider = provider
        self._ttl_s = ttl_s
        self._clock = clock
        self._cache: dict[str, tuple[Quote, float]] = {}
        #: Zuletzt per SSE publizierter Wertestand je Symbol (Price-Änderungs-
        #: Erkennung; GET-Abrufe schreiben hier bewusst nicht hinein).
        self._publikationen: dict[str, tuple[float | None, float | None, str | None]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_waechter = threading.Lock()

    # -- Cache/Single-Flight --------------------------------------------

    def _lock_fuer(self, symbol: str) -> threading.Lock:
        with self._locks_waechter:
            lock = self._locks.get(symbol)
            if lock is None:
                lock = threading.Lock()
                self._locks[symbol] = lock
            return lock

    def _ttl(self) -> float:
        if self._ttl_s is not None:
            return max(0.0, float(self._ttl_s))
        return live_einstellungen()["quotes_poll_s"]

    def liefere_eines(self, symbol: str, *, erzwingen: bool = False) -> Quote:
        """Ein Quote — aus dem Cache, solange frisch; sonst genau ein Fetch.

        ``erzwingen=True`` überspringt den Cache-Leseteil (Refresh; der neue
        Wert ersetzt den Cache-Eintrag).
        """
        with self._lock_fuer(symbol):
            if not erzwingen:
                gecacht = self._cache.get(symbol)
                if gecacht is not None and self._clock() - gecacht[1] < self._ttl():
                    return gecacht[0]
            quote = self._provider_objekt().fetch_quote(symbol)
            self._cache[symbol] = (quote, self._clock())
            return quote

    def liefere(self, symbole: list[str], *, erzwingen: bool = False) -> list[dict[str, Any]]:
        """Batch: je Symbol einen Quote-Eintrag oder einen Fehler-Eintrag.

        Ein fehlerhaftes Symbol kippt nie den Batch (E024/DoD d): der Fehler
        steht im Feld dieses Symbols (``error: {code, message}``), alle
        anderen werden geliefert.
        """
        eintraege: list[dict[str, Any]] = []
        for symbol in symbole:
            try:
                eintraege.append(self.liefere_eines(symbol, erzwingen=erzwingen).zu_dict())
            except ProviderError as exc:
                eintraege.append(
                    {"symbol": symbol, "error": {"code": exc.code, "message": exc.message}}
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — Isolation je Symbol
                logger.exception("Quote-Abruf für %s fehlgeschlagen", symbol)
                eintraege.append(
                    {
                        "symbol": symbol,
                        "error": {
                            "code": "provider_fehler",
                            "message": str(exc) or exc.__class__.__name__,
                        },
                    }
                )
        return eintraege

    def _provider_objekt(self) -> QuotesProvider:
        return self._provider if self._provider is not None else get_provider()

    # -- SSE: nur geänderte Werte ----------------------------------------

    def geaenderte_seit_publikation(
        self, eintraege: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filtert Fehler-freie Einträge mit neuem ``(price, previous_close, currency)``.

        Side-effect: die als „geändert“ erkannten Werte werden als neuer
        Publikations-Stand vermerkt — derselbe Wert wird kein zweites Mal
        gepusht (E024 „kein Spam“).
        """
        geaendert: list[dict[str, Any]] = []
        for eintrag in eintraege:
            if "error" in eintrag or eintrag.get("price") is None:
                continue
            schluessel = (
                eintrag.get("price"),
                eintrag.get("previous_close"),
                eintrag.get("currency"),
            )
            if self._publikationen.get(eintrag["symbol"]) != schluessel:
                self._publikationen[eintrag["symbol"]] = schluessel
                geaendert.append(eintrag)
        return geaendert

    # -- Poll-Tick --------------------------------------------------------

    async def tick(self, *, erzwingen: bool = False) -> dict[str, Any]:
        """Ein Poll-Durchlauf: Watchlist-Quotes holen, Änderungen per SSE pushen.

        Rückgabe: Tick-Details für ``scheduler_tick`` (``ok`` = mind. ein
        Symbol geliefert; Fehler je Symbol im ``fehler``-Text).
        """
        symbole = await asyncio.to_thread(watchlist_symbole)
        if not symbole:
            return {"ok": True, "symbole": 0, "geaendert": 0, "hinweis": "Watchlist ist leer."}
        eintraege = await asyncio.to_thread(self.liefere, symbole, erzwingen=erzwingen)
        geaenderte = self.geaenderte_seit_publikation(eintraege)
        fehler: list[str] = [
            f"{eintrag['symbol']}: {eintrag['error']['message']}"
            for eintrag in eintraege
            if "error" in eintrag
        ]
        if geaenderte:
            zeitstempel = max(
                (eintrag["quote_ts"] for eintrag in geaenderte if "quote_ts" in eintrag),
                default=jetzt_us(),
            )
            events.quotes(geaenderte, zeitstempel)
        return {
            "ok": len(fehler) < len(symbole),
            "symbole": len(symbole),
            "geaendert": len(geaenderte),
            "fehler": "; ".join(fehler) or None,
        }


# ---------------------------------------------------------------------------
# Prozess-Singletons + Scheduler-Anschluss (T21-Gerüst ruft tick_quotes)
# ---------------------------------------------------------------------------

_SERVICE: QuotesService | None = None


def get_service() -> QuotesService:
    """Aktiver Quotes-Service (lazy; TTL aus ``live``-Settings, echte Clock).

    Tests ersetzen ihn per ``monkeypatch.setattr(quotes_service, "get_service",
    …)`` — Router und Scheduler holen den Service **zur Laufzeit**.
    """
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = QuotesService()
    return _SERVICE


async def tick_quotes(*, erzwingen: bool = False) -> dict[str, Any]:
    """Poll-Tick für den Scheduler (``run-now`` erzwingt den Refresh)."""
    return await get_service().tick(erzwingen=erzwingen)


__all__ = [
    "DEFAULT_QUOTES_POLL_S",
    "MAX_QUOTES_POLL_S",
    "MIN_QUOTES_POLL_S",
    "QuotesService",
    "get_provider",
    "get_service",
    "live_einstellungen",
    "tick_quotes",
    "watchlist_symbole",
]
