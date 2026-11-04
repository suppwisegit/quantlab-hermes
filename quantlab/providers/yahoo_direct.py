"""Yahoo-Direkt-Provider (T-F9, E085): httpx auf ``v8/finance/chart`` als Primärbein.

Ersetzt den yfinance-Primärabruf durch eine eigene dünne HTTP-Hülle — yfinance
wurde bereits nur als solche gefahren (``threads=False``, eigener Timeout); der
v8/chart-Endpunkt liefert dieselben Daten ohne Library-Overhead. Vertrag
(RECHERCHE F.7, LIVE-verifiziert 2026-09-21):

- **Endpoint:** ``GET https://query2.finance.yahoo.com/v8/finance/chart/{symbol}``
  mit ``interval``, ``range`` **oder** ``period1``/``period2`` (Unix-Sekunden),
  ``events=div,splits,capitalGains`` und ``includeAdjustedClose=true`` —
  **ohne Cookie/Crumb**, Browser-User-Agent genügt [LIVE].
- **Struktur:** ``chart.result[0]`` mit ``meta`` (``exchangeTimezoneName``,
  ``currency``, ``instrumentType``), ``timestamp[]`` (Unix-Sekunden UTC),
  ``indicators.quote[0]{open,high,low,close,volume}`` (**null-Lücken möglich**)
  und ``indicators.adjclose[0].adjclose`` (Rohbasis für ``adj_factor``, E022).
- **Events:** ``events.dividends = {ts: {amount, date}}``, ``events.splits =
  {ts: {numerator, denominator, date}}``; ``capitalGains`` wird angefordert
  (gleicher String wie yfinance, F.7), aber nicht persistiert — das Schema
  ``corporate_actions`` kennt nur ``dividend``/``split`` (das yfinance-Bein
  schreibt dieselben zwei Typen).
- **Grenzen:** identisch zum yfinance-Bein — Intraday max. 60 Tage zurück
  (422-Vorabprüfung), ``1m`` in ≤7-Tage-Chunks (serverseitig erzwungen,
  [LIVE] ``interval=1m&range=1mo`` → HTTP 422).
- **Fehler:** HTTP 404 / ``chart.error`` „Not Found“ → 404
  ``symbol_nicht_gefunden``; 429 sowie die yfinance-Heuristik „Will be right
  back“ (history.py:292) → sofort ``verzeichne_rate_limit()`` + 429
  ``yahoo_gedrosselt`` (kein Backoff); 422/``chart.error`` „Invalid Input“ →
  422; alles Übrige → Backoff, erschöpft 502.
- **Härtung 1:1 vom yfinance-Bein übernommen:** Thread-Semaphore, Pacing +
  Circuit-Breaker über den **geteilten** prozessweiten
  :class:`~quantlab.providers.rate_limit.YahooRateLimiter` — mit korrekter
  Reihenfolge: Semaphore **zuerst**, Pacing-Slot **innerhalb** (ARCH §4
  „Fix passend zur Struktur“; das yfinance-Bein zieht parallel in T-F12 nach).
- **Zeitzonen (E006):** Intraday-Timestamps sind absolute Unix-Sekunden → µs
  UTC ohne Verschiebung; Tages-Balken liegen auf dem **Exchange-Datum**
  (``meta.exchangeTimezoneName`` → lokale Mitternacht, Parität zum
  ``ignore_tz=True``-Verhalten des yfinance-Beins) und werden von
  :func:`parquet_store.normalize_ohlcv` auf Mitternacht UTC gefloort.

Struktur (Chunking, Backoff, Kanon-Mapping, Delisted-Erkennung) ist 1:1 aus
:mod:`quantlab.providers.yfinance_provider` übernommen; gemeinsame Helfer
werden importiert statt kopiert. Netzcall = eigener httpx-GET über einen
wiederverwendeten :class:`httpx.Client` (Session/Verbindungspooling, Timeout
10 s, Browser-UA); Tests fahren :class:`httpx.MockTransport` (E013, Muster
:mod:`quantlab.providers.stooq_provider`).
"""

from __future__ import annotations

import inspect
import logging
import math
import random
import threading
import time
from collections.abc import Callable
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from quantlab.providers import rate_limit
from quantlab.providers.base import (
    DAILY_INTERVALS,
    DataProvider,
    ProviderError,
    SymbolInfo,
    SymbolNotFoundError,
    validate_interval,
)
from quantlab.providers.symbols import validate_symbol
from quantlab.providers.yfinance_provider import (
    CHUNK_TAGE_1M,
    _asset_type,
    _leere_balken,
    fenster_erwartet_handelstage,
    jetzt_us,
    pruefe_intraday_fenster,
)
from quantlab.services import data_service
from quantlab.storage import parquet_store

logger = logging.getLogger(__name__)

#: v8/chart-Endpunkt (F.7 [LIVE]; query2 wie yfinance 1.7.0 ``_BASE_URL_``).
CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart"

#: Event-Auswahl — exakt der String, den yfinance 1.7.0 sendet (F.7).
EVENTS_PARAM = "div,splits,capitalGains"

#: Default-Fenster ohne explizites start/end — Parität zum yfinance-Default
#: (``period="1mo"``): bewusst begrenzt („nie komplette Historie ohne Not“).
DEFAULT_RANGE = "1mo"
DEFAULT_RANGE_TAGE = 30

#: Marker der Yahoo-Störungsseite — yfinance-Heuristik (F.7, history.py:292):
#: „Will be right back“ → Yahoo vorübergehend nicht lieferfähig; analog zur
#: Drosselung behandelt (Circuit öffnet, kein Backoff-Retry).
_ZURUECK_MARKER = "Will be right back"

#: UA wie ein Browser (F.7 [LIVE]: v8/chart kommt ohne Cookie/Crumb aus).
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 QuantLab/0.1"
)

#: ``meta.instrumentType`` → ``symbols.asset_type`` (ANNAHME: direkte Auf-
#: lösung; unbekannte Werte fallen auf die Yahoo-Kürzel-Heuristik zurück).
_INSTRUMENT_TYP: dict[str, str] = {
    "EQUITY": "equity",
    "ETF": "etf",
    "INDEX": "index",
    "CURRENCY": "forex",
    "FUTURE": "future",
    "CRYPTOCURRENCY": "crypto",
    "MUTUALFUND": "fund",
}

#: Events (v8-Feldname) → Aktionstyp der DB — gleiche zwei Typen wie das
#: yfinance-Bein; ``capitalGains`` bleibt unbepersistet (Schema-Grenze).
_EVENT_FELDER: tuple[tuple[str, str], ...] = (
    ("dividends", "dividend"),
    ("splits", "split"),
)


class _YahooDirektGedrosselt(Exception):
    """Interner Marker: HTTP 429 bzw. „Will be right back“ (→ Circuit öffnet)."""


def _tz_aus_meta(meta: dict[str, Any]) -> str | None:
    """``meta.exchangeTimezoneName`` (IANA) oder ``None`` bei Fehl/Fremdformat."""
    name = meta.get("exchangeTimezoneName")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _chart_result(payload: Any) -> dict[str, Any] | None:
    """``chart.result[0]`` — oder ``None`` bei fehlender/leerer Struktur."""
    if not isinstance(payload, dict):
        return None
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        return None
    resultate = chart.get("result")
    if not isinstance(resultate, list) or not resultate:
        return None
    erster = resultate[0]
    return erster if isinstance(erster, dict) else None


def _instrument_typ(meta: dict[str, Any], symbol: str) -> str:
    """Asset-Type aus ``meta.instrumentType``, Fallback: Kürzel-Heuristik."""
    roh = meta.get("instrumentType")
    if isinstance(roh, str):
        typ = _INSTRUMENT_TYP.get(roh.strip().upper())
        if typ is not None:
            return typ
    return _asset_type(symbol)


def _ereignisse_aus_payload(ereignisse_roh: Any) -> list[dict]:
    """v8-``events`` → Corporate-Actions-Liste (Vertrag des yfinance-Beins).

    Dividenden: ``{ts: {amount, date}}``; Splits: ``{ts: {numerator,
    denominator, date}}`` → Wert = Zähler/Nenner (Split-Ratio, yfinance-
    Konvention, F.7). ``capitalGains`` wird bewusst übersprungen (s. Modul-
    Docstring). Defekte Einträge fallen einzeln weg.
    """
    ereignisse: list[dict] = []
    if not isinstance(ereignisse_roh, dict):
        return ereignisse
    for feld, typ in _EVENT_FELDER:
        block = ereignisse_roh.get(feld)
        if not isinstance(block, dict):
            continue
        for schluessel, eintrag in block.items():
            if not isinstance(eintrag, dict):
                continue
            wert: float | None = None
            if typ == "dividend":
                roh = eintrag.get("amount")
                if isinstance(roh, (int, float)) and not isinstance(roh, bool):
                    wert = float(roh)
            else:
                zaehler = eintrag.get("numerator")
                nenner = eintrag.get("denominator")
                if (
                    isinstance(zaehler, (int, float))
                    and isinstance(nenner, (int, float))
                    and not isinstance(zaehler, bool)
                    and not isinstance(nenner, bool)
                    and float(nenner) != 0.0
                ):
                    wert = float(zaehler) / float(nenner)
            if wert is None or wert == 0.0 or not math.isfinite(wert):
                continue
            roh_datum = eintrag.get("date", schluessel)
            try:
                ts_s = int(float(roh_datum))
            except (TypeError, ValueError):
                continue
            if ts_s <= 0:
                continue
            ereignisse.append(
                {
                    "ts": ts_s * 1_000_000,
                    "action": typ,
                    "value": wert,
                    "currency": None,
                }
            )
    ereignisse.sort(key=lambda ereignis: ereignis["ts"])
    return ereignisse


def _tage_auf_exchange_mitternacht(ts_us: pd.Series, tz_name: str) -> pd.Series:
    """Tages-Timestamps auf Exchange-lokale Mitternacht (in UTC-µs).

    Parität zum yfinance-Bein (``ignore_tz=True``): der Balken gehört zum
    **Exchange-Kalendertag**; v8/chart liefert Unix-Sekunden (z. B. Session-
    Beginn 13:30 UTC bei US-Börsen) → in die Exchange-Zone verschoben, dort
    auf Mitternacht gefloort, zurück nach UTC. Ungültige Zonennamen →
    unverändert (``normalize_ohlcv`` floort dann auf UTC-Mitternacht).
    """
    try:
        zone = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001 — fremder Zonenname darf nicht scheitern
        logger.warning("Unbekannte Exchange-Zeitzone „%s“ — floore auf UTC.", tz_name)
        return ts_us
    lokal = pd.to_datetime(ts_us, unit="us", utc=True).dt.tz_convert(zone)
    mitternacht = lokal.dt.normalize()  # Mitternacht in der Exchange-Zone
    return ((mitternacht - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(1, "us")).astype(
        "int64"
    )


def _chart_fehler_aus_payload(fehler: Any, symbol: str) -> ProviderError:
    """``chart.error`` → passender :class:`ProviderError` (F.7-Heuristik).

    Exakte description-Strings sind unverifiziert (F.7) — gemappt wird auf
    Teilstrings: „not found“ → 404 ``symbol_nicht_gefunden``, „invalid“/
    „bad request“/„null descriptor“ → 422, alles Weitere → 502 (retry-fähig).
    """
    meldung = ""
    if isinstance(fehler, dict):
        meldung = f"{fehler.get('code') or ''} {fehler.get('description') or ''}".strip()
    kombiniert = meldung.lower()
    if "not found" in kombiniert:
        return SymbolNotFoundError(
            f"Symbol „{symbol}“ ist bei Yahoo nicht bekannt "
            f"({meldung or 'chart.error'}).",
            hint="Yahoo-Kürzel prüfen, z. B. „AAPL“, „^GSPC“, „EURUSD=X“, „ES=F“.",
        )
    if "invalid" in kombiniert or "bad request" in kombiniert or "null descriptor" in kombiniert:
        return ProviderError(
            f"Yahoo wies die Anfrage für „{symbol}“ zurück ({meldung or 'chart.error'}).",
            422,
            "provider_fehler",
        )
    return ProviderError(
        f"Yahoo-Fehler für „{symbol}“ ({meldung or 'chart.error'}).",
        502,
        "quelle_nicht_erreichbar",
    )


def unterstuetzt_chunk_fortschritt(provider: Any) -> bool:
    """Duck-Typing-Probe (T-F10, Muster ``letzte_quelle``/base.py:107-109):
    akzeptiert ``provider.fetch_ohlcv`` den kw-only-Vertragszusatz
    ``fortschritt``?

    Der Zusatz ist dokumentiert an :class:`YahooDirectProvider` und — für die
    Durchreichung zum Direkt-Bein — am
    :class:`~quantlab.providers.fallback_provider.FallbackProvider`. Schlichte
    Provider (yfinance-/Stooq-Bein, Test-Fakes) kennen ihn nicht und bekommen
    den klassischen Call ohne Callback; die Lesung der Signatur (statt
    ``isinstance``) hält die Stelle offen für künftige Beine mit Chunk-Progress.
    """
    fetch = getattr(provider, "fetch_ohlcv", None)
    if not callable(fetch):
        return False
    try:
        return "fortschritt" in inspect.signature(fetch).parameters
    except (TypeError, ValueError):  # Signatur nicht lesbar → konservativ nein
        return False


class YahooDirectProvider(DataProvider):
    """Direkter :class:`DataProvider` auf den Yahoo-v8/chart-Endpunkt (E085).

    Alle Konstruktionsparameter sind injizierbar — Tests verstellen Versuche/
    Backoff/Transport; der Betrieb nutzt die Defaults (identisch zum
    yfinance-Bein: Timeout 10 s, Backoff ab 2 s, Semaphore 4).
    """

    def __init__(
        self,
        *,
        max_threads: int = 4,
        timeout_s: float = 10.0,
        max_versuche: int = 3,
        backoff_start_s: float = 2.0,
        jitter_faktor: float = 0.5,
        rate_limiter: rate_limit.YahooRateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._semaphore = threading.Semaphore(max_threads)
        self._timeout_s = timeout_s
        self._max_versuche = max(1, int(max_versuche))
        self._backoff_start_s = backoff_start_s
        self._jitter_faktor = jitter_faktor
        # Prozessweiter Limiter (T19/E035) — geteilt mit dem yfinance-Bein;
        # Tests injizieren eine Instanz mit eigener Clock (Muster yfinance).
        self._limiter = (
            rate_limiter if rate_limiter is not None else rate_limit.RATE_LIMITER
        )
        self._transport = transport
        self._client_instanz: httpx.Client | None = None

    # ------------------------------------------------------------------
    # DataProvider-ABC
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        repair: bool = False,
        *,
        fortschritt: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Kanonische Balken (ABC-Vertrag); Exchange-TZ interessiert hier nicht.

        ``repair`` wird akzeptiert, aber **ignoriert** (Signatur-Vertrag des
        ABC, Muster Stooq-Provider): der clientseitige yfinance-Repair
        (100x-Währungsfehler) hat keine Direkt-Entsprechung — korrigiert wird
        übers volle Fenster (T19 lädt ohnehin neu).

        ``fortschritt`` (T-F10, duck-geoptimer Vertragszusatz): optionaler
        Callback für den Chunk-Fortschritt — Details
        :meth:`fetch_mit_exchange_tz`.
        """
        df, _ = self.fetch_mit_exchange_tz(
            symbol, interval, start=start, end=end, fortschritt=fortschritt
        )
        return df

    def resolve_symbol(self, symbol: str) -> SymbolInfo:
        """Validiert das Kürzel per leichter v8-Probe (``1d``/``range=5d``).

        Netzärmer als die yfinance-Ticker-Probe: **die ``meta``-Sektion genügt**
        (``currency``/``exchangeTimezoneName``/``instrumentType``) — kein
        adjclose/events- und kein Historien-Download. Die v8-meta führt i. d. R.
        keinen Anzeigenamen → ``name`` bleibt ``None``.
        """
        geprueft = validate_symbol(symbol)
        try:
            meta = self._mit_wiederholung(self._probe_meta, geprueft)
        except ProviderError:
            raise
        except Exception as exc:  # Netz-/Formatfehler der Probe selbst
            raise ProviderError(
                f"Symbolprüfung für „{geprueft}“ fehlgeschlagen: {exc}",
                502,
                "quelle_nicht_erreichbar",
            ) from exc
        return SymbolInfo(
            symbol=geprueft,
            name=meta.get("shortName") or meta.get("longName"),
            asset_type=_instrument_typ(meta, geprueft),
            exchange_tz=_tz_aus_meta(meta),
            currency=(
                meta.get("currency")
                if isinstance(meta.get("currency"), str) and meta.get("currency")
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Erweiterter Fetch (nur dieser Provider): Exchange-TZ zurückgeben
    # ------------------------------------------------------------------

    def fetch_mit_exchange_tz(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        *,
        fortschritt: Callable[[int, int], None] | None = None,
    ) -> tuple[pd.DataFrame, str | None]:
        """Lädt Balken und liefert ``(kanonisches DataFrame, exchange_tz | None)``.

        ``exchange_tz`` kommt aus ``meta.exchangeTimezoneName`` (IANA-Name) und
        wird vom Download-Job-Manager in ``symbols.exchange_tz`` gespeichert.

        ``fortschritt`` (T-F10): optionaler Callback für den Chunk-Fortschritt
        des ``1m``-Chunkings — er wird nach **jedem** abgeschlossenen Chunk
        (auch nach einem toleriert gescheiterten Einzel-Chunk) mit
        ``(index, gesamt)`` gerufen, wobei ``index`` die 1-basierte Zahl der
        abgeschlossenen Chunks ist (nach dem letzten gilt ``index == gesamt``;
        nicht gechunkte Abrufe melden ``(1, 1)`` — der Aufrufer filtert das).
        Exceptions aus dem Callback werden geschluckt und geloggt: Progress
        darf den Fetch nie scheitern lassen (Wahl gemäß T-F10-Spez).
        """
        geprueft = validate_symbol(symbol)
        validate_interval(interval)
        moment = jetzt_us()

        tagesaufloesung = interval in DAILY_INTERVALS
        if not tagesaufloesung:
            pruefe_intraday_fenster(start, end, moment)

        fenster_start, fenster_ende = self._normalisiere_fenster(start, end)
        teile = self._teile(interval, fenster_start, fenster_ende, moment)
        gesamt = len(teile)

        stuecke: list[pd.DataFrame] = []
        ereignisse: list[dict] = []
        exchange_tz: str | None = None
        for index, (teil_start, teil_ende) in enumerate(teile, start=1):
            try:
                payload = self._mit_wiederholung(
                    self._roh_download, geprueft, interval, teil_start, teil_ende
                )
            except ProviderError:
                if gesamt == 1:
                    raise
                self._fortschritt_melden(fortschritt, index, gesamt)
                continue  # einzelne 1m-Chunks dürfen scheitern/leer sein
            stueck, tz, stueck_ereignisse = self._zu_kanonisch(payload, interval)
            ereignisse.extend(stueck_ereignisse)
            if tz is not None and exchange_tz is None:
                exchange_tz = tz
            if not stueck.empty:
                stuecke.append(stueck)
            self._fortschritt_melden(fortschritt, index, gesamt)

        if not stuecke:
            # T20: komplett leere History bei bekanntem Symbol → delisted
            # vermerken (best-effort) und klar melden statt generischem 404.
            # M2b: nur bei Fenstern, die plausibel Handelstage erwarten.
            if fenster_erwartet_handelstage(
                fenster_start, fenster_ende, moment
            ) and data_service.vermerke_delisted_wenn_bekannt(geprueft):
                raise ProviderError(
                    f"„{geprueft}“ liefert keine Kursdaten mehr — das Symbol "
                    "wurde als delistet vermerkt (Bestand bleibt erhalten).",
                    404,
                    "symbol_delisted",
                    hint="Der gespeicherte Bestand bleibt lesbar; neue Downloads "
                    "sind für delistierte Symbole nicht mehr möglich.",
                )
            raise ProviderError(
                f"Für „{geprueft}“ liefert Yahoo keine Daten im Intervall „{interval}“.",
                404,
                "keine_daten",
                hint="Prüfe das Symbol und den Zeitraum (Intraday: max. 60 Tage zurück).",
            )
        # M2a — Self-Healing: Daten trotz Delisted-Vermerk → Marke zurücknehmen.
        data_service.hebe_delisted_vermerk_auf(geprueft)
        self._actions_persistieren(geprueft, ereignisse)
        if len(stuecke) == 1:
            return stuecke[0], exchange_tz
        kombiniert = (
            pd.concat(stuecke, ignore_index=True)
            .drop_duplicates(subset="ts", keep="last")
            .sort_values("ts", kind="stable")
            .reset_index(drop=True)
        )
        return kombiniert, exchange_tz

    # ------------------------------------------------------------------
    # Innere Mechanik
    # ------------------------------------------------------------------

    @staticmethod
    def _fortschritt_melden(
        fortschritt: Callable[[int, int], None] | None, index: int, gesamt: int
    ) -> None:
        """Meldet einen abgeschlossenen Chunk an den optionalen Callback (T-F10).

        Bewusst best-effort: eine Panne im Callback (z. B. eine tote SSE-Queue
        beim Empfänger) wird geschluckt und geloggt — der Fetch darf am
        Progress nie scheitern.
        """
        if fortschritt is None:
            return
        try:
            fortschritt(index, gesamt)
        except Exception:  # noqa: BLE001 — Progress darf den Fetch nie killen
            logger.warning(
                "Fortschritts-Callback fehlgeschlagen (Chunk %s/%s).",
                index,
                gesamt,
                exc_info=True,
            )

    def close(self) -> None:
        """Gibt den wiederverwendeten httpx-Client frei (Hygiene, optional)."""
        if self._client_instanz is not None:
            self._client_instanz.close()
            self._client_instanz = None

    def _client(self) -> httpx.Client:
        """Wiederverwendeter Client (Session/Verbindungspooling, Browser-UA)."""
        if self._client_instanz is None:
            self._client_instanz = httpx.Client(
                transport=self._transport,
                timeout=self._timeout_s,
                headers={"User-Agent": _BROWSER_UA},
                follow_redirects=True,
            )
        return self._client_instanz

    def _normalisiere_fenster(
        self, start: int | None, end: int | None
    ) -> tuple[int | None, int | None]:
        """µs-Fenster unverändert durchreichen (None = v8-``range``-Pfad)."""
        if start is not None and end is not None and start >= end:
            raise ProviderError(
                "„start“ liegt nach „end“.",
                422,
                "ungueltiger_zeitraum",
                hint="start ist inklusive, end exklusiv (Mikrosekunden UTC).",
            )
        return start, end

    def _teile(
        self, interval: str, start: int | None, end: int | None, moment: int
    ) -> list[tuple[int | None, int | None]]:
        """Zerlegt ``1m``-Abrufe in ≤7-Tage-Fenster (1:1 yfinance-Bein)."""
        if interval != "1m":
            return [(start, end)]
        schritt = CHUNK_TAGE_1M * parquet_store.US_PER_DAY
        eff_ende = end if end is not None else moment
        eff_start = start if start is not None else max(eff_ende - schritt, 0)
        teile: list[tuple[int | None, int | None]] = []
        zeiger = eff_start
        while zeiger < eff_ende:
            teile.append((zeiger, min(zeiger + schritt, eff_ende)))
            zeiger += schritt
        return teile

    def _fenster_params(
        self, start_us: int | None, ende_us: int | None
    ) -> dict[str, Any]:
        """µs-Fenster → v8-Fensterparameter (``range`` oder ``period1/2``).

        Ohne beides: ``range=1mo`` (yfinance-Default-Parität). Sonst
        ``period1``/``period2`` in Unix-**Sekunden** (µs // 10⁶); fehlender
        Anfang → Default-Rückfenster vor dem bekannten Ende (ANNAHME: ~1 Monat,
        begrenzter Abruf statt Vollhistorie), fehlendes Ende → jetzt.
        """
        if start_us is None and ende_us is None:
            return {"range": DEFAULT_RANGE}
        moment = jetzt_us()
        if start_us is None:
            referenz = ende_us if ende_us is not None else moment
            start_us = max(0, referenz - DEFAULT_RANGE_TAGE * parquet_store.US_PER_DAY)
        return {
            "period1": int(start_us) // 1_000_000,
            "period2": int(ende_us if ende_us is not None else moment) // 1_000_000,
        }

    def _roh_download(
        self,
        symbol: str,
        interval: str,
        start_us: int | None,
        ende_us: int | None,
    ) -> dict[str, Any]:
        """Der **einzige** HTTP-Netzcall dieses Beins (Semaphore + Pacing).

        Reihenfolge gemäß ARCH §4 („Fix passend zur Struktur“, zieht im
        yfinance-Bein mit T-F12 nach): Semaphore **zuerst**, Pacing-Slot
        **innerhalb** — ein schlafender Call hält einen der Semaphore-Plätze,
        aber der Slot verstreicht nie wartend (kein Race mehr).
        """
        params: dict[str, Any] = {
            "interval": interval,
            "events": EVENTS_PARAM,
            "includeAdjustedClose": "true",
        }
        params.update(self._fenster_params(start_us, ende_us))
        return self._chart_abrufen(symbol, params)

    def _chart_abrufen(self, symbol: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET auf ``v8/finance/chart/{symbol}`` — Guard, Timeout, Fehlerklassen."""
        with self._semaphore:
            self._limiter.vor_yahoo_aufruf()
            try:
                antwort = self._client().get(f"{CHART_URL}/{symbol}", params=params)
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"Yahoo-Direktaufruf für „{symbol}“ fehlgeschlagen: {exc}",
                    502,
                    "quelle_nicht_erreichbar",
                ) from exc
            return self._antwort_als_payload(antwort, symbol)

    def _antwort_als_payload(
        self, antwort: httpx.Response, symbol: str
    ) -> dict[str, Any]:
        """HTTP-Antwort → v8-Payload; Fehlerverhalten gemäß F.7-Fehlertabelle."""
        text = antwort.text
        if antwort.status_code == 429 or _ZURUECK_MARKER in text:
            # Drosselung/Störung: kein Backoff — Circuit öffnet sofort
            # (``_mit_wiederholung`` baut daraus den ProviderError 429).
            raise _YahooDirektGedrosselt(f"HTTP {antwort.status_code} für „{symbol}“")
        if antwort.status_code == 404:
            raise SymbolNotFoundError(
                f"Symbol „{symbol}“ ist bei Yahoo nicht bekannt (HTTP 404).",
                hint="Yahoo-Kürzel prüfen, z. B. „AAPL“, „^GSPC“, „EURUSD=X“, „ES=F“.",
            )
        try:
            payload = antwort.json()
        except ValueError as exc:
            raise ProviderError(
                f"Yahoo-Direktaufruf für „{symbol}“ lieferte kein JSON "
                f"(HTTP {antwort.status_code}).",
                502,
                "quelle_nicht_erreichbar",
            ) from exc
        if antwort.status_code != 200:
            raise ProviderError(
                f"Yahoo-Direktaufruf für „{symbol}“ antwortete mit "
                f"HTTP {antwort.status_code}.",
                502,
                "quelle_nicht_erreichbar",
            )
        fehler = (
            payload.get("chart", {}).get("error")
            if isinstance(payload, dict) and isinstance(payload.get("chart"), dict)
            else None
        )
        if fehler:
            raise _chart_fehler_aus_payload(fehler, symbol)
        return payload if isinstance(payload, dict) else {}

    def _probe_meta(self, symbol: str) -> dict[str, Any]:
        """Leichte Probe (``1d``/``range=5d``, ohne Events/Adjclose) → ``meta``."""
        payload = self._chart_abrufen(
            symbol,
            {"interval": "1d", "range": "5d", "includeAdjustedClose": "false"},
        )
        ergebnis = _chart_result(payload)
        zeiten = ergebnis.get("timestamp") if ergebnis is not None else None
        if not isinstance(zeiten, list) or not zeiten:
            raise ProviderError(
                f"Symbol „{symbol}“ ist bei Yahoo nicht bekannt "
                "oder liefert keine Daten.",
                404,
                "symbol_nicht_gefunden",
                hint="Yahoo-Kürzel prüfen, z. B. „AAPL“, „^GSPC“, „EURUSD=X“, „ES=F“.",
            )
        roh_meta = ergebnis.get("meta") if ergebnis is not None else None
        return roh_meta if isinstance(roh_meta, dict) else {}

    def _mit_wiederholung(self, func, *args, **kwargs):
        """Führt ``func`` mit exponentiellem Backoff + Jitter aus (Start 2 s).

        1:1-Vertrag zum yfinance-Bein: Drosselung
        (:class:`_YahooDirektGedrosselt` — HTTP 429/„Will be right back“)
        **ohne** Retry → ``verzeichne_rate_limit()`` + :class:`ProviderError`
        429 ``yahoo_gedrosselt``; Guard-Block (offener Circuit) unangetastet;
        deterministische Ablehnungen (404/422) ohne Backoff; erschöpfte
        Versuche → 502 ``quelle_nicht_erreichbar``.
        """
        pause = self._backoff_start_s
        letzter_fehler: BaseException | None = None
        for versuch in range(self._max_versuche):
            try:
                ergebnis = func(*args, **kwargs)
                self._limiter.verzeichne_erfolg()  # Erfolg schließt den Circuit
                return ergebnis
            except rate_limit.YahooGedrosseltError:
                # Guard-Block (offener Circuit): der Call fand nicht statt —
                # kein Backoff, kein Erfolg, kein Zustandswechsel am Limiter.
                raise
            except _YahooDirektGedrosselt as exc:
                dauer = self._limiter.verzeichne_rate_limit()
                # Stooq-neutral (E087): dieses Bein kennt die Fallback-Kette
                # nicht — die ehrliche, dynamische Quellen-Erwähnung baut der
                # FallbackProvider (_stooq_oder_fehler).
                raise ProviderError(
                    "Yahoo drosselt die Anfragen (429 Too Many Requests). "
                    f"Weitere Yahoo-Abrufe werden für {dauer:.0f} s ausgesetzt.",
                    429,
                    "yahoo_gedrosselt",
                    hint="Etwas seltener anfragen — der Cool-down verdoppelt "
                    "sich bei jeder weiteren Drosselung (max. 900 s).",
                ) from exc
            except ProviderError as exc:
                if exc.status_code in (404, 422):
                    raise  # deterministische Ablehnung — Backoff wäre sinnlos
                letzter_fehler = exc
            except Exception as exc:  # noqa: BLE001 — jede Netz-/Formatpanne zählt
                letzter_fehler = exc
            if versuch < self._max_versuche - 1:
                time.sleep(pause + random.uniform(0, pause * self._jitter_faktor))
                pause *= 2
        raise ProviderError(
            f"Yahoo-Abruf nach {self._max_versuche} Versuchen fehlgeschlagen: "
            f"{letzter_fehler}",
            502,
            "quelle_nicht_erreichbar",
            hint="Yahoo drosselt ggf. — bitte wenig später erneut versuchen.",
        ) from letzter_fehler

    def _zu_kanonisch(
        self, payload: dict[str, Any] | None, interval: str
    ) -> tuple[pd.DataFrame, str | None, list[dict]]:
        """v8-Payload → Schema-Vertrag des ABC (E006); Exchange-TZ + Events.

        Null-Lücken in den quote-Arrays bleiben als ``NaN`` stehen und fallen
        wie im Bestand durch :func:`parquet_store.normalize_ohlcv` weg
        (Zeilen ohne vollständiges OHLC). ``adj_factor`` nur, wenn
        ``indicators.adjclose`` tatsächlich geliefert wurde (Lax-Vertrag —
        fehlt sie, füllen Store/Lesen neutral 1.0).
        """
        ergebnis = _chart_result(payload)
        if ergebnis is None:
            return _leere_balken(), None, []
        roh_meta = ergebnis.get("meta")
        meta = roh_meta if isinstance(roh_meta, dict) else {}
        zeiten = ergebnis.get("timestamp")
        roh_indikatoren = ergebnis.get("indicators")
        indikatoren = roh_indikatoren if isinstance(roh_indikatoren, dict) else {}
        quote_liste = indikatoren.get("quote")
        quote = (
            quote_liste[0]
            if isinstance(quote_liste, list)
            and quote_liste
            and isinstance(quote_liste[0], dict)
            else {}
        )
        tz_name = _tz_aus_meta(meta)
        if not isinstance(zeiten, list) or not zeiten:
            return _leere_balken(), tz_name, []
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) for t in zeiten):
            logger.warning(
                "Yahoo lieferte defekte Timestamps (Intervall „%s“) — Chunk verworfen.",
                interval,
            )
            return _leere_balken(), tz_name, []

        anzahl = len(zeiten)
        ts_us = [int(t) * 1_000_000 for t in zeiten]

        def feld(name: str) -> list[float | None]:
            roh = quote.get(name)
            if not isinstance(roh, list) or len(roh) != anzahl:
                return [None] * anzahl
            return [
                float(w)
                if isinstance(w, (int, float)) and not isinstance(w, bool)
                else None
                for w in roh
            ]

        roh = pd.DataFrame(
            {
                "ts": pd.Series(ts_us, dtype="int64"),
                "open": pd.Series(feld("open"), dtype="float64"),
                "high": pd.Series(feld("high"), dtype="float64"),
                "low": pd.Series(feld("low"), dtype="float64"),
                "close": pd.Series(feld("close"), dtype="float64"),
                "volume": pd.Series(feld("volume"), dtype="float64"),
            }
        )
        adj_liste = indikatoren.get("adjclose")
        adj = (
            adj_liste[0].get("adjclose")
            if isinstance(adj_liste, list)
            and adj_liste
            and isinstance(adj_liste[0], dict)
            else None
        )
        if isinstance(adj, list) and len(adj) == anzahl:
            adj_serie = pd.Series(
                [
                    float(w)
                    if isinstance(w, (int, float)) and not isinstance(w, bool)
                    else None
                    for w in adj
                ],
                dtype="float64",
            )
            roh["adj_factor"] = self._adj_faktor(roh["close"], adj_serie)
        if interval in DAILY_INTERVALS and tz_name is not None:
            roh["ts"] = _tage_auf_exchange_mitternacht(roh["ts"], tz_name)
        ereignisse = _ereignisse_aus_payload(ergebnis.get("events"))
        return parquet_store.normalize_ohlcv(roh, interval), tz_name, ereignisse

    @staticmethod
    def _adj_faktor(close: pd.Series, adj_close: pd.Series) -> pd.Series:
        """``Adj Close/Close`` je Bar (E022) — identischer Vertrag zum yfinance-Bein.

        Division nur bei vorhandenem, endlichem und positivem ``Adj Close``
        sowie positivem, endlichem ``Close`` — 0/NaN/±inf und negative Werte
        führen zu neutral 1.0 statt inf/NaN/negativem Faktor.
        """
        faktor = pd.Series(1.0, index=close.index, dtype="float64")
        adj = pd.to_numeric(adj_close, errors="coerce")
        basis = pd.to_numeric(close, errors="coerce")
        gueltig = (
            adj.notna()
            & basis.notna()
            & (basis > 0)
            & (adj > 0)
            & np.isfinite(adj)
            & np.isfinite(basis)
        )
        faktor.loc[gueltig] = (adj.loc[gueltig] / basis.loc[gueltig]).astype("float64")
        return faktor

    def _actions_persistieren(self, symbol: str, ereignisse: list[dict]) -> None:
        """Schreibt Dividenden-/Split-Events idempotent in ``corporate_actions``.

        Bewusst best-effort (E022, identisch zum yfinance-Bein): eine DB-Panne
        beim Event-Upsert darf den Fetch nicht scheitern lassen.
        """
        if not ereignisse:
            return
        try:
            data_service.speichere_corporate_actions(symbol, ereignisse)
        except Exception:  # noqa: BLE001 — Fetch darf an der DB nicht scheitern
            logger.warning(
                "Corporate Actions für „%s“ konnten nicht gespeichert werden",
                symbol,
                exc_info=True,
            )


__all__ = [
    "CHART_URL",
    "DEFAULT_RANGE",
    "EVENTS_PARAM",
    "YahooDirectProvider",
    "unterstuetzt_chunk_fortschritt",
]
