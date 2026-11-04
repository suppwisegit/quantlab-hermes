"""yfinance-Provider (T4): zentrale Fetch-Schicht vor Yahoo Finance.

Alle Yahoo-Abrufe laufen durch genau diesen Provider — die Härtung sitzt an
einer Stelle (RECHERCHE Block 2, EMPFEHLUNG „zentrale Fetch-Schicht“):

- **Thread-Limit:** eine :class:`threading.Semaphore` begrenzt parallele
  Yahoo-Calls (der Job-Manager lädt je Symbol in einem Executor-Thread,
  ``asyncio.to_thread``); Drosselungsrisiko R2 wird so klein gehalten.
- **Pacing + Circuit-Breaker (T19/E035):** vor **jedem** Yahoo-Call fragt der
  Provider den prozessweiten :class:`~quantlab.providers.rate_limit.YahooRateLimiter`
  (:meth:`~quantlab.providers.rate_limit.YahooRateLimiter.vor_yahoo_aufruf`)
  — Mindestabstand 0,75 s und, nach einer ``YFRateLimitError``, ein sofort
  öffnender Circuit mit exponentiellem Cool-down. ``YFRateLimitError`` wird
  gezielt gefangen (yfinance retryt intern **nicht**, RECHERCHE v2 Block 1.8):
  kein Backoff-Versuch, sondern direkt :class:`ProviderError` 429.
- **Backoff + Jitter:** fehlschlagende Aufrufe werden bis ``max_versuche``
  wiederholt, exponentiell (Start ``backoff_start_s`` = 2 s) mit Zufalls-Jitter.
  Erschöpfte Versuche → :class:`ProviderError` (502).
- **Pro-Request-Timeout:** an ``yf.download(timeout=…)`` durchgereicht.
- **Zeitzonen (E006):** ``ignore_tz`` intervallabhängig — Tages-Intervalle
  (:data:`quantlab.providers.base.DAILY_INTERVALS`) ``True`` (tz-naiv, wird als
  UTC interpretiert), Intraday ``False`` (Exchange-TZ) → nach UTC konvertiert;
  die Exchange-TZ liefert :meth:`fetch_mit_exchange_tz` für die
  ``symbols``-Registry zurück.
- **Grenzen:** Intraday jenseits 60 Tage → deutscher Fehler (422);
  ``1m`` wird automatisch in ≤7-Tage-Chunks geladen (ANNAHME, RECHERCHE
  Block 2 Fakt 3) — leere Einzel-Chunks (z. B. Wochenend-Fenster) sind kein
  Fehler, erst ein komplett leerer Abruf wird als „keine Daten“ (404) gemeldet.
- **``repair=True`` (T19 Gap-Repair):** wird an ``yf.download(repair=…)``
  durchgereicht — yfinance erkennt/korrigiert Bestandsfehler (z. B.
  100x-Währungsfehler) im geladenen Fenster.
- **Rohkurse + Faktor (T20/E022):** ``auto_adjust=False, actions=True`` —
  Yahoo liefert ``Close`` (roh) plus ``Adj Close``; der Faktor je Bar
  ``f_t = AdjClose_t / Close_t`` (Split- **und** Dividenden-Effekt, an den
  yfinance-Utils ``ratio = Adj Close/Close`` verifiziert — RECHERCHE v2
  Block 1.7) wird als Spalte ``adj_factor`` mitgeliefert und im Parquet
  persistiert; die Adjustierung selbst passiert on-read im Daten-Service.
  Fehlt ``Adj Close`` (z. B. Intraday-Chunks), gilt neutral 1.0. Die
  Dividenden-/Split-Spalten aus ``actions=True`` werden als Events über
  :func:`quantlab.services.data_service.speichere_corporate_actions` in die
  ``corporate_actions``-Tabelle geschrieben (idempotenter Upsert; DB-Pannen
  dürfen den Fetch nicht scheitern lassen — Warnung im Log).
- **Delisted (T20):** liefert Yahoo für ein **bekanntes** Symbol (Registry-
  Eintrag) eine komplett leere History, setzt der Provider
  ``symbols.delisted=1`` und wirft 404 ``symbol_delisted`` statt des
  generischen „keine Daten“-Fehlers — Bestand/Abdeckung bleiben sichtbar.
  M2-Fix (T20): Nur Fenster, die plausibel Handelstage erwarten
  (:func:`fenster_erwartet_handelstage`, ≥ 2 Wochentage bzw. ``start=None``),
  dürfen delisten — Wochenend-/Feiertags-Kleinstfenster sind kein Signal. Ein
  späterer nicht-leerer Fetch hebt den Vermerk bedingt wieder auf
  (Self-Healing, :func:`quantlab.services.data_service.hebe_delisted_vermerk_auf`).

yfinance wird lazy importiert (in :meth:`YfinanceProvider._roh_download` bzw.
``_ticker_probe``) — Import und Testsuite bleiben netzfrei; Tests patchen die
Öffnungen ``yfinance.download`` / ``yfinance.Ticker`` direkt.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from quantlab.providers import rate_limit
from quantlab.providers.base import (
    DAILY_INTERVALS,
    DataProvider,
    ProviderError,
    SymbolInfo,
    validate_interval,
)
from quantlab.providers.symbols import validate_symbol
from quantlab.services import data_service
from quantlab.storage import parquet_store

logger = logging.getLogger(__name__)

#: Intraday-Rückreichweite von Yahoo (RECHERCHE Block 2, Fakt 3 — verifiziert).
INTRADAY_MAX_TAGE = 60

#: ANNAHME: ``1m`` ist pro Request auf ~7 Tage limitiert → Chunking (RECHERCHE Block 2).
CHUNK_TAGE_1M = 7

#: Deutsche Fachmeldung zur Intraday-Grenze (DoD-Text; Provider **und**
#: Vorab-Prüfung im Download-Service nutzen dieselbe Konstante).
INTRADAY_LIMIT_NACHRICHT = "Intraday-Daten reichen maximal 60 Tage zurück."

_EPOCHE = datetime(1970, 1, 1, tzinfo=UTC)

#: OHLCV-Feldnamen (nach Kleinbuchstaben-Umbenennung) im kanonischen Schema.
_FELDNAMEN: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Event-Spalten des Downloads (nach Umbenennung) → Aktionstyp der DB (T20).
_EREIGNIS_SPALTEN: tuple[tuple[str, str], ...] = (
    ("dividends", "dividend"),
    ("stock splits", "split"),
)


def jetzt_us() -> int:
    """Aktuelle Zeit als int64 Mikrosekunden UTC (kanonische Zeiteinheit, E006)."""
    return (datetime.now(UTC) - _EPOCHE) // timedelta(microseconds=1)


def us_zu_datetime(ts_us: int) -> datetime:
    """int64 µs UTC → tz-aware :class:`datetime` (für yfinance-Argumente)."""
    return _EPOCHE + timedelta(microseconds=ts_us)


def pruefe_intraday_fenster(
    start: int | None, end: int | None, jetzt: int | None = None
) -> None:
    """Wirft 422, wenn der früheste gewünschte Balken älter als 60 Tage ist.

    Yahoo behält Intraday-Daten nur 60 Tage („cannot extend last 60 days“);
    ``start`` ist inklusive — es zählt also der früheste gewünschte Zeitpunkt.
    Ohne ``start`` zählt ``end`` (dann wäre das gesamte Fenster zu alt); ohne
    beide ist der Abruf implizit „aktuell“ und stets erlaubt.
    """
    if start is None and end is None:
        return
    moment = jetzt_us() if jetzt is None else jetzt
    grenze = moment - INTRADAY_MAX_TAGE * parquet_store.US_PER_DAY
    fruehester = start if start is not None else end
    assert fruehester is not None  # durch die None-Prüfung oben ausgeschlossen
    if fruehester < grenze:
        raise ProviderError(
            INTRADAY_LIMIT_NACHRICHT,
            422,
            "intraday_limit",
            hint="Wähle ein aktuelleres Zeitfenster oder Tagesdaten („1d“).",
        )


#: Mindestanzahl kalenderseitig erwarteter Wochentage (Mo–Fr), ab der ein
#: komplett leerer Abruf eines bekannten Symbols überhaupt als Delisted-Signal
#: gelten darf (M2b, T20-Fix). Fenster darunter (z. B. Sa–Mo über ein Wochenende
#: oder ein einzelner Feiertag) können legitimerweise leer sein.
FENSTER_MIN_HANDELSTAGE = 2


def fenster_erwartet_handelstage(
    start: int | None, end: int | None, moment: int
) -> bool:
    """Verdient das angeforderte Fenster plausibel echte Handelstage? (M2b)

    ``True`` bei offenem Anfang (``start=None`` → yfinance-Default/„period“-
    Variante — Yahoo wählt das Fenster selbst) oder wenn ``[start, end)``
    kalenderseitig mindestens :data:`FENSTER_MIN_HANDELSTAGE` Wochentage
    (Mo–Fr) überdeckt; ein offenes Ende (``end=None``) zählt bis ``moment``.
    Feiertage melden sich dabei als „zu wenig“ (dokumentierte Übermeldung,
    Muster: ``downloads.pruefe_lücken``) — lieber kein Delisted-Vermerk als
    ein Fehlalarm über ein Wochenend-/Feiertags-Kleinstfenster.
    """
    if start is None:
        return True
    eff_ende = end if end is not None else moment
    if eff_ende <= start:
        return False
    erster_tag = start // parquet_store.US_PER_DAY
    letzter_tag = (eff_ende - 1) // parquet_store.US_PER_DAY  # end exklusiv
    wochentage = 0
    for tag in range(erster_tag, letzter_tag + 1):
        moment_dt = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=tag)
        if moment_dt.weekday() < 5:
            wochentage += 1
            if wochentage >= FENSTER_MIN_HANDELSTAGE:
                return True
    return False


def _asset_type(symbol: str) -> str:
    """Grobe Yahoo-Kürzel-Heuristik für die ``symbols``-Registry (best-effort)."""
    if symbol.startswith("^"):
        return "index"
    if symbol.endswith("=X"):
        return "forex"
    if symbol.endswith("=F"):
        return "future"
    if "-" in symbol and symbol.split("-", 1)[1] in {"USD", "EUR", "USDT", "BTC", "ETH"}:
        return "crypto"
    return "equity"


class YfinanceProvider(DataProvider):
    """Konkrete :class:`DataProvider`-Implementierung über ``yfinance``.

    Alle Konstruktionsparameter sind bewusst injizierbar — Tests verstellen
    Versuche/Backoff, der Betrieb nutzt die Defaults (RECHERCHE Block 2,
    EMPFEHLUNG: Backoff ab 2 s, Timeout ~10 s).
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
    ) -> None:
        self._semaphore = threading.Semaphore(max_threads)
        self._timeout_s = timeout_s
        self._max_versuche = max(1, int(max_versuche))
        self._backoff_start_s = backoff_start_s
        self._jitter_faktor = jitter_faktor
        # Prozessweiter Limiter (T19/E035); Tests injizieren eine Instanz mit
        # eigener Clock, statt den Module-Default zu verschmutzen.
        self._limiter = (
            rate_limiter if rate_limiter is not None else rate_limit.RATE_LIMITER
        )

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
    ) -> pd.DataFrame:
        """Kanonische Balken (ABC-Vertrag); Exchange-TZ interessiert hier nicht."""
        df, _ = self.fetch_mit_exchange_tz(
            symbol, interval, start=start, end=end, repair=repair
        )
        return df

    def resolve_symbol(self, symbol: str) -> SymbolInfo:
        """Validiert das Kürzel per Probe-Request (5 Tage Historie) gegen Yahoo."""
        geprueft = validate_symbol(symbol)
        try:
            existiert, meta = self._mit_wiederholung(self._ticker_probe, geprueft)
        except ProviderError:
            raise
        except Exception as exc:  # Netz-/Formatfehler der Probe selbst
            raise ProviderError(
                f"Symbolprüfung für „{geprueft}“ fehlgeschlagen: {exc}",
                502,
                "quelle_nicht_erreichbar",
            ) from exc
        if not existiert:
            raise ProviderError(
                f"Symbol „{geprueft}“ ist bei Yahoo nicht bekannt oder liefert keine Daten.",
                404,
                "symbol_nicht_gefunden",
                hint="Yahoo-Kürzel prüfen, z. B. „AAPL“, „^GSPC“, „EURUSD=X“, „ES=F“.",
            )
        return SymbolInfo(
            symbol=geprueft,
            name=meta.get("shortName") or meta.get("longName"),
            asset_type=_asset_type(geprueft),
            exchange_tz=meta.get("exchangeTimezoneName"),
            currency=meta.get("currency"),
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
        repair: bool = False,
    ) -> tuple[pd.DataFrame, str | None]:
        """Lädt Balken und liefert ``(kanonisches DataFrame, exchange_tz | None)``.

        ``exchange_tz`` ist die IANA-Zone des Yahoo-Index (nur bei Intraday
        gesetzt — dort kommt der Index tz-aware; Tagesdaten sind tz-naiv/UTC).
        Der Download-Job-Manager speichert sie in ``symbols.exchange_tz``.
        ``repair=True`` reicht yfinance ``repair=True`` durch (T19).
        """
        geprueft = validate_symbol(symbol)
        validate_interval(interval)
        moment = jetzt_us()

        tagesaufloesung = interval in DAILY_INTERVALS
        if not tagesaufloesung:
            pruefe_intraday_fenster(start, end, moment)

        ignore_tz = tagesaufloesung
        fenster_start, fenster_ende = self._normalisiere_fenster(start, end, moment)
        teile = self._teile(interval, fenster_start, fenster_ende, moment)

        stuecke: list[pd.DataFrame] = []
        ereignisse: list[dict] = []
        exchange_tz: str | None = None
        for teil_start, teil_ende in teile:
            try:
                roh = self._mit_wiederholung(
                    self._roh_download,
                    geprueft,
                    interval,
                    teil_start,
                    teil_ende,
                    ignore_tz,
                    repair,
                )
            except ProviderError:
                if len(teile) == 1:
                    raise
                continue  # einzelne 1m-Chunks dürfen scheitern/leer sein
            stueck, tz, stueck_ereignisse = self._zu_kanonisch(roh, geprueft, interval)
            ereignisse.extend(stueck_ereignisse)
            if tz is not None and exchange_tz is None:
                exchange_tz = tz
            if not stueck.empty:
                stuecke.append(stueck)

        if not stuecke:
            # T20: komplett leere History bei bekanntem Symbol → delisted
            # vermerken (best-effort) und klar melden statt generischem 404.
            # M2b (T20-Fix): Nur, wenn das angeforderte Fenster überhaupt
            # plausibel Handelstage erwartet — Wochenend-/Feiertags-Kleinst-
            # fenster dürfen nicht delisten (Fehlalarm-Vermeidung).
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
        # M2a (T20-Fix) — Self-Healing: Daten trotz Delisted-Vermerk (Fehlalarm
        # oder zurückgekehrte Notierung) → Marke bedingt zurücknehmen.
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

    def _normalisiere_fenster(
        self, start: int | None, end: int | None, moment: int
    ) -> tuple[int | None, int | None]:
        """µs-Fenster unverändert durchreichen (None = yfinance-Default „aktuell“)."""
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
        """Zerlegt ``1m``-Abrufe in ≤7-Tage-Fenster; alle anderen bleiben ganz.

        ``start``/``end`` sind an dieser Stelle bereits validiert (start < end
        oder beides ``None``); ``None`` heißt „ab/bis aktuell“.
        """
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

    def _roh_download(
        self,
        symbol: str,
        interval: str,
        start_us: int | None,
        ende_us: int | None,
        ignore_tz: bool,
        repair: bool = False,
    ) -> pd.DataFrame:
        """Der **einzige** yfinance-Netzcall dieser App (Thread-Limit + Timeout).

        Vor dem Call (nach Semaphore-Erwerb): Pacing/Circuit-Guard des
        prozessweiten Limiters (T19/E035) — bei offenem Circuit fliegt hier
        sofort :class:`~quantlab.providers.rate_limit.YahooGedrosseltError`,
        ``yf.download`` wird gar nicht erreicht.

        Seit T20/E022 mit ``auto_adjust=False, actions=True`` (Signatur an
        yfinance 1.7.0 verifiziert, RECHERCHE v2 Block 1.7): ``Close`` ist der
        Rohkurs, ``Adj Close`` liefert über :meth:`_zu_kanonisch` den je-Bar-
        Faktor ``adj_factor``; ``Dividends``/``Stock Splits`` werden als
        Corporate-Actions-Events eingesammelt.
        """
        import yfinance as yf  # lazy: Import/Testsuite bleibt netzfrei

        # T-F12 (Race-Fix): Semaphore ZUERST, Pacing-Slot erst INNENHALB —
        # ein an der Semaphore wartender Thread reserviert keinen Slot mehr,
        # dessen Zeit beim Warten verstreicht (ehemalige M1-Doku-Grenze).
        # Der Slot-Zeitstempel liegt damit garantiert am realen Yahoo-Call.
        # Nebenwirkung (akzeptiert): der schlafende Pacing-Call hält einen
        # der max_threads-Semaphore-Plätze; Downloads laufen FIFO-serialisiert
        # — das Aufrufmuster bleibt gleich (s. ``YahooRateLimiter``).
        with self._semaphore:
            self._limiter.vor_yahoo_aufruf()
            return yf.download(
                tickers=symbol,
                interval=interval,
                start=us_zu_datetime(start_us) if start_us is not None else None,
                end=us_zu_datetime(ende_us) if ende_us is not None else None,
                auto_adjust=False,
                actions=True,
                ignore_tz=ignore_tz,
                repair=repair,
                progress=False,
                threads=False,  # Parallelisierung macht diese Schicht (Semaphore)
                timeout=self._timeout_s,
            )

    def _actions_persistieren(self, symbol: str, ereignisse: list[dict]) -> None:
        """Schreibt Dividenden-/Split-Events idempotent in ``corporate_actions``.

        Bewusst best-effort (E022): Die Balken selbst sind der wertvollere
        Teil des Fetch — eine DB-Panne beim Event-Upsert darf den Fetch nicht
        scheitern lassen, sie wird nur protokolliert (Nächster Download
        wiederholt den idempotenten Upsert). Ohne Events (z. B. Intraday)
        passiert hier nichts.
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

    def _ticker_probe(self, symbol: str) -> tuple[bool, dict[str, object]]:
        """Probe-Request: existiert das Kürzel und liefert es aktuelle Daten?"""
        import yfinance as yf  # lazy, s. _roh_download

        self._limiter.vor_yahoo_aufruf()  # auch die Probe ist ein Yahoo-Call
        ticker = yf.Ticker(symbol)
        historie = ticker.history(period="5d", auto_adjust=True)
        meta = getattr(ticker, "history_metadata", None) or {}
        existiert = historie is not None and len(historie) > 0
        return existiert, dict(meta)

    def _mit_wiederholung(self, func, *args, **kwargs):
        """Führt ``func`` mit exponentiellem Backoff + Jitter aus (Start 2 s).

        **Ausnahme Drosselung (T19/E035):** ``YFRateLimitError`` retryt
        yfinance-intern nicht (RECHERCHE v2 Block 1.8) — auch wir nicht:
        der Circuit öffnet sofort und der Fehler erscheint direkt als
        :class:`ProviderError` 429 ``yahoo_gedrosselt``.

        Erschöpfte Versuche → :class:`ProviderError` (502, ``quelle_nicht_erreichbar``);
        die gemessenen Schlafzeiten sind ``[2,3) · [4,6) · [8,12) …`` Sekunden
        (Basis × (1 … 1+jitter)) — deterministisch genug für Test-Asserts.
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
            except Exception as exc:  # noqa: BLE001 — jede Netz-/Formatpanne zählt
                letzter_fehler = exc
                if rate_limit.ist_rate_limit_fehler(exc):
                    dauer = self._limiter.verzeichne_rate_limit()
                    # Stooq-neutral (E087): dieses Bein kennt die Fallback-
                    # Kette nicht — die ehrliche, dynamische Quellen-Erwähnung
                    # baut der FallbackProvider (_stooq_oder_fehler).
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

    def _zu_kanonisch(
        self, df: pd.DataFrame | None, symbol: str, interval: str
    ) -> tuple[pd.DataFrame, str | None, list[dict]]:
        """yfinance-DataFrame → Schema-Vertrag des ABC (E006); Exchange-TZ lesen.

        Seit T20/E022 liefert der dritte Rückgabewert die Corporate-Actions-
        Events (Dividenden/Splits ≠ 0) des Fensters als
        ``[{"ts": µs, "action": "dividend"|"split", "value": float,
        "currency": str | None}, …]``; die Spalte ``adj_factor``
        (``Adj Close/Close``, fehlende Werte neutral 1.0) wird mitgegeben und
        fließt über :func:`parquet_store.upsert_bars` ins Parquet.
        """
        if df is None or len(df) == 0:
            return _leere_balken(), None, []

        arbeit = df.copy()
        if isinstance(arbeit.columns, pd.MultiIndex):
            # Neuere yfinance-Versionen liefern auch für Einzel-Ticker einen
            # MultiIndex (Feld, Ticker) — die Feld-Ebene gewinnt. Der Schnitt
            # ist case-insensitiv: Yahoo schreibt „Close“/„Adj Close“,
            # _FELDNAMEN sind klein (ohne das würde kein Flattening greifen und
            # der spätere Spaltenzugriff über pandas-Partial-Matching 2-d
            # liefern — sichtbar geworden über „Adj Close“, T20). Das direkte
            # Setzen der Ebene ist pandas-3-sicherer als ``droplevel(list)``.
            for ebene in range(arbeit.columns.nlevels):
                werte = {str(wert).lower() for wert in arbeit.columns.get_level_values(ebene)}
                if werte & set(_FELDNAMEN):
                    arbeit.columns = arbeit.columns.get_level_values(ebene)
                    break
        arbeit = arbeit.rename(columns=str.lower)

        index = pd.to_datetime(arbeit.index)
        exchange_tz: str | None = None
        if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
            exchange_tz = str(index.tz)
            index = index.tz_convert("UTC")
        else:
            index = index.tz_localize("UTC")  # tz-naiv (ignore_tz=True) gilt als UTC
        ts_us = (index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(1, "us")

        aus = pd.DataFrame({"ts": pd.Series(ts_us, index=arbeit.index)})
        for spalte in _FELDNAMEN:
            aus[spalte] = arbeit[spalte] if spalte in arbeit.columns else float("nan")
        # Lax-Vertrag (wie Store): Faktor-Spalte nur, wenn die Quelle „Adj Close“
        # tatsächlich lieferte — sonst füllt upsert_bars/read_bars neutral 1.0.
        if "adj close" in arbeit.columns:
            aus["adj_factor"] = self._adj_faktor(aus["close"], arbeit["adj close"])
        ereignisse = self._ereignisse(aus["ts"], arbeit)
        aus = aus.reset_index(drop=True)
        return parquet_store.normalize_ohlcv(aus, interval), exchange_tz, ereignisse

    @staticmethod
    def _adj_faktor(close: pd.Series, adj_close: pd.Series) -> pd.Series:
        """``Adj Close/Close`` je Bar (E022); neutral 1.0 bei defekten Werten.

        Division nur bei vorhandenem, **endlichem** und positivem ``Adj Close``
        sowie positivem, endlichem ``Close`` — 0/NaN/±inf (M1-Fix: Yahoo kann
        defekte „Adj Close“-Werte liefern; ``pd.to_numeric`` wandelt inf nicht
        um, es bleibt stehen) und negative Werte führen zu neutral 1.0 statt
        inf/NaN/negativem Faktor in der persistierten Spalte.
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

    @staticmethod
    def _ereignisse(ts: pd.Series, arbeit: pd.DataFrame) -> list[dict]:
        """Dividenden-/Split-Zeilen (Wert ≠ 0) aus dem Download-DataFrame.

        Spaltennamen nach der Kleinbuchstaben-Umbenennung (``dividends``,
        ``stock splits``); die optionale Währungsspalte (``dividends fx``)
        wird best-effort übernommen.
        """
        ereignisse: list[dict] = []
        for spalte, typ in _EREIGNIS_SPALTEN:
            if spalte not in arbeit.columns:
                continue
            serie = pd.to_numeric(arbeit[spalte], errors="coerce").fillna(0.0)
            for stelle in serie.index[serie != 0.0]:
                waehrung: str | None = None
                if "dividends fx" in arbeit.columns:
                    roh = arbeit.at[stelle, "dividends fx"]
                    if isinstance(roh, str) and roh.strip():
                        waehrung = roh.strip()
                ereignisse.append(
                    {
                        "ts": int(ts.loc[stelle]),
                        "action": typ,
                        "value": float(serie.loc[stelle]),
                        "currency": waehrung,
                    }
                )
        return ereignisse


def _leere_balken() -> pd.DataFrame:
    """Leeres DataFrame im kanonischen Schema (Spalten + dtypes, ABC-Vertrag)."""
    dtypes: dict[str, str] = {
        "ts": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    }
    return pd.DataFrame({spalte: pd.Series(dtype=dtype) for spalte, dtype in dtypes.items()})


__all__ = [
    "CHUNK_TAGE_1M",
    "FENSTER_MIN_HANDELSTAGE",
    "INTRADAY_LIMIT_NACHRICHT",
    "INTRADAY_MAX_TAGE",
    "YfinanceProvider",
    "fenster_erwartet_handelstage",
    "jetzt_us",
    "pruefe_intraday_fenster",
    "us_zu_datetime",
]
