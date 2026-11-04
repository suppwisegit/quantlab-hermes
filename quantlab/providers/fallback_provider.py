"""Fallback-Composite (T19, E023; Vier-Bein seit T-G2/E088): Direkt → yfinance → leeway → Stooq.

Das Composite sitzt als :class:`~quantlab.providers.base.DataProvider` vor dem
Download-Job-Manager (:func:`quantlab.services.downloads.get_provider`) —
nach außen unverändert der ABC-Vertrag, nach innen Quell-Auswahl:

1. **Bein 1 — YahooDirect** (:mod:`quantlab.providers.yahoo_direct`, httpx auf
   v8/chart) bedient **alle** Intervalle; ``yahoo_direct=None`` deaktiviert das
   Bein (Bestands-Tests bleiben netzfrei; der Betrieb verdrahtet es in
   ``downloads.get_provider`` explizit).
2. **Bein 2 — yfinance** übernimmt bei jedem Direct-Fehler — **erstmals auch
   für Intraday** (T-F9). Vor jedem Eintritt gilt der Circuit-Guard je Bein
   (E087: ``direct_limiter``/``yahoo_limiter``, Legacy-Vererbung von
   ``rate_limiter=``): sind **beide** Circuits offen, wird **gar kein**
   Yahoo-Call geführt — Daily-Abfragen gehen dann direkt an leeway/Stooq,
   Intraday erhält den Drossel-Fehler (429). Ist nur der Direct-Circuit offen,
   wird nur das Direct-Bein übersprungen — das yfinance-Bein versucht es
   trotzdem (der Anlass-Fall E087: ein Direct-429 machte früher beide Beine
   tot).
3. **Bein 3 — leeway (T-G2/E088, nur ``1d``):** ``quelle2=None`` deaktiviert
   das Bein (ohne konfigurierten Key baut der Betrieb gar kein Bein); ist es
   verdrahtet, wird es bei jedem Yahoo-Fehler **und** im Circuit-offen-Pfad
   vor Stooq versucht. Es ist eine reine Tagesdaten-Quelle (Free-Tier 50/Tag,
   „1 Request/Symbol/Tag“ dimensioniert) — Intraday überspringt es. Ein
   leeway-Fehler (inkl. „Key fehlt“) wird wie „Bein nicht verfügbar“
   behandelt: die Kette läuft weiter, kein Abbruch.
4. **Nur Intervall ``1d``** fällt weiter bis Stooq. Intraday endet am
   yfinance-Bein — schlägt es dort, erscheint dessen Fehler.
5. **Fehlerführung:** der Fehler des **primären** Yahoo-Beins (Direct) bleibt
   diagnoseführend: scheitern die Vorgänger-Beine, erscheint bei Stooq-
   Fehlschlag der Direct-Fehler. Der Ergänzungs-Hint nennt Stooq nur, wenn
   das Bein verdrahtet ist und laut ``stooq_provider.status()`` verfügbar —
   sonst ehrlich „alle Tagesdaten-Quellen nicht erreichbar“ (E087); die
   zweite Quelle wird je Konfigurationsstand erwähnt
   (:meth:`FallbackProvider._quelle2_hint`).
6. **Herkunft:** :meth:`letzte_quelle` meldet ``'yahoo_direct' | 'yahoo' |
   'leeway' | 'stooq'`` des letzten erfolgreichen Fetch — der Download-Service
   schreibt es in ``bar_meta.source`` (E022/E023/E088), sobald Balken
   gesichert wurden.

``resolve_symbol`` delegiert an das yfinance-Bein (Namens-/TZ-Metadaten; die
v8-meta führt keinen Anzeigenamen); der Composite fügt keine eigene
Validierung ein.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import pandas as pd

from quantlab.providers import rate_limit as rate_limit_mod
from quantlab.providers import stooq_provider
from quantlab.providers.base import DataProvider, ProviderError, SymbolInfo
from quantlab.providers.rate_limit import YahooRateLimiter
from quantlab.providers.stooq_provider import StooqProvider
from quantlab.providers.yahoo_direct import unterstuetzt_chunk_fortschritt
from quantlab.providers.yfinance_provider import YfinanceProvider


class FallbackProvider(DataProvider):
    """Yahoo-primär/Mehrbein-Fallback hinter dem ABC (Details: Modul-Docstring).

    Alle Beine sind injizierbar — Tests stecken Fakes bzw. einen Limiter mit
    gemockter Clock hinein (E013: kein echtes Netz).
    """

    def __init__(
        self,
        *,
        yahoo_direct: DataProvider | None = None,
        yahoo: DataProvider | None = None,
        stooq: DataProvider | None = None,
        quelle2: DataProvider | None = None,
        rate_limiter: YahooRateLimiter | None = None,
        direct_limiter: YahooRateLimiter | None = None,
        yahoo_limiter: YahooRateLimiter | None = None,
    ) -> None:
        # ``yahoo_direct=None`` = Bein deaktiviert (Kette startet bei yfinance) —
        # so bleiben Bestands-Tests, die nur yahoo/stooq injizieren, netzfrei.
        self._yahoo_direct = yahoo_direct
        self._yahoo = yahoo if yahoo is not None else YfinanceProvider()
        self._stooq = stooq if stooq is not None else StooqProvider()
        # Zweite Tagesdaten-Quelle (E088, T-G2): ``None`` = Bein nicht
        # verdrahtet (z. B. kein leeway-Key konfiguriert) — die Kette verhält
        # sich exakt wie im Drei-Bein-Bestand.
        self._quelle2 = quelle2
        # Circuits je Bein (E087): ``rate_limiter=`` erbt an beide (Legacy-
        # Vertrag — explizit injiziert heißt weiterhin geteilter Circuit,
        # Bestands-Tests bleiben semantics-preserving grün); ohne Injektion
        # gelten die getrennten Prozess-Defaults.
        self._direct_limiter = (
            direct_limiter or rate_limiter or rate_limit_mod.RATE_LIMITER
        )
        self._yahoo_limiter = (
            yahoo_limiter or rate_limiter or rate_limit_mod.RATE_LIMITER
        )
        self._quellen_lock = threading.Lock()
        self._quelle = "yahoo_direct" if yahoo_direct is not None else "yahoo"

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
        """ABC-Vertrag; Quell-Auswahl gemäß E023/E085 (Modul-Docstring).

        ``fortschritt`` (T-F10, duck-geoptimer Vertragszusatz): optionaler
        Chunk-Progress-Callback, der **ausschließlich** ans YahooDirect-Bein
        durchgereicht wird (und nur, wenn dieses den Zusatz kennt — Probe
        :func:`quantlab.providers.yahoo_direct.unterstuetzt_chunk_fortschritt`;
        Test-Fakes als Direkt-Bein bleiben so invariant). Das yfinance-Bein
        erhält den Callback bewusst NICHT: es ist nur Fallback (keine
        Chunk-Progress-Pflicht gemäß T-F10 — Besitz des Beins lief parallel in
        T-F12; Aufwand/Nutzen sprach gegen eine zweite Verkabelung).
        """
        if self._yahoo_direct is not None:
            direct_gesperrt = self._direct_limiter.ist_gesperrt()
            yahoo_gesperrt = self._yahoo_limiter.ist_gesperrt()
            if direct_gesperrt and yahoo_gesperrt:
                # Beide Circuits offen: kein einziger Yahoo-Call (E035) —
                # 1d über leeway/Stooq, Intraday kann nicht liefern.
                return self._ohne_yahoo(symbol, interval, start, end)
            if direct_gesperrt:
                # Nur der Direct-Circuit ist offen (E087): nur dieses Bein
                # überspringen — das yfinance-Bein versucht es trotzdem.
                return self._ueber_yfinance(symbol, interval, start, end, repair, None)
            try:
                if fortschritt is not None and unterstuetzt_chunk_fortschritt(
                    self._yahoo_direct
                ):
                    df = self._yahoo_direct.fetch_ohlcv(  # type: ignore[call-arg]
                        symbol,
                        interval,
                        start=start,
                        end=end,
                        repair=repair,
                        fortschritt=fortschritt,
                    )
                else:
                    df = self._yahoo_direct.fetch_ohlcv(
                        symbol, interval, start=start, end=end, repair=repair
                    )
            except ProviderError as direct_fehler:
                # Erstes Bein gescheitert → yfinance-Bein (T-F9: auch Intraday).
                return self._ueber_yfinance(
                    symbol, interval, start, end, repair, direct_fehler
                )
            with self._quellen_lock:
                self._quelle = "yahoo_direct"
            return df
        # Direct-Bein nicht verdrahtet → klassische Zweibein-Kette (Bestand).
        if self._yahoo_limiter.ist_gesperrt():
            return self._ohne_yahoo(symbol, interval, start, end)
        return self._ueber_yfinance(symbol, interval, start, end, repair, None)

    def resolve_symbol(self, symbol: str) -> SymbolInfo:
        """Metadaten kommen vom yfinance-Bein (Name/TZ); die v8-meta hat keinen Namen."""
        return self._yahoo.resolve_symbol(symbol)

    # ------------------------------------------------------------------
    # Quell-Auswahl
    # ------------------------------------------------------------------

    def letzte_quelle(self) -> str:
        """Herkunft des letzten erfolgreichen Fetch:
        ``'yahoo_direct' | 'yahoo' | 'leeway' | 'stooq'``."""
        with self._quellen_lock:
            return self._quelle

    def _ohne_yahoo(
        self, symbol: str, interval: str, start: int | None, end: int | None
    ) -> pd.DataFrame:
        """Circuit-offen-Pfad (E035): kein weiterer Yahoo-Call bei **beiden** Beinen.

        Daily geht über das leeway-Bein (E088, falls verdrahtet) an Stooq;
        Intraday hat ohne Yahoo keine Quelle und erhält den Drossel-Fehler
        (429) mit der aktuellen Restdauer des Direct-Circuits
        (diagnoseführend, E087).
        """
        if interval != "1d":
            raise self._direct_limiter.gesperrt_fehler()
        return self._ueber_leeway(symbol, start, end, None)

    def _tagesquellen_hint(self) -> str:
        """Ehrlicher Hinweis zu den Tagesdaten-Quellen (E087, D-G1).

        Stooq wird nur genannt, wenn das Bein verdrahtet ist **und** der
        Prozess-Status es als verfügbar meldet
        (``stooq_provider.status()["state"] == "ok"``) — nach erkanntem
        Tageslimit/Blockade schweigt der Hinweis über Stooq (der Anlass-Fall:
        der Text versprach den toten Fallback). Der Konfigurationsstand der
        zweiten Quelle (leeway) geht ergänzend ein
        (:meth:`_quelle2_hint`, E088). Ohne verfügbares Stooq-Bein die ehrliche
        Brücke zur Auto-Wiederholung (E089, ``wiederholung_um``).
        """
        if self._stooq is not None and stooq_provider.status().get("state") == "ok":
            basis = (
                "Tagesdaten werden automatisch über Stooq geladen; "
                "Intraday-Daten warten den Cool-down ab."
            )
        else:
            basis = (
                "Alle Tagesdaten-Quellen sind derzeit nicht erreichbar — "
                "automatische Wiederholung folgt."
            )
        quelle2_satz = self._quelle2_hint()
        return f"{basis} {quelle2_satz}" if quelle2_satz is not None else basis

    def _quelle2_hint(self) -> str | None:
        """Satz zur zweiten Tagesdaten-Quelle (E088/T-G2; T-G1-Platzhalter gefüllt).

        Der Hinweis läuft nur in Fehlerpfaden, in denen das Bein seine Chance
        bereits hatte: verdrahtet → ehrlich „ist konfiguriert, hatte aber
        keinen Erfolg“; ohne Verdrahtung (betriebsseitig: kein Key in den
        Einstellungen → Daten) → Konfigurations-Hinweis. ``None`` nie im
        Fehlerkontext — die zweite Quelle gehört in jedes ehrliche Lagebild.
        """
        if self._quelle2 is not None:
            return (
                "Die zweite Tagesquelle (leeway.tech) ist konfiguriert, "
                "hatte aber ebenfalls keinen Erfolg."
            )
        return (
            "Hinweis: Für die zweite Tagesquelle (leeway.tech) fehlt der "
            "API-Key — optional in den Einstellungen (→ Daten) hinterlegen."
        )

    def _mit_tagesquellen_hint(self, fehler: ProviderError) -> ProviderError:
        """Versieht ``fehler`` mit dem dynamischen Quellen-Hint (E087).

        Fehlerklasse, ``rest_s``, Meldung, Status und Code bleiben unangetastet
        — nur der Hinweis wird ehrlich je Tagesquellen-Status gesetzt.
        """
        fehler.hint = self._tagesquellen_hint()
        return fehler

    def _ueber_yfinance(
        self,
        symbol: str,
        interval: str,
        start: int | None,
        end: int | None,
        repair: bool,
        direct_fehler: ProviderError | None,
    ) -> pd.DataFrame:
        """Zweites Bein (yfinance) — erstmals auch für Intraday erreichbar (T-F9).

        Daily fällt bei dessen Fehlschlag über das leeway-Bein (E088) weiter
        auf Stooq; dabei bleibt der **Direct**-Fehler diagnoseführend
        (primäre Quelle gewinnt). Intraday endet hier — leeway/Stooq liefern
        keine Tages-unabhängigen Balken.
        """
        try:
            df = self._yahoo.fetch_ohlcv(
                symbol, interval, start=start, end=end, repair=repair
            )
        except ProviderError as yahoo_fehler:
            if interval != "1d":
                raise
            return self._ueber_leeway(
                symbol, start, end, direct_fehler or yahoo_fehler
            )
        with self._quellen_lock:
            self._quelle = "yahoo"
        return df

    def _ueber_leeway(
        self,
        symbol: str,
        start: int | None,
        end: int | None,
        yahoo_fehler: ProviderError | None,
    ) -> pd.DataFrame:
        """Drittes Bein (leeway, E088/T-G2) — reine Tagesquelle, vor Stooq.

        Nur für ``1d`` (leeway liefert keine Intraday-Daten); das Bein wurde
        betriebsseitig nur dann verdrahtet, wenn ein Key konfiguriert ist.
        Jeder leeway-Fehler — auch „Key fehlt“/Drosselung — wird wie
        „Bein nicht verfügbar“ behandelt: die Kette fällt auf den
        BESTEHENDEN Stooq-Pfad durch, der primäre Yahoo-Fehler bleibt
        diagnoseführend (kein Abbruch). Ein leeres Ergebnis zählt ebenfalls
        als „Bein ohne Erfolg“.
        """
        if self._quelle2 is not None:
            try:
                df = self._quelle2.fetch_ohlcv(symbol, "1d", start=start, end=end)
            except ProviderError:
                return self._stooq_oder_fehler(symbol, start, end, yahoo_fehler)
            if not df.empty:
                with self._quellen_lock:
                    self._quelle = "leeway"
                return df
        return self._stooq_oder_fehler(symbol, start, end, yahoo_fehler)

    def _stooq_oder_fehler(
        self,
        symbol: str,
        start: int | None,
        end: int | None,
        yahoo_fehler: ProviderError | None = None,
    ) -> pd.DataFrame:
        """Ein Stooq-Versuch; scheitert er, gewinnt der primäre Yahoo-Fehler.

        Ohne vorherigen Yahoo-Fehler (Cool-down-Pfad) entscheidet die Stooq-
        Fehlerart: 404 („keine Daten“) und der selbsterklärende Stooq-Block
        erscheinen direkt, alles andere wird vom Drossel-Fehler des Direct-
        Circuits überdeckt — der Nutzer soll wissen, **warum** Yahoo gerade
        nicht gefragt wurde. Der Ergänzungs-Hint nennt Stooq nur bei realer
        Verfügbarkeit (E087, :meth:`_tagesquellen_hint`).
        """
        try:
            df = self._stooq.fetch_ohlcv(symbol, "1d", start=start, end=end)
        except ProviderError as stooq_fehler:
            if yahoo_fehler is not None:
                raise self._mit_tagesquellen_hint(yahoo_fehler) from stooq_fehler
            if stooq_fehler.status_code == 404 or stooq_fehler.code == "stooq_blockiert":
                raise
            gedrosselt = self._direct_limiter.gesperrt_fehler()
            raise self._mit_tagesquellen_hint(gedrosselt) from stooq_fehler
        if df.empty:
            fehler = yahoo_fehler or ProviderError(
                f"Für „{symbol}“ liegen gerade keine Daten an (Yahoo im Cool-down, "
                "Stooq lieferte nichts).",
                404,
                "keine_daten",
            )
            raise fehler
        with self._quellen_lock:
            self._quelle = "stooq"
        return df


__all__ = ["FallbackProvider"]
