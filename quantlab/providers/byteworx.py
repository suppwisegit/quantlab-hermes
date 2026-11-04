"""Byteworx-Provider (E094, Welle H2): ISIN-natives Datenbein für 1d/1m.

Die Planspiel-Börse zeichnet ihre Instrument-Charts über den Byteworx-Chart-
Endpunkt — ein Highcharts-Konfigurations-JSON mit den **echten Seriendaten**,
ohne Auth **und ohne API-Key** (RECHERCHE §Welle H, H.4; Referenz-Vertrag
``planspiel_data_tools/planspiel-app/planspiel-app/app/psb/charts.py::
chart_punkte``, Z. 23–78 — die Request-Parameter sind 1:1 übernommen). Rolle
im Verbund: im Download-Job-Manager laufen **ISIN-förmige Symbole VOR der
Yahoo-Fallback-Kette ausschließlich über dieses Bein** (autoritative
Wettbewerbskurse; Verdrahtung ``quantlab.services.downloads._symbol_laden``).
Der Nicht-ISIN-Pfad bleibt unverändert.

- **Endpoint:** ``GET https://charts2.byteworx.de/bwcharts/js/profile/data.json``
  mit ``key.isin``, ``key.codeMarket=_1`` (Stuttgart — Standardmarkt der
  Website) und den festen Layout-Parametern der Referenz (``f=1``,
  ``profile=mmcjs``, ``chart.title``, 700×400, ``user=MMC``,
  ``dataProfile=infront``).
- **Intervall-Mapping:** ``1d`` → ``timeSpan=5Y`` (Wochen-Schlusskurse, ~5
  Jahre gratis je ISIN), ``1m`` → ``timeSpan=1D`` (Intraday-Minuten). Alle
  anderen Intervalle → 422 ``byteworx_nur_1d_1m`` — Vorabprüfung **ohne Netz**.
- **Struktur:** ``series[0].data`` = ``[epoch_ms, close]``-Paare; Punkte mit
  ``null``-Kurs oder defektem Format fallen **einzeln** weg (Referenz
  Z. 67–72). Keine Highcharts-Serie / keine ``data``-Liste → 502
  ``byteworx_fehler`` (unerwartete Struktur, kein stiller Müll); leere Serie
  bzw. nur weggefallene Punkte → 404 ``keine_daten`` (Muster leeway-Bein:
  ehrliche deutsche Fehlerklasse statt still leerer Balken — die Referenz
  gibt ``[]`` zurück, weil ihr Aufrufer über Fallbacks entscheidet; hier
  entscheidet der Provider-Vertrag).
- **Zeit (E006):** Millisekunden → int64 **Mikrosekunden UTC**; Tages-Balken
  floort :func:`parquet_store.normalize_ohlcv` auf Mitternacht UTC. ANNAHME:
  Wochen-Schlusskurse gehören zum UTC-Kalendertag ihres Stempels — die
  Referenz rechnet für die Anzeige nach Berlin, QuantLab speichert kanonisch
  UTC (die Store-Invariante erzwingt den Floor ohnehin).
- **Close-only (E094):** geliefert wird nur der Schlusskurs — open/high/low
  werden **nicht erfunden**, sondern auf den Schlusskurs gesetzt. Volumen ist
  nicht verfügbar → **0.0**. ANNAHME (konsistent zum Bestand): fehlendes
  Volumen füllen auch das Stooq-Bein (``stooq_provider.py`` Z. 339) und
  :func:`parquet_store.normalize_ohlcv` neutral mit 0.0. Keine
  ``adj_factor``-Spalte — Store/Lesen füllen neutral 1.0 (Muster leeway-Bein).
- **Fenster:** Byteworx kennt kein start/end — die Serie kommt immer
  vollständig für den gewählten Zeitraum; das ABC-Fenster (``start`` inklusiv,
  ``end`` exklusiv, µs UTC) wird klientseitig auf die kanonischen µs
  angewendet, ohne Fenster bleibt die ganze Serie stehen.
- **ISIN-Kanonik (REV-Fix H-1):** ``key.isin`` wird stets in UPPER gesendet
  (Downloads-Normalisierung vor dem Branch, ``downloads._symbol_laden``) —
  psb_repo/Backfill-Konvention; Kleinschreib-Inputs landen store-kanonisch.
- **Fehler:** 404 → ``symbol_nicht_gefunden``; Netz-/HTTP-/JSON-Pannen → 502
  ``byteworx_fehler`` (deutsche Fachmeldungen im Stil der anderen Beine).
- **Pacing (eigen, ≥ 0,5 s):** prozessweiter Mindestabstand zwischen zwei
  Requests (Muster ``leeway_provider.py`` Z. 110–118) — Uhr/Schlaf sind
  modulweit injizierbar (Tests protokollieren statt schlafen). Bewusst NICHT
  der :class:`~quantlab.providers.rate_limit.YahooRateLimiter`: Byteworx ist
  nicht Yahoo, ein Byteworx-Pacing darf keine Yahoo-Circuits berühren.
- **Kein API-Key, keine Settings (E094):** der Endpunkt ist frei zugänglich —
  dieses Bein braucht keinerlei Konfiguration und ist im Download-Manager
  deshalb **immer verdrahtet** (Kontrast: leeway-Key in den Einstellungen).

Abruf via **httpx** mit injizierbarem Transport — Tests fahren
:class:`httpx.MockTransport`, kein echtes Netz (E013, Muster Stooq-/
YahooDirect-/leeway-Bein).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import httpx
import pandas as pd

from quantlab.providers.base import (
    DataProvider,
    ProviderError,
    SymbolInfo,
    SymbolNotFoundError,
    validate_interval,
)
from quantlab.providers.symbols import validate_symbol
from quantlab.storage import parquet_store

#: Chart-Daten-Endpunkt (RECHERCHE §H.4; Referenz charts.py Z. 23).
CHART_URL = "https://charts2.byteworx.de/bwcharts/js/profile/data.json"

#: ``key.codeMarket`` — Stuttgart (Standardmarkt der Planspiel-Website;
#: Referenz charts.py Z. 25, ``STANDARD_CODE_MARKET``).
CODE_MARKET = "_1"

#: Request-Timeout (Sekunden) — Referenz charts.py Z. 41 (15.0).
TIMEOUT_S = 15.0

#: Intervall (yfinance-Schreibweise) → Byteworx-``timeSpan`` (RECHERCHE §H.4;
#: E094: 1D→1m, 5Y→1d). Alle anderen Intervalle werden vorab abgelehnt.
ZEITSPAN_PRO_INTERVAL: dict[str, str] = {"1d": "5Y", "1m": "1D"}

#: Mindestabstand zwischen zwei Byteworx-Requests (prozessweit; E094: eigenes
#: Pacing ≥ 0,5 s — Muster leeway_provider.py Z. 110–118).
_PACING_S = 0.5

_ZUSTAND_LOCK = threading.Lock()
_LETZTER_SLOT: float | None = None  # _uhr()-Wert des letzten Pacing-Slots


# ---------------------------------------------------------------------------
# Prozessweites Pacing (Muster leeway_provider.py Z. 110–118)
# ---------------------------------------------------------------------------


def _uhr() -> float:
    """Monotone Uhr — Tests monkeypatchen diese Funktion (netzfreie Zeit)."""
    return time.monotonic()


def _schlaf(sekunden: float) -> None:
    """Schläft die Pacing-Restzeit — Tests protokollieren statt schlafen."""
    time.sleep(sekunden)


def zustand_zuruecksetzen() -> None:
    """Setzt den Pacing-Slot zurück (Test-Hygiene, Muster leeway-Bein)."""
    global _LETZTER_SLOT
    with _ZUSTAND_LOCK:
        _LETZTER_SLOT = None


def _wahre_pacing_abstand() -> float:
    """Prozessweiter Mindestabstand zwischen zwei Byteworx-Calls (Slot unter Lock).

    Slot-Reservierung unter Lock, geschlafen wird **außerhalb** — der Schlaf
    blockiert keine parallelen Reservierungen (Muster ``websuche``/leeway).
    """
    global _LETZTER_SLOT
    with _ZUSTAND_LOCK:
        moment = _uhr()
        warte = 0.0
        if _LETZTER_SLOT is not None:
            warte = _LETZTER_SLOT + _PACING_S - moment
            if warte < 0.0:
                warte = 0.0
        _LETZTER_SLOT = moment + warte
    if warte > 0.0:
        _schlaf(warte)
    return warte


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ByteworxProvider(DataProvider):
    """ISIN-natives Bein auf den Byteworx-Chart-Endpunkt (E094/H2).

    **Kein Key, keine Settings** (Modul-Docstring) — der Konstruktor nimmt nur
    Betriebshygiene-Parameter; ``transport`` ist injizierbar
    (:class:`httpx.MockTransport` in Tests), produktiv läuft ein
    wiederverwendeter :class:`httpx.Client` (Timeout :data:`TIMEOUT_S`).
    """

    def __init__(
        self,
        *,
        timeout_s: float = TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._transport = transport
        self._client_instanz: httpx.Client | None = None

    # -- DataProvider-ABC ------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        repair: bool = False,
    ) -> pd.DataFrame:
        """Kanonische Close-only-Balken (ABC-Vertrag) — EIN Netzcall.

        ``repair`` wird akzeptiert, aber **ignoriert** (Signatur-Vertrag des
        ABC, Muster Stooq-/leeway-Bein): clientseitiger yfinance-Repair hat
        keine Byteworx-Entsprechung.

        Reihenfolge der Vorabprüfungen (alles **ohne Netz**): Symbol →
        Intervall (nur ``1d``/``1m``). Dann ein Request mit dem vollen
        ``timeSpan``; das µs-Fenster wird klientseitig gefiltert (Modul-
        Docstring „Fenster“).
        """
        geprueft = validate_symbol(symbol)
        validate_interval(interval)
        zeitraum = ZEITSPAN_PRO_INTERVAL.get(interval)
        if zeitraum is None:
            raise ProviderError(
                f"Byteworx liefert ausschließlich Tages- („1d“) und "
                f"Minutenbalken („1m“) — das Intervall „{interval}“ ist dort "
                "nicht verfügbar.",
                422,
                "byteworx_nur_1d_1m",
                hint="Verfügbar: „1d“ (5 Jahre Wochen-Schlusskurse) und „1m“ "
                "(Intraday des Handelstags).",
            )
        payload = self._data_json(geprueft, zeitraum)
        return self._zu_kanonisch(payload, geprueft, interval, start, end)

    def resolve_symbol(self, symbol: str) -> SymbolInfo:
        """Nur Form-Prüfung **ohne Netz** — ISIN bleibt ISIN (E094, 1:1).

        Byteworx liefert keine Namens-/Währungs-Metadaten im Chart-JSON —
        nichts wird erfunden (``name``/``asset_type``/… bleiben ``None``;
        Muster leeway-Bein, dort ohne Erfindungs-Heuristik). Eine ISIN wird
        unverändert durchgereicht; das Chart-JSON selbst ist die Güteprobe.
        """
        geprueft = validate_symbol(symbol)
        return SymbolInfo(symbol=geprueft)

    # -- Herkunftsmarke (Vertrag des Download-Managers, E023/E094) --------

    def letzte_quelle(self) -> str:
        """``'byteworx'`` — landet via Download-Manager in ``bar_meta.source``."""
        return "byteworx"

    # -- Innere Mechanik -------------------------------------------------

    def close(self) -> None:
        """Gibt den wiederverwendeten httpx-Client frei (Hygiene, optional)."""
        if self._client_instanz is not None:
            self._client_instanz.close()
            self._client_instanz = None

    def _client(self) -> httpx.Client:
        """Wiederverwendeter Client (Verbindungspooling, injizierbarer Transport)."""
        if self._client_instanz is None:
            self._client_instanz = httpx.Client(
                transport=self._transport,
                timeout=self._timeout_s,
            )
        return self._client_instanz

    def _data_json(self, isin: str, zeitraum: str) -> Any:
        """Der **einzige** Netzcall dieses Beins — Pacing, Fehlerklassen.

        Request-Parameter exakt nach Referenz ``chart_punkte`` (Z. 42–55);
        Header wie die Referenz (keine Sonderbehandlung). Fehlertexte deutsch:
        404 → ``symbol_nicht_gefunden``, Netz-/HTTP-/JSON-Pannen → 502
        ``byteworx_fehler``.
        """
        _wahre_pacing_abstand()
        params: dict[str, Any] = {
            "f": "1",
            "profile": "mmcjs",
            "key.isin": isin,
            "timeSpan": zeitraum,
            "key.codeMarket": CODE_MARKET,
            "chart.title": isin,
            "chart.width": "700",
            "chart.height": "400",
            "user": "MMC",
            "dataProfile": "infront",
        }
        try:
            antwort = self._client().get(CHART_URL, params=params)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Byteworx-Abruf für „{isin}“ fehlgeschlagen: {exc}",
                502,
                "byteworx_fehler",
            ) from exc
        if antwort.status_code == 404:
            raise SymbolNotFoundError(
                f"ISIN „{isin}“ ist bei Byteworx nicht bekannt (HTTP 404).",
                hint="ISIN prüfen (ISO 6166, z. B. „DE0007164600“).",
            )
        if antwort.status_code != 200:
            raise ProviderError(
                f"Byteworx antwortete mit HTTP {antwort.status_code} für „{isin}“.",
                502,
                "byteworx_fehler",
            )
        try:
            return antwort.json()
        except ValueError as exc:
            raise ProviderError(
                f"Byteworx lieferte für „{isin}“ kein gültiges JSON.",
                502,
                "byteworx_fehler",
            ) from exc

    def _zu_kanonisch(
        self,
        payload: Any,
        symbol: str,
        interval: str,
        start: int | None,
        end: int | None,
    ) -> pd.DataFrame:
        """Highcharts-JSON → kanonisches Schema (E006); defekte Punkte fallen weg.

        Strukturpannen (keine Serie, keine ``data``-Liste) → 502
        ``byteworx_fehler``; leere Serie bzw. nach Fensterfilter leeres
        Ergebnis → 404 ``keine_daten``. Das µs-Fenster ist klientseitig (inklusiv/
        exklusiv — ABC-Vertrag), weil Byteworx keine Fenster kennt.
        """
        if not isinstance(payload, dict):
            raise ProviderError(
                f"Byteworx lieferte für „{symbol}“ eine unerwartete "
                "Antwortstruktur (Highcharts-Objekt erwartet).",
                502,
                "byteworx_fehler",
            )
        serien = payload.get("series")
        if not isinstance(serien, list) or not serien or not isinstance(serien[0], dict):
            raise ProviderError(
                f"Byteworx lieferte für „{symbol}“ eine unerwartete "
                "Antwortstruktur (Highcharts-Serie erwartet).",
                502,
                "byteworx_fehler",
            )
        roh_daten = serien[0].get("data")
        if not isinstance(roh_daten, list):
            raise ProviderError(
                f"Byteworx lieferte für „{symbol}“ eine unerwartete "
                "Antwortstruktur (Serie ohne „data“-Liste).",
                502,
                "byteworx_fehler",
            )
        zeilen = [
            zeile
            for zeile in (self._punkt_lesen(paar) for paar in roh_daten)
            if zeile is not None
        ]
        if start is not None:
            zeilen = [zeile for zeile in zeilen if zeile["ts"] >= start]
        if end is not None:
            zeilen = [zeile for zeile in zeilen if zeile["ts"] < end]
        if not zeilen:
            raise ProviderError(
                f"Für „{symbol}“ liefert Byteworx keine Daten im Intervall "
                f"„{interval}“.",
                404,
                "keine_daten",
            )
        roh = pd.DataFrame(
            zeilen, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        # normalize_ohlcv floort Tages-ts auf Mitternacht UTC (E006) und
        # erzwingt Spalten/dtypes; keine adj_factor-Spalte → Store/Lesen
        # füllen neutral 1.0 (Adjust-Entscheid, Modul-Docstring).
        return parquet_store.normalize_ohlcv(roh, interval)

    @staticmethod
    def _punkt_lesen(paar: Any) -> dict[str, float | int] | None:
        """Ein Highcharts-Paar ``[epoch_ms, close]`` → Kanon-Zeile (oder ``None``).

        ``null``-Kurse (Highcharts-Lücken) und defekte Paare fallen einzeln
        weg (Referenz Z. 67–72); Zahlenformate werden tolerant gelesen
        (String-Kurse wie im leeway-Bein), NaN/±inf führt zum Verwerfen.
        """
        if not isinstance(paar, (list, tuple)) or len(paar) < 2:
            return None
        try:
            ms = int(paar[0])
            kurs = float(paar[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(kurs):
            return None
        return {
            "ts": ms * 1_000,  # Millisekunden → kanonische Mikrosekunden (E006)
            "open": kurs,
            "high": kurs,
            "low": kurs,
            "close": kurs,
            "volume": 0.0,  # Byteworx liefert kein Volumen (Modul-Docstring)
        }


__all__ = [
    "CHART_URL",
    "CODE_MARKET",
    "TIMEOUT_S",
    "ByteworxProvider",
    "ZEITSPAN_PRO_INTERVAL",
    "zustand_zuruecksetzen",
]
