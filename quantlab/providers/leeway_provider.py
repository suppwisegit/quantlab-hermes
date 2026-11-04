"""Leeway-Provider (T-G2, E088): zweite Tagesdaten-Quelle als Fallback-Bein.

Schlüsselpflichtiger Tagesdaten-Endpunkt von **leeway.tech** (G-0a-SIEGER,
RECHERCHE.md §Welle G — Vertrag dort LIVE gegen OpenAPI/401/429 geprüft,
2026-09-21). Rolle im Verbund: **Fallback dimensioniert** — das Free-Tier
umfasst 50 Abfrageeinheiten/Tag, deshalb ist dieses Bein bewusst als
„1 Request/Symbol/Tag“ dimensioniert; **Yahoo bleibt Primär** (E088), leeway
springt im
:class:`~quantlab.providers.fallback_provider.FallbackProvider` erst an, wenn
beide Yahoo-Beine scheitern.

- **Endpoint:** ``GET https://api.leeway.tech/api/v1/public/historicalquotes/
  {symbolExchange}?from=YYYY-MM-DD&to=YYYY-MM-DD&apitoken=TOKEN`` — ``from``/
  ``to`` (inklusiv bei leeway; unser ``end`` ist exklusiv → ``to`` = Vortag).
  Ein Request trägt bis zu 100 Jahre Historie → **5y in EINEM Request** (kein
  Chunking wie bei Yahoo-1m; :meth:`fetch_ohlcv` führt genau einen Netzcall).
- **Auth:** ``apitoken``-Query-Parameter. Der Key wird **niemals** im Klartext
  geloggt oder in Fehlermeldungen wiederholt — Ausnahmetexte werden vor der
  Weitergabe maskiert (:meth:`LeewayProvider._maskiere`).
- **Symbole:** ``SYMBOL.EXCHANGE`` (offizielles Beispiel ``SAP.XETRA``), US
  ``AAPL`` (ANNAHME laut G-0a analog ``ACWI.US``-Muster), **ISIN direkt
  zulässig** (doku-belegt: „Ticker und ISIN“). Der Provider nimmt das Symbol
  **1:1** — kein Mapping (anders als :mod:`quantlab.providers.stooq_provider`).
- **Response:** JSON-Liste je Tag mit den Feldern ``date, o, h, l, c, v``
  (Shape aus offizieller Doku; Detailschachtelung mit echtem Token noch
  gegenzuprüfen — UNVERIFIZIERT, G-0a). Zeilen ohne``date``/unlesbare Zahlen
  fallen einzeln weg; gar keine Liste → 502 ``leeway_fehler``, leere Liste →
  404 ``keine_daten``.
- **Fehler (G-0a, live beobachtet):** 401 „Apitoken is not valid“ → 401
  ``leeway_key_ungueltig`` (Hint auf Einstellungen → Daten); 404 Symbol
  unbekannt → 404 ``symbol_nicht_gefunden``; 429 ``RATE_LIMIT_EXCEEDED`` → 429
  ``leeway_gedrosselt`` **mit Cooldown im EIGENEN Limiter** — bewusst NICHT in
  den Yahoo-Circuits (:mod:`quantlab.providers.rate_limit`): leeway ist nicht
  Yahoo, ein leeway-429 darf kein Yahoo-Bein berühren (E087-Entflechtung).
- **Eigener Schutz (nicht YahooRateLimiter):** einfache prozessweite Pacing-/
  429-Behandlung analog :mod:`quantlab.services.websuche` — Mindestabstand
  zwischen zwei Calls (:data:`_PACING_S`) plus Backoff-Sperrfenster nach 429
  (:data:`_SPERRFENSTER_S`); Uhr/Schlaf sind modulweit injizierbar (Tests
  monkeypatchen :func:`_uhr`/:func:`_schlaf`).
- **Kein Key → sofortiger Fachfehler** (``quelle2_nicht_konfiguriert``, 502)
  **ohne Netz** — der Composite behandelt das wie „Bein nicht verfügbar“ und
  läuft in der Kette weiter.
- **Nur Tagesdaten:** Intervall ``1d``; alles andere → 422
  ``leeway_nur_tagesdaten`` (Muster Stooq-Bein). Intraday bleibt Yahoo-
  vorbehalten.

**Adjust-Entscheidung (D-G2/E088, bindend):** leeway liefert Schlusskurse
(UNVERIFIZIERT im Detail) ohne separaten AdjClose-Track — dieser Provider
schreibt daher **keine** ``adj_factor``-Spalte (der Store/das Lesen füllen
neutral mit 1.0) und der Bestand wird mit ``bar_meta.source = 'leeway'``
markiert. Der ``data_service``-Sonderfall ``bar_quelle(...) == "stooq"`` wird
bewusst **nicht** erweitert: weil der Faktor exakt 1.0 ist, ist
``adjust="adjusted"`` (OHLC × Faktor) mathematisch ein **No-op** — beide
Lesemodi liefern dieselben, bereits adjustierten Werte, es entsteht also kein
„adjusted lügt still“. Das Stooq-Sonderverhalten (Multiplikation überspringen
+ UI-Hinweis) bleibt allein für ``stooq`` bestehen.

**Tages-Stempel (ANNAHME, dokumentiert):** leeway liefert das Datum OHNE
Intraday-Zeit. Wir interpretieren es als **UTC-Mitternacht** (Muster
Stooq-Bein; ``normalize_ohlcv`` floort Tages-ts ohnehin auf UTC-Mitternacht).
Für US-Symbole ist das identisch zum ``yahoo_direct``-Daily-Pfad (Exchange-
Mitternacht New York liegt im selben UTC-Tag). Für XETRA-Tage kann der
Stempel um einen UTC-Tag vom Yahoo-Stempel abweichen (Exchange-Mitternacht
Europa liegt im **vor**hergehenden UTC-Tag) — im Upsert (Dedup auf ``ts``,
last-write-wins) führt das im Grenzfall zu zwei Zeilen je Handelstag; die
Konsolidierung wäre ein Live-Punkt mit echtem Token (G-0a offener Punkt).

Abruf via **httpx** mit injizierbarem Transport — Tests fahren
:class:`httpx.MockTransport`, kein echtes Netz (E013, Muster
Stooq-/YahooDirect-Bein).
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
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

#: historicalquotes-Endpunkt (G-0a-Vertrag, Live-OpenAPI 2026-09-21).
LEEWAY_URL = "https://api.leeway.tech/api/v1/public/historicalquotes"

#: Rückfenster in Tagen ohne explizites ``start`` — 5 Jahre (Projekt-Standard-
#: periode „5y“; „nie komplette Historie ohne Not“ E023, ein Request trägt es).
DEFAULT_FENSTER_TAGE = 1825

#: Request-Timeout (Sekunden) — G-0a-Implementierungs-Vertrag.
TIMEOUT_S = 15.0

#: Mindestabstand zwischen zwei leeway-Calls (prozessweit; Free-Tier 50/Tag
#: ist pro Symbol/Paarung großzügig, Serien-Downloads sollen sich aber
#: messbar verhalten). Nicht-Yahoo-Pacing, bewusst nicht der YahooRateLimiter.
_PACING_S = 1.0

#: Backoff-Sperrfenster nach einem leeway-429 (Sekunden) — danach ist ein
#: erneuter Versuch erlaubt (der Tages-Limit-Fall liefert dann wieder 429 und
#: schiebt das Fenster neu).
_SPERRFENSTER_S = 60.0


# ---------------------------------------------------------------------------
# Prozessweiter leeway-Zustand (Pacing + 429-Sperrfenster; Muster websuche)
# ---------------------------------------------------------------------------


def _uhr() -> float:
    """Monotone Uhr — Tests monkeypatchen diese Funktion (netzfreie Zeit)."""
    return time.monotonic()


def _schlaf(sekunden: float) -> None:
    """Schläft die Pacing-Restzeit — Tests protokollieren statt schlafen."""
    time.sleep(sekunden)


_ZUSTAND_LOCK = threading.Lock()
_GEDROSSELT_BIS: float | None = None  # _uhr()-Wert; None = kein Sperrfenster
_LETZTER_SLOT: float | None = None  # _uhr()-Wert des letzten Pacing-Slots


def zustand_zuruecksetzen() -> None:
    """Setzt Sperrfenster + Pacing-Slot zurück (Test-Hygiene, Muster websuche)."""
    global _GEDROSSELT_BIS, _LETZTER_SLOT
    with _ZUSTAND_LOCK:
        _GEDROSSELT_BIS = None
        _LETZTER_SLOT = None


def _pruefe_sperrfenster() -> None:
    """Offenes 429-Backoff-Fenster → :class:`ProviderError` 429 **ohne** Call."""
    with _ZUSTAND_LOCK:
        if _GEDROSSELT_BIS is None:
            return
        rest = _GEDROSSELT_BIS - _uhr()
        if rest <= 0.0:
            return  # abgelaufen → leeway wieder erlaubt (Fenster bleibt stehen)
    raise ProviderError(
        "leeway.tech drosselt die Anfragen (Rate-Limit). Das Bein ist für "
        f"{max(rest, 1.0):.0f} s ausgesetzt.",
        429,
        "leeway_gedrosselt",
        hint="Das Free-Tier erlaubt 50 Abfragen/Tag — Yahoo bleibt Primär, "
        "leeway greift nur im Fallback.",
    )


def _verzeichne_drossel() -> float:
    """Öffnet das Backoff-Fenster nach einem 429; liefert die Sperrdauer."""
    global _GEDROSSELT_BIS
    with _ZUSTAND_LOCK:
        _GEDROSSELT_BIS = _uhr() + _SPERRFENSTER_S
        return _SPERRFENSTER_S


def _wahre_pacing_abstand() -> float:
    """Prozessweiter Mindestabstand zwischen zwei leeway-Calls (Slot unter Lock).

    Slot-Reservierung unter Lock, geschlafen wird **außerhalb** — Muster
    ``websuche._wahre_suche_abstand`` (der Schlaf blockiert keine parallelen
    Reservierungen).
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


class LeewayProvider(DataProvider):
    """Zweite Tagesdaten-Quelle: leeway.tech ``historicalquotes`` (E088).

    ``(api_key)`` ist Pflicht-Positionsparameter — ein leerer/``None``-Key
    bedeutet „Bein nicht konfiguriert“: :meth:`fetch_ohlcv` wirft dann sofort
    den deutschen Fachfehler ``quelle2_nicht_konfiguriert`` **ohne Netz**.
    ``transport`` ist injizierbar (:class:`httpx.MockTransport` in Tests);
    produktiv läuft ein wiederverwendeter :class:`httpx.Client` (Timeout
    :data:`TIMEOUT_S`).
    """

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout_s: float = TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key.strip() if isinstance(api_key, str) else ""
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
        """Tages-Balken im kanonischen Schema (ABC-Vertrag) — EIN Netzcall.

        ``repair`` wird akzeptiert, aber **ignoriert** (Signatur-Vertrag des
        ABC, Muster Stooq-Bein): leeway kennt keinen clientseitigen Gap-Repair;
        Korrekturen laufen übers Yahoo-Bein (T19).

        Reihenfolge der Vorabprüfungen (alles **ohne Netz**): Symbol →
        Intervall (nur ``1d``) → Key. Dann ein Request mit dem vollen
        ``from/to``-Fenster (5y in einem Call — kein Chunking).
        """
        geprueft = validate_symbol(symbol)
        if validate_interval(interval) != "1d":
            raise ProviderError(
                "leeway.tech liefert ausschließlich Tagesdaten (Intervall „1d“).",
                422,
                "leeway_nur_tagesdaten",
                hint="Intraday-Daten sind nur über Yahoo verfügbar.",
            )
        if not self._api_key:
            raise ProviderError(
                "Die zweite Tagesdaten-Quelle (leeway.tech) ist nicht "
                "konfiguriert — es fehlt ein API-Key.",
                502,
                "quelle2_nicht_konfiguriert",
                hint="Optional: Key (kostenlose Registrierung, 50 Abfragen/Tag) "
                "in den Einstellungen unter „Daten“ hinterlegen.",
            )
        params = self._fenster_params(start, end)
        payload = self._historicalquotes(geprueft, params)
        return self._zu_kanonisch(payload, geprueft)

    def resolve_symbol(self, symbol: str) -> SymbolInfo:
        """Nur Form-Prüfung **ohne Netz** — keine Brenn-Probe (Free-Tier 50/Tag).

        Der Composite delegiert ``resolve_symbol`` ans yfinance-Bein (Name/TZ);
        dieser Pfad existiert nur für die direkte Nutzung des Beins. Metadaten
        jenseits der Heuristik liefert leeway im historquotes-Shape nicht.
        """
        geprueft = validate_symbol(symbol)
        return SymbolInfo(symbol=geprueft, asset_type=_asset_type(geprueft))

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

    def _fenster_params(self, start: int | None, end: int | None) -> dict[str, str]:
        """µs-Fenster → ``from``/``to`` (YYYY-MM-DD; ``to`` = Tag vor ``end``).

        Unser ``end`` ist exklusiv (ABC-Vertrag), leeways ``to`` inklusiv.
        Ohne ``start``: begrenztes Rückfenster (:data:`DEFAULT_FENSTER_TAGE`);
        ohne ``end``: heute (UTC).
        """
        ende_datum = (
            (
                datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=end)
            ).date()
            - timedelta(days=1)
            if end is not None
            else datetime.now(UTC).date()
        )
        if start is not None:
            start_datum = (
                datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=start)
            ).date()
        else:
            start_datum = ende_datum - timedelta(days=DEFAULT_FENSTER_TAGE)
        return {
            "from": start_datum.strftime("%Y-%m-%d"),
            "to": ende_datum.strftime("%Y-%m-%d"),
        }

    def _maskiere(self, text: str) -> str:
        """Entfernt den Key aus Ausnahmetexten (httpx embeddet ggf. die URL)."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "••••")
        return text

    def _historicalquotes(
        self, symbol: str, params: dict[str, str]
    ) -> Any:
        """Der **einzige** Netzcall dieses Beins — Guard, Pacing, Fehlerklassen.

        Guard/Sperrfenster des EIGENEN Limiter-Zustands zuerst (429 ohne
        zweiten Call), dann Pacing-Slot, dann der GET. Fehlerführung: 401 →
        ``leeway_key_ungueltig`` (401), 404 → ``symbol_nicht_gefunden``, 429 →
        Sperrfenster + ``leeway_gedrosselt``; Netz-/HTTP-/JSON-Pannen → 502
        ``leeway_fehler`` (Key vor Weitergabe maskiert).
        """
        _pruefe_sperrfenster()
        _wahre_pacing_abstand()
        try:
            antwort = self._client().get(
                f"{LEEWAY_URL}/{symbol}",
                params={**params, "apitoken": self._api_key},
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"leeway.tech-Abruf für „{symbol}“ fehlgeschlagen: "
                f"{self._maskiere(str(exc))}",
                502,
                "leeway_fehler",
            ) from exc
        if antwort.status_code == 401:
            raise ProviderError(
                "Der leeway.tech API-Key ist ungültig oder fehlt (HTTP 401).",
                401,
                "leeway_key_ungueltig",
                hint="Key in den Einstellungen unter „Daten“ prüfen — "
                "kostenlose Registrierung auf leeway.tech.",
            )
        if antwort.status_code == 404:
            raise SymbolNotFoundError(
                f"Symbol „{symbol}“ ist bei leeway.tech nicht bekannt "
                "(HTTP 404).",
                hint="Format „SYMBOL.EXCHANGE“ (z. B. „SAP.XETRA“, „AAPL“) "
                "oder ISIN verwenden.",
            )
        if antwort.status_code == 429:
            dauer = _verzeichne_drossel()
            raise ProviderError(
                "leeway.tech drosselt die Anfragen (Rate-Limit) — das Bein ist "
                f"für {dauer:.0f} s ausgesetzt.",
                429,
                "leeway_gedrosselt",
                hint="Das Free-Tier erlaubt 50 Abfragen/Tag — Yahoo bleibt "
                "Primär, leeway greift nur im Fallback.",
            )
        if antwort.status_code != 200:
            raise ProviderError(
                f"leeway.tech antwortete mit HTTP {antwort.status_code} "
                f"für „{symbol}“.",
                502,
                "leeway_fehler",
            )
        try:
            return antwort.json()
        except ValueError as exc:
            raise ProviderError(
                f"leeway.tech lieferte für „{symbol}“ kein gültiges JSON.",
                502,
                "leeway_fehler",
            ) from exc

    def _zu_kanonisch(self, payload: Any, symbol: str) -> pd.DataFrame:
        """leeway-JSON → kanonisches Schema (E006); defekte Zeilen fallen weg.

        Erwartet eine **Liste** von Tages-Objekten ``{date, o, h, l, c, v}``
        (G-0a-Vertrag). Nicht-Liste → 502 ``leeway_fehler`` (unerwartete
        Struktur, kein stiller Müll); leere Liste bzw. nur unlesbare Zeilen →
        404 ``keine_daten`` (deutsche Fehlerklasse statt still leeren Balken).
        ``volume`` kommt teils als String/Float → robust in float64 überführt
        (Kanonvertrag), fehlend → 0.0.
        """
        if not isinstance(payload, list):
            raise ProviderError(
                f"leeway.tech lieferte für „{symbol}“ eine unerwartete "
                "Antwortstruktur (Tagesliste erwartet).",
                502,
                "leeway_fehler",
            )
        zeilen: list[dict[str, float | int]] = []
        for eintrag in payload:
            zeile = self._zeile_lesen(eintrag)
            if zeile is not None:
                zeilen.append(zeile)
        if not zeilen:
            raise ProviderError(
                f"Für „{symbol}“ liefert leeway.tech keine Daten im "
                "gewünschten Zeitraum.",
                404,
                "keine_daten",
            )
        roh = pd.DataFrame(zeilen, columns=["ts", "open", "high", "low", "close", "volume"])
        # normalize_ohlcv floort Tages-ts auf Mitternacht UTC (E006) und
        # erzwingt Spalten/dtypes; leere Eingabe läuft leer durch. Keine
        # adj_factor-Spalte → Store/Lesen füllen neutral 1.0 (Adjust-Entscheid,
        # Modul-Docstring).
        return parquet_store.normalize_ohlcv(roh, "1d")

    @staticmethod
    def _zeile_lesen(eintrag: Any) -> dict[str, float | int] | None:
        """Eine Tages-Zeile ``{date,o,h,l,c,v}`` → Kanon-Dict (oder ``None``).

        ``date`` ohne Intraday-Zeit → UTC-Mitternacht (ANNAHME, Modul-Docstring);
        ISO-Formate mit Zeitanteil werden toleriert (Datumsteil gewinnt).
        """
        if not isinstance(eintrag, dict):
            return None
        roh_datum = eintrag.get("date")
        if not isinstance(roh_datum, str) or not roh_datum.strip():
            return None
        try:
            moment = datetime.fromisoformat(roh_datum.strip())
        except ValueError:
            return None
        # ANNAHME: reines Datum (= leeway-Vertrag) gilt als UTC-Mitternacht;
        # ein ISO-Zeitanteil mit Zone wird nach UTC verschoben, naive Zeitanteile
        # werden als UTC gelesen.
        moment = (
            moment.astimezone(UTC) if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
        )
        try:
            werte = {
                spalte: float(eintrag[feld])
                for spalte, feld in (
                    ("open", "o"),
                    ("high", "h"),
                    ("low", "l"),
                    ("close", "c"),
                )
            }
        except (KeyError, TypeError, ValueError):
            return None
        for wert in werte.values():
            if wert != wert:  # NaN — Zeile ohne vollständiges OHLC verwerfen
                return None
        volumen = eintrag.get("v", 0)
        try:
            volumen_f = float(volumen) if volumen is not None else 0.0
        except (TypeError, ValueError):
            volumen_f = 0.0
        return {
            "ts": int((moment - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds())
            * 1_000_000,
            **werte,
            "volume": volumen_f,
        }


def _asset_type(symbol: str) -> str:
    """Gleiche Grob-Heuristik wie die anderen Beine (Konsistenz der Registry)."""
    if symbol.startswith("^"):
        return "index"
    return "equity"


__all__ = [
    "DEFAULT_FENSTER_TAGE",
    "LEEWAY_URL",
    "LeewayProvider",
    "TIMEOUT_S",
    "zustand_zuruecksetzen",
]
