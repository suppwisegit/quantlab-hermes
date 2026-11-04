"""Stooq-Provider (T19, E023): reiner Daily-Fallback über CSV + httpx.

Stooq ist ein **inoffizieller**, schlüsselloser CSV-Endpunkt (RECHERCHE v2
Block 2): ``https://stooq.com/q/d/l/?s={symbol}&d1=YYYYMMDD&d2=YYYYMMDD&i=d``
mit den Spalten ``Date,Open,High,Low,Close,Volume`` (Datum ``YYYY-MM-DD``).
Vertrag dieser Implementierung:

- **Nur Tagesdaten:** Intervall ``1d``; alles andere → :class:`ProviderError`
  422 („Stooq liefert ausschließlich Tagesdaten“). Intraday-Fallback ist
  bewusst ausgeschlossen (E023).
- **Immer ``d1/d2``-Fenster** („nie komplette Historie ohne Not“): ``start``/
``end`` sind int64 µs UTC wie im ABC; ``end`` ist exklusiv → ``d2`` = Tag vor
``end``. Ohne ``start`` gilt ein begrenztes Rückfenster
  (:data:`DEFAULT_FENSTER_TAGE`), ohne ``end`` heute.
- **Close ist bereits adjustiert** (QuantStart, R1 Block 2.4) — Stooq-Bestand
  wird über ``bar_meta.source='stooq'`` gekennzeichnet; T20/E022 behandelt die
  fehlende Roh-/Adjustiert-Dualität beim Lesen.
- **Symbol-Mapping** (:func:`stooq_symbol`): US-Aktien ``<sym>.us`` (klein),
  Indizes ``^``-Präfix klein; ``^GSPC → ^spx`` ist **ANNAHME** (R1 Block 2.3)
  und Live-Test-Punkt. Nicht mappbare Kürzel (FX ``=X``, Futures ``=F``,
  Krypto ``-USD``, Punktsymbole) → 422 deutsch — der Fallback schweigt dann
  kontrolliert statt Müll zu liefern.
- **Fehler-Texte:** „Przekroczony dzienny limit wywolan“ (tägliches Limit
  überschritten) → 502 ``stooq_limit``; ``{"Message":"No data"}`` → 404
  ``keine_daten``. Netz-/HTTP-Fehler → 502 ``stooq_fehler``.
- **Browser-Verifizierung (Live-Befund 2026-09-14):** stooq.com stellt aktuell
  **alle** Antworten hinter eine JavaScript-Challenge („This site requires
  JavaScript to verify your browser“, PoW via ``/__verify``). Der Provider
  erkennt diese Seite und meldet 502 ``stooq_blockiert`` — er löst die
  Challenge bewusst **nicht** (keine Umgehung einer Zugangsbeschränkung);
  Yahoo bleibt dann die einzige Quelle, bis Stooq wieder offen ist.

Abruf via **httpx** (Haupt-Dependency seit T19, E023) mit injizierbarem
Transport — Tests fahren :class:`httpx.MockTransport`, kein echtes Netz (E013).
"""

from __future__ import annotations

import csv
import io
import re
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from quantlab.providers.base import DataProvider, ProviderError, SymbolInfo, validate_interval
from quantlab.providers.symbols import validate_symbol
from quantlab.storage import parquet_store

#: CSV-Endpunkt (R1 Block 2.1) — Parameter: s, d1, d2, i=d.
STOOQ_URL = "https://stooq.com/q/d/l/"

#: Explicit bekanntes Yahoo→Stooq-Mapping. ANNAHME: ``^GSPC → ^spx``
#: (R1 Block 2.3, Suffix-Logik belegt, konkretes Mapping Live-Test-Punkt).
STOOQ_SYMBOL_MAPPING: dict[str, str] = {
    "^GSPC": "^spx",
}

#: US-Aktien-Kürzel: nur Buchstaben ( Punktsymbole wie ``BRK.B`` sind auf
#: Stooq anders geschrieben (``brk-b.us``) und bleiben bewusst unmapped —
#: konservativ, bis ein Live-Punkt das Format verifiziert).
_US_AKTIE = re.compile(r"^[A-Za-z]+$")

#: Rückfenster in Tagen, wenn der Aufrufer kein ``start`` liefert (begrenzt,
#: E023 „nie komplette Historie ohne Not“).
DEFAULT_FENSTER_TAGE = 3650

#: Request-Timeout für den CSV-Abruf (Sekunden).
TIMEOUT_S = 15.0

#: Stooq-Meldung bei überschrittenem Tageslimit (R1 Block 2.5).
_LIMIT_MARKER = "Przekroczony"

#: Stooq-Meldung bei unbekanntem Symbol (R1 Block 2.5).
_NO_DATA_MARKER = "No data"

#: Marker der JavaScript-Browser-Verifizierung (Live-Befund 2026-09-14) —
#: Stooq schaltet diese Seite aktuell vor ALLE Antworten. Bewusst kein
#: Automatisieren der Challenge (keine Umgehung einer Zugangsbeschränkung).
_BLOCK_MARKER = "verify your browser"

#: UA wie ein Browser (konsistent zu yfinance, das Browser-Impersonation als
#: Kernmethode nutzt) — mit dem httpx-Default-UA antwortet Stooq nur 404;
#: browserähnlich erreichen wir wenigstens die echte Antwort/Challenge.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 QuantLab/0.1"
)

# ---------------------------------------------------------------------------
# Prozessweiter Stooq-Zustand (für GET /api/data/provider-status, E035)
# ---------------------------------------------------------------------------

_ZUSTAND_LOCK = threading.Lock()
_ZUSTAND: dict[str, Any] = {"state": "ok"}


def status() -> dict[str, Any]:
    """Stooq-Zustand für ``GET /api/data/provider-status`` (§C.4).

    ``{"state": "ok"}`` bzw. ``{"state": "limit", "seit": <ISO>}`` nach einem
    erkannten Tageslimit (daily — ein Reset passiert erst mit dem Prozess).
    """
    with _ZUSTAND_LOCK:
        return dict(_ZUSTAND)


def _zustand_limit() -> None:
    with _ZUSTAND_LOCK:
        _ZUSTAND["state"] = "limit"
        _ZUSTAND["seit"] = datetime.now(UTC).isoformat()


def _zustand_ok() -> None:
    with _ZUSTAND_LOCK:
        if _ZUSTAND.get("state") != "ok":
            _ZUSTAND.clear()
            _ZUSTAND["state"] = "ok"


# ---------------------------------------------------------------------------
# Symbol-Mapping
# ---------------------------------------------------------------------------


def stooq_symbol(symbol: str) -> str:
    """Yahoo-Kürzel → Stooq-Kürzel; Unmappbares → :class:`ProviderError` (422).

    Regeln (E023): explizite Tabelle (``^GSPC → ^spx``, ANNAHME) → Index
    ``^ABC`` → ``^abc`` → Plain-Kürzel ``ABC`` → ``abc.us``. FX/Futures/Krypto/
    Punktsymbole sind auf Stooq nicht verlässlich adressierbar → 422 mit
    deutscher Meldung (der Composite fängt das als „Stooq nicht möglich“).
    """
    kandidat = validate_symbol(symbol)
    if kandidat in STOOQ_SYMBOL_MAPPING:
        return STOOQ_SYMBOL_MAPPING[kandidat]
    if kandidat.startswith("^"):
        return kandidat.lower()
    if _US_AKTIE.match(kandidat):
        return f"{kandidat.lower()}.us"
    raise ProviderError(
        f"Für „{kandidat}“ gibt es kein Stooq-Symbol (nur US-Aktien und Indizes).",
        422,
        "kein_stooq_symbol",
        hint="Stooq deckt diesen Symboltyp nicht ab — Yahoo bleibt die einzige Quelle.",
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class StooqProvider(DataProvider):
    """Tagesdaten-Fallback über den Stooq-CSV-Endpunkt (E023).

    ``transport`` ist injizierbar (:class:`httpx.MockTransport` in Tests);
    produktiv baut jeder Fetch einen kurzen :class:`httpx.Client`
    (Timeout :data:`TIMEOUT_S`).
    """

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = TIMEOUT_S,
    ) -> None:
        self._transport = transport
        self._timeout_s = timeout_s

    # -- DataProvider-ABC ------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        repair: bool = False,
    ) -> pd.DataFrame:
        """CSV → kanonisches OHLCV-Schema (E006: Tages-Bars Mitternacht UTC).

        ``repair`` wird akzeptiert, aber **ignoriert** (M1: Signatur-Vertrag
        des ABC): Stooq kennt keinen Gap-Repair — der Daily-Fallback liefert
        das Fenster unverändert; Repair (T19) läuft über Yahoo.
        """
        geprueft = validate_symbol(symbol)
        if validate_interval(interval) != "1d":
            raise ProviderError(
                "Stooq liefert ausschließlich Tagesdaten (Intervall „1d“).",
                422,
                "stooq_nur_tagesdaten",
                hint="Intraday-Daten sind nur über Yahoo verfügbar.",
            )
        stooq_kuerzel = stooq_symbol(geprueft)
        params = self._fenster_params(start, end)
        text = self._csv_abrufen(stooq_kuerzel, params)
        return self._zu_kanonisch(text, geprueft)

    def resolve_symbol(self, symbol: str) -> SymbolInfo:
        """Leichte Probe (7 Tage Fenster): gibt es Stooq-Daten zum Kürzel?

        Liefert nur Minimal-Metadaten — Namens-/TZ-Anreicherung macht Yahoo
        (der Composite delegiert ``resolve_symbol`` dorthin).
        """
        geprueft = validate_symbol(symbol)
        heute = datetime.now(UTC)
        ende_us = int((heute - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1_000_000)
        start_us = ende_us - 7 * parquet_store.US_PER_DAY
        df = self.fetch_ohlcv(geprueft, "1d", start=start_us, end=ende_us)
        if df.empty:
            raise ProviderError(
                f"Symbol „{geprueft}“ ist bei Stooq nicht bekannt oder liefert keine Daten.",
                404,
                "keine_daten",
            )
        return SymbolInfo(symbol=geprueft, asset_type=_asset_type(geprueft))

    # -- Innere Mechanik -------------------------------------------------

    def _fenster_params(
        self, start: int | None, end: int | None
    ) -> dict[str, str]:
        """µs-Fenster → ``d1``/``d2`` (YYYYMMDD, beide inklusiv bei Stooq).

        Unser ``end`` ist exklusiv (ABC-Vertrag) → ``d2`` = Tag vor ``end``.
        Ohne ``start``: begrenztes Rückfenster (E023); ohne ``end``: heute.
        """
        ende_datum = (
            (
                datetime(1970, 1, 1, tzinfo=UTC)
                + timedelta(microseconds=end)
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
            "d1": start_datum.strftime("%Y%m%d"),
            "d2": ende_datum.strftime("%Y%m%d"),
        }

    def _csv_abrufen(self, stooq_kuerzel: str, params: dict[str, str]) -> str:
        """GET auf den CSV-Endpunkt; Stooq-Fehlertexte → ProviderError."""
        anfrage_params = {"s": stooq_kuerzel, **params, "i": "d"}
        try:
            with httpx.Client(
                transport=self._transport,
                timeout=self._timeout_s,
                headers={"User-Agent": _BROWSER_UA},
            ) as client:
                antwort = client.get(STOOQ_URL, params=anfrage_params)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Stooq-Abruf für „{stooq_kuerzel}“ fehlgeschlagen: {exc}",
                502,
                "stooq_fehler",
            ) from exc
        if antwort.status_code != 200:
            raise ProviderError(
                f"Stooq antwortete mit HTTP {antwort.status_code}.",
                502,
                "stooq_fehler",
            )
        text = antwort.text
        if _BLOCK_MARKER in text:
            _zustand_limit()  # Fallback formal unbenutzbar, solange geblockt
            raise ProviderError(
                "Stooq verlangt derzeit eine Browser-Verifizierung (JavaScript) — "
                "der Tagesdaten-Fallback ist vorerst nicht verfügbar.",
                502,
                "stooq_blockiert",
                hint="Yahoo bleibt die Datenquelle; der Zustand ist unter "
                "/api/data/provider-status (stooq) sichtbar.",
            )
        if _LIMIT_MARKER in text:
            _zustand_limit()
            raise ProviderError(
                "Stooq hat das tägliche Abruflimit erreicht — bitte später erneut versuchen.",
                502,
                "stooq_limit",
            )
        if _NO_DATA_MARKER in text:
            raise ProviderError(
                f"Für „{stooq_kuerzel}“ liefert Stooq keine Daten.",
                404,
                "keine_daten",
            )
        _zustand_ok()
        return text

    def _zu_kanonisch(self, text: str, symbol: str) -> pd.DataFrame:
        """CSV-Text → kanonisches Schema; defekte Zeilen fallen weg.

        Nur Header (keine Datenzeilen) → leeres DataFrame im Kanonformat —
        der Aufrufer (Composite/Job) entscheidet, ob das „keine Daten“ ist.
        """
        try:
            zeilen = list(csv.DictReader(io.StringIO(text)))
        except csv.Error as exc:
            raise ProviderError(
                f"Stooq-Antwort für „{symbol}“ ist kein lesbares CSV.",
                502,
                "stooq_fehler",
            ) from exc
        daten: list[dict[str, float | int]] = []
        for zeile in zeilen:
            roh_datum = (zeile.get("Date") or "").strip()
            if not roh_datum:
                continue
            try:
                moment = datetime.strptime(roh_datum, "%Y-%m-%d").replace(tzinfo=UTC)
                werte = {
                    spalte: float(zeile[spalte])
                    for spalte in ("Open", "High", "Low", "Close")
                }
            except (ValueError, TypeError, KeyError):
                continue  # Einzeldefekt verwirft nur diese Zeile
            volumen = (zeile.get("Volume") or "").strip()
            daten.append(
                {
                    "ts": int(
                        (moment - datetime(1970, 1, 1, tzinfo=UTC))
                        // timedelta(microseconds=1)
                    ),
                    "open": werte["Open"],
                    "high": werte["High"],
                    "low": werte["Low"],
                    "close": werte["Close"],
                    "volume": float(volumen) if volumen else 0.0,
                }
            )
        roh = pd.DataFrame(
            daten,
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        # normalize_ohlcv floors Tages-ts auf Mitternacht UTC (E006) und
        # erzwingt Spalten/dtypes; leere Eingabe läuft leer durch.
        return parquet_store.normalize_ohlcv(roh, "1d")


def _asset_type(symbol: str) -> str:
    """Gleiche Grob-Heuristik wie der Yahoo-Provider (Konsistenz der Registry)."""
    if symbol.startswith("^"):
        return "index"
    return "equity"


__all__ = [
    "DEFAULT_FENSTER_TAGE",
    "STOOQ_SYMBOL_MAPPING",
    "STOOQ_URL",
    "StooqProvider",
    "status",
    "stooq_symbol",
]
