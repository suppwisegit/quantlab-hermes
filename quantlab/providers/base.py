"""Provider-Abstraktion für Marktdaten (T3) — Schnittstelle für yfinance (T4) und Test-Fakes.

Ein :class:`DataProvider` kapselt eine externe Quelle (Yahoo, …) bzw. einen
Ersatz für Tests. Der Vertrag ist bewusst eng, damit beide Implementierungen
(a) austauschbar sind und (b) direkt in den Parquet-Store (§D.2) einspeisen:

- **Zeit** ist kanonisch int64 Mikrosekunden UTC (E006); ``start`` ist
  inklusive, ``end`` exklusiv (yfinance-Konvention), ``None`` = unbegrenzt.
- **Schema** von ``fetch_ohlcv``: Spalten ``ts`` (int64 µs UTC) sowie
  ``open/high/low/close/volume`` (float64), aufsteigend sortiert, ohne
  Duplikate auf ``ts``; Tages-Intervalle liegen auf Mitternacht UTC.
  Nichts zu holen → leeres DataFrame mit genau diesen Spalten.
- **Fehler** immer als :class:`ProviderError` — mit HTTP-mappbaren Feldern
  (Vertrag C.1), damit Router sie über ``to_api_error()`` 1:1 übernehmen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from quantlab.errors import ApiError

#: Gültige Intervalle in yfinance-Schreibweise (RECHERCHE Block 2, Fakt 1) —
#: logisch sortiert für Fehlerhinweise und UI-Auswahllisten.
INTERVAL_ORDER: tuple[str, ...] = (
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo",
)

#: Whitelist-Set aus :data:`INTERVAL_ORDER` (O(1)-Mitgliedschaft, Pfad-sicher).
VALID_INTERVALS: frozenset[str] = frozenset(INTERVAL_ORDER)

#: Tages- und gröber aufgelöste Intervalle. E006: ihre Balken liegen auf
#: Mitternacht UTC. (Wird auch vom Parquet-Store und von T4 für ``ignore_tz``
#: verwendet — deshalb hier in der providers-Schicht verortet.)
DAILY_INTERVALS: frozenset[str] = frozenset({"1d", "5d", "1wk", "1mo", "3mo"})


class ProviderError(Exception):
    """Fachlicher Fehler der Daten-Domäne, direkt in den ApiError-Vertrag mappbar.

    ``message`` ist deutsch (UI-Sprache, E001), ``code`` stabil und
    maschinenlesbar. ``status_code`` ist der HTTP-Status, unter dem ein Router
    den Fehler ausgeben soll (Default 502: Quelle nicht erreichbar/fehlerhaft;
    404 für fehlende Daten via :class:`SymbolNotFoundError`; 422 für
    ungültige Eingaben wie Symbol/Intervall).
    """

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        code: str = "provider_fehler",
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.hint = hint

    def to_api_error(self) -> ApiError:
        """Überführt den Fehler in den ApiError-Vertrag (C.1) — Felder unverändert."""
        return ApiError(self.status_code, self.code, self.message, hint=self.hint)


class SymbolNotFoundError(ProviderError):
    """Gesuchtes Symbol/Intervall hat keine gespeicherten Daten — HTTP 404."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message, 404, "symbol_nicht_gefunden", hint)


@dataclass(frozen=True)
class SymbolInfo:
    """Provider-Metadaten zu einem Symbol (Spalten der Tabelle ``symbols``, D.1)."""

    symbol: str
    name: str | None = None
    asset_type: str | None = None
    exchange_tz: str | None = None
    currency: str | None = None


class DataProvider(ABC):
    """Abstrakte OHLCV-Datenquelle (Implementierungen: yfinance T4, FakeProvider Tests)."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        repair: bool = False,
    ) -> pd.DataFrame:
        """Lädt OHLCV-Balken im kanonischen Schema (Klassendocstring, Abschnitt Schema).

        ``start``/``end`` sind int64 Mikrosekunden UTC (inklusiv/exklusiv).
        Fehler — unbekanntes Symbol, Quelle nicht erreichbar — als
        :class:`ProviderError`.

        ``repair=True`` (T19 Gap-Repair): die Quelle soll Bestandsfehler im
        Fenster korrigieren (yfinance ``repair=True`` — erkennt/fixt z. B.
        100x-Währungsfehler). Implementierungen, die kein Repair kennen,
        ignorieren das Flag; Aufrufer dürfen es nur setzen, wenn der Provider
        es unterstützt (Duck-Typing am konkreten Provider).
        """

    @abstractmethod
    def resolve_symbol(self, symbol: str) -> SymbolInfo:
        """Validiert das Symbol gegen die Quelle und liefert die Metadaten.

        Ungültige/unbekannte Symbole als :class:`ProviderError` (404/422).
        """


def validate_interval(interval: str) -> str:
    """Prüft das Intervall gegen die yfinance-Menge; liefert es unverändert zurück.

    Die Prüfung ist zugleich Pfad-Schutz für das Parquet-Layout
    ``market/<interval>/<SYMBOL>.parquet`` (D.2): Intervalle wie ``../x``
    werden verworfen.
    """
    if not isinstance(interval, str) or interval not in VALID_INTERVALS:
        raise ProviderError(
            f"Unbekanntes Intervall „{interval}“.",
            422,
            "ungueltiges_intervall",
            hint=f"Gültige Intervalle: {', '.join(INTERVAL_ORDER)}.",
        )
    return interval
