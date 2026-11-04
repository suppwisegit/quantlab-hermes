"""Daten-Service: OHLC-Read-Pfad über dem Parquet-Store (T3) + E022-Baustein (T20).

Der Read-Pfad ist bewusst schmal: Validierung von Symbol/Intervall plus klare
404-Semantik für fehlende Daten, danach Weiterreichung an
:mod:`quantlab.storage.parquet_store`. Router (T4/T5) übersetzen einen
:class:`~quantlab.providers.base.ProviderError` über ``to_api_error()`` in den
ApiError-Vertrag (C.1) — fehlende Symbole erscheinen damit als 404 mit
deutscher Meldung, ungültige Eingaben als 422.

Seit T20 (E022 — Dual-Preis-Modell, Adjustierung on-read):

- :func:`read_ohlcv` kennt ``adjust='raw'|'adjusted'`` (Default ``raw``):
  ``adjusted`` multipliziert OHLC **beim Lesen** mit der je Bar persistierten
  Spalte ``adj_factor`` (Faktor ``AdjClose/Close``) — Volume bleibt unverändert.
  Keine zweite Preisspur im Speicher. Bestand mit ``bar_meta.source='stooq'``
  wird **nicht** multipliziert (Stooq-Close ist bereits adjustiert, E023) —
  :func:`adjustierungs_hinweis` liefert dazu den deutschen Hinweis für die
  API-Response statt einer falschen Dualität.
- :func:`speichere_corporate_actions` / :func:`lese_actions`: Dividenden und
  Splits aus dem yfinance-Download (``actions=True``) landen idempotent
  (Upsert je symbol+datum+typ) in der Tabelle ``corporate_actions`` — Events
  getrennt vom Faktor (Portfolio-Dividenden, Chart-Marker, Übersichten).
  ANNAHME (T20): T18 hat die Tabelle angelegt, aber kein Repo-Modul für sie
  bereitgestellt — die SQL-Muster folgen den Repo-Konventionen
  (``BEGIN IMMEDIATE``-Transaktion, eine Verbindung pro Operation) und können
  später auf ein T18-Repo umgestellt werden.
- :func:`vermerke_delisted_wenn_bekannt`: komplett leere Yahoo-History bei
  einem bereits bekannten Symbol (Registry-Eintrag) setzt ``symbols.delisted=1``
  (Spalte aus Migration 3) statt nur einen pseudo-unbekannt-404 zu werfen —
  Bestand und Abdeckung bleiben damit sichtbar und kennzeichnungsfähig (T22
  get_quote/T28 UI können ``ist_delisted`` auswerten).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from quantlab import db
from quantlab.providers.base import (
    ProviderError,
    SymbolNotFoundError,
    validate_interval,
)
from quantlab.providers.symbols import validate_symbol
from quantlab.storage import parquet_store
from quantlab.storage.repos import symbols_repo

#: Gültige Werte des ``adjust``-Query-Parameters (E022; Default ``raw``).
ADJUST_WERTE: tuple[str, ...] = ("raw", "adjusted")

#: Deutscher Hinweis bei adjust=adjusted über Stooq-Bestand (E022/E023).
STOOQ_ADJUST_HINWEIS = (
    "Der Bestand dieser Reihe stammt (zuletzt) von Stooq — dort ist „close“ "
    "bereits adjustiert, roh und adjustiert können daher nicht sauber "
    "unterschieden werden."
)

#: Deutscher Hinweis bei Legacy-Blendung (M3, T20-Fix): Der Bestand enthält
#: v1-adjustierte Kurse als raw — alte Balken (v1-Datei ohne ``adj_factor``)
#: wurden mit neuen Rohkursen gemischt; erst ein Neudownload mit explizitem
#: ``start`` über das gesamte Bestandsfenster baut die Faktorkette komplett.
LEGACY_BLENDUNG_HINWEIS = (
    "Der Bestand dieser Reihe enthält v1-adjustierte Kurse als raw — alte "
    "Balken ohne adj_factor wurden mit neuen gemischt. Einmalig mit "
    "explizitem „start“ neu laden, dann stimmen Roh- und Adjustierungsbild."
)

#: Event-Typen der Tabelle ``corporate_actions`` (CHECK-Constraint, Migration 2).
ACTION_TYPEN: tuple[str, ...] = ("dividend", "split")

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Zeit-Helfer (E006: kanonische int64 µs UTC)
# ---------------------------------------------------------------------------


def _us_zu_datum_iso(ts_us: int) -> str:
    """int64 µs UTC → ISO-Datum (``YYYY-MM-DD``, Tagesauflösung der Tabelle)."""
    moment = _EPOCH_UTC + timedelta(microseconds=ts_us)
    return moment.date().isoformat()


def _datum_iso_zu_us(text: str) -> int:
    """ISO-Datum/-Zeit der ``corporate_actions.ts``-Spalte → Mitternacht-µs UTC."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - _EPOCH_UTC) // timedelta(microseconds=1)


# ---------------------------------------------------------------------------
# OHLC-Read mit adjust (E022)
# ---------------------------------------------------------------------------


def bar_quelle(symbol: str, interval: str) -> str:
    """Liest die Herkunfts-Marke ``bar_meta.source`` ('yahoo' | 'stooq').

    Lesender Zwilling zu ``downloads.bar_meta_source`` — die **Schreibstelle**
    bleibt allein beim Download-Service (T19-Übergabe); dieser Pfad hier
    vermeidet einen Import-Zyklus (downloads → yfinance_provider → data_service).
    ``'yahoo'`` gilt auch für fehlende Zeile/Spalte (v1-Bestand).
    """
    meta = parquet_store.get_bar_meta(symbol, interval)
    return "yahoo" if meta is None else meta.source


def ist_legacy_blendung(symbol: str, interval: str) -> bool:
    """Trägt der Bestand die v1-Blendungs-Marke (M3, T20-Fix)?

    ``bar_meta.legacy_blendung`` wird von :func:`parquet_store.upsert_bars`
    gesetzt, sobald erstmals in eine v1-Datei (ohne ``adj_factor``-Spalte im
    Bestand vor dem Merge) geschrieben wird — der Bestand mischt dann
    v1-adjustierte Kurse als raw mit neuen Rohkursen.
    """
    meta = parquet_store.get_bar_meta(symbol, interval)
    return meta is not None and meta.legacy_blendung


def adjustierungs_hinweis(symbol: str, interval: str, adjust: str) -> str | None:
    """Deutscher Hinweis, wenn die Dualität raw/adjusted nicht sauber ist.

    ``None`` in allen anderen Fällen (kein Feld in der API-Response).
    M3 (T20-Fix): Bei Legacy-Blendung stimmt schon „raw“ nicht (v1-adjustierte
    Kurse liegen als Rohwerte) — der Hinweis erscheint daher unabhängig vom
    ``adjust``-Wert. Daneben unverändert: ``adjust=adjusted`` über Stooq-
    Bestand (Close dort bereits adjustiert, E023).
    """
    if ist_legacy_blendung(symbol, interval):
        return LEGACY_BLENDUNG_HINWEIS
    if adjust != "adjusted":
        return None
    return STOOQ_ADJUST_HINWEIS if bar_quelle(symbol, interval) == "stooq" else None


def read_ohlcv(
    symbol: str,
    interval: str,
    start: int | None = None,
    end: int | None = None,
    limit: int | None = None,
    adjust: str = "raw",
) -> pd.DataFrame:
    """Liest gespeicherte OHLCV-Balken, aufsteigend nach ``ts`` (int64 µs UTC).

    ``start`` inklusive, ``end`` exklusiv; ``limit`` kürzt auf die jüngsten
    Balken (Details: :func:`parquet_store.read_bars`). Ein Symbol ohne
    gespeicherte Daten löst :class:`SymbolNotFoundError` (HTTP 404) aus;
    ungültiges Symbol/Intervall bzw. sinnwidrige Filter (:class:`ProviderError`,
    HTTP 422). Ein vorhandenes Symbol mit leerem Zeitraum-Fenster liefert
    demgegenüber ein leeres DataFrame — das ist ein gültiges Ergebnis, kein
    Fehler.

    ``adjust`` (E022): ``'raw'`` (Default) liefert die unveränderten Roh-Kurse;
    ``'adjusted'`` multipliziert open/high/low/close mit der je Bar
    gespeicherten Spalte ``adj_factor`` (Volume unverändert). Bestand der
    Quelle Stooq wird nicht multipliziert — dort ist ``close`` bereits
    adjustiert (Hinweis via :func:`adjustierungs_hinweis`, E023). Andere Werte
    → :class:`ProviderError` 422 ``ungueltiger_adjust``.
    """
    if adjust not in ADJUST_WERTE:
        raise ProviderError(
            f"Unbekannter „adjust“-Wert „{adjust}“.",
            422,
            "ungueltiger_adjust",
            hint="Gültige Werte: raw (Rohkurse, Standard), adjusted (OHLC × Faktor).",
        )
    checked = validate_symbol(symbol)
    validate_interval(interval)
    if not parquet_store.has_data(checked, interval):
        raise _symbol_nicht_gefunden(checked, interval)
    balken = parquet_store.read_bars(checked, interval, start=start, end=end, limit=limit)
    if adjust == "adjusted" and not balken.empty and bar_quelle(checked, interval) != "stooq":
        for spalte in ("open", "high", "low", "close"):
            balken[spalte] = balken[spalte] * balken["adj_factor"]
    return balken


def _symbol_nicht_gefunden(symbol: str, interval: str) -> ProviderError:
    """Einheitliche 404 für fehlende Bestände (Meldung wie T3-Bestand)."""
    return SymbolNotFoundError(
        f"Für das Symbol „{symbol}“ sind keine Daten im Intervall "
        f"„{interval}“ gespeichert.",
        hint="Lade zuerst Marktdaten herunter (Dashboard → Download).",
    )


# ---------------------------------------------------------------------------
# Corporate Actions (Tabelle ``corporate_actions``, T18-Schema — E022)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorporateAction:
    """Ein Dividenden-/Split-Event (Zeile aus ``corporate_actions``).

    ``ts_us`` ist der Ex-Tag als Mitternacht-UTC-µs (API-Vertrag, E006);
    ``datum`` das ISO-Datum der Speicherung (Tagesauflösung).
    """

    symbol: str
    datum: str
    action: str  # 'dividend' | 'split'
    value: float
    currency: str | None
    source: str
    ts_us: int = 0

    def __post_init__(self) -> None:
        if not self.ts_us:
            object.__setattr__(self, "ts_us", _datum_iso_zu_us(self.datum))


def speichere_corporate_actions(symbol: str, ereignisse: list[dict]) -> int:
    """Idempotenter Upsert von Dividenden-/Split-Events je (symbol, datum, typ).

    ``ereignisse`` kommen aus dem yfinance-Download (``actions=True``): je
    Event ``{"ts": <int64 µs UTC>, "action": "dividend"|"split", "value":
    float, "currency": str | None}``. Der Upsert überschreibt ``value``/
    ``currency`` (Yahoo korrigiert Dividendenwerte gelegentlich) — Events, die
    Yahoo später nicht mehr meldet, bleiben bewusst erhalten (historisch
    stabil). Rückgabe: Anzahl geschriebener Zeilen.

    ANNAHME (T20): kein T18-Repo vorhanden — SQL direkt nach Repo-Muster.
    """
    checked = validate_symbol(symbol)
    zeilen: list[tuple[str, str, str, float, str | None]] = []
    for ereignis in ereignisse:
        action = str(ereignis.get("action", ""))
        if action not in ACTION_TYPEN:
            continue  # unbekannte Typen still verwerfen (Schema-CHECK sonst 500)
        wert = float(ereignis.get("value", 0.0))
        waehrung = ereignis.get("currency")
        zeilen.append(
            (
                _us_zu_datum_iso(int(ereignis["ts"])),
                action,
                wert,
                str(waehrung).strip() if waehrung else None,
            )
        )
    if not zeilen:
        return 0

    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT INTO corporate_actions(symbol, ts, action, value, currency, source) "
            "VALUES (?, ?, ?, ?, ?, 'yahoo') "
            "ON CONFLICT(symbol, ts, action) DO UPDATE SET "
            "value = excluded.value, currency = excluded.currency, "
            "source = excluded.source",
            [(checked, *zeile) for zeile in zeilen],
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return len(zeilen)


def lese_actions(
    symbol: str,
    start: int | None = None,
    end: int | None = None,
) -> list[CorporateAction]:
    """Gespeicherte Events aufsteigend nach Datum; ``start`` inklusive, ``end``
    exklusiv (beides Mitternacht-Vergleich auf Tagesebene, E006-Konvention).

    ÜBERGABE T23/T24 (MINOR 5, bewusst NICHT Teil dieser Runde): Für Balken-
    Fenster, die vor T20 (``actions=True``-Download) gespeichert wurden, kann
    ``corporate_actions`` leer sein — Events werden nur für künftig geladene
    Fenster geschrieben; ein Backfill der Altfenster steht aus.
    """
    checked = validate_symbol(symbol)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM corporate_actions WHERE symbol = ? ORDER BY ts, action",
            (checked,),
        ).fetchall()
    finally:
        conn.close()
    aktionen: list[CorporateAction] = []
    for row in rows:
        ts_us = _datum_iso_zu_us(row["ts"])
        if start is not None and ts_us < start:
            continue
        if end is not None and ts_us >= end:
            continue
        aktionen.append(
            CorporateAction(
                symbol=checked,
                datum=row["ts"],
                action=row["action"],
                value=float(row["value"]),
                currency=row["currency"],
                source=row["source"],
                ts_us=ts_us,
            )
        )
    return aktionen


def hat_actions(symbol: str) -> bool:
    """Existiert mindestens eine ``corporate_actions``-Zeile zum Symbol?"""
    checked = validate_symbol(symbol)
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM corporate_actions WHERE symbol = ? LIMIT 1", (checked,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def actions_symbol_bekannt(symbol: str) -> bool:
    """Ist das Symbol für den Actions-Endpunkt „bekannt``?

    Bekannt = Registry-Eintrag, Bestand in ``bar_meta`` oder bereits
    gespeicherte Events. Unbekannte Symbole beantwortet der Router mit 404.
    """
    checked = validate_symbol(symbol)
    return (
        symbols_repo.get_symbol(checked) is not None
        or bool(parquet_store.list_bar_meta(checked))
        or hat_actions(checked)
    )


# ---------------------------------------------------------------------------
# Delisted-Handling (T20: leere History bei bekanntem Symbol)
# ---------------------------------------------------------------------------


def vermerke_delisted_wenn_bekannt(symbol: str) -> bool:
    """Setzt ``symbols.delisted=1``, wenn die Registry das Symbol kennt.

    Best-effort: DB-Pannen (fehlendes Schema, gesperrte Datei) werden
    stillscheigend zu ``False`` — der Delisted-Vermerk darf einen Fetch nie
    scheitern lassen (Muster: ``downloads._registry_metadata_pflegen``).
    Rückgabe ``True`` genau dann, wenn ein Registry-Eintrag vermerkt wurde.
    """
    try:
        checked = validate_symbol(symbol)
    except ProviderError:
        return False
    try:
        conn = db.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE symbols SET delisted = 1 WHERE symbol = ?", (checked,)
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return cursor.rowcount > 0
    except sqlite3.Error:
        return False


def hebe_delisted_vermerk_auf(symbol: str) -> bool:
    """Setzt ``symbols.delisted`` wieder auf 0 (M2a, T20-Fix — Self-Healing).

    Bedingtes ``UPDATE … WHERE delisted = 1``: Ein nachfolgender Fetch mit
    Daten beweist, dass das Symbol doch Kurse liefert (der Vermerk war z. B.
    ein Fehlalarm oder die Notierung ist zurückgekehrt). Best-effort wie
    :func:`vermerke_delisted_wenn_bekannt` — DB-Pannen werden zu ``False``.
    Rückgabe ``True`` genau dann, wenn ein Vermerk tatsächlich aufgehoben wurde.
    """
    try:
        checked = validate_symbol(symbol)
    except ProviderError:
        return False
    try:
        conn = db.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE symbols SET delisted = 0 WHERE symbol = ? AND delisted = 1",
                (checked,),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return cursor.rowcount > 0
    except sqlite3.Error:
        return False


def ist_delisted(symbol: str) -> bool:
    """Liest ``symbols.delisted`` (``False`` bei unbekanntem/v1-Symbol)."""
    try:
        checked = validate_symbol(symbol)
    except ProviderError:
        return False
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT delisted FROM symbols WHERE symbol = ?", (checked,)
        ).fetchone()
    except sqlite3.Error:  # v1-DB ohne Spalte — kein Delisted-Konzept
        return False
    finally:
        conn.close()
    return bool(row is not None and row["delisted"])
