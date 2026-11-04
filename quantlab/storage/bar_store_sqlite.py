"""SQLite-Bar-Store-Adapter (Welle M2b, E-M03) — Backend hinter dem Dispatcher.

Mobiles Pendant zum Parquet-Store für Android/Chaquopy, wo pyarrow nicht
installierbar ist (PLAN.md F3). Dieses Modul wird **nicht direkt importiert** —
der Zugang läuft über :mod:`quantlab.storage.parquet_store` (stabile Modul-
funktionen-API, E-M07) mit ``QUANTLAB_BAR_STORE=sqlite``. Alle öffentlichen
Funktionen tragen dieselben Signaturen und Semantik wie ihre Parquet-Pendants
(Parität via ``tests/test_mobile_bar_store.py``, parametrisiert über beide
Backends).

Schema und Speicherort:

- Tabelle ``bars`` **in der bestehenden** ``quantlab.db`` (respektiert
  ``QUANTLAB_DATA_DIR``): ``symbol, interval, ts`` (int64 µs UTC) als
  Komposit-Primärschlüssel plus ``open/high/low/close/volume/adj_factor``
  (REAL). Angelegt wird sie **idempotent zur Laufzeit** per
  ``CREATE TABLE IF NOT EXISTS`` (Muster: ``_legacy_blendung_setzen`` —
  bewusst keine neue Migration, da 13 Bestandstests die Migrationskette 1..9
  hart prüfen und migrations.py in dieser Runde Sperrgebiet ist; ENTSCHEIDUNG
  des Implementers gem. PLAN.md §M2b „oder eigener Weg — Implementer
  entscheidet").
- ``bar_meta`` bleibt in der bestehenden Tabelle mit identischer Semantik
  (first_ts/last_ts/row_count/updated_at/source) — Lesezugriff und Meta-Pflege
  werden aus :mod:`quantlab.storage.parquet_store` wiederverwendet, nicht
  dupliziert (E-M07). Ebenso :func:`normalize_ohlcv` und :class:`BarMeta`/
  :data:`BAR_COLUMNS`.

Verhalten und dokumentierte Abweichungen zum Parquet-Backend:

- **Dedup last-write-wins** auf gleichem ``ts`` wird direkt in SQL abgebildet:
  ``INSERT … ON CONFLICT(symbol, interval, ts) DO UPDATE`` — der letzte Wert
  der Lieferung gewinnt (executemany verarbeitet die Zeilen in Folge).
- **``adj_factor`` ist immer vorhanden** (Schema-Spalte, Default 1.0): das
  v1-Kompatibilitätsproblem fehlender Spalten entfällt mobil komplett;
  ``legacy_blendung`` wird deshalb von diesem Backend nie gesetzt (bleibt
  ``False``). NULL-Faktoren aus Lieferungen bereinigt ``normalize_ohlcv``
  bereits vor dem Schreiben zu 1.0.
- Kein Read-merge-write und kein tmp-File: SQLite-Transaktionen
  (``BEGIN IMMEDIATE``, Muster :mod:`quantlab.db`) machen den Upsert atomar.
- ``has_data`` prüft Zeilen in ``bars`` (kein File-Existenz-Check);
  ``delete_symbol_data`` zählt gelöschte (Intervall-)Gruppen statt Dateien.
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from quantlab import db
from quantlab.providers.base import validate_interval
from quantlab.providers.symbols import validate_symbol

logger = logging.getLogger(__name__)

#: Idempotente DDL der Balken-Tabelle (additiv, kein ALTER auf Bestand).
_BARS_DDL = """
CREATE TABLE IF NOT EXISTS bars(
    symbol TEXT NOT NULL,                     -- Kürzel wie im Parquet-Store
    interval TEXT NOT NULL,                   -- z. B. '1d', '1h' (Whitelist)
    ts INTEGER NOT NULL,                      -- int64 Mikrosekunden UTC (E006)
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL DEFAULT 0,
    adj_factor REAL NOT NULL DEFAULT 1.0,     -- immer vorhanden (v1-Fall entfällt)
    PRIMARY KEY(symbol, interval, ts)
)
"""

_UPSERT_SQL = (
    "INSERT INTO bars(symbol, interval, ts, open, high, low, close, volume, "
    "adj_factor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT(symbol, interval, ts) DO UPDATE SET "
    "open = excluded.open, high = excluded.high, low = excluded.low, "
    "close = excluded.close, volume = excluded.volume, "
    "adj_factor = excluded.adj_factor"
)


def _dispatcher():
    """Läd das Dispatcher-Modul lazy (kein circular Import beim Laden hier)."""
    from quantlab.storage import parquet_store

    return parquet_store


def _verbinde_und_ensure() -> sqlite3.Connection:
    """Verbindung mit garantiert vorhandener ``bars``-Tabelle (idempotent).

    Die DDL ist ``CREATE TABLE IF NOT EXISTS`` — nach dem ersten Schreiben ist
    sie ein No-Op; Lese-/Löschpfade vor dem je ersten Upsert sehen damit
    dieselbe „kein Bestand“-Semantik wie der Parquet-Check auf fehlende Files.
    """
    conn = db.connect()
    conn.execute(_BARS_DDL)
    return conn


# ---------------------------------------------------------------------------
# Lesen
# ---------------------------------------------------------------------------


def has_data(symbol: str, interval: str) -> bool:
    """Existiert mindestens ein Balken für (Symbol, Intervall) in ``bars``?"""
    conn = _verbinde_und_ensure()
    try:
        zeile = conn.execute(
            "SELECT 1 FROM bars WHERE symbol = ? AND interval = ? LIMIT 1",
            (symbol, interval),
        ).fetchone()
    finally:
        conn.close()
    return zeile is not None


def read_bars(
    symbol: str,
    interval: str,
    start: int | None = None,
    end: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Liest Balken im kanonischen Schema, aufsteigend nach ``ts`` (Parität).

    ``start`` inklusive, ``end`` exklusiv (int64 µs UTC); ``limit`` kürzt nach
    dem Filtern auf die letzten (jüngsten) Balken. Unbekanntes Symbol →
    leeres kanonisches DataFrame (404-Semantik setzt der Service).
    """
    parquet_store = _dispatcher()
    validate_symbol(symbol)
    validate_interval(interval)
    parquet_store._validate_read_args(start, end, limit)

    abfrage = "SELECT ts, open, high, low, close, volume, adj_factor FROM bars "
    bedingungen: list[str] = []
    parameter: list[object] = [symbol, interval]
    if start is not None:
        bedingungen.append("ts >= ?")
        parameter.append(int(start))
    if end is not None:
        bedingungen.append("ts < ?")
        parameter.append(int(end))
    if bedingungen:
        abfrage += "WHERE symbol = ? AND interval = ? AND " + " AND ".join(bedingungen)
    else:
        abfrage += "WHERE symbol = ? AND interval = ?"
    abfrage += " ORDER BY ts ASC"

    conn = _verbinde_und_ensure()
    try:
        zeilen = conn.execute(abfrage, parameter).fetchall()
    finally:
        conn.close()

    spalten = parquet_store.BAR_COLUMNS
    if not zeilen:
        return parquet_store._empty_bars()
    bars = pd.DataFrame(zeilen, columns=list(spalten))
    bars = bars.astype(parquet_store._BAR_DTYPES)
    if limit is not None:  # jüngste Balken behalten (Parquet-Semantik: tail)
        bars = bars.tail(int(limit))
    return bars.reset_index(drop=True)


# ---------------------------------------------------------------------------
# bar_meta (gemeinsame Tabelle — Delegation auf den Dispatcher, E-M07)
# ---------------------------------------------------------------------------


def get_bar_meta(symbol: str, interval: str):
    """Abdeckungs-Metadaten — identisch zum Parquet-Backend (gleiche Tabelle)."""
    return _dispatcher().get_bar_meta(symbol, interval)


def list_bar_meta(symbol: str | None = None):
    """Alle Abdeckungs-Metadaten — identisch zum Parquet-Backend (gleiche Tabelle)."""
    return _dispatcher().list_bar_meta(symbol)


# ---------------------------------------------------------------------------
# Upsert und Löschen
# ---------------------------------------------------------------------------


def upsert_bars(symbol: str, interval: str, bars: pd.DataFrame):
    """Mergt Balken atomar in ``bars`` (Dedup last-write-wins via ON CONFLICT).

    Semantik wie :func:`quantlab.storage.parquet_store.upsert_bars`: leerer/
    bereinigt leerer Upload wird nicht geschrieben (Rückgabe: unveränderte
    Bestands-Meta bzw. Null-Meta); die ``bar_meta``-Zeile wird in derselben
    Transaktion aus dem frischen Tabellenstand (MIN/MAX/COUNT) gepflegt.
    """
    parquet_store = _dispatcher()
    validate_symbol(symbol)
    neue = parquet_store.normalize_ohlcv(bars, interval)

    conn = _verbinde_und_ensure()
    try:
        if neue.empty:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(_BARS_DDL)
            conn.execute("COMMIT")
            return get_bar_meta(symbol, interval) or parquet_store.BarMeta(
                symbol, interval, None, None, 0, updated_at=""
            )

        if "adj_factor" not in neue.columns:  # Lieferung ohne Faktoren → 1.0
            neue = neue.copy()
            neue["adj_factor"] = 1.0
        neue = neue[list(parquet_store.BAR_COLUMNS)]

        daten = [
            (
                symbol,
                interval,
                int(zeile.ts),
                float(zeile.open),
                float(zeile.high),
                float(zeile.low),
                float(zeile.close),
                float(zeile.volume),
                float(zeile.adj_factor),
            )
            for zeile in neue.itertuples(index=False)
        ]

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(_BARS_DDL)
            conn.executemany(_UPSERT_SQL, daten)
            erste, letzte, anzahl = conn.execute(
                "SELECT MIN(ts), MAX(ts), COUNT(*) FROM bars WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            ).fetchone()
            conn.execute(
                "INSERT INTO bar_meta(symbol, interval, first_ts, last_ts, "
                "row_count, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol, interval) DO UPDATE SET "
                "first_ts = excluded.first_ts, last_ts = excluded.last_ts, "
                "row_count = excluded.row_count, updated_at = excluded.updated_at",
                (
                    symbol,
                    interval,
                    parquet_store._ts_to_iso(int(erste)),
                    parquet_store._ts_to_iso(int(letzte)),
                    int(anzahl),
                    db.utc_now_iso(),
                ),
            )
            zeile = conn.execute(
                "SELECT * FROM bar_meta WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            ).fetchone()
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:  # kein blindes ROLLBACK (Muster: db.py)
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    assert zeile is not None  # frisch geschrieben — immer lesbar
    return parquet_store._row_to_meta(zeile)


def delete_bars(symbol: str, interval: str) -> bool:
    """Löscht Balken + bar_meta-Zeile für (Symbol, Intervall); ``True`` wenn vorhanden."""
    validate_symbol(symbol)
    validate_interval(interval)
    conn = _verbinde_und_ensure()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = conn.execute(
                "DELETE FROM bars WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            )
            vorhanden = cursor.rowcount > 0
            meta_cursor = conn.execute(
                "DELETE FROM bar_meta WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    return vorhanden or meta_cursor.rowcount > 0


def delete_symbol_data(symbol: str) -> int:
    """Löscht alle Balken des Symbols (über Intervalle hinweg) + bar_meta-Zeilen.

    Liefert die Anzahl gelöschter (Intervall-)Gruppen — das Pendant zur
    Dateizahl des Parquet-Backends. Räumt bewusst auch Meta-Zeilen ohne
    Bestand (konsistenter Zustand).
    """
    validate_symbol(symbol)
    conn = _verbinde_und_ensure()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            gruppen = conn.execute(
                "SELECT COUNT(DISTINCT interval) FROM bars WHERE symbol = ?",
                (symbol,),
            ).fetchone()[0]
            conn.execute("DELETE FROM bars WHERE symbol = ?", (symbol,))
            conn.execute("DELETE FROM bar_meta WHERE symbol = ?", (symbol,))
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    return int(gruppen)
