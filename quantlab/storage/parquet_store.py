"""Parquet-Store: kanonischer OHLCV-Speicher (§D.2, E005/E006).

Layout: ``<Datenverzeichnis>/market/<interval>/<SAFE_SYMBOL>.parquet`` — ein
File pro (Symbol, Intervall); der Dateiname wird über
:func:`quantlab.providers.symbols.safe_filename` abgeleitet (``^GSPC`` →
``_GSPC``). Geschlüsselt wird überall über den ursprünglichen Symbol-Kürzel.

Schema (bindend, E006): Spalten ``ts`` (int64 Mikrosekunden UTC) sowie
``open/high/low/close/volume`` (float64), aufsteigend nach ``ts`` sortiert,
``ts`` ohne Duplikate. Tages-/Wochen-/Monats-Intervalle
(:data:`quantlab.providers.base.DAILY_INTERVALS`) werden beim Schreiben auf
Mitternacht UTC gefloort — die Invariante gilt damit unabhängig vom Provider.

Seit T20/E022 gehört die Spalte ``adj_factor`` (float64, Faktor
``AdjClose/Close`` je Bar) zum Store-Schema (:data:`BAR_COLUMNS`): die Balken
werden **roh** gespeichert, die Adjustierung passiert on-read (Service).
Bestandsdateien ohne die Spalte liest :func:`read_bars` mit Faktor 1.0
(Kompatibilität, R11) — echter Faktorstand per Re-Download des Fensters
(expliziter ``start`` am Download-Job erzwingt den Voll-Fenster-Fetch). Die
5-Balken-Inkrement-Überlappung (Download-Service, §D.2) lädt damit nach neuen
Dividenden automatisch die rückwirkend geänderten Faktoren der letzten Balken
nach (Yahoo-Faktorkette ändert sich retrograd; Dedup last-write-wins regelt
Konflikte — dokumentiert und in ``tests/test_adj_factor.py`` getestet).

Upsert-Protokoll (E005, atomar/idempotent/crash-safe):
1. Neues DataFrame + bestehendes File im Speicher mergen: stabil sortieren,
   dann ``drop_duplicates(subset="ts", keep="last")`` — last-write-wins, der
   neuere Wert gewinnt auf gleichem ``ts``.
2. Nach ``<file>.tmp.parquet`` schreiben und mit ``os.replace`` atomar
   tauschen. Wirft irgendetwas (auch ein simulierter Crash), bleibt das alte
   File intakt und das tmp-File wird entfernt.
3. ``bar_meta`` (first_ts/last_ts/row_count/updated_at) in SQLite pflegen —
   TEXT-Spalten im kanonischen ISO-8601-UTC-Format (Muster ``db.utc_now_iso``);
   die API dieser Schicht arbeitet konsequent mit int64 µs.

Ein leerer Upload wird nicht geschrieben (§D.2 „Null-Download“). Schreib- und
Meta-Pflege laufen pro Prozess unter einem Lock (genau ein Sidecar-Prozess);
SQLite-Zugriffe folgen dem Verbindungs-Muster aus :mod:`quantlab.db`
(``BEGIN IMMEDIATE``, konditionales ``ROLLBACK``).

**Dispatcher (Welle M2b/E-M03):** Dieses Modul ist die stabile Import-Ober-
fläche (Modulfunktionen-API, E-M07 — alle Konsumenten nutzen Attributzugriff).
Die Env ``QUANTLAB_BAR_STORE`` wählt **pro Aufruf** die Implementierung:
``parquet`` (Default, unverändertes Desktop-Verhalten) oder ``sqlite``
(:mod:`quantlab.storage.bar_store_sqlite` — Android/Chaquopy, dort ist
pyarrow nicht installierbar, F3). Der pyarrow-Import ist lazy — mit
``QUANTLAB_BAR_STORE=sqlite`` wird pyarrow nie geladen. :func:`normalize_ohlcv`
ist backend-unabhängig und bleibt hier als gemeinsame Implementierung
(auch die Provider rufen sie direkt); equally gemeinsam sind ``bar_meta``-
Lesezugriff (:func:`get_bar_meta`/:func:`list_bar_meta`) — ``bar_meta`` liegt
in beiden Backends in derselben SQLite-Tabelle.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab import db
from quantlab.config import get_market_dir
from quantlab.providers.base import (
    DAILY_INTERVALS,
    ProviderError,
    validate_interval,
)
from quantlab.providers.symbols import safe_filename, validate_symbol

#: Kanonische Spalten (Reihenfolge im Parquet-File) und dtypes (E006 + E022).
#: ``adj_factor`` (T20): Faktor ``AdjClose/Close`` je Bar — 1.0, wenn die
#: Quelle keinen liefert (Stooq/Fake) oder die Spalte fehlt (v1-Bestand).
BAR_COLUMNS: tuple[str, ...] = (
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_factor",
)
_BAR_DTYPES: dict[str, str] = {
    "ts": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "adj_factor": "float64",
}

#: Mikrosekunden pro Tag (für das Midnight-Flooring der Tages-Intervalle).
US_PER_DAY: int = 86_400_000_000

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)

#: Serialisiert Upserts pro Prozess (Read-merge-write auf dem Parquet-File).
_UPSERT_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatcher (M2b/E-M03): Backend-Wahl pro Aufruf über QUANTLAB_BAR_STORE
# ---------------------------------------------------------------------------

#: Gültige Werte der Env ``QUANTLAB_BAR_STORE`` (Default: ``parquet``).
BAR_STORE_BACKENDS: tuple[str, ...] = ("parquet", "sqlite")


def bar_store_backend() -> str:
    """Aktives Bar-Store-Backend aus der Env (``parquet`` | ``sqlite``).

    Pro Aufruf gelesen (nicht beim Import gebunden), damit Tests beide
    Backends im selben Prozess parametrisiert treiben können.
    """
    wert = os.environ.get("QUANTLAB_BAR_STORE", "parquet").strip().lower()
    if wert not in BAR_STORE_BACKENDS:
        raise RuntimeError(
            f"QUANTLAB_BAR_STORE=„{wert}“ ist ungültig — erlaubt: {', '.join(BAR_STORE_BACKENDS)}"
        )
    return wert


def _sqlite_impl():
    """Läd den SQLite-Adapter lazy (kein circular Import, kein pyarrow nötig)."""
    from quantlab.storage import bar_store_sqlite

    return bar_store_sqlite


def _pq():
    """pyarrow.parquet lazy — mit QUANTLAB_BAR_STORE=sqlite wird es nie geladen."""
    import pyarrow.parquet as pq

    return pq


@dataclass(frozen=True)
class BarMeta:
    """Abdeckungs-Metadaten zu (Symbol, Intervall) — Zeile aus ``bar_meta``.

    ``first_ts``/``last_ts`` sind int64 µs UTC (in SQLite als ISO-8601-UTC-Text
    abgelegt); ``updated_at`` ist der ISO-Text des letzten Schreibens.
    ``source`` ist die grobe Herkunftsmarke ``'yahoo' | 'stooq'`` (E022/E023 —
    geschrieben vom Download-Service; ``'yahoo'``, wenn die Spalte/Zeile fehlt).
    ``legacy_blendung`` (M3, T20-Fix): der Bestand enthält v1-adjustierte Kurse
    als raw — beim ersten Schreiben in eine v1-Datei (ohne ``adj_factor``-Spalte
    im Bestand vor dem Merge) gesetzt; erst ein Neudownload mit explizitem
    ``start`` über das gesamte Bestandsfenster (Faktoren komplett) nimmt die
    Marke wieder zurück (``False``, wenn die Spalte/Zeile fehlt).
    """

    symbol: str
    interval: str
    first_ts: int | None
    last_ts: int | None
    row_count: int
    updated_at: str
    source: str = "yahoo"
    legacy_blendung: bool = False


# ---------------------------------------------------------------------------
# Zeit-Helfer (E006)
# ---------------------------------------------------------------------------


def _ts_to_iso(ts_us: int) -> str:
    """int64 µs UTC → ISO-8601-UTC-Text (kanonisches Format für TEXT-Spalten)."""
    return (_EPOCH_UTC + timedelta(microseconds=ts_us)).isoformat()


def _iso_to_ts(text: str | None) -> int | None:
    """ISO-8601-UTC-Text → int64 µs UTC; ``None`` bei leerem/defektem Text."""
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - _EPOCH_UTC) // timedelta(microseconds=1)


# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------


def parquet_path(symbol: str, interval: str) -> Path:
    """Pfad ``market/<interval>/<SAFE_SYMBOL>.parquet``; validiert beide Teile.

    Die Validierung (Symbol-Whitelist, Intervall-Whitelist) ist zugleich
    Pfad-Schutz — keine ``/``, ``..`` o. Ä. gelangen in den Dateinamen.
    """
    validate_symbol(symbol)
    validate_interval(interval)
    return get_market_dir() / interval / f"{safe_filename(symbol)}.parquet"


def has_data(symbol: str, interval: str) -> bool:
    """Existieren für (Symbol, Intervall) Balken im aktiven Backend?"""
    if bar_store_backend() == "sqlite":
        return _sqlite_impl().has_data(symbol, interval)
    return _has_data_parquet(symbol, interval)


def _has_data_parquet(symbol: str, interval: str) -> bool:
    """Existiert für (Symbol, Intervall) ein Parquet-File?"""
    return parquet_path(symbol, interval).is_file()


def _empty_bars() -> pd.DataFrame:
    """Leeres DataFrame im kanonischen Schema (Spalten + dtypes garantiert)."""
    return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in _BAR_DTYPES.items()})[
        list(BAR_COLUMNS)
    ]


# ---------------------------------------------------------------------------
# Normalisierung (Eingabe → kanonisches Schema)
# ---------------------------------------------------------------------------


def normalize_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Bringt ein OHLCV-DataFrame ins kanonische Schema (§D.2/E006).

    Akzeptiert die Spalte ``ts`` als int64 (Mikrosekunden UTC — der kanonische
    Fall) oder als datetime (tz-aware wird nach UTC konvertiert, tz-naiv als
    UTC interpretiert — ANNAHME); alternativ einen DatetimeIndex. Andere
    ``ts``-Typen (z. B. float) werden bewusst verworfen (422), um stille
    Fehl-Interpretationen auszuschließen.

    Zeilen ohne ``ts`` oder mit NULL in open/high/low/close fallen weg;
    fehlende ``volume``-Spalte bzw. NULL-Volumen wird mit 0.0 gefüllt.
    Zusätzliche Spalten werden verworfen — **Ausnahme ``adj_factor`` (T20/E022):**
    ist sie im Input vorhanden, wird sie übernommen (NULL/defekt → 1.0); fehlt
    sie, bleibt sie auch im Ergebnis weg (z. B. Stooq-/Fake-Provider) —
    :func:`upsert_bars` ergänzt vor dem Schreiben 1.0, :func:`read_bars` beim
    Lesen v1-Bestandsdateien. Für Tages-Intervalle wird ``ts``
    auf Mitternacht UTC gefloort. Die Reihenfolge bleibt erhalten — Sortieren
    und Dedup macht :func:`upsert_bars`.
    """
    validate_interval(interval)
    if not isinstance(df, pd.DataFrame):
        raise ProviderError(
            "OHLC-Daten müssen ein pandas-DataFrame sein.", 422, "ungueltige_ohlc_daten"
        )
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.columns[0]: "ts"})
        else:
            raise ProviderError(
                "Die Spalte „ts“ fehlt (alternativ: DatetimeIndex verwenden).",
                422,
                "ungueltige_ohlc_daten",
            )

    out = out[out["ts"].notna()]
    ts = out["ts"]
    if pd.api.types.is_integer_dtype(ts):
        out["ts"] = ts.astype("int64")
    elif pd.api.types.is_float_dtype(ts):
        raise ProviderError(
            "Spalte „ts“ muss int64-Mikrosekunden UTC oder datetime sein "
            "(float wird nicht akzeptiert, um Einheiten-Fehldeutungen auszuschließen).",
            422,
            "ungueltige_ohlc_daten",
        )
    else:
        try:
            as_utc = pd.to_datetime(ts, utc=True)
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                "Spalte „ts“ ist weder int64-Mikrosekunden UTC noch datetime.",
                422,
                "ungueltige_ohlc_daten",
            ) from exc
        # Einheiten-unabhängig (pandas darf ns/us/ms führen): Differenz zur
        # Epoche, dann ganzzahlige Division durch 1 µs → int64 µs UTC.
        out["ts"] = (as_utc - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(1, "us")

    if interval in DAILY_INTERVALS:  # E006: Tages-Bars auf Mitternacht UTC
        out["ts"] = out["ts"] - (out["ts"] % US_PER_DAY)

    missing = [column for column in ("open", "high", "low", "close") if column not in out.columns]
    if missing:
        raise ProviderError(
            f"Fehlende OHLC-Spalten: {', '.join(missing)}.",
            422,
            "ungueltige_ohlc_daten",
        )
    if "volume" not in out.columns:
        out["volume"] = 0.0

    hat_faktor = "adj_factor" in out.columns
    if hat_faktor:  # NULL/defekte Faktoren → neutral 1.0 (kein stiller Datenverlust)
        # M1-Härtung (T20-Fix, symmetrisch zu YfinanceProvider._adj_faktor):
        # Direkt-Upsererte Faktoren können ±inf/negativ sein (``fillna`` allein
        # greift bei inf nicht) — alles Nicht-Endliche/Nicht-Positive → 1.0.
        faktoren = pd.to_numeric(out["adj_factor"], errors="coerce")
        out["adj_factor"] = faktoren.where(np.isfinite(faktoren) & (faktoren > 0), 1.0).astype(
            "float64"
        )
    spalten = list(BAR_COLUMNS) if hat_faktor else [c for c in BAR_COLUMNS if c != "adj_factor"]
    out = out[spalten]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    dtypes = (
        dict(_BAR_DTYPES)
        if hat_faktor
        else {name: typ for name, typ in _BAR_DTYPES.items() if name != "adj_factor"}
    )
    return out.astype(dtypes).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lesen
# ---------------------------------------------------------------------------


def _validate_read_args(start: int | None, end: int | None, limit: int | None) -> None:
    """Gemeinsame Argumentprüfung für den Lesepfad (422 bei Unsinn)."""
    if start is not None and end is not None and start > end:
        raise ProviderError(
            "„start“ liegt nach „end“.",
            422,
            "ungueltiger_zeitraum",
            hint="start ist inklusive, end exklusiv (Mikrosekunden UTC).",
        )
    if limit is not None and limit <= 0:
        raise ProviderError("„limit“ muss eine positive ganze Zahl sein.", 422, "ungueltiges_limit")


def read_bars(
    symbol: str,
    interval: str,
    start: int | None = None,
    end: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Liest Balken im kanonischen Schema, aufsteigend nach ``ts`` (Dispatcher).

    Delegiert an das aktive Backend (``QUANTLAB_BAR_STORE``); Semantik der
    Filter siehe :func:`_read_bars_parquet` bzw. Modul-Docstring von
    :mod:`quantlab.storage.bar_store_sqlite` — beide Backends verhalten sich
    identisch (Parität via ``tests/test_mobile_bar_store.py``).
    """
    if bar_store_backend() == "sqlite":
        return _sqlite_impl().read_bars(symbol, interval, start, end, limit)
    return _read_bars_parquet(symbol, interval, start, end, limit)


def _read_bars_parquet(
    symbol: str,
    interval: str,
    start: int | None = None,
    end: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Liest Balken im kanonischen Schema, aufsteigend nach ``ts``.

    ``start`` inklusive, ``end`` exklusiv (int64 µs UTC, yfinance-Konvention);
    ``limit`` kürzt nach dem Filtern auf die letzten (jüngsten) Balken.
    Fehlendes File → leeres kanonisches DataFrame (die 404-Semantik für
    fehlende Symbole setzt der darüberliegende Service). v1-Bestandsdateien
    ohne ``adj_factor``-Spalte werden mit Faktor 1.0 gelesen (E022, R11).
    """
    _validate_read_args(start, end, limit)
    path = parquet_path(symbol, interval)
    if not path.is_file():
        return _empty_bars()
    bars = pd.read_parquet(path, engine="pyarrow")
    if "adj_factor" not in bars.columns:  # v1-Bestand → neutraler Faktor
        bars["adj_factor"] = 1.0
    bars = bars[list(BAR_COLUMNS)].astype(_BAR_DTYPES)
    bars = bars.sort_values("ts", kind="stable").reset_index(drop=True)
    if start is not None:
        bars = bars[bars["ts"] >= start]
    if end is not None:
        bars = bars[bars["ts"] < end]
    if limit is not None:
        bars = bars.tail(int(limit))
    return bars.reset_index(drop=True)


# ---------------------------------------------------------------------------
# bar_meta (SQLite, Muster: quantlab.db)
# ---------------------------------------------------------------------------


def _row_to_meta(row: sqlite3.Row) -> BarMeta:
    """sqlite3.Row → :class:`BarMeta` (TEXT-Zeitstempel zurück nach µs)."""
    quelle = str(row["source"]) if "source" in row.keys() else "yahoo"
    blendung = bool(row["legacy_blendung"]) if "legacy_blendung" in row.keys() else False
    return BarMeta(
        symbol=row["symbol"],
        interval=row["interval"],
        first_ts=_iso_to_ts(row["first_ts"]),
        last_ts=_iso_to_ts(row["last_ts"]),
        row_count=int(row["row_count"]),
        updated_at=row["updated_at"],
        source=quelle,
        legacy_blendung=blendung,
    )


def get_bar_meta(symbol: str, interval: str) -> BarMeta | None:
    """Abdeckungs-Metadaten zu (Symbol, Intervall) — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM bar_meta WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _row_to_meta(row)


def list_bar_meta(symbol: str | None = None) -> list[BarMeta]:
    """Alle Abdeckungs-Metadaten (optional je Symbol), sortiert nach Symbol/Intervall."""
    conn = db.connect()
    try:
        if symbol is None:
            rows = conn.execute("SELECT * FROM bar_meta ORDER BY symbol, interval").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM bar_meta WHERE symbol = ? ORDER BY symbol, interval",
                (symbol,),
            ).fetchall()
    finally:
        conn.close()
    return [_row_to_meta(row) for row in rows]


def _write_bar_meta(symbol: str, interval: str, merged: pd.DataFrame) -> BarMeta:
    """Schreibt die bar_meta-Zeile zum frisch gesicherten Bestand (Upsert).

    Die Rückgabe liest die Zeile nach dem Commit frisch zurück — so trägt sie
    die aktuelle ``source``-Marke (E023: der UPDATE-Zweig des Upserts erhält
    eine bestehende Herkunft, bis der Download-Service sie neu setzt).
    """
    first_ts = int(merged["ts"].iloc[0])
    last_ts = int(merged["ts"].iloc[-1])
    row_count = int(len(merged))
    updated_at = db.utc_now_iso()
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — Meta-Pflege ist Read-modify-write
        # auf dem Primärschlüssel (symbol, interval).
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO bar_meta(symbol, interval, first_ts, last_ts, row_count, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol, interval) DO UPDATE SET "
            "first_ts = excluded.first_ts, last_ts = excluded.last_ts, "
            "row_count = excluded.row_count, updated_at = excluded.updated_at",
            (symbol, interval, _ts_to_iso(first_ts), _ts_to_iso(last_ts), row_count, updated_at),
        )
        zeile = conn.execute(
            "SELECT * FROM bar_meta WHERE symbol = ? AND interval = ?", (symbol, interval)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:  # kein blindes ROLLBACK (Muster: quantlab.db)
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert zeile is not None  # frisch geschrieben — immer lesbar
    return _row_to_meta(zeile)


def _legacy_blendung_setzen(symbol: str, interval: str, wert: bool) -> None:
    """Setzt/löscht die Notiz ``bar_meta.legacy_blendung`` (M3, T20-Fix).

    Speicher: additive SQLite-Spalte ``legacy_blendung INTEGER NOT NULL DEFAULT 0``
    — bewusst per Runtime-``ALTER`` statt einer neuen Migration (migrations.py
    ist in dieser Runde Sperrgebiet; die Spalte ist rein additiv und für alle
    Bestandsleser unsichtbar: ``_row_to_meta`` bewacht sie wie ``source``).
    Best-effort wie der Delisted-Vermerk: eine DB-Panne darf einen erfolgreichen
    Parquet-Write nachträglich nicht kippen — sie wird nur protokolliert.
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        spalten = {zeile[1] for zeile in conn.execute("PRAGMA table_info(bar_meta)")}
        if "legacy_blendung" not in spalten:
            if not wert:
                conn.execute("COMMIT")  # nichts zu löschen — Spalte existiert nie
                return
            conn.execute(
                "ALTER TABLE bar_meta ADD COLUMN legacy_blendung INTEGER NOT NULL DEFAULT 0"
            )
        conn.execute(
            "UPDATE bar_meta SET legacy_blendung = ? WHERE symbol = ? AND interval = ?",
            (1 if wert else 0, symbol, interval),
        )
        conn.execute("COMMIT")
    except sqlite3.Error:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        logger.warning(
            "legacy_blendung-Marker für (%s, %s) konnte nicht %s werden",
            symbol,
            interval,
            "gesetzt" if wert else "gelöscht",
            exc_info=True,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Upsert (E005) und Löschen
# ---------------------------------------------------------------------------


def upsert_bars(symbol: str, interval: str, bars: pd.DataFrame) -> BarMeta:
    """Mergt Balken atomar in den Bestand (Dispatcher, Protokoll: E005).

    Delegiert an das aktive Backend (``QUANTLAB_BAR_STORE``) — Dedup
    last-write-wins auf ``ts``, Semantik siehe :func:`_upsert_bars_parquet`
    bzw. :mod:`quantlab.storage.bar_store_sqlite`.
    """
    if bar_store_backend() == "sqlite":
        return _sqlite_impl().upsert_bars(symbol, interval, bars)
    return _upsert_bars_parquet(symbol, interval, bars)


def _upsert_bars_parquet(symbol: str, interval: str, bars: pd.DataFrame) -> BarMeta:
    """Mergt Balken atomar in den Bestand (Protokoll: Modul-Docstring, E005).

    Dedup last-write-wins auf ``ts``: Bei Konflikten gewinnt der Wert aus
    ``bars`` (und innerhalb von ``bars`` die spätere Zeile). Ein leerer/bereinigt
    leerer Upload wird nicht geschrieben — Rückgabe ist dann der unveränderte
    Bestand (bzw. eine Null-Meta, falls nie etwas gespeichert war).

    M3 (T20-Fix) — v1-Blendung: Fehlte dem Bestandsfile **vor** dem Merge die
    ``adj_factor``-Spalte (v1-Datei), werden alte, v1-adjustierte Kurse mit
    neuen Rohkursen gemischt — :func:`_legacy_blendung_setzen` markiert das
    dauerhaft in ``bar_meta.legacy_blendung`` (Lesepfad: :class:`BarMeta`).
    Zurückgenommen wird die Marke nur durch eine Lieferung **mit** Faktoren,
    die das gesamte bisherige Bestandsfenster abdeckt (Neudownload mit
    explizitem ``start`` beim ersten ``first_ts``).
    """
    validate_symbol(symbol)
    neue = normalize_ohlcv(bars, interval)
    with _UPSERT_LOCK:
        path = parquet_path(symbol, interval)
        if neue.empty:
            existing = get_bar_meta(symbol, interval)
            if existing is not None:
                return existing
            return BarMeta(symbol, interval, None, None, 0, updated_at="")

        v1_bestand = False
        meta_vorher: BarMeta | None = None
        if path.is_file():
            # Footer-Only-Schemaread: Stand VOR dem Merge (read_bars würde die
            # fehlende Spalte bereits mit 1.0 auffüllen und die Information tilgen).
            v1_bestand = "adj_factor" not in _pq().read_schema(path).names
            bestand = read_bars(symbol, interval)
            meta_vorher = get_bar_meta(symbol, interval)
            merged = pd.concat([bestand, neue], ignore_index=True)
        else:
            merged = neue

        # Stabil sortieren (Eingabe-Reihenfolge bleibt innerhalb gleicher ts
        # erhalten) → dedup keep="last": der neuere Wert gewinnt deterministisch.
        merged = (
            merged.sort_values("ts", kind="stable")
            .drop_duplicates(subset="ts", keep="last")
            .reset_index(drop=True)
        )
        # E022: adj_factor ist Bestandteil des Store-Schemas — Lieferungen ohne
        # die Spalte (Stooq, Fake, v1-Persistenzpfad) bekommen neutral 1.0;
        # Lücken nach dem Merge (concat kann NaN erzeugen) ebenso.
        if "adj_factor" not in merged.columns:
            merged["adj_factor"] = 1.0
        merged["adj_factor"] = merged["adj_factor"].fillna(1.0)
        merged = merged[list(BAR_COLUMNS)]

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(f"{path}.tmp.parquet")
        try:
            merged.to_parquet(tmp, engine="pyarrow", index=False)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)  # kein .tmp-Rest; altes File blieb intakt
            raise
        meta = _write_bar_meta(symbol, interval, merged)

        # M3: Marker erst NACH der Meta-Zeile setzen/löschen (UPDATE braucht die
        # Zeile) — danach frisch zurücklesen, damit die Rückgabe ihn trägt.
        if v1_bestand:
            _legacy_blendung_setzen(symbol, interval, True)
            meta = get_bar_meta(symbol, interval) or meta
        elif (
            meta_vorher is not None
            and meta_vorher.legacy_blendung
            and "adj_factor" in neue.columns
            and meta_vorher.first_ts is not None
            and meta_vorher.last_ts is not None
            and int(neue["ts"].min()) <= meta_vorher.first_ts
            and int(neue["ts"].max()) >= meta_vorher.last_ts
        ):
            _legacy_blendung_setzen(symbol, interval, False)
            meta = get_bar_meta(symbol, interval) or meta
        return meta


def delete_bars(symbol: str, interval: str) -> bool:
    """Löscht Bestand + bar_meta-Zeile für (Symbol, Intervall) im aktiven Backend.

    ``True``, wenn Bestand oder Meta-Zeile existierte (idempotenter No-Op
    sonst) — gleiche Semantik in beiden Backends.
    """
    if bar_store_backend() == "sqlite":
        return _sqlite_impl().delete_bars(symbol, interval)
    return _delete_bars_parquet(symbol, interval)


def _delete_bars_parquet(symbol: str, interval: str) -> bool:
    """Löscht Parquet-File + bar_meta-Zeile für (Symbol, Intervall).

    ``True``, wenn File oder Meta-Zeile existierte (idempotenter No-Op sonst).
    """
    validate_symbol(symbol)
    validate_interval(interval)
    path = parquet_path(symbol, interval)
    file_existed = path.is_file()
    path.unlink(missing_ok=True)

    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM bar_meta WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        )
        meta_existed = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return file_existed or meta_existed


def delete_symbol_data(symbol: str) -> int:
    """Löscht den gesamten Bestand des Symbols im aktiven Backend + bar_meta-Zeilen.

    Liefert die Anzahl gelöschter (Intervall-)Bestände — Parquet: Dateien,
    SQLite: belegte (Symbol, Intervall)-Gruppen.
    """
    if bar_store_backend() == "sqlite":
        return _sqlite_impl().delete_symbol_data(symbol)
    return _delete_symbol_data_parquet(symbol)


def _delete_symbol_data_parquet(symbol: str) -> int:
    """Löscht alle Parquet-Files des Symbols (über Intervalle hinweg) + bar_meta-Zeilen.

    Liefert die Anzahl gelöschter Files. Räumt bewusst auch Meta-Zeilen, deren
    File bereits fehlte (konsistenter Zustand; „Parquet-Dateien räumt die
    darüberliegende Datenschicht ab“ — symbols_repo).
    """
    validate_symbol(symbol)
    safe = safe_filename(symbol)
    market_dir = get_market_dir()

    deleted = 0
    if market_dir.is_dir():
        for interval_dir in market_dir.iterdir():
            if not interval_dir.is_dir():
                continue
            candidate = interval_dir / f"{safe}.parquet"
            if candidate.is_file():
                candidate.unlink()
                deleted += 1

    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM bar_meta WHERE symbol = ?", (symbol,))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return deleted
