"""Versionierte, idempotente Schema-Migrationen (§D.1 + v2 §D.2).

- Jede Migration ist ein nummeriertes SQL-Skript (``MIGRATIONEN``); angewendete
  Versionen werden in der Buchhaltungstabelle ``schema_migrations`` protokolliert
  (keine fachliche Tabelle — Migrations-Metadaten).
- Ein Lauf wendet nur fehlende Versionen an; jedes Skript läuft in genau einer
  Transaktion (atomar, crash-sicher — halb angewendete Schemata gibt es nicht).
- :func:`schema_hash` liefert einen deterministischen Hash über das gesamte
  Schema (Tests: zweimal migrieren → identischer Hash; DoD T2).

Migrationskette (E028 — nur T18 schreibt DDL):

1. ``kernschema_d1`` — Kernschema wörtlich aus PLAN.md §D.1 (14 Tabellen).
2. ``v2_tabellen`` — neue v2-Domänen als reine CREATEs (§D.2): Portfolios,
   Corporate Actions, Fundamentals (Long-Format), Multi-Watchlists, Alerts,
   Chart-Zeichnungen, User-Strategien, Screener-Presets, Suche-Cache,
   Freshness-Marker, Backtest-Jobs. Bestandsdaten unberührt.
3. ``v2_flags_und_watchlist`` — ``symbols.delisted``/``bar_meta.source`` plus
   Daten-Überführung der v1-``watchlist`` in die Default-Liste 1 der neuen
   ``watchlists``/``watchlist_symbols`` (E033); die Legacy-Tabelle wird in
   derselben Transaktion gedroppt. Die Überführung läuft im Daten-Hook
   ``Migration.nach_sql`` (in derselben Transaktion wie die Statements), weil
   ``<now>`` aus :func:`db.utc_now_iso` stammt und die Reihenfolge
   Default-Liste → Symbole → DROP garantiert sein muss.
4. ``websuche_cache`` — Cache der Agent-Websuche (T-F6/E083) als reine neue
   Tabelle (E028: kein ALTER auf Bestand); TTL-Bewertung wie bei T19 beim
   Aufrufer (:mod:`quantlab.services.websuche`).
5. ``download_retries`` — Wiederholungsplan fehlgeschlagener Download-Symbole
   (E089/T-G3, Auto-Retry-Queue): je (Job, Symbol, Intervall) ein Versuchszähler
   plus fälliger nächster Versuch; gepflegt von :mod:`quantlab.services.downloads`.
6. ``symbols_isin`` — ``symbols.isin`` als nullable Spalte (T-G5/E091,
   ISIN-Brücke); ALTER-Präzedenz auf ``symbols`` ist Migration 3
   (``delisted``).
7. ``psb_instrumente`` — Planspiel-Universum als reine neue Tabelle (H1/
   E093): ISIN-Schlüssel, deutsche Fachfelder, Gruppen-Union der Tab-
   Zugehörigkeit, Yahoo-Kürzel-Brücke; gepflegt von
   :mod:`quantlab.storage.repos.psb_repo`.
8. ``psb_kennzahlen`` — VWD-Fundamentaldaten-Cache (H3/E095): je (ISIN,
   Sektion) die geparste Snippet-Sektion als JSON; gepflegt von
   :mod:`quantlab.storage.repos.psb_repo` (``kennzahlen_speichern``/
   ``kennzahlen_lesen``).
9. ``psb_backfill_fortschritt`` — Fortschritts-Buchhaltung des Bulk-
   Historien-Backfills (H5/E097): je (ISIN, Zeitraum) Status, Punkte-Zahl
   und Fehlversuche — macht den Byteworx-Bulk-Lauf **resumabel** (Backend-
   Neustart/Abbruch: fertige ISINs werden übersprungen). Pflege in
   :mod:`quantlab.services.planspiel.backfill` (die Tabelle ist H5-eigenes
   Schema; ``psb_repo`` bleibt Domäne der Universum-/Kennzahlen-Zeilen).

Vor der ersten Anwendung einer Version ≥ 2 legt :func:`run_migrations` einmalig
die Dateisicherung ``<db>.v1.bak`` neben die Datenbank (nur wenn noch nicht
vorhanden) — Bestandsschutz für v1-Datenbestände.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from quantlab import db

# Kernschema Migration 1 — exakt die DDL aus PLAN.md §D.1 (nur an die SQLite-
# Konvention „ein Statement pro CREATE" angepasst, Fachinhalt unverändert).
_KERNSCHEMA_SQL = """
CREATE TABLE settings(key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL);

CREATE TABLE symbols(
    symbol TEXT PRIMARY KEY, name TEXT, asset_type TEXT, exchange_tz TEXT,
    currency TEXT, added_at TEXT NOT NULL
);

CREATE TABLE watchlist(
    symbol TEXT PRIMARY KEY REFERENCES symbols(symbol),
    position INTEGER NOT NULL,
    added_at TEXT NOT NULL
);

CREATE TABLE bar_meta(
    symbol TEXT NOT NULL, interval TEXT NOT NULL, first_ts TEXT, last_ts TEXT,
    row_count INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
    PRIMARY KEY(symbol, interval)
);

CREATE TABLE downloads(
    job_id TEXT PRIMARY KEY, symbols_json TEXT NOT NULL, interval TEXT NOT NULL,
    params_json TEXT, status TEXT NOT NULL, error TEXT,
    created_at TEXT NOT NULL, finished_at TEXT
);

CREATE TABLE projects(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, description TEXT DEFAULT '',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE project_symbols(
    project_id INTEGER NOT NULL, symbol TEXT NOT NULL,
    PRIMARY KEY(project_id, symbol)
);

CREATE TABLE tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'todo' CHECK(status IN ('todo','doing','done')),
    position INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE notes(
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, title TEXT NOT NULL,
    body_md TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE project_context(
    project_id INTEGER PRIMARY KEY, content TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL
);

CREATE TABLE artifacts(
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('report','chart','backtest','analysis')),
    title TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);

CREATE TABLE chat_runs(
    id TEXT PRIMARY KEY, project_id INTEGER, title TEXT, model TEXT NOT NULL,
    thinking_type TEXT NOT NULL, reasoning_effort TEXT,
    status TEXT NOT NULL CHECK(status IN ('running','done','cancelled','error')),
    error TEXT, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
    created_at TEXT NOT NULL, finished_at TEXT
);

CREATE TABLE chat_messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('system','user','assistant','tool')),
    content TEXT, reasoning_content TEXT, tool_calls_json TEXT, tool_call_id TEXT,
    name TEXT, created_at TEXT NOT NULL
);

CREATE TABLE backtest_results(
    id TEXT PRIMARY KEY, project_id INTEGER, strategy TEXT NOT NULL,
    symbol TEXT NOT NULL, interval TEXT NOT NULL, params_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL, equity_json TEXT NOT NULL, trades_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_SCHEMA_MIGRATIONS_SQL = (
    "CREATE TABLE IF NOT EXISTS schema_migrations("
    "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
)

# Migration 2 — neue v2-Domänen (PLAN.md §D.2, wörtlich; Kommentare im Skript
# tragen bewusst keine Semikolons — der Statement-Splitter trennt an „;“).
_V2_TABELLEN_SQL = """
CREATE TABLE portfolios(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,                    -- NULL = global, sonst Projekt-Depot (E025)
    name TEXT NOT NULL,
    base_currency TEXT NOT NULL DEFAULT 'USD',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_portfolios_project ON portfolios(project_id);

CREATE TABLE portfolio_transactions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    ts TEXT NOT NULL,                      -- ISO-8601 UTC, Buchungszeitpunkt
    type TEXT NOT NULL CHECK(type IN ('buy','sell','dividend','cash_in','cash_out','transfer')),
    symbol TEXT,                           -- NULL nur bei cash_in/cash_out/transfer
    quantity REAL,                         -- Stuecke (buy/sell)
    price REAL,                            -- Stueckpreis (buy/sell, dividend je Stueck optional)
    amount REAL,                           -- Cashbetrag (dividend/cash_in/cash_out/transfer)
    fees REAL NOT NULL DEFAULT 0,
    taxes REAL NOT NULL DEFAULT 0,
    currency TEXT,                         -- NULL = base_currency des Depots (E031)
    transfer_portfolio_id INTEGER,         -- Gegen-Depot bei type='transfer'
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_ptx_portfolio_ts ON portfolio_transactions(portfolio_id, ts);
CREATE INDEX idx_ptx_symbol ON portfolio_transactions(symbol);

CREATE TABLE corporate_actions(
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,                      -- Ex-Tag (ISO-8601, Tagesaufloesung)
    action TEXT NOT NULL CHECK(action IN ('dividend','split')),
    value REAL NOT NULL,                   -- Dividende je Stueck bzw. Split-Ratio (z. B. 4.0)
    currency TEXT,
    source TEXT NOT NULL DEFAULT 'yahoo',
    PRIMARY KEY(symbol, ts, action)
);
CREATE INDEX idx_corp_actions_symbol_ts ON corporate_actions(symbol, ts);

CREATE TABLE fundamentals(                 -- Long-Format: 1 Zeile je Kennzahl/Periode
    symbol TEXT NOT NULL,
    statement TEXT NOT NULL CHECK(statement IN ('income','balance','cashflow')),
    freq TEXT NOT NULL CHECK(freq IN ('annual','quarterly')),
    line_item TEXT NOT NULL,               -- yfinance-Zeilenname, z. B. "Total Revenue"
    as_of TEXT NOT NULL,                   -- asOfDate "YYYY-MM-DD" (R1 Block 1.3)
    value REAL,                            -- NULL = gemeldet aber leer
    currency TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(symbol, statement, freq, line_item, as_of)
);

CREATE TABLE watchlists(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    position INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE watchlist_symbols(
    watchlist_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    position INTEGER NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY(watchlist_id, symbol)
);
CREATE INDEX idx_wls_order ON watchlist_symbols(watchlist_id, position);

CREATE TABLE alerts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN
        ('price_above','price_below','price_crossing_up','price_crossing_down','percent_move')),
    params_json TEXT NOT NULL,             -- z. B. {"price":150.0} | {"pct":5,"bars":5}
    state TEXT NOT NULL DEFAULT 'active' CHECK(state IN ('active','paused','triggered')),
    note TEXT,
    cooldown_min INTEGER NOT NULL DEFAULT 60,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_alerts_state_symbol ON alerts(state, symbol);

CREATE TABLE alert_events(                 -- Historie (T33)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL,
    triggered_at TEXT NOT NULL,
    value REAL NOT NULL,                   -- Kurs/Metrik beim Ausloesen
    message TEXT NOT NULL,
    quote_ts TEXT                          -- Alter der Quote (Live-Sicherheit)
);
CREATE INDEX idx_alert_events_alert ON alert_events(alert_id, triggered_at);

CREATE TABLE chart_drawings(               -- Zeichentools-Persistenz (E034)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chart_key TEXT NOT NULL,               -- "<SYMBOL>|<interval>" (z. B. "AAPL|1d")
    tool TEXT NOT NULL CHECK(tool IN
        ('trendline','ray','hline','vline','fib_retracement','rectangle','ellipse','brush','text')),
    points_json TEXT NOT NULL,             -- [{time:int64 us UTC, price:float}, ...]
    style_json TEXT,                       -- {color, width, fontSize? ...}
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_drawings_chart_key ON chart_drawings(chart_key);

CREATE TABLE user_strategies(              -- E026: eigene Strategien (Code in Versionen)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE user_strategy_versions(
    strategy_id INTEGER NOT NULL,
    version INTEGER NOT NULL,
    code TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(strategy_id, version)
);

CREATE TABLE screener_presets(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    rules_json TEXT NOT NULL,              -- Regel-AST (E027: Feld/Op/Wert, AND-Verknuepfung)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE symbol_search_cache(          -- Suche-Cache (T19, TTL via fetched_at)
    query TEXT PRIMARY KEY,
    results_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE data_freshness(               -- Staleness-Marker (T21/T22/E035)
    domain TEXT NOT NULL,                  -- 'daily_bars' | 'fundamentals' | 'quotes' | ...
    key TEXT NOT NULL,                     -- z. B. Symbol
    fetched_at TEXT NOT NULL,
    PRIMARY KEY(domain, key)
);

CREATE TABLE backtest_jobs(                -- Grid/Walk-Forward-Jobs (E032, genutzt ab T31)
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('grid','walkforward')),
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued','running','done','error','cancelled')),
    error TEXT,
    result_json TEXT,                      -- beste Parameter + Fenster-/Kombinationsuebersicht
    created_at TEXT NOT NULL,
    finished_at TEXT
);
"""

# Migration 3 — Spalten-Erweiterungen (§D.2); die Watchlist-Überführung läuft im
# Daten-Hook (nach_sql), damit Default-Liste, Symbole und DROP in garantierter
# Reihenfolge und in derselben Transaktion laufen.
_V2_FLAGS_SQL = """
-- delisted: 0/1 (delistierte Symbole bleiben im Bestand sichtbar)
ALTER TABLE symbols ADD COLUMN delisted INTEGER NOT NULL DEFAULT 0;
-- source: 'yahoo' | 'stooq' (E022/E023 — Herkunft der Balken)
ALTER TABLE bar_meta ADD COLUMN source TEXT NOT NULL DEFAULT 'yahoo';
"""

# Migration 4 — Websuche-Cache (T-F6/E083): reine neue Tabelle (E028 — kein
# ALTER auf Bestand), Aufbau analog ``symbol_search_cache`` (T19). Keine
# Semikolons in Kommentaren (Statement-Splitter trennt an „;“).
_WEBSUCHE_CACHE_SQL = """
CREATE TABLE websuche_cache(               -- Suche-Cache der Agent-Websuche (TTL via fetched_at)
    query TEXT PRIMARY KEY,
    results_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL
)
"""

# Migration 6 — ISIN-Brücke (T-G5/E091): ``symbols.isin`` als nullable Spalte.
# ALTER-Präzedenz auf ``symbols``: Migration 3 (``delisted``). Nullable und
# ohne Default — Alt-Bestand bleibt isin-los (NULL), die Registry füllt die
# Spalte im Betrieb (search_symbols-Treffer, Auflösungskette im Tool-Layer).
# Kein Semikolon im SQL (Statement-Splitter trennt an „;“).
_SYMBOLS_ISIN_SQL = """
ALTER TABLE symbols ADD COLUMN isin TEXT
"""


# Migration 5 — Auto-Retry-Queue (T-G3/E089): Wiederholungsplan je fehl-
# geschlagenem Download-Symbol. Eine Zeile je (Job, Symbol, Intervall);
# ``job_id`` ist der Job, dessen Fehlversuch die Zeile zuletzt schrieb (UNIQUE
# hält den Plan dedupiziert). ``naechster_versuch_us`` int64 µs UTC (E006) —
# der Retry-Tick (``downloads.faellige_retries_starten``) liest genau die
# fälligen Zeilen. Kein Semikolon im SQL (Statement-Splitter trennt an „;“).
_DOWNLOAD_RETRIES_SQL = """
CREATE TABLE download_retries(             -- Wiederholungsplan (E089/T-G3)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,                  -- Job, dessen Fehlversuch die Zeile schrieb
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    versuch INTEGER NOT NULL DEFAULT 0,    -- Fehlversuche (ab 1 nach dem ersten)
    naechster_versuch_us INTEGER NOT NULL, -- fälliger nächster Versuch (µs UTC, E006)
    letzte_fehler_code TEXT,               -- maschinenlesbarer Code des letzten Fehlers
    angelegt_us INTEGER NOT NULL,          -- Erstanlage der Zeile (µs UTC)
    UNIQUE(job_id, symbol, interval)
)
"""


# Migration 7 — PSB-Instrumente-Universum (Welle H1, E093): lokaler Spiegel
# der Planspiel-Marktgruppen (``getTabListKeyValue``/``getInstrumentsTab``,
# RECHERCHE H.2 — ~18 Gruppen, ~800 Instrumente). Eine Zeile je ISIN;
# ``gruppen`` hält die **Vereinigung** aller Tab-Zugehörigkeiten (sortiert,
# kommasepariert — Mehrfach-Zugehörigkeit ist der Normalfall, Pflege über
# :mod:`quantlab.storage.repos.psb_repo`). ``yahoo_kuerzel`` ist die E093-
# Brücke zur symbols-Registry (best effort, NULL = unaufgelöst; Nicht-Yahoo-
# Papiere nutzen die ISIN selbst als QuantLab-Symbol). Reine neue Tabelle
# (E028 — kein ALTER auf Bestand). Kein Semikolon im SQL (Statement-Splitter).
_PSB_INSTRUMENTE_SQL = """
CREATE TABLE psb_instrumente(               -- Planspiel-Universum (H1/E093)
    isin TEXT PRIMARY KEY,
    wkn TEXT,
    name TEXT,
    typ TEXT,                              -- instrumentType (STOCK, ETC, FUND, ...)
    branchen TEXT,                         -- branch-Feld der Boerse
    gruppen TEXT NOT NULL DEFAULT '',      -- Vereinigung der Tab-Zugehoerigkeit (kommasepariert)
    id_external TEXT,                      -- idExternal (Boersen-Referenz)
    waehrung TEXT,                         -- isoCurrency
    yahoo_kuerzel TEXT,                    -- E093-Bruecke ISIN → Yahoo (NULL = unaufgeloest)
    aktualisiert_am TEXT NOT NULL          -- ISO-8601 UTC (letzte Upsert-Beruehrung, E006)
)
"""


# Migration 8 — PSB-Kennzahlen-Cache (Welle H3, E095): die geparsten VWD-
# Snippet-Sektionen (``getKennzahlen``/``getKursdaten``/``getPerformance``/
# ``getAnalystenempfehlungen``/``getUnternehmensprofil``, RECHERCHE H.3) je
# Instrument. Eine Zeile je (ISIN, Sektion); ``daten`` hält die bereits
# **geparste** Sektion als JSON (keine Roh-HTML-Persistenz — der Parser ist
# der Vertrag). ISIN im UPPER-Kanon (H-1-Fix-Konvention, wie
# ``psb_instrumente``). Reine neue Tabelle (E028 — kein ALTER auf Bestand).
# Kein Semikolon im SQL (Statement-Splitter).
_PSB_KENNZAHLEN_SQL = """
CREATE TABLE psb_kennzahlen(                -- VWD-Fundamentaldaten-Cache (H3/E095)
    isin TEXT NOT NULL,                    -- UPPER-Kanon (ISIN-Bruecke)
    sektion TEXT NOT NULL,                 -- VWD-Sektion (getKennzahlen, …)
    daten TEXT NOT NULL,                   -- geparste Sektion als JSON
    aktualisiert_am TEXT NOT NULL,         -- ISO-8601 UTC (letzte Aktualisierung, E006)
    PRIMARY KEY(isin, sektion)
)
"""


# Migration 9 — Bulk-Historien-Backfill-Fortschritt (Welle H5, E097): Buch-
# haltung des resumablen Byteworx-Bulk-Laufs („5Y/1d fürs gesamte Universum“).
# **Struktur-Entscheidung (H5, dokumentiert):** Komposit-Schlüssel
# ``(isin, zeitraum)`` statt ``isin`` allein — der Zeitraum gehört fachlich
# zur getrackten Einheit (Referenz ``HistorienBackfill`` führt 5Y/1Y/1M/5D/1D
# je ISIN, RECHERCHE H.5); eine spätere Erweiterung um weitere Zeiträume
# braucht dann KEIN Schema-Änderung. Der H5-Lauf führt ausschließlich
# ``'5Y'`` (Byteworx-timeSpan der 1d-Serie). ISIN im UPPER-Kanon (H-1-Fix-
# Konvention, wie ``psb_instrumente``). ``fehlversuche``/``grund`` machen die
# 3-Fehler-Grenze **neustartfest** (ein Abbruch vergisst keine Versuche).
# ``status`` ist per CHECK auf die vier Lauf-Zustände begrenzt. Reine neue
# Tabelle (E028 — kein ALTER auf Bestand). Kein Semikolon im SQL (Splitter).
_PSB_BACKFILL_FORTSCHRITT_SQL = """
CREATE TABLE psb_backfill_fortschritt(      -- Bulk-Backfill-Fortschritt (H5/E097)
    isin TEXT NOT NULL,                    -- UPPER-Kanon (ISIN-Bruecke)
    zeitraum TEXT NOT NULL,                -- Byteworx-timeSpan ('5Y' = 1d-Serie)
    status TEXT NOT NULL DEFAULT 'offen'
        CHECK(status IN ('offen','laeuft','fertig','fehler')),
    punkte INTEGER NOT NULL DEFAULT 0,     -- Balken im Bestand (Store-Zeilen nach Upsert)
    fehlversuche INTEGER NOT NULL DEFAULT 0,  -- Fehlversuche je ISIN (max. 3)
    grund TEXT,                            -- letzter Fehlergrund deutsch (leer nach Erfolg)
    aktualisiert_am TEXT NOT NULL,         -- ISO-8601 UTC (letzte Aktualisierung, E006)
    PRIMARY KEY(isin, zeitraum)
)
"""


def _fuehre_watchlist_ueber(conn: sqlite3.Connection) -> None:
    """Daten-Hook Migration 3 (E033): Default-Liste 1 + Überführung + Legacy-Drop.

    Läuft **in derselben Transaktion** wie die ALTER-Statements: (1) Default-
    Liste ``id=1 'Watchlist'`` anlegen (ON CONFLICT DO NOTHING — der Hook ist
    damit auch ohne Protokollzeile wiederholbar), (2) alle v1-``watchlist``-
    Zeilen mit Position **und** ``added_at`` in Liste 1 überführen
    (INSERT OR IGNORE ↔ PK (watchlist_id, symbol)), (3) Legacy-Tabelle droppen.
    Die Reihenfolge ist die geforderte „Reihenfolge-Garantie" aus §D.2.
    """
    conn.execute(
        "INSERT INTO watchlists(id, name, position, created_at) "
        "VALUES (1, 'Watchlist', 1, ?) ON CONFLICT(id) DO NOTHING",
        (db.utc_now_iso(),),
    )
    conn.execute(
        "INSERT OR IGNORE INTO watchlist_symbols(watchlist_id, symbol, position, added_at) "
        "SELECT 1, symbol, position, added_at FROM watchlist ORDER BY position"
    )
    conn.execute("DROP TABLE IF EXISTS watchlist")


@dataclass(frozen=True)
class Migration:
    """Eine versionierte Schema-Migration.

    ``nach_sql`` ist ein optionaler Daten-Hook, der **in derselben Transaktion**
    nach den Statements läuft (Migration 3: Watchlist-Überführung — ``<now>``
    aus :func:`db.utc_now_iso` und Reihenfolge-Garantie, §D.2-Mechanik).
    """

    version: int
    name: str
    sql: str
    nach_sql: Callable[[sqlite3.Connection], None] | None = None


MIGRATIONEN: list[Migration] = [
    Migration(version=1, name="kernschema_d1", sql=_KERNSCHEMA_SQL),
    Migration(version=2, name="v2_tabellen", sql=_V2_TABELLEN_SQL),
    Migration(version=3, name="v2_flags_und_watchlist", sql=_V2_FLAGS_SQL,
              nach_sql=_fuehre_watchlist_ueber),
    Migration(version=4, name="websuche_cache", sql=_WEBSUCHE_CACHE_SQL),
    Migration(version=5, name="download_retries", sql=_DOWNLOAD_RETRIES_SQL),
    Migration(version=6, name="symbols_isin", sql=_SYMBOLS_ISIN_SQL),
    Migration(version=7, name="psb_instrumente", sql=_PSB_INSTRUMENTE_SQL),
    Migration(version=8, name="psb_kennzahlen", sql=_PSB_KENNZAHLEN_SQL),
    Migration(version=9, name="psb_backfill_fortschritt", sql=_PSB_BACKFILL_FORTSCHRITT_SQL),
]


def _statements(sql: str) -> list[str]:
    """Splittet ein Migrations-Skript in einzelne Statements (Semikolon-getrennt).

    ANNAHME: Migrations-Skripte enthalten keine Semikolons innerhalb eines
    Statements (keine Trigger/SQL-Strings) — für das Schema aus §D.1 erfüllt.
    """
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


def _ist_angewendet(conn: sqlite3.Connection, version: int) -> bool:
    row = conn.execute("SELECT 1 FROM schema_migrations WHERE version = ?", (version,)).fetchone()
    return row is not None


def _datenbank_pfad(conn: sqlite3.Connection) -> Path | None:
    """Dateipfad der ``main``-Datenbank der Verbindung — ``None`` bei In-Memory."""
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None:
        return None
    datei = str(row[2])  # Spalte „file" (main = erste Zeile)
    return Path(datei) if datei else None


def _sichere_v1_bestand(conn: sqlite3.Connection) -> None:
    """Einmalige Dateisicherung ``<db>.v1.bak`` vor der ersten v2-Migration.

    Bestandsschutz (§D.2-Mechanik): Die Sicherung wird nur angelegt, wenn sie
    noch nicht existiert — ein zweiter Migrationslauf überschreibt sie nie.
    Geschrieben wird **atomar**: erst in eine eindeutige temporäre Datei
    (:func:`tempfile.mkstemp`, O_EXCL — parallele Migrationsläufe kollidieren
    nicht) neben der DB, dann per :func:`os.replace` umbenannt. Schlägt
    ``connect``/``backup``/``replace`` fehl, bleibt keine partielle ``.bak``
    liegen, die den ``exists()``-Guard (und damit jeden Retry) blockieren
    würde; die temporäre Datei wird im Fehlerpfad entfernt. Die SQLite-Online-
    Backup-API (:meth:`sqlite3.Connection.backup`) liefert auch unter WAL
    einen konsistenten Stand. In-Memory-Datenbanken (Tests ohne Datei) und
    nicht existierende Dateien werden übersprungen.
    """
    db_pfad = _datenbank_pfad(conn)
    if db_pfad is None or not db_pfad.exists():
        return
    backup_pfad = db_pfad.with_name(f"{db_pfad.name}.v1.bak")
    if backup_pfad.exists():
        return
    fd, temp_name = tempfile.mkstemp(
        dir=db_pfad.parent, prefix=f"{db_pfad.name}.v1.bak.", suffix=".tmp"
    )
    os.close(fd)  # sqlite3 öffnet die Datei selbst; nur Name/Platz reservieren
    temp_pfad = Path(temp_name)
    try:
        ziel = sqlite3.connect(temp_pfad)
        try:
            conn.backup(ziel)
        finally:
            ziel.close()
        os.replace(temp_pfad, backup_pfad)  # atomar (rename im selben Verzeichnis)
    except BaseException:
        temp_pfad.unlink(missing_ok=True)  # keine partielle Datei liegen lassen
        raise


def run_migrations(conn: sqlite3.Connection | None = None) -> int:
    """Wendet alle ausstehenden Migrationen an (idempotent) — Anzahl neu angewendet.

    Ohne ``conn`` wird eine eigene, kurzlebige Verbindung geöffnet (Muster:
    :mod:`quantlab.db`). Bereits protokollierte Versionen werden übersprungen,
    daher ist ein zweiter Lauf ein No-Op mit identischem Schema-Hash. Vor der
    ersten Anwendung einer Version ≥ 2 wird einmalig ``<db>.v1.bak`` gesichert.

    Parallelschutz: Der Applied-Check wird **in** der ``BEGIN IMMEDIATE``-
    Transaktion wiederholt — hat ein paralleler Lauf die Version zwischen dem
    Check darüber und dem Lock-Zugriff angewendet, wird sie klaglos übersprungen
    (COMMIT ohne Statements) statt mit ``OperationalError`` zu sterben. Die
    Migrationen selbst (IDs, DDL) bleiben davon unberührt (E028).
    """
    own_connection = conn is None
    if own_connection:
        conn = db.connect()
    assert conn is not None  # für die Typprüfung (conn ist sonst nie None)
    try:
        conn.execute(_SCHEMA_MIGRATIONS_SQL)
        applied = 0
        for migration in MIGRATIONEN:
            if _ist_angewendet(conn, migration.version):
                continue
            if migration.version >= 2:
                _sichere_v1_bestand(conn)
            conn.execute("BEGIN IMMEDIATE")
            try:
                # TOCTOU: Ein paralleler Lauf kann die Version zwischen dem
                # Check oben und BEGIN IMMEDIATE angewendet haben — der Check
                # wird deshalb in der Transaktion (unter Schreib-Lock) wiederholt.
                if _ist_angewendet(conn, migration.version):
                    conn.execute("COMMIT")  # inzwischen angewendet — nichts zu tun
                    continue
                for statement in _statements(migration.sql):
                    conn.execute(statement)
                if migration.nach_sql is not None:
                    migration.nach_sql(conn)
                conn.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.name, db.utc_now_iso()),
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            applied += 1
        return applied
    finally:
        if own_connection:
            conn.close()


def schema_hash(conn: sqlite3.Connection) -> str:
    """Deterministischer SHA-256-Hash über alle Nutzertabellen (Name + DDL, sortiert).

    Internes SQLite (``sqlite_%``) bleibt außen vor; der Hash ist für
    Idempotenz-Tests gedacht (gleicher Schema-Stand → gleicher Hash).
    """
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    material = "\n".join(f"{name}:{(ddl or '').strip()}" for name, ddl in rows)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
