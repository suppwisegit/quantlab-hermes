"""SQLite-Verbindungs-Helper für den Persistenz-Kern (E005).

Zugriffsmuster (bindend für alle Repos und Migrationen):

- **Eine Verbindung pro Operation:** Jede Repo-/Migrationsfunktion öffnet eine
  eigene, kurzlebige Verbindung (:func:`connect`) und schließt sie wieder.
  Verbindungen werden nie über Thread-Grenzen geteilt
  (``check_same_thread`` bleibt auf dem sicheren Standard ``True``).
- **Sync-Aufrufe im Threadpool:** ``sqlite3`` ist synchron. FastAPI-Pfad-
  Operationen werden deshalb als normale ``def`` (nicht ``async def``)
  deklariert — FastAPI führt sie in seinem Threadpool aus, der asyncio-
  Event-Loop wird nicht blockiert. ``asyncio.to_thread`` ist nur nötig, wenn
  ein ``async``-Endpunkt doch direkt auf ein Repo zugreift.
- **WAL (RECHERCHE Block 6):** ``journal_mode=WAL`` erlaubt parallele Leser
  neben einer offenen Writer-Transaktion; ``busy_timeout`` entschärft kurze
  Kollisionen zusätzlich. WAL setzt ein lokales Dateisystem voraus (R6 —
  nicht für Netz-Freigaben).
- **Transaktionen:** Autocommit-Modus (``isolation_level=None``); Transaktionen
  werden explizit gesteuert. Bindendes Muster für alle schreibenden Funktionen
  (Repos **und** Migrationen):
  1. Schreib-Transaktionen immer mit ``BEGIN IMMEDIATE`` eröffnen — deferred
     ``BEGIN`` + Read-modify-write führt unter Konkurrenz zu
     SQLITE_BUSY_SNAPSHOT („database is locked") erst beim COMMIT/Upgrade;
     ``IMMEDIATE`` holt den Schreib-Lock sofort und serialisiert sauber
     (``busy_timeout`` trägt die Wartezeit).
  2. ``ROLLBACK`` im Fehlerpfad nur, wenn eine Transaktion offen ist
     (``conn.in_transaction``) — wirft bereits ``BEGIN``, würde ein blindes
     ROLLBACK den Originalfehler maskieren.
  Referenzimplementierungen: ``migrations.run_migrations``,
  ``settings_repo.update_section``, ``symbols_repo.upsert_symbol/delete_symbol``.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from quantlab.config import get_database_path

# Gerätelokale Desktop-App: kurze Wartezeit auf gesperrte DB genügt; WAL macht
# Lock-Kollisionen zwischen Lesern und dem einen Writer ohnehin selten.
SQLITE_TIMEOUT_S = 5.0
_BUSY_TIMEOUT_MS = 5000

#: Kurze, ansteigende Pausen (Sekunden) für den WAL-Umschalt-Retry (M1).
_WAL_RETRY_PAUSEN_S = (0.05, 0.1, 0.2)


def _pragma_wal_aktivieren(conn: sqlite3.Connection) -> None:
    """``journal_mode=WAL`` mit kurzem Retry gegen den Kaltstart-Race (M1).

    Das Umschalten auf WAL braucht kurz exklusiven Zugriff; verbinden sich zwei
    Prozesse zeitgleich zum ERSTEN Mal mit einer frischen DB, liefert diese
    Operation trotz ``busy_timeout`` sofort BUSY — der Busy-Handler greift beim
    Journal-Mode-Wechsel nicht zuverlässig (Re-Review T19, ~3/9 beobachtet;
    reproduzierbar im Paralleltest). Drei kurze, ansteigende Versuche entzerren
    das; ist WAL bereits gesetzt, ist das Pragma ohnehin ein No-Op."""
    for pause in _WAL_RETRY_PAUSEN_S:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            text = str(exc).lower()
            if "locked" not in text and "busy" not in text:
                raise  # kein Lock-Konflikt (z. B. io_error) — nicht wegretryen
            time.sleep(pause)
    conn.execute("PRAGMA journal_mode=WAL")  # letzter Versuch, Fehler darf fliegen


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Öffnet eine neue Verbindung mit den Projekt-Pragmas (WAL, Row-Factory).

    Legt das übergeordnete Verzeichnis bei Bedarf an. ``path=None`` nutzt den
    kanonischen Pfad aus :func:`quantlab.config.get_database_path` (respektiert
    den ``QUANTLAB_DATA_DIR``-Override). Der Aufrufer schließt die Verbindung.
    """
    db_path = path if path is not None else get_database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_S, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # busy_timeout VOR journal_mode=WAL setzen (M1, Re-Review T19): Das
    # Umschalten auf WAL braucht kurz den Schreib-Lock — verbinden sich zwei
    # Prozesse zeitgleich zum ERSTEN Mal mit einer frischen DB (2-Prozess-
    # Kaltstart, ~3/9 beobachtet), kann der zweite sonst an ``PRAGMA
    # journal_mode`` BUSY sterben, statt die Wartezeit zu nutzen.
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    _pragma_wal_aktivieren(conn)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def utc_now_iso() -> str:
    """UTC-Zeitstempel als ISO-8601-Text — kanonisches Format für TEXT-Spalten (E006)."""
    return datetime.now(UTC).isoformat()
