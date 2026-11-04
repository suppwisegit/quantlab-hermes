"""Backtest-Repository: ``backtest_results`` (D.1) + ``backtest_jobs`` (v2 §D.2).

Ein Lauf wird synchron ausgeführt (T10: Backtests sind schnell, kein Job-
System) und vollständig persistiert: Strategie, Symbol/Intervall, Parameter,
Metriken (JSON), Equity-Kurve (JSON, bereits auf ≤ 2000 Punkte resampled)
und Trades (JSON). ``run_id`` ist eine vom Router vergebene UUID.

T29-Anreicherung: Drawdown-Kurve, Monatsrenditen, Periode/Symbol-Alignment,
Fills und Benchmark werden im reservierten Schlüssel ``_enrichment`` **in der
bestehenden Spalte** ``metrics_json`` mitgeführt (keine Schema-Änderung) —
das Repo trennt Kennzahlen und Anreicherung beim Lesen sauber
(:attr:`BacktestRun.enrichment`); Bestandsleser sehen unveränderte Metriken.

v2 (E032, genutzt ab T31): Grid- und Walk-Forward-Optimierungen laufen als
asynchrone Jobs (``backtest_jobs``) — Buchhaltung analog ``downloads_repo``
(Status-Werte, Terminal-Status setzen ``finished_at``, ``result_json`` nimmt
die besten Parameter + Fenster-/Kombinationsübersicht auf). Der Job-Runner
(``services/backtest_jobs.py``) nutzt ausschließlich dieses Repo.

SQLite-Zugriffe nach dem Verbindungs-Muster aus :mod:`quantlab.db`
(``BEGIN IMMEDIATE``, konditionales ``ROLLBACK``, Verbindung pro Operation).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from quantlab import db

#: Reservierter Schlüssel in ``metrics_json`` für den T29-Anreicherungs-Block.
ENRICHMENT_KEY = "_enrichment"

# ---------------------------------------------------------------------------
# Backtest-Jobs (v2 §D.2, E032 — Grid/Walk-Forward, Runner folgt in T31)
# ---------------------------------------------------------------------------

#: Job-Arten (CHECK-Constraint §D.2).
JOB_KINDS: frozenset[str] = frozenset({"grid", "walkforward"})

#: Job-Lebenszyklus (analog Downloads: queued→running→done | error | cancelled).
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_ERROR = "error"
JOB_STATUS_CANCELLED = "cancelled"

JOB_STATUS: frozenset[str] = frozenset(
    {
        JOB_STATUS_QUEUED,
        JOB_STATUS_RUNNING,
        JOB_STATUS_DONE,
        JOB_STATUS_ERROR,
        JOB_STATUS_CANCELLED,
    }
)

#: Rangfolge für den Vorwärts-Guard (:func:`update_job_status_vorwaerts`) —
#: Terminal-Status teilen sich den höchsten Rang (unumkehrbar, keine
#: Rangfolge untereinander; Muster: ``downloads_repo._STATUS_RANG``).
_JOB_STATUS_RANG: dict[str, int] = {
    JOB_STATUS_QUEUED: 0,
    JOB_STATUS_RUNNING: 1,
    JOB_STATUS_DONE: 2,
    JOB_STATUS_ERROR: 2,
    JOB_STATUS_CANCELLED: 2,
}

#: Terminal-Status setzen ``finished_at`` (nicht-Terminal-Updates lassen es unberührt).
_JOB_TERMINAL: frozenset[str] = frozenset(
    {JOB_STATUS_DONE, JOB_STATUS_ERROR, JOB_STATUS_CANCELLED}
)


@dataclass(frozen=True)
class BacktestJob:
    """Ein Eintrag der backtest_jobs-Tabelle (Optimierungs-Job-Buchhaltung)."""

    job_id: str
    kind: str  # grid | walkforward
    payload: dict[str, Any]
    status: str  # queued | running | done | error | cancelled
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = ""
    finished_at: str | None = None


@dataclass(frozen=True)
class BacktestRun:
    """Ein Eintrag der backtest_results-Tabelle (D.1, ohne Laufzeitlogik)."""

    run_id: str
    project_id: int | None
    strategy: str
    symbol: str
    interval: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    equity: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    created_at: str
    #: T29-Anreicherung (Drawdown/Monate/Periode/Fills/Benchmark) — leer bei
    #: Bestandszeilen ohne ``_enrichment``-Block.
    enrichment: dict[str, Any] = field(default_factory=dict)


def _loads_json(text: str, default: Any) -> Any:
    """JSON-Spalte tolerant dekodieren (defekte Zeile werfen nicht)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _row_to_run(row: sqlite3.Row) -> BacktestRun:
    """sqlite3.Row → :class:`BacktestRun` (JSON-Spalten dekodiert)."""
    project_raw = row["project_id"]
    params = _loads_json(row["params_json"], {})
    metrics_raw = _loads_json(row["metrics_json"], {})
    equity = _loads_json(row["equity_json"], [])
    trades = _loads_json(row["trades_json"], [])
    enrichment: dict[str, Any] = {}
    if isinstance(metrics_raw, dict):
        block = metrics_raw.pop(ENRICHMENT_KEY, None)
        if isinstance(block, dict):
            enrichment = block
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    return BacktestRun(
        run_id=row["id"],
        project_id=None if project_raw is None else int(project_raw),
        strategy=row["strategy"],
        symbol=row["symbol"],
        interval=row["interval"],
        params=params if isinstance(params, dict) else {},
        metrics=metrics,
        equity=equity if isinstance(equity, list) else [],
        trades=trades if isinstance(trades, list) else [],
        created_at=row["created_at"],
        enrichment=enrichment,
    )


def create_run(
    run_id: str,
    *,
    project_id: int | None,
    strategy: str,
    symbol: str,
    interval: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    equity: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    enrichment: dict[str, Any] | None = None,
) -> BacktestRun:
    """Persistiert einen Backtest-Lauf und liefert ihn zurück.

    ``enrichment`` (T29, optional) wird im reservierten Schlüssel
    ``ENRICHMENT_KEY`` innerhalb von ``metrics_json`` abgelegt (Tabelle
    unverändert) und beim Lesen wieder als eigenes Feld getrennt.
    """
    created_at = db.utc_now_iso()
    metrics_payload: dict[str, Any] = dict(metrics)
    if enrichment:
        metrics_payload[ENRICHMENT_KEY] = dict(enrichment)
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — einheitliches Schreibmuster.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO backtest_results(id, project_id, strategy, symbol, interval, "
            "params_json, metrics_json, equity_json, trades_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                project_id,
                strategy,
                symbol,
                interval,
                json.dumps(params, ensure_ascii=False, sort_keys=True),
                json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True),
                json.dumps(equity, ensure_ascii=False),
                json.dumps(trades, ensure_ascii=False),
                created_at,
            ),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return BacktestRun(
        run_id=run_id,
        project_id=project_id,
        strategy=strategy,
        symbol=symbol,
        interval=interval,
        params=dict(params),
        metrics=dict(metrics),
        equity=list(equity),
        trades=list(trades),
        created_at=created_at,
        enrichment=dict(enrichment) if enrichment else {},
    )


def get_run(run_id: str) -> BacktestRun | None:
    """Ein Lauf anhand seiner ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM backtest_results WHERE id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _row_to_run(row)


def list_runs(limit: int = 50, project_id: int | None = None) -> list[BacktestRun]:
    """Läufe, neueste zuerst (``created_at`` DESC, ``rowid`` als Tiebreaker).

    Optional auf ein Projekt gefiltert. Liefert vollständige Zeilen — die
    Liste-Antwort des Routers trennt Equity/Trades selbst ab.
    """
    query = "SELECT * FROM backtest_results"
    clauses: list[str] = []
    args: list[Any] = []
    if project_id is not None:
        clauses.append("project_id = ?")
        args.append(int(project_id))
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    args.append(int(limit))
    conn = db.connect()
    try:
        rows = conn.execute(query, args).fetchall()
    finally:
        conn.close()
    return [_row_to_run(row) for row in rows]


def delete_run(run_id: str) -> bool:
    """Löscht einen Lauf; ``True``, wenn er existierte."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute("DELETE FROM backtest_results WHERE id = ?", (run_id,))
        deleted = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return deleted


# ---------------------------------------------------------------------------
# Jobs: Grid/Walk-Forward (v2 §D.2, E032 — Buchhaltung analog downloads_repo)
# ---------------------------------------------------------------------------


def _row_to_job(row: sqlite3.Row) -> BacktestJob:
    """sqlite3.Row → :class:`BacktestJob` (JSON-Spalten tolerant dekodiert)."""
    payload = _loads_json(row["payload_json"], {})
    result_raw = row["result_json"]
    result = _loads_json(result_raw, None) if result_raw else None
    return BacktestJob(
        job_id=row["id"],
        kind=row["kind"],
        payload=payload if isinstance(payload, dict) else {},
        status=row["status"],
        error=row["error"],
        result=result if isinstance(result, dict) else None,
        created_at=row["created_at"],
        finished_at=row["finished_at"],
    )


def create_job(job_id: str, kind: str, payload: dict[str, Any]) -> BacktestJob:
    """Legt einen Optimierungs-Job an (Status ``queued``) und liefert ihn zurück.

    ``ValueError`` bei unbekannter Job-Art (CHECK-Spiegelung mit deutscher Meldung).
    """
    if kind not in JOB_KINDS:
        raise ValueError(f"Unbekannte Job-Art: {kind!r}")
    created_at = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO backtest_jobs(id, kind, payload_json, status, error, "
            "result_json, created_at, finished_at) "
            "VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL)",
            (
                job_id,
                kind,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                JOB_STATUS_QUEUED,
                created_at,
            ),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return BacktestJob(
        job_id=job_id,
        kind=kind,
        payload=dict(payload),
        status=JOB_STATUS_QUEUED,
        created_at=created_at,
    )


def get_job(job_id: str) -> BacktestJob | None:
    """Ein Job anhand seiner ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM backtest_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _row_to_job(row)


def list_jobs(limit: int = 50) -> list[BacktestJob]:
    """Jobs, neueste zuerst (``created_at`` DESC, ``rowid`` als Tiebreaker)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT *, rowid FROM backtest_jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_job(row) for row in rows]


def update_job_status(
    job_id: str,
    status: str,
    *,
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> bool:
    """Setzt Status (und optional Fehler/Ergebnis); ``True`` wenn der Job existierte.

    Terminal-Status (done/error/cancelled) schreiben ``finished_at``; sonstige
    Updates lassen es unverändert (Muster: ``downloads_repo.update_status``).
    ``result`` wird nur geschrieben, wenn es übergeben wird — typischerweise mit
    dem Terminal-Status ``done`` (beste Parameter + Übersicht, E032).
    ``ValueError`` bei unbekanntem Status.
    """
    if status not in JOB_STATUS:
        raise ValueError(f"Unbekannter Job-Status: {status!r}")
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE backtest_jobs SET status = ?, error = ?, "
            "result_json = CASE WHEN ? = 1 THEN ? ELSE result_json END, "
            "finished_at = CASE WHEN ? = 1 THEN ? ELSE finished_at END "
            "WHERE id = ?",
            (
                status,
                error,
                1 if result is not None else 0,
                json.dumps(result, ensure_ascii=False, sort_keys=True)
                if result is not None
                else None,
                1 if status in _JOB_TERMINAL else 0,
                db.utc_now_iso(),
                job_id,
            ),
        )
        updated = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return updated


def update_job_status_vorwaerts(
    job_id: str,
    status: str,
    *,
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> bool:
    """Setzt den Job-Status nur aufwärts — Monotonie-Guard in derselben Transaktion.

    T31 (E032): Grundlage für den kooperativen Abbruch der Grid/Walk-Forward-
    Jobs. Der Rang-Vergleich läuft als ``CASE`` direkt in der ``WHERE``-Klausel
    des ``UPDATE`` — Prüfen und Schreiben sind **eine** atomare Operation
    (``BEGIN IMMEDIATE``); ein gleichzeitiges Cancel und das Job-Ende können
    sich nicht überholen (``done`` nach ``cancelled`` ist ausgeschlossen).
    Muster: ``downloads_repo.update_status_vorwaerts``. ``result`` (beste
    Parameter + Übersicht) wird wie in :func:`update_job_status` nur bei
    Übergabe geschrieben — typischerweise zusammen mit ``done`` in **einer**
    atomaren Transaktion. ``True`` nur, wenn der Übergang akzeptiert wurde
    (Job existiert und neuer Status hat strikt höheren Rang). Unbekannter
    Ziel-Status → ``ValueError``.
    """
    ziel_rang = _JOB_STATUS_RANG.get(status)
    if ziel_rang is None:
        raise ValueError(f"Unbekannter Job-Status: {status!r}")
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE backtest_jobs SET status = ?, error = ?, "
            "result_json = CASE WHEN ? = 1 THEN ? ELSE result_json END, "
            "finished_at = CASE WHEN ? = 1 THEN ? ELSE finished_at END "
            "WHERE id = ? AND ("
            "  CASE status WHEN ? THEN 0 WHEN ? THEN 1 ELSE 2 END"
            ") < ?",
            (
                status,
                error,
                1 if result is not None else 0,
                json.dumps(result, ensure_ascii=False, sort_keys=True)
                if result is not None
                else None,
                1 if status in _JOB_TERMINAL else 0,
                db.utc_now_iso(),
                job_id,
                JOB_STATUS_QUEUED,
                JOB_STATUS_RUNNING,
                ziel_rang,
            ),
        )
        akzeptiert = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return akzeptiert
