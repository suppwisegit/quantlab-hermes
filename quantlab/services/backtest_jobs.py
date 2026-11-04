"""Backtest-Optimierungs-Jobs (T31, E032): Grid und Walk-Forward als Jobs.

Konventionen (E032, LEAN-orientiert):

- **Grid:** kartesisches Produkt der Parameterbereiche, harte Obergrenze
  :data:`GRID_CAP` = 200 Kombinationen (überschritten → 422 ``grid_zu_gross``).
  Jede Kombination läuft über **denselben Kern** wie ``POST /api/backtest/runs``
  (Router ``_fuehre_request_aus``) — das Ergebnisverhalten ist dadurch
  identisch (inkl. User-Strategie-Verdrahtung und Gewichte-422).
- **Walk-Forward:** rollierend (nicht anchored) — Train = N Balken,
  Test = M Balken, Schrittweite = M; Fenster beginnen am Datenanfang;
  **mindestens** :data:`WF_MIN_FENSTER` = 2 vollständige Fenster, sonst 422
  (:func:`walkforward_fenster`). Die Signale werden je Kombination **einmal**
  über die Gesamthistorie erzeugt und je Fenster nur geschnitten — der
  Gewichte-Vertrag (Zeile i aus Daten ≤ i) macht das look-ahead-frei, und
  User-Strategien brauchen so nur eine Sandbox-Ausführung je Symbol.
- **Zeitraum:** ``start``/``end`` im Job-Payload sind ISO-Datum/-Zeit oder
  int64 µs UTC (Vertrag ``BacktestRunRequest``) und werden vor ``_lade_daten``
  einheitlich via :func:`zeitstempel_zu_us` normalisiert — der Router delegiert
  auf dieselbe Funktion (T31-QA-FUND 1, keine Duplikat-Logik).
- **Zielfunktion der Parameterwahl = Ø Out-of-Sample-Sharpe** — nie In-Sample
  (Kurven-Fitting-Abwehr). Fallback je Fenster: OOS-Total-Return, wenn der
  OOS-Sharpe ``None`` ist (z. B. Vol = 0 in flachen Fenstern; dokumentierte
  ANNAHME: der Fallback greift fensterweise, der Zielwert mittelt dann über
  die Einzelfenster-Werte). Das OOS-Segment misst ab der Equity-Basis des
  letzten Train-Balkens — die Train-Phase dient als Signal-Warm-up, bewertet
  wird ausschließlich Out-of-Sample.
- **Aggregation:** Kennzahlen je Fenster **und** die verkettete OOS-Gesamt-
  sequenz aller Test-Fenster der besten Kombination (je Fenster auf das
  Vorfenster-Ende rebased) — persistiert als Lauf in ``backtest_results``
  (``run_id`` im Job-Ergebnis, „Lauf-ID-verbunden“ E032).

Jobs laufen als ``backtest_jobs``-Zeilen (Repo T18) mit dem Status-Vertrag
``queued→running→done|error|cancelled``, serialisiert je Event-Loop (Muster
Download-Schleuse, R15), kooperativem Abbruch zwischen Kombinationen/Fenstern
und SSE-Fortschritt auf dem globalen Bus (§C.5):
``backtest_job_progress {jobId, done, total, phase}`` und
``backtest_job_finished {jobId, ok, error?}``. Seit B-Fix-4 (E056) geht eine
Daten-Vorbereitungsphase mit ``phase: "preparing_data"`` voraus (nur bei
explizitem ``auto_download`` im Payload — die API-Endpunkte setzen es via
``model_dump``), Details :func:`_bereite_job_daten`.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from quantlab import events
from quantlab.errors import ApiError
from quantlab.storage.repos import backtest_repo

logger = logging.getLogger(__name__)

#: Harte Obergrenze der Grid-Kombinationen (E032).
GRID_CAP = 200

#: Mindestanzahl vollständiger Walk-Forward-Fenster (E032).
WF_MIN_FENSTER = 2

#: SSE-Event-Namen (§C.5) — bewusst hier (statt events.py) definiert, damit
#: die T31-Verträge an einem Ort stehen; der Bus transportiert beliebige Namen.
BACKTEST_JOB_PROGRESS = "backtest_job_progress"
BACKTEST_JOB_FINISHED = "backtest_job_finished"

#: Job-spezifische Payload-Schlüssel, die nicht in ``BacktestRunRequest`` gehören.
_JOB_ZUSATZ_SCHLUESSEL = ("grid", "train_bars", "test_bars")


def zeitstempel_zu_us(value: str | int | None, feldname: str) -> int | None:
    """ISO-Datum/-Zeit oder µs (int bzw. Ziffern-String) → int64 µs UTC.

    Ein Vertrag an **einer** Stelle (T31-QA-FUND 1): ``start``/``end`` der
    Backtest-Requests (synchroner Lauf, Grid- und Walk-Forward-Job) akzeptieren
    ISO-Datum/-Zeit oder int64 Mikrosekunden UTC — vorher hier, damit der
    WF-Job denselben Vertrag erfüllt wie der Router, der delegiert.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ApiError(
            422, "ungueltiger_zeitraum", f"„{feldname}“ muss Datum oder µs-Zeitstempel sein."
        )
    if isinstance(value, int):
        return value
    text = value.strip()
    if text.lstrip("-").isdigit():  # µs als String (JS-Number-Verlust umgehen)
        return int(text)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApiError(
            422,
            "ungueltiger_zeitraum",
            f"„{feldname}“ ist weder ISO-Datum/-Zeit noch µs-Zeitstempel: „{value}“.",
            hint="Beispiele: 2023-01-01, 2023-01-01T09:30:00Z.",
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)  # ANNAHME: naive Angaben sind UTC
    return int(moment.timestamp()) * 1_000_000


# ---------------------------------------------------------------------------
# Grid-Kombinationen (kartesisches Produkt, Cap 200)
# ---------------------------------------------------------------------------


def grid_kombinationen(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Kartesisches Produkt der Parameterbereiche (deterministische Reihenfolge).

    Leeres Grid → ``[{}]`` (genau eine Kombination = Basis-Parameter). Eine
    leere Werteliste und mehr als :data:`GRID_CAP` Kombinationen → 422 mit
    deutscher Meldung (E032).
    """
    if not grid:
        return [{}]
    for name, werte in grid.items():
        if not isinstance(werte, list) or not werte:
            raise ApiError(
                422,
                "grid_leer",
                f"Der Parameterbereich „{name}“ enthält keine Werte.",
                hint="Mindestens einen Wert je Parameterbereich angeben.",
            )
    namen = list(grid.keys())
    kombis = [
        dict(zip(namen, werte, strict=True))
        for werte in itertools.product(*(grid[name] for name in namen))
    ]
    if len(kombis) > GRID_CAP:
        raise ApiError(
            422,
            "grid_zu_gross",
            f"Das Parameter-Grid umfasst {len(kombis)} Kombinationen "
            f"(Obergrenze: {GRID_CAP}).",
            hint="Bereiche verkleinern oder die Schrittweite erhöhen.",
        )
    return kombis


def walkforward_fenster(
    gesamt: int, train_bars: int, test_bars: int
) -> list[tuple[int, int, int, int]]:
    """Fensterliste des rollierenden Walk-Forward (E032) — Goldwert-Vertrag.

    Rollierend (nicht anchored), Schrittweite = ``test_bars``, Fenster beginnen
    am Datenanfang: Fenster ``i`` trainiert auf Balken ``[i·M, i·M + N)`` und
    testet auf ``[i·M + N, i·M + N + M)``. Vollständig ist ein Fenster, wenn
    sein Test-Ende innerhalb der Daten liegt. Weniger als
    :data:`WF_MIN_FENSTER` vollständige Fenster → 422.

    Rückgabe: ``[(train_start, train_end, test_start, test_end), …]`` als
    Balken-Indizes (``*_end`` exklusiv) — deterministisch, datenunabhängig.
    """
    if isinstance(train_bars, bool) or not isinstance(train_bars, int) or train_bars < 1:
        raise ApiError(
            422,
            "walkforward_parameter",
            "„train_bars“ (N) muss eine ganze Zahl ≥ 1 sein.",
        )
    if isinstance(test_bars, bool) or not isinstance(test_bars, int) or test_bars < 1:
        raise ApiError(
            422,
            "walkforward_parameter",
            "„test_bars“ (M) muss eine ganze Zahl ≥ 1 sein.",
        )
    n = int(gesamt)
    anzahl = (n - train_bars) // test_bars if n > train_bars else 0
    if anzahl < WF_MIN_FENSTER:
        raise ApiError(
            422,
            "zu_wenige_wf_fenster",
            f"Die Historie umfasst {n} Balken — für rollierendes Walk-Forward "
            f"(Train {train_bars}, Test {test_bars}) entstehen nur {anzahl} "
            f"vollständige Fenster (mindestens {WF_MIN_FENSTER} erforderlich).",
            hint="Train/Test verkürzen oder die Historie erweitern.",
        )
    return [
        (
            i * test_bars,
            i * test_bars + train_bars,
            i * test_bars + train_bars,
            i * test_bars + train_bars + test_bars,
        )
        for i in range(anzahl)
    ]


def oos_fenster_wert(oos_sharpe: float | None, oos_total_return: float | None) -> float:
    """Zielwert eines OOS-Fensters: Sharpe, Fallback Total-Return bei ``None`` (E032)."""
    if oos_sharpe is not None:
        return float(oos_sharpe)
    return float(oos_total_return) if oos_total_return is not None else 0.0


# ---------------------------------------------------------------------------
# Job-Verwaltung (Muster: services/downloads.py — Schleuse je Event-Loop)
# ---------------------------------------------------------------------------


@dataclass
class _JobHandle:
    """Laufzeit-Handle eines Optimierungs-Jobs (Abbruch-Event + Task)."""

    abbruch: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


@dataclass
class _LoopZustand:
    """Serialisierung je Event-Loop: Optimierungs-Jobs laufen nacheinander (R15)."""

    schleuse: asyncio.Lock = field(default_factory=asyncio.Lock)


_LOOPZUSTAENDE: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopZustand] = (
    weakref.WeakKeyDictionary()
)

#: Offene Handles je job_id (nur nicht-terminaler Lebensabschnitt).
_JOBS: dict[str, _JobHandle] = {}

_TERMINAL: frozenset[str] = frozenset(
    {
        backtest_repo.JOB_STATUS_DONE,
        backtest_repo.JOB_STATUS_ERROR,
        backtest_repo.JOB_STATUS_CANCELLED,
    }
)


def _loop_zustand() -> _LoopZustand:
    """Zustand (Schleuse) des laufenden Loops (Sidecar hat genau einen)."""
    loop = asyncio.get_running_loop()
    zustand = _LOOPZUSTAENDE.get(loop)
    if zustand is None:
        zustand = _LoopZustand()
        _LOOPZUSTAENDE[loop] = zustand
    return zustand


def job_als_dict(job: backtest_repo.BacktestJob) -> dict[str, Any]:
    """Job-Zeile → Antwort-Darstellung (GET-Vertrag)."""
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "error": job.error,
        "result": job.result,
        "payload": job.payload,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }


def _progress(job_id: str, done: int, total: int, phase: str) -> None:
    events.publizieren(
        BACKTEST_JOB_PROGRESS,
        {"jobId": job_id, "done": done, "total": total, "phase": phase},
    )


def _finished(job_id: str, ok: bool, error: str | None = None) -> None:
    daten: dict[str, Any] = {"jobId": job_id, "ok": ok}
    if error:
        daten["error"] = error
    events.publizieren(BACKTEST_JOB_FINISHED, daten)


async def _status_error(job_id: str, meldung: str) -> None:
    await asyncio.to_thread(
        backtest_repo.update_job_status_vorwaerts,
        job_id,
        backtest_repo.JOB_STATUS_ERROR,
        error=meldung,
    )
    _finished(job_id, ok=False, error=meldung)


# ---------------------------------------------------------------------------
# Öffentliche API (Router)
# ---------------------------------------------------------------------------


async def submit_job(kind: str, payload: dict[str, Any]) -> backtest_repo.BacktestJob:
    """Legt den Job an (``queued``) und startet seine Verarbeitung im Loop.

    Der Aufrufer validiert fachlich **vorher** synchron (422 vor Job-Anlage) —
    diese Funktion legt nur die Zeile an und reiht ein. Mehrere Jobs laufen
    nacheinander (FIFO-Schleuse je Event-Loop, R15).
    """
    job_id = f"btjob-{uuid.uuid4().hex}"
    zeile = await asyncio.to_thread(backtest_repo.create_job, job_id, kind, payload)
    handle = _JobHandle()
    _JOBS[job_id] = handle
    handle.task = asyncio.get_running_loop().create_task(
        _job_lauf(job_id, kind, payload, _loop_zustand().schleuse)
    )
    return zeile


def cancel_job_anfordern(job_id: str) -> backtest_repo.BacktestJob | None:
    """DB-Seite des Abbruchs (thread-safe): Status nur vorwärts auf ``cancelled``.

    ``None`` bei unbekannter ID; bereits terminale Jobs bleiben unverändert.
    Das Wecken des Jobs macht danach :func:`abbruch_signalieren` im Loop-Thread.
    """
    job = backtest_repo.get_job(job_id)
    if job is None:
        return None
    if job.status not in _TERMINAL:
        backtest_repo.update_job_status_vorwaerts(
            job_id, backtest_repo.JOB_STATUS_CANCELLED
        )
    return backtest_repo.get_job(job_id)


def abbruch_signalieren(job_id: str) -> None:
    """Setzt das Abbruch-Event (muss im Loop-Thread des Jobs laufen)."""
    handle = _JOBS.get(job_id)
    if handle is not None:
        handle.abbruch.set()


async def warte_auf_abschluss(
    job_id: str, timeout_s: float = 120.0
) -> backtest_repo.BacktestJob | None:
    """Wartet auf einen Terminal-Status (Hilfe für Agent-Tools und Tests)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        job = await asyncio.to_thread(backtest_repo.get_job, job_id)
        if job is not None and job.status in _TERMINAL:
            return job
        await asyncio.sleep(0.02)
    return None


async def _job_lauf(
    job_id: str, kind: str, payload: dict[str, Any], schleuse: asyncio.Lock
) -> None:
    """Task-Leib eines Jobs: serialisiert via Schleuse, fängt Restfehler ab."""
    try:
        async with schleuse:
            await _ausfuehren(job_id, kind, payload)
    except asyncio.CancelledError:
        raise
    except ApiError as exc:
        await _status_error(job_id, exc.message)
    except Exception as exc:  # letztes Netz: Job als error verbuchen
        logger.exception("Backtest-Job %s unerwartet fehlgeschlagen", job_id)
        await _status_error(job_id, f"Unerwarteter Fehler im Backtest-Job: {exc}")


async def _ausfuehren(job_id: str, kind: str, payload: dict[str, Any]) -> None:
    """Führt einen dequeue-ten Job aus (Grid oder Walk-Forward)."""
    handle = _JOBS.get(job_id)
    abbruch = handle.abbruch if handle is not None else None

    job = await asyncio.to_thread(backtest_repo.get_job, job_id)
    if job is None or job.status in _TERMINAL:
        # z. B. Cancel, während der Job noch in der Schleuse wartete.
        _finished(
            job_id,
            job is not None and job.status == backtest_repo.JOB_STATUS_DONE,
            job.error if job is not None else None,
        )
        return

    await asyncio.to_thread(
        backtest_repo.update_job_status_vorwaerts,
        job_id,
        backtest_repo.JOB_STATUS_RUNNING,
    )

    # Daten-Vorbereitungsphase (B-Fix-4, E056): fehlende Marktdaten werden vor
    # dem Kern geladen; der Fortschritt läuft über dasselbe Progress-Event mit
    # ``phase: "preparing_data"`` (vor ``"kombinationen"``/``"fenster"``).
    # Bewusst nur bei explizit gesetztem Feld: Die API-Endpunkte übernehmen es
    # via ``model_dump`` (Default True), direkte ``submit_job``-Aufrufe mit
    # Alt-Payloads (ohne Feld) verhalten sich unverändert.
    if payload.get("auto_download"):
        if not await _bereite_job_daten(job_id, payload):
            return

    if kind == "grid":
        await _grid_ausfuehren(job_id, payload, abbruch)
    else:
        await _walkforward_ausfuehren(job_id, payload, abbruch)


async def _bereite_job_daten(job_id: str, payload: dict[str, Any]) -> bool:
    """Daten-Vorbereitung für einen Optimierungs-Job (B-Fix-4, E056).

    Lädt fehlende Symbole über ``daten_vorbereitung.bereite_daten_vor`` und
    leitet deren Fortschritt als ``backtest_job_progress`` mit
    ``phase: "preparing_data"`` weiter (keine separaten Prepare-Events —
    ``token=None``). Fachfehler (Intraday-Grenze, Timeout, unvollständige
    Symbole) werden als deutscher Job-Fehler verbucht und beenden den Job
    vor dem Kern; ``False`` = Job ist terminiert, kein weiterer Schritt.
    """
    from quantlab.providers.base import ProviderError
    from quantlab.services import daten_vorbereitung

    symbole: list[str] = list(
        payload.get("symbols") or ([payload["symbol"]] if payload.get("symbol") else [])
    )
    if not symbole:
        return True

    def _fortschritt(done: int, total: int, symbol: str) -> None:
        _progress(job_id, done, total, "preparing_data")

    try:
        await daten_vorbereitung.bereite_daten_vor(
            symbole,
            str(payload.get("interval") or "1d"),
            start_us=zeitstempel_zu_us(payload.get("start"), "start"),
            end_us=zeitstempel_zu_us(payload.get("end"), "end"),
            token=None,
            fortschritt=_fortschritt,
        )
    except (ApiError, ProviderError) as exc:
        await _status_error(job_id, exc.message)
        return False
    return True


def _abbruch_gefordert(abbruch: asyncio.Event | None) -> bool:
    return abbruch is not None and abbruch.is_set()


# ---------------------------------------------------------------------------
# Grid-Ausführung
# ---------------------------------------------------------------------------


def _request_aus_payload(payload: dict[str, Any], params: dict[str, Any]):
    """Payload → ``BacktestRunRequest`` (Job-Zusatzschlüssel entfernt)."""
    from quantlab.api.routers.backtest import BacktestRunRequest

    basis = {k: v for k, v in payload.items() if k not in _JOB_ZUSATZ_SCHLUESSEL}
    basis["params"] = dict(params)
    return BacktestRunRequest(**basis)


async def _grid_ausfuehren(
    job_id: str,
    payload: dict[str, Any],
    abbruch: asyncio.Event | None,
) -> None:
    """Grid: je Kombination ein Lauf über den Router-Kern; persistiert wird der Beste."""
    from quantlab.api.routers.backtest import _fuehre_request_aus

    kombis = grid_kombinationen(dict(payload.get("grid") or {}))
    basis_params = dict(payload.get("params") or {})
    total = len(kombis)
    _progress(job_id, 0, total, "kombinationen")

    zeilen: list[dict[str, Any]] = []
    fehler: str | None = None
    done = 0
    for kombi in kombis:
        if _abbruch_gefordert(abbruch):
            break
        kombi_params = {**basis_params, **kombi}
        try:
            request = _request_aus_payload(payload, kombi_params)
            _, detail = await asyncio.to_thread(_fuehre_request_aus, request, False)
        except ApiError as exc:
            # Eine ungültige Kombination (z. B. fast ≥ slow) verdirbt die
            # Optimierung — abbrechen und den Fachfehler melden.
            fehler = exc.message
            break
        metrik = detail["metrics"]
        zeilen.append(
            {
                "params": kombi_params,
                "zielwert": oos_fenster_wert(
                    metrik.get("sharpe"), metrik.get("total_return")
                ),
                "metrics": metrik,
            }
        )
        done += 1
        _progress(job_id, done, total, "kombinationen")

    if _abbruch_gefordert(abbruch):
        await asyncio.to_thread(
            backtest_repo.update_job_status_vorwaerts,
            job_id,
            backtest_repo.JOB_STATUS_CANCELLED,
        )
        _finished(job_id, ok=False)
        return
    if fehler is not None and not zeilen:
        await _status_error(job_id, fehler)
        return

    ranking = sorted(zeilen, key=lambda zeile: zeile["zielwert"], reverse=True)
    beste: dict[str, Any] | None = None
    if ranking:
        # Beste Kombination erneut ausführen — diesmal persistiert (deterministisch).
        request = _request_aus_payload(payload, dict(ranking[0]["params"]))
        run, detail = await asyncio.to_thread(_fuehre_request_aus, request, True)
        beste = {
            **ranking[0],
            "run_id": run.run_id if run is not None else None,
            "metrics": detail["metrics"],
        }

    ergebnis = {
        "art": "grid",
        # Fallback-Anzeige: Zielwert der besten Zeile basiert auf Total-Return,
        # wenn ihr Sharpe None ist (E032-Fallback, fensterweise).
        "zielfunktion": (
            "total_return"
            if beste is not None and beste["metrics"].get("sharpe") is None
            else "sharpe"
        ),
        "anzahl_kombinationen": total,
        "ausgefuehrt": len(zeilen),
        "beste": beste,
        "ranking": ranking,
    }
    akzeptiert = await asyncio.to_thread(
        backtest_repo.update_job_status_vorwaerts,
        job_id,
        backtest_repo.JOB_STATUS_DONE,
        error=fehler,
        result=ergebnis,
    )
    if not akzeptiert:
        # Cancel hat dazwischengesprochen — Ergebnis verwerfen, Status respektieren.
        _finished(job_id, ok=False)
        return
    _progress(job_id, total, total, "fertig")
    _finished(job_id, ok=True)


# ---------------------------------------------------------------------------
# Walk-Forward-Ausführung
# ---------------------------------------------------------------------------


def _gewichte_fuer_kombi(
    strategie_ref: str,
    params: dict[str, Any],
    daten: dict[str, pd.DataFrame],
    common_ts: np.ndarray,
    *,
    user_code: str | None,
) -> pd.DataFrame:
    """Volle Gewichte-Matrix einer Kombination über die Gesamthistorie.

    Katalog-Strategien laufen über ``_multi_gewichte`` des Routers — exakt
    dieselbe Semantik wie ``POST /api/backtest/runs`` (je-Symbol-Auswertung,
    Multi-Strategien aufs gemeinsame Grid, Normalisierung Σ|w|→1). User-
    Strategien laufen je Symbol einmal durch die Sandbox und werden **nicht**
    normalisiert (Σ|w| > 1 → 422 ``gewichte_summe`` durch die Engine, T29-
    Übergabe: keine stille Normalisierung für User-Gewichte).
    """
    from quantlab.services import sandbox

    symbole = list(daten.keys())
    if user_code is not None:
        grid = pd.Index(common_ts)
        spalten: dict[str, pd.Series] = {}
        for sym in symbole:
            ergebnis = sandbox.fuehre_strategie_aus(code=user_code, daten=daten[sym], params=params)
            if not ergebnis.ok:
                raise ApiError(422, f"sandbox_{ergebnis.error_kind}", ergebnis.meldung)
            werte = [0.0 if w is None else float(w) for w in ergebnis.gewichte]
            eigene = pd.Series(
                werte, index=pd.Index(np.asarray(daten[sym]["ts"], dtype="int64"))
            )
            spalten[sym] = eigene.reindex(grid)
        return pd.DataFrame(spalten)[symbole].fillna(0.0)

    from quantlab.api.routers.backtest import STRATEGIES, _multi_gewichte, _resolve_params

    spec = STRATEGIES.get(strategie_ref)
    if spec is None:
        raise ApiError(422, "unbekannte_strategie", f"Unbekannte Strategie „{strategie_ref}“.")
    resolved = _resolve_params(spec.params, dict(params))
    return _multi_gewichte(spec, resolved, daten, common_ts)


def _fenster_lauf(
    aligned: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    symbole: list[str],
    *,
    train_start: int,
    test_start: int,
    test_end: int,
    fill: str,
    initial_capital: float,
    rebalance_every: int,
    costs,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Engine-Lauf über ein Fenster-Slice; liefert (Resultat, ts, Equity-Werte)."""
    from quantlab.backtest.engine import run_backtest, run_portfolio_backtest

    geschnitte_daten = {
        sym: aligned[sym].iloc[train_start:test_end] for sym in symbole
    }
    geschnittene_gewichte = weights.iloc[train_start:test_end]
    if len(symbole) == 1 and rebalance_every == 0:
        resultat = run_backtest(
            geschnitte_daten[symbole[0]],
            geschnittene_gewichte,
            costs,
            symbol=symbole[0],
            initial_capital=initial_capital,
            fill=fill,
        )
    else:
        resultat = run_portfolio_backtest(
            geschnitte_daten,
            geschnittene_gewichte,
            costs,
            initial_capital=initial_capital,
            fill=fill,
            rebalance_every=rebalance_every,
        )
    ts = resultat.equity["ts"].to_numpy(dtype="int64")
    werte = resultat.equity["equity"].to_numpy(dtype=float)
    return resultat, ts, werte


def _bewerte_kombi(
    kombi: dict[str, Any],
    *,
    strategie_ref: str,
    basis_params: dict[str, Any],
    daten: dict[str, pd.DataFrame],
    aligned: dict[str, pd.DataFrame],
    common_ts: np.ndarray,
    fenster: list[tuple[int, int, int, int]],
    user_code: str | None,
    fill: str,
    initial_capital: float,
    rebalance_every: int,
    costs,
) -> dict[str, Any]:
    """Eine Kombination über alle Fenster — OOS-Metriken je Test-Segment.

    Die Signale werden einmal über die Gesamthistorie erzeugt und je Fenster
    geschnitten (look-ahead-frei, siehe Modul-Docstring); das OOS-Segment
    beginnt an der Equity-Basis des letzten Train-Balkens.
    """
    from quantlab.backtest.engine import compute_equity_metrics

    weights = _gewichte_fuer_kombi(
        strategie_ref,
        {**basis_params, **kombi},
        daten,
        common_ts,
        user_code=user_code,
    )
    symbole = list(daten.keys())
    fenster_zeilen: list[dict[str, Any]] = []
    for f_index, (train_start, train_end, test_start, test_end) in enumerate(fenster):
        _, _, werte = _fenster_lauf(
            aligned,
            weights,
            symbole,
            train_start=train_start,
            test_start=test_start,
            test_end=test_end,
            fill=fill,
            initial_capital=initial_capital,
            rebalance_every=rebalance_every,
            costs=costs,
        )
        segment = werte[test_start - train_start - 1 :]
        metrik = compute_equity_metrics(segment)
        fenster_zeilen.append(
            {
                "index": f_index,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "oos_sharpe": metrik["sharpe"],
                "oos_total_return": metrik["total_return"],
            }
        )
    zielwerte = [
        oos_fenster_wert(zeile["oos_sharpe"], zeile["oos_total_return"])
        for zeile in fenster_zeilen
    ]
    sharpes = [
        zeile["oos_sharpe"] for zeile in fenster_zeilen if zeile["oos_sharpe"] is not None
    ]
    return {
        "params": {**basis_params, **kombi},
        "zielwert": float(np.mean(zielwerte)) if zielwerte else 0.0,
        "durchschnitt_oos_sharpe": float(np.mean(sharpes)) if sharpes else None,
        "fenster": fenster_zeilen,
    }


def _verkette_oos(
    beste: dict[str, Any],
    *,
    strategie_ref: str,
    daten: dict[str, pd.DataFrame],
    aligned: dict[str, pd.DataFrame],
    common_ts: np.ndarray,
    user_code: str | None,
    fill: str,
    initial_capital: float,
    rebalance_every: int,
    costs,
    interval: str,
    project_id: int | None,
) -> tuple[backtest_repo.BacktestRun, list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Beste Kombination: Fenster erneut laufen lassen und OOS-Sequenz verketten.

    Je Fenster wird die Equity ab der Basis des letzten Train-Balkens auf das
    Vorfenster-Ende rebased und angehängt (verkettete OOS-Gesamtsequenz,
    E032). Der resultierende Lauf wird in ``backtest_results`` persistiert.
    """
    from quantlab.backtest.engine import (
        MAX_EQUITY_POINTS,
        build_enrichment,
        compute_equity_metrics,
        resample_equity,
    )

    weights = _gewichte_fuer_kombi(
        strategie_ref, dict(beste["params"]), daten, common_ts, user_code=user_code
    )
    symbole = list(daten.keys())
    verkettet_ts: list[int] = []
    verkettet_werte: list[float] = []
    fenster_report: list[dict[str, Any]] = []
    vorh_end: float | None = None
    for zeile in beste["fenster"]:
        _, ts, werte = _fenster_lauf(
            aligned,
            weights,
            symbole,
            train_start=zeile["train_start"],
            test_start=zeile["test_start"],
            test_end=zeile["test_end"],
            fill=fill,
            initial_capital=initial_capital,
            rebalance_every=rebalance_every,
            costs=costs,
        )
        basis_pos = zeile["test_start"] - zeile["train_start"] - 1
        segment_ts = [int(t) for t in ts[basis_pos:]]
        segment_w = [float(w) for w in werte[basis_pos:]]
        basis = segment_w[0] if segment_w else 1.0
        for ts_punkt, wert in zip(segment_ts, segment_w, strict=True):
            skaliert = wert if vorh_end is None else (wert / basis) * vorh_end
            verkettet_ts.append(ts_punkt)
            verkettet_werte.append(skaliert)
        vorh_end = verkettet_werte[-1] if verkettet_werte else None
        fenster_report.append(
            {
                **zeile,
                "train_start_ts": int(common_ts[zeile["train_start"]]),
                "test_start_ts": int(common_ts[zeile["test_start"]]),
                "test_letzter_ts": int(common_ts[zeile["test_end"] - 1]),
            }
        )

    verkettet = pd.DataFrame(
        {
            "ts": pd.Series(verkettet_ts, dtype="int64"),
            "equity": pd.Series(verkettet_werte, dtype="float64"),
        }
    )
    oos_metrik = compute_equity_metrics(verkettet_werte)
    _, aligned_leer, symbols_info = _align_infos(daten)
    run = backtest_repo.create_run(
        uuid.uuid4().hex,
        project_id=project_id,
        strategy=strategie_ref,
        symbol=",".join(symbole),
        interval=interval,
        params=dict(beste["params"]),
        metrics=oos_metrik,
        equity=resample_equity(verkettet, MAX_EQUITY_POINTS),
        trades=[],
        enrichment=build_enrichment(
            verkettet,
            [],
            symbols_info,
            benchmark_symbol=symbole[0],
            benchmark=None,
        ),
    )
    return run, fenster_report, verkettet, oos_metrik


def _align_infos(daten: dict[str, pd.DataFrame]):
    """Alignment-Info (``bars_loaded``/``bars_used``) für den Lauf-Vertrag."""
    from quantlab.backtest.engine import align_frames

    _, _, symbols_info = align_frames(daten)
    return None, None, symbols_info


async def _walkforward_ausfuehren(
    job_id: str,
    payload: dict[str, Any],
    abbruch: asyncio.Event | None,
) -> None:
    """Walk-Forward: rollierend N/M (E032), Zielfunktion Ø OOS-Sharpe."""
    from quantlab.api.routers.backtest import _lade_daten, _user_strategie_code
    from quantlab.backtest.costs import CostModel
    from quantlab.backtest.engine import (
        MAX_EQUITY_POINTS,
        align_frames,
        resample_equity,
    )

    symbole: list[str] = list(
        payload.get("symbols") or ([payload["symbol"]] if payload.get("symbol") else [])
    )
    strategie_ref = str(payload.get("strategy") or "")
    train_bars = int(payload["train_bars"])
    test_bars = int(payload["test_bars"])
    basis_params = dict(payload.get("params") or {})
    fill = str(payload.get("fill") or "open")
    initial_capital = float(payload.get("initial_capital") or 10_000.0)
    rebalance_every = int(payload.get("rebalance_every") or 0)
    interval = str(payload.get("interval") or "1d")
    roh_costs = dict(payload.get("costs") or {})
    # H7/E100 (additiv): gebuehren_modell='psb' ⇒ PSB-Kostenmodell (0,3 % min
    # 15 €); Default bleibt commission/slippage wie bisher (byte-identisch).
    if str(roh_costs.get("gebuehren_modell") or "standard") == "psb":
        from quantlab.services.planspiel import regeln as psb_regeln

        costs = psb_regeln.psb_kostenmodell()
    else:
        costs = CostModel(
            commission_bps=float(roh_costs.get("commission_bps", 10.0)),
            slippage_bps=float(roh_costs.get("slippage_bps", 5.0)),
        )

    # User-Strategie auflösen (404/422 wie im Router) — vor der Datenlast.
    user_code: str | None = None
    if strategie_ref.startswith("user:"):
        user_code = await asyncio.to_thread(_user_strategie_code, strategie_ref)

    try:
        # Zeitraum-Normalisierung wie im Router-Kern (T31-QA-FUND 1): der
        # Payload enthält Rohwerte (ISO-Datum/-Zeit ODER int64 µs) — int geht
        # unverändert durch, Strings werden hier (nicht erst in _lade_daten)
        # nach int64 µs konvertiert; ungültige Angaben → fachlicher Job-Fehler.
        start_us = zeitstempel_zu_us(payload.get("start"), "start")
        end_us = zeitstempel_zu_us(payload.get("end"), "end")
        daten = await asyncio.to_thread(_lade_daten, symbole, interval, start_us, end_us)
    except ApiError as exc:
        await _status_error(job_id, exc.message)
        return

    common_ts, aligned, _ = align_frames(daten)
    fenster = walkforward_fenster(len(common_ts), train_bars, test_bars)
    kombis = grid_kombinationen(dict(payload.get("grid") or {}))
    total = len(kombis) * len(fenster)
    _progress(job_id, 0, total, "fenster")

    kombi_ergebnisse: list[dict[str, Any]] = []
    fehler: str | None = None
    schritt = 0
    for kombi in kombis:
        if _abbruch_gefordert(abbruch):
            break
        try:
            kombi_ergebnisse.append(
                await asyncio.to_thread(
                    _bewerte_kombi,
                    kombi,
                    strategie_ref=strategie_ref,
                    basis_params=basis_params,
                    daten=daten,
                    aligned=aligned,
                    common_ts=common_ts,
                    fenster=fenster,
                    user_code=user_code,
                    fill=fill,
                    initial_capital=initial_capital,
                    rebalance_every=rebalance_every,
                    costs=costs,
                )
            )
        except ApiError as exc:
            fehler = exc.message
            break
        schritt += len(fenster)
        _progress(job_id, schritt, total, "fenster")

    if _abbruch_gefordert(abbruch):
        await asyncio.to_thread(
            backtest_repo.update_job_status_vorwaerts,
            job_id,
            backtest_repo.JOB_STATUS_CANCELLED,
        )
        _finished(job_id, ok=False)
        return
    if fehler is not None:
        await _status_error(job_id, fehler)
        return
    if not kombi_ergebnisse:
        await _status_error(job_id, "Keine Parameter-Kombination ausgewertet.")
        return

    beste = max(kombi_ergebnisse, key=lambda zeile: zeile["zielwert"])
    run, fenster_report, verkettet, oos_metrik = await asyncio.to_thread(
        _verkette_oos,
        beste,
        strategie_ref=strategie_ref,
        daten=daten,
        aligned=aligned,
        common_ts=common_ts,
        user_code=user_code,
        fill=fill,
        initial_capital=initial_capital,
        rebalance_every=rebalance_every,
        costs=costs,
        interval=interval,
        project_id=payload.get("project_id"),
    )

    fallback_genutzt = any(
        zeile["oos_sharpe"] is None for zeile in beste["fenster"]
    )
    ergebnis = {
        "art": "walkforward",
        "train_bars": train_bars,
        "test_bars": test_bars,
        "zielfunktion": "oos_total_return" if fallback_genutzt else "oos_sharpe",
        "anzahl_fenster": len(fenster),
        "anzahl_kombinationen": len(kombis),
        "ausgefuehrt": len(kombi_ergebnisse),
        "beste_params": dict(beste["params"]),
        "zielwert": beste["zielwert"],
        "durchschnitt_oos_sharpe": beste["durchschnitt_oos_sharpe"],
        "fenster": fenster_report,
        "run_id": run.run_id,
        "oos_equity": resample_equity(verkettet, MAX_EQUITY_POINTS),
        "oos_metrics": oos_metrik,
    }
    akzeptiert = await asyncio.to_thread(
        backtest_repo.update_job_status_vorwaerts,
        job_id,
        backtest_repo.JOB_STATUS_DONE,
        result=ergebnis,
    )
    if not akzeptiert:
        _finished(job_id, ok=False)
        return
    _progress(job_id, total, total, "fertig")
    _finished(job_id, ok=True)
