"""Backtest-Tools (T14 §E 9–10, T34-Erweiterung): Strategien speichern, Backtests
(inkl. Multi-Symbol, User-Strategien, Grid-/Walk-Forward-Jobs) starten und lesen.

Der Lauf nutzt exakt die T10/T29/T31-Pipeline (Strategie-Registry + Engine +
Repo + Sandbox). Läuft der Run **in einem Projekt**, wird das Ergebnis
zusätzlich als ``kind="backtest"``-Artefakt im Projekt abgelegt (Payload
``{run_id, strategy, symbol, interval}``) und erscheint damit live im
Artifacts-Panel. Ohne Projekt wird nur der Lauf persistiert
(``backtest_results.project_id`` ist nullable, ``artifacts.project_id`` laut
§D.1 nicht — Entscheidung dokumentiert in :mod:`quantlab.agent.tools.charts`).

T34-Verträge:

- ``save_strategy`` validiert User-Strategie-Code **ohne Ausführung** über die
  T30-AST-Prüfung (:func:`quantlab.services.sandbox.pruefe_code_als_dict` —
  Verstöße → ``ok:false`` mit Zeilennummer) und hängt eine neue monotone
  Version an (E026). Rückgabe enthält die Backtest-Referenz
  ``user:<id>@<version>``.
- ``run_backtest`` versteht Multi-Symbol (``symbols``, Schnittmengen-Alignment,
  Short-fähig), ``start``/``end``-Fenster, User-Referenzen (Sandbox-Ausführung,
  **keine** stille Normalisierung — Σ|w| > 1 → deutscher Fachfehler
  ``gewichte_summe``) und die Optimierungs-Modi ``grid``/``walkforward`` als
  ``backtest_jobs``-Jobs (E032) mit ``warte_auf_abschluss``-Timeout und
  Ergebnis-Zusammenfassung. Seit B-Fix-4 (E051/E053): ``auto_download``
  (Default an) lädt fehlende Marktdaten vor dem Lauf — Sync-Pfad via
  ``daten_vorbereitung.bereite_daten_vor``, Optimierungs-Jobs über die
  ``auto_download``-Phase des Job-Runners (``phase: "preparing_data"``).
- ANNAHME: Für den Sync-Lauf und die Job-Submit-Prüfung konsumiert das Tool
  die Router-Kernfunktionen (:func:`quantlab.api.routers.backtest.
  _fuehre_request_aus`/``_validiere_job_anfrage``) per Funktionsimport ohne
  HTTP-Umweg — dieselbe Vertragsquelle nutzt der ``backtest_jobs``-Service
  intern (T31); eine Kopie der Engine-Orchestrierung im Tool würde zwei
  Vertragstrainings erzeugen.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, daten_fehler, pruefe_args, schema_von
from quantlab.agent.tools.kontext import aktives_projekt
from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.services import backtest_jobs, sandbox
from quantlab.storage.repos import backtest_repo, projects_repo, strategies_repo

#: Maximale Punkte des Equity-Samples in ``get_backtest_result`` (§T14).
_EQUITY_SAMPLE_POINTS = 50

#: Eigenes Warte-Limit für Optimierungs-Jobs (s) — bewusst unter dem Dispatch-
#: Timeout von 120 s (Muster download_data), damit das Tool noch berichten kann.
_JOB_TIMEOUT_S = 110.0


# ---------------------------------------------------------------------------
# 9) run_backtest
# ---------------------------------------------------------------------------


class RunBacktestArgs(BaseModel):
    """Argumente von ``run_backtest`` (flach — LLM-freundlich, keine ``$defs``-Refs)."""

    strategy: str = Field(
        description=(
            'Katalog-Strategie (z. B. "sma_crossover", "momentum", "momentum_rotation") '
            'oder User-Referenz "user:<id>@<version>" (aus save_strategy).'
        )
    )
    symbol: str | None = Field(
        default=None,
        min_length=1,
        description='Ein Symbol-Kürzel, z. B. "AAPL" (oder „symbols“ für Multi-Asset).',
    )
    symbols: list[str] | None = Field(
        default=None,
        description=(
            "Mehrere Symbole (Multi-Asset-Lauf auf der Kalender-Schnittmenge, "
            'Short über negative Gewichte), z. B. ["AAPL", "MSFT"].'
        ),
    )
    interval: str = Field(default="1d", description='yfinance-Intervall, Default "1d".')
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            'Strategie-Parameter je Strategie-Schema, z. B. {"fast": 50, "slow": 200} '
            "für sma_crossover; bei User-Strategien unverändert an den Code übergeben."
        ),
    )
    start: str | None = Field(
        default=None, description='Fensterbeginn inklusive, z. B. "2023-01-01".'
    )
    end: str | None = Field(
        default=None, description='Fensterende exklusiv, z. B. "2024-12-31".'
    )
    commission_bps: float = Field(
        default=10.0, ge=0, description="Provision in Basispunkten, Default 10."
    )
    slippage_bps: float = Field(
        default=5.0, ge=0, description="Slippage in Basispunkten, Default 5."
    )
    gebuehren_modell: Literal["standard", "psb"] = Field(
        default="standard",
        description=(
            'Kostenmodell: "standard" (Default: commission_bps/slippage_bps) oder '
            '"psb" — Gebührenmodell der Planspiel-Börse (0,3 % vom Kurswert, '
            "mindestens 15 € je Order; commission_bps/slippage_bps werden dann "
            "ignoriert). Für Wettbewerb-Simulationen."
        ),
    )
    initial_capital: float = Field(
        default=10_000.0, gt=0, description="Startkapital, Default 10000."
    )
    rebalance_every: int = Field(
        default=0, ge=0, description="Rebalancing alle n Balken (Multi-Asset), Default 0."
    )
    modus: Literal["lauf", "grid", "walkforward"] = Field(
        default="lauf",
        description=(
            '"lauf" (Default): einzelner Backtest. "grid": Parameter-Optimierung '
            'über „grid“ als Job. "walkforward": rollierendes Walk-Forward '
            "(braucht train_bars/test_bars) als Job."
        ),
    )
    grid: dict[str, list[Any]] = Field(
        default_factory=dict,
        description=(
            "Parameter-Bereiche für grid/walkforward, z. B. {\"fast\": [10, 20], "
            "\"slow\": [100, 200]} — max. 200 Kombinationen."
        ),
    )
    train_bars: int | None = Field(
        default=None, ge=1, description="Train-Fenster N in Balken (nur walkforward)."
    )
    test_bars: int | None = Field(
        default=None, ge=1, description="Test-Fenster M in Balken (nur walkforward)."
    )
    auto_download: bool = Field(
        default=True,
        description=(
            "Fehlende Marktdaten vor dem Lauf automatisch herunterladen "
            "(Default: ja). false = deutscher Datenfehler, wenn Daten fehlen."
        ),
    )


def _symbole_als_liste(args: RunBacktestArgs) -> list[str]:
    """``symbol``/``symbols`` → deduplizierte Symbol-Liste (deutsche Fehler)."""
    if args.symbols:
        symbole = [s.strip() for s in args.symbols if s and s.strip()]
        if len(set(symbole)) != len(symbole):
            raise ToolFehler(
                "Die Symbolliste „symbols“ enthält Duplikate — jedes Symbol nur einmal."
            )
        return symbole
    if args.symbol and args.symbol.strip():
        return [args.symbol.strip()]
    raise ToolFehler(
        'Bitte „symbol“ (z. B. "AAPL") oder „symbols“ (Multi-Asset-Liste) angeben.'
    )


def _metriken_kompakt(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Metriken-Block JSON-sicher runden (Muster Bestands-Tool)."""
    return {
        str(schluessel): (round(float(wert), 6) if wert is not None else None)
        for schluessel, wert in dict(metrics or {}).items()
    }


def _artefakt_anlegen(
    projekt_id: int | None, run_id: str | None, strategy: str, symbol: str, interval: str
) -> int | None:
    """Backtest-Artefakt im aktiven Projekt (Muster Bestands-Tool, Payload §T14)."""
    if projekt_id is None or run_id is None:
        return None
    artefakt = projects_repo.create_artifact(
        projekt_id,
        "backtest",
        f"Backtest {strategy} — {symbol}",
        {
            "run_id": run_id,
            "strategy": strategy,
            "symbol": symbol,
            "interval": interval,
        },
    )
    return artefakt.id


def _router_backtest():
    """Router-Kernfunktionen lazy importieren (ANNAHME, siehe Modul-Docstring)."""
    from quantlab.api.routers import backtest as backtest_router

    return backtest_router


async def run_backtest_tool(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Führt einen Backtest (Sync-Lauf oder Optimierungs-Job) aus; legt bei
    Projektbezug ein Artefakt an."""
    args = pruefe_args(RunBacktestArgs, roh_args, "run_backtest")
    projekt_id = aktives_projekt()
    symbole = _symbole_als_liste(args)

    if args.modus in ("grid", "walkforward"):
        return await _optimierungs_job(args, symbole, projekt_id)
    return await _einzellauf(args, symbole, projekt_id)


def _basis_payload(
    args: RunBacktestArgs, symbole: list[str], projekt_id: int | None
) -> dict[str, Any]:
    """Args → Backtest-Request-Payload (Vertrag ``BacktestRunRequest``, T31)."""
    return {
        "strategy": args.strategy,
        "symbol": symbole[0] if len(symbole) == 1 else None,
        "symbols": symbole if len(symbole) > 1 else None,
        "params": dict(args.params),
        "interval": args.interval,
        "start": args.start,
        "end": args.end,
        "costs": {
            "commission_bps": args.commission_bps,
            "slippage_bps": args.slippage_bps,
            "gebuehren_modell": args.gebuehren_modell,
        },
        "initial_capital": args.initial_capital,
        "rebalance_every": args.rebalance_every,
        "auto_download": args.auto_download,
        "project_id": projekt_id,
    }


async def _bereite_tool_daten(
    args: RunBacktestArgs, symbole: list[str]
) -> None:
    """Daten-Vorbereitung vor dem Sync-Lauf (B-Fix-4, E051).

    Fehlgeschlagene Vorbereitung wird ins bestehende deutsche Fehlermuster
    des Tools gemappt (:func:`daten_fehler` bzw. :class:`ToolFehler`) — der
    Agent-Run bricht nicht ab, das Tool-Ergebnis ist ``ok:false`` mit Grund.
    """
    from quantlab.services import daten_vorbereitung

    start_us = backtest_jobs.zeitstempel_zu_us(args.start, "start")
    end_us = backtest_jobs.zeitstempel_zu_us(args.end, "end")
    try:
        await daten_vorbereitung.bereite_daten_vor(
            symbole,
            args.interval,
            start_us=start_us,
            end_us=end_us,
            token=None,  # Agent-Pfad: keine Prepare-SSE (nur der Lauf zählt)
        )
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    except ApiError as exc:
        raise ToolFehler(exc.message) from exc


async def _einzellauf(
    args: RunBacktestArgs, symbole: list[str], projekt_id: int | None
) -> dict[str, Any]:
    """Synchroner Lauf über den Router-Kern (identischer Vertrag wie POST /runs)."""
    router = _router_backtest()

    if args.auto_download:
        await _bereite_tool_daten(args, symbole)

    def _ausfuehren() -> dict[str, Any]:
        request = router.BacktestRunRequest(
            **_basis_payload(args, symbole, projekt_id),
            fill="open",
            sizing="target_weights",
            sizing_fraction=1.0,
        )
        _, detail = router._fuehre_request_aus(request, True)
        return detail

    try:
        detail = await asyncio.to_thread(_ausfuehren)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    except ApiError as exc:
        raise ToolFehler(exc.message) from exc

    run_id = detail.get("run_id")
    strategy = str(detail.get("strategy") or args.strategy)
    symbol_feld = str(detail.get("symbol") or ",".join(symbole))
    artefakt_id = _artefakt_anlegen(projekt_id, run_id, strategy, symbol_feld, args.interval)
    antwort: dict[str, Any] = {
        "run_id": run_id,
        "strategy": strategy,
        "symbol": symbol_feld,
        "interval": detail.get("interval", args.interval),
        "params": detail.get("params", args.params),
        "metrics": _metriken_kompakt(detail.get("metrics")),
        "trades_anzahl": len(detail.get("trades") or []),
        "zeitraum": detail.get("period") or {},
        "artefakt_id": artefakt_id,
        "hinweis": (
            "Details (Metriken + Equity-Kurve) via get_backtest_result mit dieser "
            "run_id; im Projekt wurde ein Backtest-Artefakt angelegt."
            if artefakt_id is not None
            else "Details via get_backtest_result mit dieser run_id."
        ),
    }
    if len(symbole) > 1:
        antwort["symbols"] = detail.get("symbols") or symbole
    return antwort


async def _optimierungs_job(
    args: RunBacktestArgs, symbole: list[str], projekt_id: int | None
) -> dict[str, Any]:
    """Grid/Walk-Forward als ``backtest_jobs``-Job (E032) mit Warte-Timeout."""
    if args.modus == "walkforward" and (args.train_bars is None or args.test_bars is None):
        raise ToolFehler(
            '„walkforward“ benötigt "train_bars" (N) und "test_bars" (M) — '
            "z. B. train_bars=200, test_bars=60."
        )
    router = _router_backtest()

    def _validieren() -> None:
        # Submit-Prüfung synchron (422 vor Job-Anlage): Strategie auflösbar,
        # Grid innerhalb des Caps von 200 Kombinationen.
        router._validiere_job_anfrage(args.strategy, args.grid)

    try:
        await asyncio.to_thread(_validieren)
    except ApiError as exc:
        raise ToolFehler(exc.message) from exc

    payload = _basis_payload(args, symbole, projekt_id)
    payload["grid"] = {k: list(v) for k, v in args.grid.items()}
    if args.modus == "walkforward":
        payload["train_bars"] = args.train_bars
        payload["test_bars"] = args.test_bars

    job = await backtest_jobs.submit_job(args.modus, payload)
    fertig = await backtest_jobs.warte_auf_abschluss(job.job_id, timeout_s=_JOB_TIMEOUT_S)
    if fertig is None:
        return {
            "status": "laeuft_noch",
            "job_id": job.job_id,
            "modus": args.modus,
            "meldung": (
                "Die Optimierung läuft noch (Timeout der Wartezeit erreicht). "
                "Sie läuft im Hintergrund weiter; das Ergebnis (beste Kombination "
                "bzw. OOS-Auswertung) ist anschließend in der Backtest-UI "
                "sichtbar — bitte den Nutzer, dort später nachzusehen."
            ),
        }
    if fertig.status == "error":
        raise ToolFehler(
            f"Der {args.modus}-Job ist fehlgeschlagen: {fertig.error or 'unbekannter Fehler'}"
        )
    if fertig.status == "cancelled":
        return {"status": "abgebrochen", "job_id": job.job_id, "modus": args.modus}
    return _job_zusammenfassung(fertig)


def _job_zusammenfassung(job: backtest_repo.BacktestJob) -> dict[str, Any]:
    """Terminaler Optimierungs-Job → kompakte deutsche Zusammenfassung.

    Bewusst **nicht** das volle ``result`` (Ranking/Fensterlisten blähen den
    Kontext) — die Details liegen über die job_id in der Backtest-UI bzw. im
    persistierten Lauf.
    """
    result = job.result if isinstance(job.result, dict) else {}
    basis: dict[str, Any] = {
        "status": "fertig",
        "job_id": job.job_id,
        "modus": result.get("art", job.kind),
    }
    if result.get("art") == "grid":
        beste = result.get("beste") or {}
        basis.update(
            zielfunktion=result.get("zielfunktion"),
            anzahl_kombinationen=result.get("anzahl_kombinationen"),
            ausgefuehrt=result.get("ausgefuehrt"),
            beste_params=beste.get("params"),
            zielwert=beste.get("zielwert"),
            bester_run_id=beste.get("run_id"),
            hinweis=(
                "Beste Kombination wurde als Lauf persistiert — Details via "
                "get_backtest_result mit bester_run_id; das vollständige Ranking "
                "liegt in der Backtest-UI unter der job_id."
            ),
        )
    else:
        basis.update(
            train_bars=result.get("train_bars"),
            test_bars=result.get("test_bars"),
            zielfunktion=result.get("zielfunktion"),
            anzahl_fenster=result.get("anzahl_fenster"),
            anzahl_kombinationen=result.get("anzahl_kombinationen"),
            ausgefuehrt=result.get("ausgefuehrt"),
            beste_params=result.get("beste_params"),
            zielwert=result.get("zielwert"),
            durchschnitt_oos_sharpe=result.get("durchschnitt_oos_sharpe"),
            oos_run_id=result.get("run_id"),
            oos_kennzahlen=_metriken_kompakt(result.get("oos_metrics")),
            hinweis=(
                "Die verkettete Out-of-Sample-Sequenz wurde als Lauf persistiert — "
                "Details via get_backtest_result mit oos_run_id."
            ),
        )
    return basis


# ---------------------------------------------------------------------------
# 10) get_backtest_result
# ---------------------------------------------------------------------------


class GetBacktestResultArgs(BaseModel):
    """Argumente von ``get_backtest_result``."""

    run_id: str = Field(min_length=1, description='Lauf-ID aus run_backtest, z. B. "a1b2c3…".')


async def get_backtest_result(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Liest ein Backtest-Ergebnis: Kennzahlen + Equity-Sample (≤ 50 Punkte)."""
    args = pruefe_args(GetBacktestResultArgs, roh_args, "get_backtest_result")

    def _lesen() -> dict[str, Any]:
        lauf = backtest_repo.get_run(args.run_id)
        if lauf is None:
            raise ToolFehler(f"Backtest-Lauf „{args.run_id}“ wurde nicht gefunden.")
        equity: list[dict[str, Any]] = lauf.equity if isinstance(lauf.equity, list) else []
        punkte: list[dict[str, Any]] = []
        if equity:
            # Gleichmäßiges Sample ≤ 50 Punkte; erster und letzter bleiben immer
            # (Muster: engine.resample_equity, deterministisch ohne Off-by-one).
            if len(equity) <= _EQUITY_SAMPLE_POINTS:
                indizes = list(range(len(equity)))
            else:
                teiler = _EQUITY_SAMPLE_POINTS - 1
                indizes = sorted(
                    {
                        round(i * (len(equity) - 1) / teiler)
                        for i in range(_EQUITY_SAMPLE_POINTS)
                    }
                )
            for index in indizes:
                eintrag = equity[index]
                if isinstance(eintrag, dict):
                    punkte.append({"ts": eintrag.get("ts"), "equity": eintrag.get("equity")})
        return {
            "run_id": lauf.run_id,
            "strategy": lauf.strategy,
            "symbol": lauf.symbol,
            "interval": lauf.interval,
            "params": lauf.params,
            "created_at": lauf.created_at,
            "metrics": lauf.metrics,
            "trades_anzahl": len(lauf.trades) if isinstance(lauf.trades, list) else 0,
            "equity_sample": punkte,
        }

    return await asyncio.to_thread(_lesen)


# ---------------------------------------------------------------------------
# 11) save_strategy (T34: E026-Strategie + neue Version, AST-geprüft)
# ---------------------------------------------------------------------------


class SaveStrategyArgs(BaseModel):
    """Argumente von ``save_strategy``."""

    name: str = Field(
        min_length=1,
        max_length=200,
        description='Eindeutiger Strategie-Name, z. B. "Mein Momentum-Trend".',
    )
    code: str = Field(
        min_length=1,
        max_length=200_000,
        description=(
            "Vollständiger Python-Code mit der Vertragsfunktion "
            "generate_weights(daten, **params) → Gewichte (z. B. DataFrame-Spalte "
            '"weight" oder Liste); erlaubt sind pandas/numpy, verboten os/sys/'
            "subprocess/socket/open (Sandbox E030)."
        ),
    )
    description: str = Field(
        default="", max_length=5000, description="Optionale Kurzbeschreibung der Idee."
    )
    strategy_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Bestehende Strategie-ID (dann neue Version dort); ohne ID wird über "
            "den Namen gesucht und ggf. neu angelegt."
        ),
    )


async def save_strategy(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Validiert Strategie-Code (AST, ohne Ausführung) und legt eine Version an."""
    args = pruefe_args(SaveStrategyArgs, roh_args, "save_strategy")

    def _speichern() -> dict[str, Any]:
        verstoesse = sandbox.pruefe_code_als_dict(args.code)
        if verstoesse:
            erste = verstoesse[0]
            details = "; ".join(
                f"Zeile {v.get('zeile')}: {v.get('meldung')}" for v in verstoesse[:5]
            )
            mehr = "" if len(verstoesse) <= 5 else f" (+{len(verstoesse) - 5} weitere)"
            raise ToolFehler(
                f"Der Strategie-Code wurde abgelehnt ({len(verstoesse)} Verstoß/"
                f"Verstöße) — {details}{mehr}. Erste Zeile {erste.get('zeile')}: "
                "bitte korrigieren und erneut speichern."
            )
        strategie = None
        if args.strategy_id is not None:
            strategie = strategies_repo.get_strategy(args.strategy_id)
            if strategie is None:
                raise ToolFehler(
                    f"Strategie mit ID {args.strategy_id} nicht gefunden."
                )
        else:
            treffer = [
                s for s in strategies_repo.list_strategies() if s.name == args.name.strip()
            ]
            strategie = treffer[0] if treffer else None
        if strategie is None:
            try:
                strategie = strategies_repo.create_strategy(
                    args.name, description=args.description
                )
            except ValueError as exc:
                raise ToolFehler(str(exc)) from exc
        try:
            version = strategies_repo.add_version(strategie.id, args.code)
        except ValueError as exc:
            raise ToolFehler(str(exc)) from exc
        return {
            "status": "gespeichert",
            "strategy_id": strategie.id,
            "name": strategie.name,
            "version": version.version,
            "referenz": f"user:{strategie.id}@{version.version}",
            "hinweis": (
                "Backtest über run_backtest mit strategy=\"user:<id>@<version>\" "
                "— der Code läuft sandboxed (Zeit-/Speicherlimits)."
            ),
        }

    return await asyncio.to_thread(_speichern)


# ---------------------------------------------------------------------------
# Registrierung (§E-Reihenfolge 9–10, dann T34-Zugang)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="run_backtest",
            description=(
                "Führt einen Strategie-Backtest auf gespeicherten Kursdaten aus und "
                "speichert das Ergebnis (Ausführung t+1, Kosten in bps; im aktiven "
                "Projekt entsteht ein Backtest-Artefakt). Einzel- oder Multi-Symbol "
                "(„symbols“), Katalog- oder User-Strategie („user:<id>@<version>“); "
                "mit modus=\"grid\" bzw. \"walkforward\" läuft eine Parameter-"
                "Optimierung als Job und liefert die beste Kombination. "
                'Beispiel: {"strategy": "sma_crossover", "symbol": "AAPL", '
                '"params": {"fast": 20, "slow": 100}}.'
            ),
            parameters=schema_von(RunBacktestArgs),
            handler=run_backtest_tool,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="get_backtest_result",
            description=(
                "Liest ein gespeichertes Backtest-Ergebnis: Kennzahlen, Trade-Anzahl "
                "und Equity-Kurve (gleichmäßig gesampelt, max. 50 Punkte). "
                'Beispiel: {"run_id": "9f8e7d6b"}.'
            ),
            parameters=schema_von(GetBacktestResultArgs),
            handler=get_backtest_result,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="save_strategy",
            description=(
                "Prüft eigenen Strategie-Code (Sandbox-AST-Regeln, ohne Ausführung) "
                "und speichert ihn als neue Version — Rückgabe enthält die Backtest-"
                "Referenz user:<id>@<version>. Bei Verstößen kommt ok:false mit "
                "Zeilennummern zurück. "
                'Beispiel: {"name": "Mein Trend", "code": "def generate_weights'
                '(daten):\\n    …"}.'
            ),
            parameters=schema_von(SaveStrategyArgs),
            handler=save_strategy,
        )
    )


_registrieren()
