"""Die fachlichen Agent-Tools (T14 §E, T34-Erweiterung) — Registrierung beim Import.

Jedes Tool-Modul registriert seine :class:`~quantlab.agent.registry.ToolSpec`-
Instanzen **beim Import** in :data:`quantlab.agent.registry.GLOBAL_REGISTRY`
(dieses Paket-``__init__`` importiert alle Module). Der Agent-Loop
(:mod:`quantlab.agent.loop`) importiert den Run-Kontext aus diesem Paket —
damit ist die Registry gefüllt, sobald der Loop benutzt wird; der Router
übergibt ``GLOBAL_REGISTRY`` explizit an :func:`loop.start_run`.

Modulaufteilung:

- :mod:`.agent_tools` — ``starte_subagent`` (Welle F/T-F7, E084 — eingeschränkter
  Subagent mit Rollen-Presets analyse/recherche/backtest, Kern in
  :mod:`quantlab.agent.subagents`)
- :mod:`.daten` — ``get_market_data``, ``list_symbols``, ``download_data``,
  ``get_quote``, ``search_symbols``, ``get_fundamentals``,
  ``get_corporate_actions``, ``manage_watchlists``, ``get_scheduler_status``,
  ``trigger_data_update``
- :mod:`.quant` — ``compute_metrics``, ``compute_covariance``, ``run_pca``,
  ``estimate_var``
- :mod:`.backtest` — ``run_backtest``, ``get_backtest_result``, ``save_strategy``
- :mod:`.charts` — ``create_chart``, ``read_chart``
- :mod:`.projekte` — ``save_report``, ``manage_tasks``, ``read_project_context``
- :mod:`.portfolio` — ``manage_portfolios``, ``record_transaction``,
  ``get_portfolio_performance``
- :mod:`.screener` — ``run_screener``
- :mod:`.alerts` — ``manage_alerts``
- :mod:`.planspiel` — ``planspiel_status``, ``planspiel_suche``,
  ``planspiel_instrument``, ``planspiel_depot``, ``planspiel_orders``,
  ``planspiel_depotwerte``, ``planspiel_regeln`` (Welle P/E073, lesend)
  sowie ``planspiel_universum`` (Welle H1/E093, Universum listen/importieren)
- :mod:`.zeitplan` — ``verwalte_zeitplaene`` (Welle H/E099 — Agent-Zeitpläne
  der Planspiel-/Hintergrund-Beobachtung auflisten/anlegen/löschen/sofort
  ausführen; Delegation auf die agent_cron-Sektion + Scheduler-run-now)
- :mod:`.web` — ``web_suche``, ``web_lesen`` (Welle F/T-F6, E083 — externe
  Websuche über den Websuche-Service mit Cache/Pacing/SSRF-Guard)

Gemeinsame Verträge (Fehler→Tool-Ergebnis, Pydantic-Args-Validierung,
Result-Kapping) liegen in :mod:`._common` bzw. :mod:`quantlab.agent.registry`.
"""

from __future__ import annotations

from quantlab.agent.tools import (  # noqa: F401 — Registrierung in GLOBAL_REGISTRY
    agent_tools,
    alerts,
    backtest,
    charts,
    daten,
    planspiel,
    portfolio,
    projekte,
    quant,
    screener,
    web,
    zeitplan,
)

#: Reihenfolge der Registrierung (für Smoke/Debug; Registry sortiert eh nach Name).
REGISTERED_TOOL_MODULES: tuple[str, ...] = (
    "daten",
    "quant",
    "backtest",
    "charts",
    "projekte",
    "portfolio",
    "screener",
    "alerts",
    "planspiel",  # Welle P (E073): 7 lesende Planspiel-Tools (additiv)
    "zeitplan",  # Welle H (E099): verwalte_zeitplaene (additiv)
    "web",  # Welle F (T-F6/E083): web_suche + web_lesen (additiv)
    "agent_tools",  # Welle F (T-F7/E084): starte_subagent (additiv)
)
