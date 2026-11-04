"""Hermes Tool-Registrierung (Welle H, Task H1) — Selektion statt Alles-Import.

Analog zum mobilen Profil (:mod:`quantlab.agent.tools.mobile_registry`) baut
dieses Modul ein **kuratiertes Tool-Profil** für Hermes, den autonomen
Trading-Agenten auf Raspberry Pi 4 (MCP-Toolset):

- Selektion statt Kopie: die für Hermes relevanten Tool-Module (daten, charts,
  screener, quant, planspiel, web) werden importiert — sie registrieren ihre
  Specs beim Import in :data:`~quantlab.agent.registry.GLOBAL_REGISTRY`
  (unveränderter Mechanismus, kein paralleles Registry-System).
- Anschließend wird ``GLOBAL_REGISTRY`` auf die Hermes-Allowlist reduziert:
  lesender/analytischer Datenzugriff plus Planspiel-Handel (inkl.
  ``platziere_order``); vom Quant-Modul bleibt bewusst nur ``compute_metrics``.
  Persistenz-, Backtest-, Portfolio-, Chart-, Projekt- und Agent-Orchestrierung-
  Tools verschwinden aus dem Katalog — der Pi-Agent soll datenbasiert handeln,
  nicht die Desktop-Datenhaltung fernsteuern.
- Desktop unberührt: das Pruning passiert nur, wenn
  :func:`registriere_hermes_tools` gerufen wird (MCP-Server/Test) — der
  Desktop-Import füllt die Registry wie bisher vollständig.

Hinweis (bewusste ANNAHME, wie E-M08): Ein Import z. B. von
``quantlab.agent.tools.daten`` initialisiert Python-seits immer das Paket-
``__init__`` (alle Module). Nach dem Lazy-Import-Refactor (M2a) sind diese
Ketten leicht (httpx/pandas/numpy/fastapi/pydantic) — die Selektion wirkt über
die Allowlist, nicht über die Import-Kette.
"""

from __future__ import annotations

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolRegistry, ToolSpec
from quantlab.agent.tools import (  # noqa: F401 — Registrierung in GLOBAL_REGISTRY
    charts,
    daten,
    planspiel,
    quant,
    screener,
    web,
)

#: Hermes Tool-Katalog (Welle-H-Vorgabe): Planspiel-Handel (10), Daten (5),
#: Analyse (4: charts/screener/quant/web) — bewusst OHNE Desktop-Persistenz,
#: Backtest, Portfolio, Projekt-Verwaltung und Agent-Orchestrierung.
HERMES_TOOL_NAMEN: frozenset[str] = frozenset(
    {
        # planspiel (lesend + platziere_order für den handelnden Agenten)
        "planspiel_status",
        "planspiel_suche",
        "planspiel_instrument",
        "planspiel_depot",
        "planspiel_orders",
        "planspiel_depotwerte",
        "planspiel_kennzahlen",
        "planspiel_regeln",
        "planspiel_universum",
        "platziere_order",
        # daten
        "get_market_data",
        "download_data",
        "get_quote",
        "get_fundamentals",
        "search_symbols",
        # analyse: charts (Indikatoren), screener, quant (nur compute_metrics), web (news)
        "get_indicators",
        "run_screener",
        "compute_metrics",
        "news_suche",
        # web
        "web_lesen",
    }
)

#: Explizit verboten im Hermes-Profil (Drift-Schutz für Tests, Welle H):
#: Backtest, Strategien, Alerts, Portfolios, Risiko-Statistik (sklearn-Last),
#: Charts/Analyse-Persistenz, Projekt-/Notiz-/Artefakt-Verwaltung,
#: Subagent-Orchestrierung, Zeitpläne, Websuche, Watchlist-/Symbol-Pflege,
#: Scheduler-Mechanik.
VERBOTENE_TOOL_NAMEN: frozenset[str] = frozenset(
    {
        "run_backtest",
        "get_backtest_result",
        "save_strategy",
        "manage_alerts",
        "manage_portfolios",
        "record_transaction",
        "get_portfolio_performance",
        "compute_covariance",
        "run_pca",
        "estimate_var",
        "create_chart",
        "read_chart",
        "manage_chart_drawings",
        "save_analyse",
        "save_report",
        "manage_tasks",
        "read_project_context",
        "manage_projects",
        "manage_notes",
        "list_artifacts",
        "starte_subagent",
        "verwalte_zeitplaene",
        "web_suche",
        "manage_watchlists",
        "list_symbols",
        "get_corporate_actions",
        "get_scheduler_status",
        "trigger_data_update",
    }
)


def hermes_specs() -> list[ToolSpec]:
    """Die Hermes-Tool-Specs aus ``GLOBAL_REGISTRY`` (sortiert nach Name)."""
    return [
        spec
        for name in GLOBAL_REGISTRY.names()
        if name in HERMES_TOOL_NAMEN and (spec := GLOBAL_REGISTRY.get(name)) is not None
    ]


def registriere_hermes_tools() -> ToolRegistry:
    """Reduziert ``GLOBAL_REGISTRY`` auf das Hermes-Profil und liefert sie.

    Idempotent: bereits entfernte Tools sind schlicht nicht mehr vorhanden.
    Fehlende Hermes-Tools können nicht auftreten, weil dieses Modul die
    Tool-Module selbst importiert. Die Funktion wird nur vom Hermes-Profil
    (MCP-Server, Task H3) gerufen — der Desktop füllt die Registry unverändert
    vollständig.
    """
    for name in list(GLOBAL_REGISTRY.names()):
        if name not in HERMES_TOOL_NAMEN:
            GLOBAL_REGISTRY.unregister(name)
    return GLOBAL_REGISTRY
