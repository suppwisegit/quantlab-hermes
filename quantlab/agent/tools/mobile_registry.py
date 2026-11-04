"""Mobile Tool-Registrierung (Welle M2a/E-M08) — Selektion statt Alles-Import.

Das Paket-``__init__`` (:mod:`quantlab.agent.tools`) importiert beim Import
**alle** Tool-Module inkl. Backtest/Alerts/Portfolio — auf Android/Chaquopy
sind die zugehörigen Schwerabhängigkeiten (pyarrow für Backtests, Desktop-
Persistenz) nicht verfügbar. Dieses Modul baut das **mobile Profil**:

- Die leichtgewichtigen Tool-Module (daten, charts, screener, planspiel,
  projekte, web, zeitplan, agent_tools) werden importiert — sie registrieren ihre
  Specs beim Import in :data:`~quantlab.agent.registry.GLOBAL_REGISTRY`
  (unveränderter Mechanismus, kein paralleles Registry-System).
- Anschließend wird ``GLOBAL_REGISTRY`` auf die mobile Allowlist reduziert:
  vom Quant-Modul bleibt bewusst nur ``compute_metrics`` (numpy+pandas,
  E-M08); Backtest/Alerts/Portfolio-Tools verschwinden aus dem Katalog.
- Desktop unberührt: das Pruning passiert nur, wenn
  :func:`registriere_mobile_tools` gerufen wird (mobiler Server/Test) — der
  Desktop-Import füllt die Registry wie bisher vollständig.

Hinweis (E-M08, bewusste ANNAHME): Ein Import z. B. von
``quantlab.agent.tools.daten`` initialisiert Python-seits immer das Paket-
``__init__`` (alle Module). Nach dem Lazy-Import-Refactor (M2a) und dem
Bar-Store-Dispatcher (M2b, pyarrow lazy) sind diese Ketten aber leicht
(httpx/pandas/numpy/fastapi/pydantic) — die Selektion wirkt über die
Allowlist, nicht über die Import-Kette.
"""

from __future__ import annotations

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolRegistry, ToolSpec
from quantlab.agent.tools import (  # noqa: F401 — Registrierung in GLOBAL_REGISTRY
    agent_tools,
    charts,
    daten,
    planspiel,
    projekte,
    screener,
    web,
    zeitplan,
)

#: Mobile Tool-Katalog je Modul (E-M08-Vorgabe: daten/charts/screener/
#: quant-nur-compute_metrics/planspiel/projekte/web/zeitplan/agent_tools).
MOBILE_TOOL_NAMEN: frozenset[str] = frozenset(
    {
        # daten
        "get_market_data",
        "list_symbols",
        "download_data",
        "get_quote",
        "search_symbols",
        "get_fundamentals",
        "get_corporate_actions",
        "manage_watchlists",
        "get_scheduler_status",
        "trigger_data_update",
        # charts (inkl. Welle I: Indikatoren/Zeichnungen/Analyse)
        "create_chart",
        "read_chart",
        "get_indicators",
        "manage_chart_drawings",
        "save_analyse",
        # screener
        "run_screener",
        # quant — bewusst NUR compute_metrics (numpy+pandas, E-M08)
        "compute_metrics",
        # planspiel (lesend + platziere_order für den Chat-Agenten)
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
        # projekte (Welle I inkl. Notizen/Artefakte)
        "save_report",
        "manage_tasks",
        "read_project_context",
        "manage_projects",
        "manage_notes",
        "list_artifacts",
        # web (Welle F + I: Suche/Lesen/News)
        "web_suche",
        "web_lesen",
        "news_suche",
        # zeitplan
        "verwalte_zeitplaene",
        # agent_tools
        "starte_subagent",
    }
)

#: Explizit verboten im mobilen Profil (Drift-Schutz für Tests, M2a-DoD).
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
    }
)


def mobile_specs() -> list[ToolSpec]:
    """Die mobilen Tool-Specs aus ``GLOBAL_REGISTRY`` (sortiert nach Name)."""
    return [
        spec
        for name in GLOBAL_REGISTRY.names()
        if name in MOBILE_TOOL_NAMEN and (spec := GLOBAL_REGISTRY.get(name)) is not None
    ]


def registriere_mobile_tools() -> ToolRegistry:
    """Reduziert ``GLOBAL_REGISTRY`` auf das mobile Profil und liefert sie.

    Idempotent: bereits entfernte Desktop-Tools sind schlicht nicht mehr
    vorhanden. Fehlende mobile Tools können nicht auftreten, weil dieses
    Modul die Tool-Module selbst importiert. Die Funktion wird nur vom
    mobilen Profil (:mod:`quantlab.mobile_app`) gerufen — der Desktop füllt
    die Registry unverändert vollständig.
    """
    for name in list(GLOBAL_REGISTRY.names()):
        if name not in MOBILE_TOOL_NAMEN:
            GLOBAL_REGISTRY.unregister(name)
    return GLOBAL_REGISTRY
