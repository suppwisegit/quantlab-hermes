"""Subagent (Welle F, T-F7/E084): EIN genesteter, eingeschränkter Agenten-Loop.

Der QuantLab-Agent kann über das Tool ``starte_subagent`` (Modul
:mod:`quantlab.agent.tools.agent_tools`) für umfangreiche Nebenanalysen einen
**Subagenten** spawnen — bewusst „minimal, kein Orchestrator“ (E084):

- **Kein** ``loop.start_run``-Wiederverwendung: Der Subagent ist **ephemär** —
  er persistiert nichts (kein ``chat_repo``, kein ``RUN_REGISTRY``, keine
  SSE-Events), sein Verlauf lebt ausschließlich im Speicher dieser Funktion.
  Deshalb läuft hier eine eigene schlanke Iterationsschleife auf
  :class:`~quantlab.agent.llm_client.LLMClient` + gefilterter
  :class:`~quantlab.agent.registry.ToolRegistry` + ``dispatch_tool``.
- **Rollen-Presets** (:data:`ROLLEN_PRESETS`, Muster ``AGENT_CRON_PRESETS``):
  je Rolle ein deutschsprachiger Systemprompt plus **feste Tool-Allowlist**
  ausschließlich lesender Tools. ``starte_subagent`` (keine Rekursion),
  ``platziere_order`` und alle Schreib-/Verwaltungs-Tools sind strukturell
  nie sichtbar (:data:`_NIEMALS`, Import-Zeit-Gegencheck unten).
- **Grenzen:** max. 10 Iterationen, Gesamt-Timeout 180 s (in Verantwortung
  dieses Moduls — F-A0 §3), Bericht-Kappung auf 6000 Zeichen. Der Parent-
  Dispatch im Loop kappt das Tool mit ``STANDARD_TIMEOUT_S +
  DISPATCH_PUFFER_S`` (200 s), damit das Eigenlimit — nicht der generische
  120-s-Registry-Default — wirkt (REV-F3/M1).
- **Sequenzierung:** max. EIN Subagent gleichzeitig (E084 „sequenziell“).
  ANNAHME: Die Falle ist als Modul-State (:data:`_subagent_aktiv`) global —
  strenger als „pro Parent-Run“ und absichtlich so; ein asyncio.Lock wäre an
  den Event-Loop der ersten Nutzung gebunden und würde über pytest-Test-Loops
  hinweg brechen. Prüfen + Setzen sind await-frei, damit atomar.
- **Fehlervertrag (§E):** Alle Fachfehler (unbekannte Rolle, Parallelitäts-
  sperre, fehlender API-Key, Timeout, LLM-Fehler) werden als
  :class:`~quantlab.agent.tools._common.ToolFehler` geworfen → „ok:false“-
  Tool-Ergebnis an den Parent; der Parent-Run läuft weiter.
- **Projekt-Kontext:** Die ContextVar ``laufendes_projekt`` erbt sich in den
  Sub-Loop (asyncio-Task erzeugt im Handler-Task, F-A0 §3) — lesende
  Projekt-Tools (z. B. ``read_project_context``) funktionieren im Subagent
  ohne Zusatzverdrahtung.
- **Thinking:** Die Thinking-Einstellungen des Parent werden **nicht**
  geerbt (ANNAHME: Modell-/Server-Defaults genügen — ``prepare_request``
  läuft mit ``thinking=None``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quantlab.agent.llm_client import (
    CompletionEvent,
    LLMClient,
    ToolCallRequest,
    assistant_message,
    deutsche_fehlermeldung,
    system_message,
    tool_message,
    user_message,
)
from quantlab.agent.registry import GLOBAL_REGISTRY, ToolRegistry, dispatch_tool
from quantlab.agent.tools._common import ToolFehler
from quantlab.storage.repos import settings_repo

logger = logging.getLogger(__name__)

#: Hartes Iterations-Limit des Sub-Loops (E084) — unabhängig vom Parent-Limit
#: (``ki.max_tool_iterationen``); als Obergrenze auch im Args-Modell des Tools.
MAX_ITERATIONEN = 10

#: Standard-Gesamt-Timeout des Subagenten (s, E084). In Verantwortung dieses
#: Moduls (F-A0 §3): ``asyncio.wait_for`` um den Sub-Loop-Task + explizites
#: Cancel mit abgewartetem Cancel-Throw (kein Task-Leak). Wird von
#: :func:`starte_subagent` zur **Laufzeit** gelesen (Default ``timeout_s=None``),
#: damit Tests ihn skalieren können (REV-F3/M1).
STANDARD_TIMEOUT_S = 180.0

#: Puffer des Parent-Dispatchs über dem Subagent-Eigenlimit (REV-F3/M1): Der
#: Agent-Loop (:mod:`quantlab.agent.loop`) kappt den ``starte_subagent``-Dispatch
#: mit ``STANDARD_TIMEOUT_S + DISPATCH_PUFFER_S`` (Default 200 s) statt dem
#: Registry-Default (120 s) — sonst stirbt ein >120-s-Subagent mit der
#: generischen Dispatch-Meldung, bevor die deutsche Zeit-Limit-Meldung dieses
#: Moduls wirken kann. Der Puffer deckt Sub-Tool-Dispatchs und Aufräumen
#: (Client-Close) nach Ablauf des Eigenlimits. Beide Konstanten werden im Loop
#: zur Laufzeit gelesen — skaliert ein Test ``STANDARD_TIMEOUT_S``, folgt der
#: Override mit (ANNAHME: Override = Eigenlimit + fester Puffer, nie darunter).
DISPATCH_PUFFER_S = 20.0

#: Timeout je Tool-Dispatch im Sub-Loop (Registry-Default unverändert).
_TOOL_TIMEOUT_S = 120.0

#: Bericht-Kappung (T-F7-DoD „~6000 Z“) — bewusst unter dem Dispatch-Kapping
#: (8000, ``registry.TOOL_RESULT_MAX_CHARS``), damit der Parent-Bericht nicht
#: doppelt und unschöner gekappt wird.
_BERICHT_MAX_ZEICHEN = 6000

#: Tools, die ein Subagent **nie** sieht (E084) — keine Rekursion, keine
#: Orders/Buchungen, keine Schreib- oder Verwaltungs-Zugriffe. Die Allowlisten
#: der Rollen enthalten diese Namen strukturell nicht; die Konstante dient dem
#: Import-Zeit-Gegencheck und den Tests.
_NIEMALS: frozenset[str] = frozenset(
    {
        "starte_subagent",
        "platziere_order",
        "record_transaction",
        "manage_portfolios",
        "manage_watchlists",
        "manage_alerts",
        "save_strategy",
        "save_report",
        "manage_tasks",
        "create_chart",
        # Welle I (E105 additiv): Projekt-/Notiz-/Zeichnungs-/Analyse-Schreib-
        # Tools — Subagenten bleiben strikt lesend. news_suche bewusst NICHT
        # hier (lesend; Recherche-Rolle erhält es in der Allowlist).
        "manage_projects",
        "manage_notes",
        "manage_chart_drawings",
        "save_analyse",
        "trigger_data_update",
        "download_data",
    }
)

#: Gemeinsame Basis aller Rollen: ausschließlich lesende Tools (E084) — lokale
#: Kursdaten, Kennzahlen/Risiko, Backtest-**Ergebnisse**, Charts, Depot- und
#: Planspiel-Lese-Zugriffe, Screener-Regel-Scan, Projektkontext.
_BASIS_TOOLS: frozenset[str] = frozenset(
    {
        "get_market_data",
        "list_symbols",
        "get_quote",
        "search_symbols",
        "get_fundamentals",
        "get_corporate_actions",
        "compute_metrics",
        "compute_covariance",
        "run_pca",
        "estimate_var",
        "get_backtest_result",
        "read_chart",
        "read_project_context",
        "get_portfolio_performance",
        "run_screener",
        "planspiel_status",
        "planspiel_suche",
        "planspiel_instrument",
        "planspiel_depot",
        "planspiel_orders",
        "planspiel_depotwerte",
        "planspiel_regeln",
    }
)


@dataclass(frozen=True)
class RollenPreset:
    """Ein Rollen-Preset (E084): deutschsprachiger Systemprompt + feste Allowlist."""

    systemprompt: str
    tools: frozenset[str]


#: Rollen-Presets (E084, Muster ``AGENT_CRON_PRESETS``): eigene Prompts, eigene
#: Allowlists. Die finale Antwort des Subagenten ist immer EIN Fachbericht.
ROLLEN_PRESETS: dict[str, RollenPreset] = {
    "analyse": RollenPreset(
        systemprompt=(
            "Du bist der QuantLab-Subagent mit der Rolle „Analyse“ — ein "
            "eingeschränkter Helfer der QuantLab-Workstation. Du bewertest "
            "lokale Daten fachlich: gespeicherte Kursdaten, Kennzahlen, "
            "Risikomaße, Depot-Performance, Screener-Treffer und gespeicherte "
            "Charts. Du arbeitest ausschließlich lesend und kannst nichts "
            "speichern, bestellen oder verändern.\n"
            "Arbeitsweise: Hol dir alle Werte über die Tools und erfinde keine "
            "Zahlen; ist ein Ergebnis gekürzt, frage kompakter nach (Zeitraum, "
            "limit). Kennzeichne Annahmen ausdrücklich.\n"
            "Deine finale Antwort ist EIN kompakter deutschsprachiger "
            "Fachbericht mit den wichtigsten Zahlen und einem kurzen Fazit — "
            "er wird unverändert an den Hauptagenten zurückgegeben."
        ),
        tools=_BASIS_TOOLS,
    ),
    "recherche": RollenPreset(
        systemprompt=(
            "Du bist der QuantLab-Subagent mit der Rolle „Recherche“ — ein "
            "eingeschränkter Helfer der QuantLab-Workstation. Du beschaffst "
            "aktuelle Informationen (news_suche — Börsen-News je Symbol, zu "
            "Marktkategorien oder per Schlagwort; web_suche, web_lesen — "
            "Quartalszahlen, Meldungen, Volltexte) und kombinierst sie mit "
            "lokalen Fundamentals und Kursdaten. Du arbeitest ausschließlich "
            "lesend und kannst nichts speichern, bestellen oder verändern.\n"
            "Arbeitsweise: Suche gezielt, lies die treffendste Quelle mit "
            "web_lesen, trenne sauber zwischen belegten Fakten (mit Quelle: "
            "Titel/URL) und Deutung; erfinde keine Zahlen. Kennzeichne "
            "Annahmen ausdrücklich.\n"
            "Deine finale Antwort ist EIN kompakter deutschsprachiger "
            "Fachbericht mit den Kernbefunden inklusive Quellenangaben — er "
            "wird unverändert an den Hauptagenten zurückgegeben."
        ),
        tools=_BASIS_TOOLS | {"web_suche", "web_lesen", "news_suche"},
    ),
    "backtest": RollenPreset(
        systemprompt=(
            "Du bist der QuantLab-Subagent mit der Rolle „Backtest“ — ein "
            "eingeschränkter Helfer der QuantLab-Workstation. Du setzt "
            "Backtests auf (run_backtest — auch Multi-Symbol und User-"
            "Strategien user:<id>@<version>) und liest Ergebnisse aus "
            "(get_backtest_result). Du arbeitest ausschließlich lesend: "
            "Strategien speicherst du nicht, Depots und Einstellungen "
            "veränderst du nicht.\n"
            "Arbeitsweise: Wähle Zeiträume und Parameter nachvollziehbar und "
            "kompakt (große Grids vermeiden), lies Metriken über die Tools — "
            "erfinde keine Zahlen; interpretiere Rendite, Risiko und Drawdown "
            "kritisch (Beispielhafte Vergangenheit, keine Prognose). "
            "Kennzeichne Annahmen ausdrücklich.\n"
            "Deine finale Antwort ist EIN kompakter deutschsprachiger "
            "Fachbericht mit Setup, Kernzahlen und kurzem Fazit — er wird "
            "unverändert an den Hauptagenten zurückgegeben."
        ),
        tools=_BASIS_TOOLS | {"run_backtest"},
    ),
}

# Import-Zeit-Gegencheck (E084-Sicherheitsversprechen): Eine Allowlist, die
# ein verbotenes Tool enthält, wäre ein Programmfehler — laut scheitern statt
# still zu liefern.
_gesichtet = _NIEMALS.intersection(*(preset.tools for preset in ROLLEN_PRESETS.values()))
if _gesichtet:  # pragma: no cover — Programmierfehler-Schutz, tritt nie auf
    raise RuntimeError(
        f"Subagent-Allowlists enthalten verbotene Tools: {sorted(_gesichtet)}"
    )


# ---------------------------------------------------------------------------
# Registry-Filter und Client-Bau
# ---------------------------------------------------------------------------


def registry_fuer_rolle(rolle: str) -> ToolRegistry:
    """Gefilterte Registry je Rolle (Allowlist; Muster ``scheduler._gefilterte_registry``).

    Neue Instanz — die ``GLOBAL_REGISTRY`` bleibt unangetastet. Fehlt ein
    erlaubtes Tool in der ``GLOBAL_REGISTRY`` (Schema-Drift), wird es
    übersprungen und per ``logger.warning`` gemeldet (REV-F3/M4 — kein
    Fail-silent mehr); der Subagent arbeitet dann mit dem Rest.
    """
    preset = _preset_fuer(rolle)
    for name in sorted(preset.tools - set(GLOBAL_REGISTRY.names())):
        logger.warning(
            "Subagent-Rolle „%s“: Das erlaubte Tool „%s“ fehlt in der "
            "Tool-Registry (Schema-Drift) und wird dem Subagenten nicht "
            "bereitgestellt.",
            rolle,
            name,
        )
    registry = ToolRegistry()
    for name in GLOBAL_REGISTRY.names():
        if name not in preset.tools:
            continue
        spec = GLOBAL_REGISTRY.get(name)
        if spec is not None:
            registry.register(spec)
    return registry


def _preset_fuer(rolle: str) -> RollenPreset:
    """Preset je Rolle; unbekannte Rolle → deutscher Fachfehler mit erlaubten Rollen."""
    preset = ROLLEN_PRESETS.get(rolle)
    if preset is None:
        raise ToolFehler(
            f"Unbekannte Subagent-Rolle: „{rolle}“. Erlaubte Rollen: "
            f"{', '.join(sorted(ROLLEN_PRESETS))}."
        )
    return preset


def _standard_client(api_key: str) -> LLMClient:
    """Client-Fabrik des Subagenten (Injektionspunkt für Tests; Muster scheduler).

    Der Subagent übernimmt die Ownership und schließt den Client am Ende
    (``finally`` in :func:`starte_subagent` — Muster ``loop.start_run``).
    """
    return LLMClient(api_key)


# ---------------------------------------------------------------------------
# Sub-Loop (ephemär — kein chat_repo, kein RUN_REGISTRY, keine SSE-Events)
# ---------------------------------------------------------------------------


def _calls_als_wire(calls: tuple[ToolCallRequest, ...]) -> list[dict[str, Any]]:
    """Tool-Calls im z.ai/OpenAI-Format (Muster ``loop._tool_calls_als_wire``)."""
    return [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments_json},
        }
        for call in calls
    ]


def _sekunden_text(wert: float) -> str:
    """Deutsche Sekundenangabe (180.0 → „180 s“, 0.25 → „0,25 s“)."""
    text = f"{wert:.1f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"{text} s"


def _kappe_bericht(text: str, max_zeichen: int = _BERICHT_MAX_ZEICHEN) -> str:
    """Kappt den Subagenten-Bericht (DoD „~6000 Z“) mit deutschem Hinweis.

    Unter dem Limit: unverändert. Darüber: auf das Limit gekürzt, wobei der
    Hinweis die letzten Zeichen ersetzt — die Gesamtlänge bleibt garantiert
    unter dem Limit (Muster ``registry.kuerze_result``).
    """
    if len(text) <= max_zeichen:
        return text
    hinweis = (
        f"… [Bericht gekürzt: {len(text)} Zeichen gesamt, Anzeige auf "
        f"{max_zeichen} begrenzt.]"
    )
    return text[: max(0, max_zeichen - len(hinweis))] + hinweis


async def _subagent_loop(
    *,
    client: LLMClient,
    model: str,
    max_tokens: int,
    registry: ToolRegistry,
    systemprompt: str,
    aufgabe: str,
    max_iterationen: int,
) -> str:
    """Schlanker genesteter Loop: Tool-Runden bis zur finalen Bericht-Message.

    Verlauf ausschließlich im Speicher (system + user(aufgabe) +
    assistant/tool-Messages); Whitelist-Verletzungen des Modells landen als
    „Unbekanntes Tool“-Ergebnis (``ok:false``) im Sub-Verlauf — der Sub-Loop
    läuft weiter (§E). Das Limit ``max_iterationen`` stoppt mit Hinweis-Bericht.
    """
    messages: list[dict[str, Any]] = [
        system_message(systemprompt),
        user_message(aufgabe),
    ]
    tools = registry.openai_tools() or None
    iteration = 0
    while True:
        request = client.prepare_request(
            model=model,
            messages=messages,
            thinking=None,  # Parent-Thinking erbt nicht (siehe Moduldocstring)
            max_tokens=max_tokens,
            tools=tools,
        )
        final: CompletionEvent | None = None
        async for event in request.execute():
            if isinstance(event, CompletionEvent):
                final = event
        if final is None:
            raise RuntimeError("Der Subagent-Stream endete ohne Abschluss-Ereignis.")
        if not final.tool_calls:
            # Finale Assistant-Message = Bericht (leer → deutscher Resttext).
            return final.content.strip() or (
                "Der Subagent hat die Aufgabe ohne Bericht-Inhalte beendet."
            )
        if iteration >= max_iterationen:
            zwischen = final.content.strip()
            hinweis = (
                f"[Hinweis: Maximale Anzahl an Iterationen erreicht "
                f"({max_iterationen}) — der Subagent wurde gestoppt, bevor er "
                "die Aufgabe abschließen konnte.]"
            )
            if zwischen:
                return f"{zwischen}\n\n{hinweis}"
            return (
                "Der Subagent hat sein Iterations-Limit erreicht, ohne einen "
                f"Bericht zu formulieren. {hinweis}"
            )
        messages.append(
            assistant_message(
                final.content or None,
                reasoning_content=final.reasoning_content,
                tool_calls=_calls_als_wire(final.tool_calls),
            )
        )
        for call in final.tool_calls:
            result = await dispatch_tool(registry, call, timeout_s=_TOOL_TIMEOUT_S)
            messages.append(tool_message(result.call_id, result.content, name=result.name))
        iteration += 1


# ---------------------------------------------------------------------------
# Öffentlicher Einsprung (Tool-Handler delegiert hierher)
# ---------------------------------------------------------------------------

#: Sequenzierungs-Falle (E084 „sequenziell“): Max EIN Subagent gleichzeitig —
#: Modul-State statt asyncio.Lock (Begründung siehe Moduldocstring).
_subagent_aktiv: bool = False


async def starte_subagent(
    aufgabe: str,
    rolle: str,
    *,
    max_iterationen: int = MAX_ITERATIONEN,
    timeout_s: float | None = None,
    client_fabrik: Callable[[], LLMClient] | None = None,
) -> str:
    """Startet einen Subagenten und liefert seinen Fachbericht (finaler Inhalt).

    ``timeout_s=None`` liest :data:`STANDARD_TIMEOUT_S` zur **Laufzeit**
    (REV-F3/M1 — so skaliert der Test den Eigenlimit-Pfad über die
    Produktionsverdrahtung, ohne echte 180 s zu warten). ``client_fabrik`` ist
    der Test-Injektionspunkt (Muster ``scheduler._standard_client``); ohne
    Fabrik wird der Key über ``settings_repo.api_key_read`` gelesen (Muster
    ``routers/agent.py``) — fehlt er, gibt es einen deutschen Fachfehler mit
    Hinweis auf die Einstellungen. Alle Fachfehler sind
    :class:`ToolFehler` → der Parent-Run läuft weiter (§E).
    """
    global _subagent_aktiv
    if timeout_s is None:
        timeout_s = STANDARD_TIMEOUT_S
    preset = _preset_fuer(rolle)
    if _subagent_aktiv:
        raise ToolFehler(
            "Es läuft bereits ein Subagent — gleichzeitig arbeitet nur EIN "
            "Subagent. Bitte warten, bis der laufende Subagent fertig ist, "
            "und die Aufgabe dann erneut geben."
        )
    _subagent_aktiv = True
    try:
        if client_fabrik is None:
            api_key = await asyncio.to_thread(settings_repo.api_key_read)
            if not api_key:
                raise ToolFehler(
                    "Kein z.ai-API-Key konfiguriert — der Subagent kann ohne "
                    "Key nicht starten. Bitte unter Einstellungen → API-Key "
                    "hinterlegen."
                )
            client = _standard_client(api_key)
        else:
            client = client_fabrik()
        try:
            ki = await asyncio.to_thread(settings_repo.get_section, "ki")
            gesamt = asyncio.get_running_loop().create_task(
                _subagent_loop(
                    client=client,
                    model=str(ki["standard_modell"]),
                    max_tokens=int(ki["max_tokens"]),
                    registry=registry_fuer_rolle(rolle),
                    systemprompt=preset.systemprompt,
                    aufgabe=aufgabe,
                    max_iterationen=max_iterationen,
                )
            )
            try:
                bericht = await asyncio.wait_for(gesamt, timeout=timeout_s)
            except asyncio.CancelledError:
                raise  # Fremd-Cancel (Parent-Run-Abbruch) — durchreichen
            except TimeoutError:
                # Kein Task-Leak: hart cancen UND den Cancel-Throw abwarten.
                # ``gather(return_exceptions=True)`` schluckt den Sub-Cancel;
                # ein eigener Parent-Cancel bliebe als CancelledError wirksam.
                gesamt.cancel()
                await asyncio.gather(gesamt, return_exceptions=True)
                raise ToolFehler(
                    f"Der Subagent (Rolle „{rolle}“) hat das Zeit-Limit von "
                    f"{_sekunden_text(timeout_s)} überschritten und wurde "
                    "abgebrochen. Bitte die Aufgabe verkleinern oder es später "
                    "erneut versuchen."
                ) from None
            except Exception as exc:  # Sub-Störung → Fachfehler ans Parent-Tool
                raise ToolFehler(deutsche_fehlermeldung(exc)) from exc
            return _kappe_bericht(bericht)
        finally:
            await client.close()
    finally:
        _subagent_aktiv = False
