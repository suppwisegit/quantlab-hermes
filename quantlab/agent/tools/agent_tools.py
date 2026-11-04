"""Subagent-Tool (Welle F, T-F7/E084): ``starte_subagent`` — dünn über subagents.

Der Handler validiert die Argumente gegen sein Pydantic-Modell
(:func:`pruefe_args`-Muster — die Registry validiert selbst nicht) und delegiert
an :func:`quantlab.agent.subagents.starte_subagent`. Alle Fachfehler (unbekannte
Rolle, Parallelitäts-Sperre, fehlender API-Key, Timeout, LLM-Fehler) kommen als
:class:`~quantlab.agent.tools._common.ToolFehler` aus dem Kern → ``ok:false``-
Tool-Ergebnis an den Parent; der Parent-Run läuft weiter (§E).

SSE/Vertrag (ANNAHME, T-F8): Der Parent sieht nur die normalen
``tool_call_started``/``tool_call_finished``-Events — Zwischenstatus je
Sub-Iteration bewusst weggelassen. Das Tool-Result ist ein JSON-Objekt
``{"rolle": …, "bericht": …}``; der Bericht (finaler Inhalt der Subagent-
Assistant-Message, auf 6000 Zeichen gekappt) ist der einklappbare Endbericht
der Tool-Line.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from quantlab.agent import subagents  # Modul-Import — Attribute erst zur Laufzeit
from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import pruefe_args, schema_von


class StarteSubagentArgs(BaseModel):
    """Argumente von ``starte_subagent``."""

    rolle: Literal["analyse", "recherche", "backtest"] = Field(
        description='Rollen-Preset des Subagenten: "analyse" (lokale Daten, '
        'Kennzahlen, Risiken, Charts), "recherche" (Websuche, Web-Lektüre, '
        'Fundamentals) oder "backtest" (Backtests aufsetzen und auslesen).',
    )
    aufgabe: str = Field(
        min_length=1,
        max_length=2000,
        description="Fachliche Aufgabe für den Subagenten — präzise und "
        "ohne Rückfragen selbstständig ausführbar, z. B. \"Analysiere die "
        'Kursentwicklung von AAPL im letzten Quartal und fasse Risiken '
        'zusammen".',
    )
    max_iterationen: int = Field(
        default=10,
        ge=3,
        le=10,
        description="Maximale Anzahl an Tool-Iterationen des Subagenten "
        "(3–10), Default 10.",
    )


async def starte_subagent(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Spawnt einen eingeschränkten Subagenten und liefert seinen Fachbericht."""
    args = pruefe_args(StarteSubagentArgs, roh_args, "starte_subagent")
    bericht = await subagents.starte_subagent(
        aufgabe=args.aufgabe,
        rolle=args.rolle,
        max_iterationen=args.max_iterationen,
    )
    return {"rolle": args.rolle, "bericht": bericht}


# ---------------------------------------------------------------------------
# Registrierung (Muster daten.py: _registrieren + Modul-Call)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="starte_subagent",
            description=(
                "Startet einen eingeschränkten Subagenten (Rollen: analyse, "
                "recherche, backtest), der eine Nebenanalyse selbstständig "
                "mit eigenen lesenden Werkzeugen ausführt und am Ende EINEN "
                "Fachbericht liefert — für umfangreiche Nebenanalysen gedacht, "
                "die den Hauptdialog nicht aufblähen sollen. Läuft mit "
                "Zeit-Limit (180 s) und maximal 10 Iterationen; kann weder "
                "Orders platzieren noch etwas speichern oder verändern. "
                'Beispiel: {"rolle": "recherche", "aufgabe": "Recherchiere '
                'die letzten Quartalszahlen von SAP und fasse sie zusammen"}.'
            ),
            parameters=schema_von(StarteSubagentArgs),
            handler=starte_subagent,
        )
    )


_registrieren()
