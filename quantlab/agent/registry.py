"""Tool-Registry & Dispatch-Mechanik (T12); die 15 fachlichen Tools registriert T14.

Vertrag (PLAN §E, RECHERCHE Block 1.3):

- Tools im OpenAI-Function-Schema (``{"type": "function", "function": {name,
  description, parameters}}``), ``tool_choice`` ausschließlich ``"auto"``.
- ``function.arguments`` ist ein **JSON-String** → Parsen mit Fehlerbehandlung.
- **Tool-Fehler sind Tool-Ergebnisse, kein Run-Abbruch:** :func:`dispatch_tool`
  wirft für fachliche Fehler bewusst keine Exception, sondern liefert ein
  ``ToolResult(ok=False, …)``, das als ``role:tool``-Message zurückgeht.

Die Registry ist bewusst eine einfache Datenstruktur (Name → ToolSpec): T14
registriert die 15 Tools aus PLAN §E beim Import der Tool-Module in
``GLOBAL_REGISTRY``; der Loop bekommt seine Instanz per Parameter (Tests
nutzen eigene Instanzen mit Fake-Tools).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from quantlab.agent.llm_client import ToolCallRequest

logger = logging.getLogger(__name__)

#: Ein Tool-Handler bekommt die geparsten Argumente und liefert den Inhalt der
#: Tool-Ergebnis-Message: ein String (wird 1:1 gesendet) oder ein JSON-serializebares
#: Objekt (wird mit ``ensure_ascii=False`` kodiert; nicht-serialisierbare Objekte
#: werden als ``ok=False``-Tool-Ergebnis gemeldet, siehe :func:`dispatch_tool`).
ToolHandler = Callable[[dict[str, Any]], Awaitable[str | dict[str, Any] | list[Any]]]

#: Timeout je Tool-Ausführung (s) — ein hängendes Tool darf einen Run nicht blockieren.
DEFAULT_TOOL_TIMEOUT_S = 120.0

#: Result-Kapping (§T14): Maximale Länge eines Tool-Ergebnisses in Zeichen.
#: Große OHLC-Antworten (get_market_data ohne start/end/limit) blähen den
#: LLM-Kontext auf und kosten Geld — alles darüber hinaus wird mit deutschem
#: Hinweis gekappt. Die Konstante ist bewusst hier (Dispatch-Schicht) und
#: damit für **alle** Tools einheitlich; die Zusammenfassung relevanter Felder
#: steht bei :func:`get_market_data` **vor** den Balken im JSON, damit beim
#: Kappen nur die rohen Balken verloren gehen.
TOOL_RESULT_MAX_CHARS = 8_000

#: Hinweis, der an gekappte Ergebnisse angehängt wird (deutsch, handlungs-
#: orientiert — der Agent soll start/end/limit nachsetzen können).
_KUERZUNGS_HINWEIS = (
    "… [Ergebnis gekürzt: {original} Zeichen übersteigen das Limit von "
    "{limit} Zeichen. Bitte mit kompakteren Parametern erneut abfragen "
    "(z. B. start/end bzw. kleineres limit bei get_market_data) oder "
    "Teilauswertungen serverseitig durchführen lassen.]"
)


def kuerze_result(text: str, max_zeichen: int = TOOL_RESULT_MAX_CHARS) -> str:
    """Kappt ein Tool-Ergebnis auf ``max_zeichen`` (Result-Kapping, §T14).

    Unter dem Limit: unverändert. Darüber: auf das Limit gekürzt plus
    deutscher Kürzungshinweis (der Hinweis ersetzt die letzten Zeichen, damit
    die Gesamtlänge garantiert eingehalten wird).
    """
    if len(text) <= max_zeichen:
        return text
    hinweis = _KUERZUNGS_HINWEIS.format(original=len(text), limit=max_zeichen)
    return text[: max(0, max_zeichen - len(hinweis))] + hinweis


@dataclass(frozen=True)
class ToolSpec:
    """Ein registriertes Agent-Tool (OpenAI-Function-Schema + Handler)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON-Schema: {"type": "object", "properties", "required"}
    handler: ToolHandler

    def openai_schema(self) -> dict[str, Any]:
        """Das Tool im z.ai/OpenAI-Request-Format (§E)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """Ergebnis eines Dispatchs — geht unverändert als Tool-Message in den Verlauf."""

    call_id: str
    name: str
    ok: bool
    content: str


class ToolRegistry:
    """Namensbasierte Tool-Registry (Dispatch-Datenstruktur, T12)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Registriert ein Tool; Doppelregistrierung desselben Namens ist ein Fehler."""
        if spec.name in self._tools:
            raise ValueError(f"Tool bereits registriert: {spec.name}")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> bool:
        """Entfernt ein Tool (Test-Hygiene, dynamische Kataloge) — ``True`` wenn entfernt."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)

    def openai_tools(self) -> list[dict[str, Any]]:
        """Alle Tools im Request-Format (§E); leere Liste, wenn nichts registriert ist."""
        return [spec.openai_schema() for spec in self._tools.values()]


#: Globale Registry — T14 füllt sie (15 Tools aus PLAN §E); der Loop nutzt sie als Default.
GLOBAL_REGISTRY = ToolRegistry()


async def dispatch_tool(
    registry: ToolRegistry, call: ToolCallRequest, *, timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
) -> ToolResult:
    """Führt einen Tool-Call aus — Fehler werden als Tool-Ergebnis (``ok=False``) gemeldet.

    Fehlerpfade (alles deutsche Nutzertexte, §E-Regel „kein Run-Abbruch“):
    unbekanntes Tool, ungültiges Argumente-JSON, nicht-Objekt-Argumente,
    Handler-Exception (zusätzlicher ``logger.warning`` mit Traceback zur
    Diagnose), Timeout und nicht-JSON-serializebares Result — auch Letzteres
    (das ``json.dumps`` läuft bewusst im selben ``try``-Block) wird als
    ``ok=False``-Tool-Ergebnis gemeldet, statt den Run abzubrechen.
    """
    spec = registry.get(call.name)
    if spec is None:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            content=f"Unbekanntes Tool: „{call.name}“. Verfügbare Tools: "
            f"{', '.join(registry.names()) or '(keine)'}",
        )
    try:
        args = json.loads(call.arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            content=f"Ungültige Argumente für „{call.name}“ (kein gültiges JSON): {exc}",
        )
    if not isinstance(args, dict):
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            content=f"Ungültige Argumente für „{call.name}“: Es wird ein JSON-Objekt erwartet.",
        )
    try:
        ergebnis = await asyncio.wait_for(spec.handler(args), timeout=timeout_s)
        # Bewusst im selben try-Block: Ein nicht-JSON-serializebares Result
        # (z. B. set()) wäre sonst eine unbehandelte Exception im Loop und
        # würde den Run abbrechen — so wird es ein ok:false-Tool-Ergebnis.
        content = (
            ergebnis if isinstance(ergebnis, str) else json.dumps(ergebnis, ensure_ascii=False)
        )
    except TimeoutError:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            content=f"Tool „{call.name}“ wurde nach {timeout_s:.0f} s abgebrochen (Timeout).",
        )
    except Exception as exc:  # noqa: BLE001 — Tool-Fehler isolieren (§E), Run läuft weiter
        logger.warning("Tool „%s“ fehlgeschlagen: %s", call.name, exc, exc_info=True)
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            content=f"Tool-Fehler in „{call.name}“: {exc}",
        )
    # Result-Kapping (§T14) — hier ist der Erfolgspfad; Fehlerpfade oben
    # liefern kurze, strukturierte deutsche Meldungen und bleiben ganz erhalten.
    content = kuerze_result(content)
    return ToolResult(call_id=call.call_id, name=call.name, ok=True, content=content)
