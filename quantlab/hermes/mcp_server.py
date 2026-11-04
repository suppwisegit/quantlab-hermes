"""Hermes MCP-Server (Welle H, Task H3) — ``python -m quantlab.hermes.mcp_server``.

Standard-IO-Server, der dem autonomen Trading-Agenten Hermes (Raspberry Pi 4)
das kuratierte Hermes-Toolprofil über MCP (Model Context Protocol) zugänglich
macht:

- Die 20 Hermes-Tools (:func:`quantlab.hermes.registry.hermes_specs`) werden
  mit ihren **eigenen JSON-Schemas** (``ToolSpec.parameters``) am Low-Level-
  Server (:class:`mcp.server.lowlevel.Server`) angemeldet — das High-Level-
  ``MCPServer``-API inferiert Schemas nur aus Funktions-Signaturen und könnte
  die handgebauten Schemas nicht 1:1 liefern.
- Der Skill-Text (:data:`quantlab.hermes.skill.HERMES_SKILL`) wird als
  ``instructions`` des initialize-Results ausgeliefert (Constructor-Argument
  des Low-Level-Servers, im SDK 2.3.0 vorhanden und verifiziert).
- Ergebnis-Serialisierung und -Kapping folgen dem Agent-Loop
  (:mod:`quantlab.agent.registry`): Strings 1:1, Objekte als JSON mit
  ``ensure_ascii=False``, Kapping bei 8.000 Zeichen (:func:`kuerze_result`).
- Fehlerisolierung wie im Agent-Loop: unbekanntes Tool und Handler-Exception
  werden als ``is_error``-Tool-Ergebnis gemeldet, statt den Server zu crashen.

Wire-Protokoll-Disziplin: stdout ist JSON-RPC-Transport — kein ``print()``;
Logging läuft ausschließlich über stderr.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)

from quantlab.agent.registry import (
    DEFAULT_TOOL_TIMEOUT_S,
    ToolSpec,
    kuerze_result,
)
from quantlab.hermes.recorder_tick import starte_recorder_ticker, stoppe_recorder_ticker
from quantlab.hermes.registry import hermes_specs, registriere_hermes_tools
from quantlab.hermes.skill import HERMES_SKILL

logger = logging.getLogger(__name__)

#: Server-Name im MCP-Handshake (protocolVersion/initialize → serverInfo.name).
SERVER_NAME = "hermes"

#: Server-Version im MCP-Handshake (folgt der Projekt-Version des Sidecars).
SERVER_VERSION = "0.1.0"


def _tool_aus_spec(spec: ToolSpec) -> Tool:
    """Baut den MCP-Tool-Eintrag aus einer ToolSpec (Schema 1:1, ohne Inferenz)."""
    return Tool(name=spec.name, description=spec.description, input_schema=spec.parameters)


def _ergebnis_als_text(ergebnis: Any) -> str:
    """Serialisiert ein Handler-Ergebnis wie der Agent-Loop (§T14).

    Strings 1:1, alles andere als JSON mit ``ensure_ascii=False``; danach
    Result-Kapping auf 8.000 Zeichen (:func:`kuerze_result`).
    """
    content = ergebnis if isinstance(ergebnis, str) else json.dumps(ergebnis, ensure_ascii=False)
    return kuerze_result(content)


def _fehler(text: str) -> CallToolResult:
    """MCP-Fehlerergebnis (``is_error=True``) — der Server läuft weiter."""
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def baue_server() -> Server[Any]:
    """Baut den Low-Level-MCP-Server mit den Hermes-Tools und dem Skill.

    Die Tools stammen aus :func:`hermes_specs` (Aufrufer muss das Pruning via
    :func:`quantlab.hermes.registry.registriere_hermes_tools` veranlasst haben
    — :func:`main` tut dies; Tests injizieren eigene Test-Tools in die
    Registry, bevor sie diese Funktion rufen, und lesen die Specs daher zum
    Aufrufzeitpunkt).
    """
    specs: dict[str, ToolSpec] = {spec.name: spec for spec in hermes_specs()}

    async def on_list_tools(
        ctx: Any, params: Any  # noqa: ANN401 — SDK-Context-Typ ist generisch
    ) -> ListToolsResult:
        return ListToolsResult(tools=[_tool_aus_spec(spec) for spec in specs.values()])

    async def on_call_tool(ctx: Any, params: CallToolRequestParams) -> CallToolResult:  # noqa: ANN401
        spec = specs.get(params.name)
        if spec is None:
            verfuegbare = ", ".join(sorted(specs)) or "(keine)"
            return _fehler(
                f"Unbekanntes Tool: „{params.name}“. Verfügbare Tools: {verfuegbare}"
            )
        args = params.arguments or {}
        try:
            ergebnis = await asyncio.wait_for(spec.handler(args), timeout=DEFAULT_TOOL_TIMEOUT_S)
            text = _ergebnis_als_text(ergebnis)
        except TimeoutError:
            return _fehler(
                f"Tool „{params.name}“ wurde nach "
                f"{DEFAULT_TOOL_TIMEOUT_S:.0f} s abgebrochen (Timeout)."
            )
        except Exception as exc:  # noqa: BLE001 — Tool-Fehler isolieren, Server läuft weiter
            logger.warning("Tool „%s“ fehlgeschlagen: %s", params.name, exc, exc_info=True)
            return _fehler(f"Tool-Fehler in „{params.name}“: {exc}")
        return CallToolResult(content=[TextContent(type="text", text=text)])

    return Server(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=HERMES_SKILL,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def _konfiguriere_logging() -> None:
    """Logging auf stderr (stdout ist MCP-Wire-Protokoll — niemals dorthin)."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


def main() -> None:
    """Startet den Hermes-MCP-Server auf stdio (blockierend).

    Reihenfolge nach dem Muster von :mod:`quantlab.mobile_server`: Verzeichnisse
    und Migrationen zuerst, dann Registry-Pruning auf das Hermes-Profil, dann
    Logging (stderr) und der blockierende stdio-Run.
    """
    from quantlab.config import ensure_directories
    from quantlab.migrations import run_migrations

    _konfiguriere_logging()
    ensure_directories()
    run_migrations()
    registriere_hermes_tools()

    import anyio
    from mcp.server.stdio import stdio_server

    server = baue_server()

    async def _run() -> None:
        # E-H05/H4: optionaler Recorder-Tick im Server-Prozess (Env-Gate
        # ``QUANTLAB_HERMES_RECORDER`` — ohne Env startet kein Task). Try/finally
        # stoppt ihn beim Server-Ende sauber (Muster scheduler.stoppe_scheduler).
        ticker = await starte_recorder_ticker()
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                    raise_exceptions=False,
                )
        finally:
            await stoppe_recorder_ticker(ticker)

    anyio.run(_run)


if __name__ == "__main__":  # pragma: no cover — manueller Start
    main()
