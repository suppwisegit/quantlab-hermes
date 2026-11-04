"""Welle H (Task H3) — Hermes-MCP-Server: Katalog, Handshake, Dispatch, Fehler.

Netzfreie Tests: der Tool-Handler wird für die Roundtrip-Tests per Monkeypatch
auf ``hermes_specs`` injiziert (kein echter Daten-/Netz-Call); der Subprocess-
Test startet ``python -m quantlab.hermes.mcp_server`` als echten stdio-Prozess
und prüft den JSON-RPC-Handshake inklusive Instructions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.hermes.mcp_server import baue_server
from quantlab.hermes.registry import (
    HERMES_TOOL_NAMEN,
    registriere_hermes_tools,
)
from quantlab.hermes.skill import HERMES_SKILL

# ---------------------------------------------------------------------------
# Fixtures — Registry-Snapshot/Restore wie in test_hermes_registry.py, damit
# Desktop-Tests derselben pytest-Session die volle Registry zurückbekommen.
# ---------------------------------------------------------------------------


def _snapshot_registry() -> dict[str, Any]:
    return {name: GLOBAL_REGISTRY.get(name) for name in GLOBAL_REGISTRY.names()}


def _restauriere_registry(snapshot: dict[str, Any]) -> None:
    for name, spec in snapshot.items():
        if spec is not None and GLOBAL_REGISTRY.get(name) is None:
            GLOBAL_REGISTRY.register(spec)


@pytest.fixture(autouse=True)
def _registry_roundtrip() -> Any:
    """Autouse: Snapshot VOR und Restore NACH jedem Test dieses Moduls."""
    import quantlab.agent.tools  # noqa: F401 — Registry VOR dem Snapshot füllen

    snapshot = _snapshot_registry()
    yield
    _restauriere_registry(snapshot)


def _test_spec(handler: Any) -> ToolSpec:
    """Baut ein Fake-Hermes-Tool (echte Handler-Signatur: async, dict→Ergebnis)."""
    return ToolSpec(
        name="planspiel_status",  # echter Hermes-Name → Schema-Katalog bleibt stimmig
        description="Test-Dummy (gemockter Handler)",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )


# ---------------------------------------------------------------------------
# tools/list über den Katalog
# ---------------------------------------------------------------------------


async def test_list_tools_liefert_20_hermes_tools_mit_schema() -> None:
    """tools/list liefert exakt die 20 Hermes-Tools mit eigenen inputSchemas."""
    registriere_hermes_tools()
    async with Client(baue_server(), raise_exceptions=True) as client:
        result = await client.list_tools()
    namen = {tool.name for tool in result.tools}
    assert namen == set(HERMES_TOOL_NAMEN)
    assert len(result.tools) == 20
    for tool in result.tools:
        assert tool.description and tool.description.strip(), tool.name
        assert tool.input_schema.get("type") == "object", tool.name


# ---------------------------------------------------------------------------
# initialize/Handshake — Instructions (Skill) via Subprocess über stdio
# ---------------------------------------------------------------------------


async def test_subprocess_handshake_liefert_instructions_und_tools() -> None:
    """`python -m quantlab.hermes.mcp_server`: echter stdio-Prozess, Handshake.

    Vollständiger JSON-RPC-Handshake über stdio: initialize liefert den
    Skill-Text als Instructions (Kern-Begriff „Planspiel-Börse“), tools/list
    den vollen 20er-Katalog. Dokumentierte Wahl (Task H3): Subprozess-Beleg
    via StdioServerParameters statt rohem JSON-RPC-Gebastel — das SDK baut
    dieselben Frames, aber robust gegen Protokolldetails.
    """
    backend = Path(__file__).resolve().parents[1]
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "quantlab.hermes.mcp_server"],
        cwd=backend,
    )
    async with Client(params) as client:
        instructions = client.instructions
        assert instructions is not None
        assert "Planspiel-Börse" in instructions
        assert HERMES_SKILL[:80] in instructions
        tools = await client.list_tools()
    assert {tool.name for tool in tools.tools} == set(HERMES_TOOL_NAMEN)


# ---------------------------------------------------------------------------
# call_tool-Roundtrip mit gemocktem Handler (kein Netz)
# ---------------------------------------------------------------------------


async def test_call_tool_roundtrip_dict_ergebnis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler-dict wird als JSON-Text (ensure_ascii=False) zurückgeliefert."""
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, " Echo ": args}

    monkeypatch.setattr(
        "quantlab.hermes.mcp_server.hermes_specs", lambda: [_test_spec(handler)]
    )
    async with Client(baue_server(), raise_exceptions=True) as client:
        result = await client.call_tool("planspiel_status", {"x": 1})
    text = result.content[0].text
    assert json.loads(text) == {"ok": True, " Echo ": {"x": 1}}
    assert result.is_error is not True


async def test_call_tool_kappt_lange_ergebnisse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ergebnisse über 8.000 Zeichen werden wie im Agent-Loop gekappt (§T14)."""
    lang = "ä" * 9_000

    async def handler(args: dict[str, Any]) -> str:
        return lang

    monkeypatch.setattr(
        "quantlab.hermes.mcp_server.hermes_specs", lambda: [_test_spec(handler)]
    )
    async with Client(baue_server(), raise_exceptions=True) as client:
        result = await client.call_tool("planspiel_status", {})
    text = result.content[0].text
    assert len(text) <= 8_000
    assert "gekürzt" in text


async def test_call_tool_unbekannt_is_error_kein_crash() -> None:
    """Unbekannter Tool-Name → is_error-Ergebnis mit deutscher Meldung."""
    registriere_hermes_tools()
    async with Client(baue_server(), raise_exceptions=True) as client:
        result = await client.call_tool("existiert_nicht", {})
        # danach lebt die Session noch (kein Crash):
        tools = await client.list_tools()
    assert result.is_error is True
    assert "Unbekanntes Tool" in result.content[0].text
    assert len(tools.tools) == 20


async def test_call_tool_handler_exception_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler-Exception → is_error-Ergebnis, Server bleibt bedienbar."""

    async def handler(args: dict[str, Any]) -> str:
        raise ValueError("Planspiel-API down")

    monkeypatch.setattr(
        "quantlab.hermes.mcp_server.hermes_specs", lambda: [_test_spec(handler)]
    )
    async with Client(baue_server(), raise_exceptions=True) as client:
        result = await client.call_tool("planspiel_status", {})
    assert result.is_error is True
    assert "Tool-Fehler" in result.content[0].text
    assert "Planspiel-API down" in result.content[0].text


async def test_call_tool_handler_timeout_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hängender Handler wird nach DEFAULT_TOOL_TIMEOUT_S abgebrochen."""
    import asyncio

    async def handler(args: dict[str, Any]) -> str:
        await asyncio.sleep(300)
        return "spät"

    monkeypatch.setattr(
        "quantlab.hermes.mcp_server.hermes_specs", lambda: [_test_spec(handler)]
    )
    monkeypatch.setattr("quantlab.hermes.mcp_server.DEFAULT_TOOL_TIMEOUT_S", 0.2)
    async with Client(baue_server(), raise_exceptions=True) as client:
        result = await client.call_tool("planspiel_status", {})
    assert result.is_error is True
    assert "Timeout" in result.content[0].text
