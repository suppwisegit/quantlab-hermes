"""Welle H (Task H1) — Hermes-Tool-Profil: Katalog, Pruning, Registry-Hygiene.

Netzfreie Registry-Tests nach dem Muster von ``test_mobile_app.py``: die
Autouse-Fixture snapshottet ``GLOBAL_REGISTRY`` vor jedem Test und stellt sie
nachher wieder her, damit Desktop-Tests derselben pytest-Session die volle
Registry zurückbekommen (``registriere_hermes_tools`` reduziert sie auf das
Hermes-Profil).
"""

from __future__ import annotations

from typing import Any

import pytest

from quantlab.agent.registry import GLOBAL_REGISTRY
from quantlab.hermes.registry import (
    HERMES_TOOL_NAMEN,
    VERBOTENE_TOOL_NAMEN,
    hermes_specs,
    registriere_hermes_tools,
)

# ---------------------------------------------------------------------------
# Fixtures — Registry-Snapshot/Restore, damit Desktop-Tests derselben Session
# die volle Registry zurückbekommen (registriere_hermes_tools reduziert sie).
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


_ERWARTETE_NAMEN: frozenset[str] = frozenset(
    {
        # Planspiel (10)
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
        # Daten (5)
        "get_market_data",
        "download_data",
        "get_quote",
        "get_fundamentals",
        "search_symbols",
        # Analyse (4)
        "get_indicators",
        "run_screener",
        "compute_metrics",
        "news_suche",
        # Web (1)
        "web_lesen",
    }
)


def test_hermes_profil_exakt_die_vorgegebenen_tools() -> None:
    """Profil umfasst exakt die vorgegebenen Hermes-Tools (Planspiel/Daten/Analyse/Web).

    ANNAHME: Der Task nennt „19 Tools“, listet aber 20 Namen (10+5+4+1) —
    die explizite Namensliste ist normativ, daher 20.
    """
    assert HERMES_TOOL_NAMEN == _ERWARTETE_NAMEN
    assert len(HERMES_TOOL_NAMEN) == 20


def test_hermes_pruning_katalog() -> None:
    """Nach dem Pruning: exakt die erlaubten Tools, sortiert, keine anderen."""
    registry = registriere_hermes_tools()
    namen = registry.names()
    assert len(namen) == 20
    assert namen == sorted(_ERWARTETE_NAMEN)
    assert set(namen) == HERMES_TOOL_NAMEN


def test_hermes_keine_verbotenen_tools() -> None:
    """Verbotene Desktop-/Orchestrierungs-Tools sind nach dem Pruning weg."""
    registriere_hermes_tools()
    namen = set(GLOBAL_REGISTRY.names())
    for name in VERBOTENE_TOOL_NAMEN:
        assert name not in namen, name


def test_hermes_pruning_idempotent() -> None:
    """Zweiter Aufruf ändert nichts (bereits entfernte Tools bleiben entfernt)."""
    erste_namen = registriere_hermes_tools().names()
    zweite_namen = registriere_hermes_tools().names()
    assert erste_namen == zweite_namen == sorted(HERMES_TOOL_NAMEN)


def test_hermes_specs_vollstaendig_und_sortiert() -> None:
    """hermes_specs: alle Specs, sortiert nach Name, mit Name/Beschreibung/Schema."""
    registriere_hermes_tools()
    specs = hermes_specs()
    assert [spec.name for spec in specs] == sorted(_ERWARTETE_NAMEN)
    for spec in specs:
        assert spec.description.strip(), spec.name
        assert isinstance(spec.parameters, dict), spec.name
        assert spec.parameters.get("type") == "object", spec.name


def test_registry_nach_test_wiederhergestellt() -> None:
    """Sanity: Desktop-Umfang liegt VOR dem Pruning vollständig vor.

    (Der eigentliche Restore-Nachweis ist die Autouse-Fixture; dieser Test
    sichert, dass der Snapshot überhaupt den vollen Desktop-Katalog umfasst.)
    """
    import quantlab.agent.tools  # noqa: F401

    desktop_namen = set(GLOBAL_REGISTRY.names())
    assert desktop_namen > set(HERMES_TOOL_NAMEN)
    assert "run_backtest" in desktop_namen
