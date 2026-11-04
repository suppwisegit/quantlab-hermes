"""Welle H (Task H4, E-H05) — optionaler Recorder-Tick des Hermes-MCP-Servers.

Netzfrei: Sweep, Client-Fabrik und Backoff-Registrierung werden per Monkeypatch
ersetzt (kein echter Planspiel-Client, kein Login, keine DB); das
Handelszeiten-Gate wird über die injizierbare ``now``-Uhr deterministisch in
die Handelszeiten gelegt, damit die Tests nicht von der Uhrzeit abhängen.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from quantlab.hermes import recorder_tick
from quantlab.hermes.recorder_tick import (
    ENV_AKTIV,
    ENV_INTERVALL_S,
    intervall_aus_env,
    starte_recorder_ticker,
    stoppe_recorder_ticker,
)


def _in_handelszeiten_uhr() -> datetime:
    """Feste Uhr innerhalb der Handelszeiten (12:00 UTC = 13:00/14:00 Berlin)."""
    return datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


_IN_HANDELSZEITEN: Callable[[], datetime] = _in_handelszeiten_uhr


async def _warte_bis(bedingung: Callable[[], bool], timeout_s: float = 2.0) -> None:
    """Pollt kurz, bis die Bedingung steht (Thread-Offload des Ticks abwarten)."""
    for _ in range(int(timeout_s / 0.01)):
        if bedingung():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Bedingung wurde innerhalb des Timeouts nicht erfüllt.")


@pytest.fixture
def gemockter_recorder(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Ersetzt Sweep/Client/Backoff durch netzfreie Zähler-Stubs."""
    zustand: dict[str, Any] = {"aufrufe": 0, "wirft": False}

    def fake_sweep(client: Any, *, jetzt: datetime) -> dict[str, Any]:
        zustand["aufrufe"] += 1
        if zustand["wirft"]:
            raise RuntimeError("Planspiel-API down")
        return {"sweeps": 3, "instrumente": 10, "fokussiert": 2, "fehler": 0}

    fortsetzung = asyncio.Event()

    async def fake_schlafe(sekunden: float) -> None:
        zustand.setdefault("takts", []).append(sekunden)
        await fortsetzung.wait()

    monkeypatch.setattr(recorder_tick.recorder_service, "sweep", fake_sweep)
    monkeypatch.setattr(recorder_tick, "_standard_client", lambda: "fake-client")
    monkeypatch.setattr(
        recorder_tick.recorder_service, "fehler_registrieren", lambda: 1
    )
    monkeypatch.setattr(recorder_tick, "_schlafe", fake_schlafe)
    zustand["fortsetzung"] = fortsetzung
    return zustand


# ---------------------------------------------------------------------------
# Env-Gate
# ---------------------------------------------------------------------------


async def test_ohne_env_gate_kein_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (Env unset bzw. falscher Wert): starte_recorder_ticker() → None."""
    for wert in (None, "", "0", "false", "no", "off"):
        if wert is None:
            monkeypatch.delenv(ENV_AKTIV, raising=False)
        else:
            monkeypatch.setenv(ENV_AKTIV, wert)
        assert await starte_recorder_ticker() is None, wert


async def test_env_gate_an_startet_und_stoppt_task(
    monkeypatch: pytest.MonkeyPatch, gemockter_recorder: dict[str, Any]
) -> None:
    """Env an: Task läuft, erster Sweep sofort (gemockt, kein Netz); Stopp sauber."""
    monkeypatch.setenv(ENV_AKTIV, "TRUE")  # case-insensitive
    task = await starte_recorder_ticker(now=_IN_HANDELSZEITEN)
    assert task is not None and not task.done()
    await _warte_bis(lambda: gemockter_recorder["aufrufe"] >= 1)
    assert gemockter_recorder["aufrufe"] == 1
    assert gemockter_recorder["takts"][0] == 60  # Default-Basis-Intervall
    await stoppe_recorder_ticker(task)
    assert task.done()


async def test_stoppe_recorder_ticker_vertraegt_none() -> None:
    """stoppe_recorder_ticker(None) ist ein No-Op (Server-Start ohne Env)."""
    await stoppe_recorder_ticker(None)


# ---------------------------------------------------------------------------
# Intervall-Env-Clamp (30–3600, Default 60)
# ---------------------------------------------------------------------------


def test_intervall_clamp_ungueltige_werte(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ungültige Env-Werte fallen auf Default 60, Fenster-Verletzungen klemmen."""
    faelle = [
        (None, 60),  # unset → Default
        ("", 60),
        ("abc", 60),
        ("10", 30),  # unter Untergrenze → hochklampen
        ("-5", 30),
        ("0", 30),
        ("99999", 3600),  # über Obergrenze → runterklampen
        ("120", 120),  # gültig bleibt gültig
    ]
    for wert, erwartet in faelle:
        if wert is None:
            monkeypatch.delenv(ENV_INTERVALL_S, raising=False)
        else:
            monkeypatch.setenv(ENV_INTERVALL_S, wert)
        assert intervall_aus_env() == erwartet, wert


# ---------------------------------------------------------------------------
# Fehler-Resilienz: Task überlebt Sweep-Exceptions (Backoff statt Absturz)
# ---------------------------------------------------------------------------


async def test_sweep_exception_task_bleibt_am_leben(
    monkeypatch: pytest.MonkeyPatch,
    gemockter_recorder: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wirft der Sweep, bleibt der Task am Leben und loggt (stderr-Logger)."""
    monkeypatch.setenv(ENV_AKTIV, "1")
    gemockter_recorder["wirft"] = True
    with caplog.at_level("ERROR", logger="quantlab.hermes.recorder_tick"):
        task = await starte_recorder_ticker(now=_IN_HANDELSZEITEN)
        assert task is not None
        await _warte_bis(lambda: gemockter_recorder["aufrufe"] >= 1)
        # zweiter Tick nach freigegebenem Schlaf — der Task lebt noch
        gemockter_recorder["fortsetzung"].set()
        await _warte_bis(lambda: gemockter_recorder["aufrufe"] >= 2)
        assert not task.done(), "Task darf an einer Sweep-Exception nicht sterben"
    assert any(
        "Hermes-Recorder-Sweep fehlgeschlagen" in r.message for r in caplog.records
    )
    # Backoff statt Basis-Takt: fehlgeschlagene Serie verdoppelt den Schlaf.
    assert gemockter_recorder["takts"][0] == 120  # 60 × 2**min(1, 4)
    await stoppe_recorder_ticker(task)
    assert task.done()
