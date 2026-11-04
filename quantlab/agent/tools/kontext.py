"""Run-Kontext für Agent-Tools (T14): das aktive Projekt des laufenden Runs.

Die Tool-Handler bekommen laut Registry-Vertrag (T12) nur den geparsten
Argumente-String — Projektbezug (``create_chart``/``save_report``/Tasks) soll
aber „aus dem Run" kommen, ohne dass das LLM ihn raten muss. Deshalb setzt
:mod:`quantlab.agent.loop` beim Start des Run-Tasks diese :class:`~contextvars.
ContextVar` auf ``handle.project_id`` (Tasks isolieren ihren Kontext — mehrere
parallele Runs stören sich nicht; direkte Tool-Aufrufe außerhalb eines Runs,
z. B. API-Smoke, sehen ``None``).
"""

from __future__ import annotations

import contextvars

#: Projekt-ID des laufenden Runs (``None`` = globaler Chat ohne Projekt).
laufendes_projekt: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "laufendes_projekt", default=None
)


def aktives_projekt() -> int | None:
    """Projekt-ID des Runs, in dem das Tool gerade ausgeführt wird (oder ``None``)."""
    return laufendes_projekt.get()
