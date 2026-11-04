"""Globaler SSE-Endpunkt (T4, PLAN §C.2): ``GET /api/events``.

Liefert die Events des globalen Buses (:mod:`quantlab.events`) als
``event:``/``data:``-Paare (JSON-Payload); Heartbeat alle 15 s als
SSE-Kommentar via ``EventSourceResponse(ping=…)``. Kein Replay — frische
Clients holen ihren Zustand per REST (Modul-Docstring events.py).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from sse_starlette import EventSourceResponse, ServerSentEvent

from quantlab import events as event_bus

router = APIRouter(tags=["data"])

#: Heartbeat-Intervall (s; PLAN §C.2 „Heartbeat alle 15 s“).
_SSE_PING_S = 15


@router.get("/events")
async def events_stream() -> EventSourceResponse:
    """App-weiter SSE-Strom: download_progress/download_finished/data_changed."""

    async def generator() -> AsyncIterator[ServerSentEvent]:
        async for nachricht in event_bus.stream():
            yield ServerSentEvent(
                event=nachricht["name"],
                data=json.dumps(nachricht["daten"], ensure_ascii=False),
            )

    return EventSourceResponse(generator(), ping=_SSE_PING_S)
