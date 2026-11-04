"""Run-Event-Buffer mit Replay (T12, PLAN §C.2, E009).

Je Run hält ein :class:`RunEventBuffer` alle SSE-Events im Speicher (In-Memory,
Prozesslebensdauer des Sidecars). Jede SSE-Verbindung — auch ein später
Reconnect oder ein Connect erst nach dem Run-Ende — bekommt via :meth:`RunEventBuffer.stream`
zuerst die **komplette Historie** und danach live weitere Events; nach dem
Terminal-Event (``run_finished`` / ``run_error`` / ``run_cancelled``) endet der
Stream sauber. Das macht EventSource-Reconnects verlustfrei (DoD T12) — das
Frontend muss Terminal-Events deshalb idempotent verarbeiten.

Heartbeat: übernimmt ``EventSourceResponse(ping=…)`` des Routers (SSE-Kommentar),
nicht dieser Puffer.

Nebenläufigkeit: ``append`` wird von der Loop-Task (synchron, ohne await)
gerufen, ``stream`` von beliebig vielen Konsumenten im selben Event-Loop;
Wake-ups laufen über private :class:`asyncio.Event` pro Konsument (Broadcast).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

#: Terminal-Events — nach ihnen endet der Stream (Puffer gilt als geschlossen).
TERMINAL_EVENTS: frozenset[str] = frozenset({"run_finished", "run_error", "run_cancelled"})


@dataclass(frozen=True)
class SSEHeartbeat:
    """Marker für einen SSE-Kommentar-Keepalive („: heartbeat“) im Stream."""


@dataclass(frozen=True)
class RunEvent:
    """Ein Run-Event mit fortlaufender Nummer (Replay-Reihenfolge)."""

    seq: int
    name: str
    data: dict[str, Any]


@dataclass
class RunEventBuffer:
    """In-Memory-Eventpuffer je Run (C.2) — Quelle für ``GET …/stream`` inkl. Replay."""

    _events: list[RunEvent] = field(default_factory=list)
    _waiter: set[asyncio.Event] = field(default_factory=set)
    _seq: int = 0
    _geschlossen: bool = False

    def append(self, name: str, data: dict[str, Any] | None = None) -> RunEvent:
        """Hängt ein Event an (nach dem Terminal-Event nicht mehr erlaubt)."""
        if self._geschlossen:
            raise RuntimeError(
                f"Event nach Terminal-Event verworfen: {name!r} (Run-Buffer ist geschlossen)."
            )
        self._seq += 1
        event = RunEvent(seq=self._seq, name=name, data=dict(data or {}))
        self._events.append(event)
        if name in TERMINAL_EVENTS:
            self._geschlossen = True
        for waiter in self._waiter:
            waiter.set()
        return event

    @property
    def geschlossen(self) -> bool:
        """Terminal-Event bereits angehängt?"""
        return self._geschlossen

    def history(self) -> list[RunEvent]:
        """Bisherige Events (Kopie der Liste, Events selbst immutable)."""
        return list(self._events)

    async def stream(
        self, *, heartbeat_interval_s: float | None = None
    ) -> AsyncIterator[RunEvent | SSEHeartbeat]:
        """Volle Historie, dann live weiter; Ende nach dem Terminal-Event.

        ``heartbeat_interval_s`` (optional): nach so vielen wartenden Sekunden wird
        ein :class:`SSEHeartbeat` geliefert, falls kein Event kam — der Router
        formatiert ihn als SSE-Kommentar. ``None`` (Default) verlässt sich auf den
        Ping des SSE-Frameworks.
        """
        position = 0
        while True:
            while position < len(self._events):
                event = self._events[position]
                position += 1
                yield event
                if event.name in TERMINAL_EVENTS:
                    return
            if self._geschlossen:
                # Terminal-Event wurde bereits konsumiert (obige Schleife) —
                # sonstiger geschlossener Zustand ist hier unerreichbar.
                return
            waiter = asyncio.Event()
            self._waiter.add(waiter)
            try:
                if heartbeat_interval_s is None:
                    await waiter.wait()
                else:
                    try:
                        await asyncio.wait_for(waiter.wait(), timeout=heartbeat_interval_s)
                    except TimeoutError:
                        yield SSEHeartbeat()
            finally:
                self._waiter.discard(waiter)
