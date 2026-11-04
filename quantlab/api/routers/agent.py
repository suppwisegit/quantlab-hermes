"""Agent-Endpunkte (T12, PLAN §C.1/C.2): Runs, Cancel, Run-SSE mit Replay, Modellkatalog.

Die Endpunkte, die auf die Repos zugreifen, sind normale ``def`` (FastAPI-
Threadpool, Muster ``quantlab.db``); ``POST /api/agent/runs`` und Cancel/Stream
sind ``async``, weil dort ein ``asyncio.Task`` erzeugt/erwartet bzw. ein
SSE-Stream zurückgegeben wird — ihre DB-Zugriffe laufen über
``asyncio.to_thread`` (db.py-Docstring: „asyncio.to_thread ist nur nötig, wenn
ein async-Endpunkt doch direkt auf ein Repo zugreift“).

Key-Vertrag (DoD h): ``POST /api/agent/runs`` prüft den z.ai-Key **vor** jedem
Client-Bau/Netzcall — fehlt er, gibt es 400 mit deutscher Meldung und es wird
kein Transport berührt. Der Client-Bau läuft über eine austauschbare Fabrik
(``_standard_client``; Tests injizieren den Fake-LLM-Transport dort).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse, ServerSentEvent

from quantlab.agent import llm_client
from quantlab.agent import loop as agent_loop
from quantlab.agent.registry import GLOBAL_REGISTRY
from quantlab.agent.sse import RunEventBuffer, SSEHeartbeat
from quantlab.errors import ApiError
from quantlab.storage.repos import chat_repo, settings_repo

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)

#: Herzbeat-Intervall des Run-Streams (s; PLAN §C.2 „Heartbeat“).
_SSE_PING_S = 15

#: Wartezeit des Cancel-Endpunkts auf das Run-Ende (s), bevor der aktuelle
#: Status schlicht zurückgegeben wird.
_CANCEL_WARTEZEIT_S = 10.0

#: Meldung für Waisen-Runs (Status ``running`` in der DB, aber kein In-Memory-
#: Handle mehr — Sidecar wurde neu gestartet). Sie werden endgültig als
#: ``error`` abgeschlossen, damit kein Lauf dauerhaft „läuft“ (REV/T12).
_WAISEN_FEHLER = "Der Sidecar wurde neu gestartet, der Lauf wurde abgebrochen."


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ThinkingRequest(BaseModel):
    """Thinking-Wahl im Request (Details/Validierung: llm_client.resolve_thinking)."""

    type: Literal["enabled", "disabled"] | None = None
    effort: Literal["minimal", "low", "medium", "high", "xhigh", "max", "none"] | None = None


class RunCreateRequest(BaseModel):
    """Body von ``POST /api/agent/runs`` (§C.1)."""

    project_id: int | None = None
    message: str = Field(min_length=1)
    model: str | None = None
    thinking: ThinkingRequest | None = None


class RunCreatedResponse(BaseModel):
    run_id: str
    status: str


class RunCancelResponse(BaseModel):
    run_id: str
    status: str


# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------


def _standard_client(api_key: str) -> llm_client.LLMClient:
    """Client-Fabrik (Injektionspunkt für Tests: Fake-LLM-Transport).

    Liefert je Aufruf einen **frischen** Client — der Loop übernimmt die
    Ownership und schließt ihn am Run-Ende (kein Pool-Leak, kein
    Cache-Invalidierungs-Problem bei Key-Wechsel; REV/T12 Punkt 3).
    """
    return llm_client.LLMClient(api_key)


def _run_als_dict(run: chat_repo.ChatRun) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "project_id": run.project_id,
        "title": run.title,
        "model": run.model,
        "thinking": {"type": run.thinking_type, "effort": run.reasoning_effort},
        "status": run.status,
        "error": run.error,
        "usage": {"input_tokens": run.input_tokens, "output_tokens": run.output_tokens},
        "created_at": run.created_at,
        "finished_at": run.finished_at,
    }


def _message_als_dict(message: chat_repo.ChatMessage) -> dict[str, Any]:
    return {
        "seq": message.seq,
        "role": message.role,
        "content": message.content,
        "reasoning_content": message.reasoning_content,
        "tool_calls": message.tool_calls(),
        "tool_call_id": message.tool_call_id,
        "name": message.name,
        "created_at": message.created_at,
    }


def _sse(event_name: str, data: dict[str, Any]) -> ServerSentEvent:
    """Event im C.2-Format: ``event: <name>`` + ``data: <json>``."""
    return ServerSentEvent(event=event_name, data=json.dumps(data, ensure_ascii=False))


async def _sse_stream(buffer: RunEventBuffer) -> AsyncIterator[ServerSentEvent | str]:
    """Puffer → SSE-Events; Herzbeat als Kommentar (``: heartbeat``)."""
    async for item in buffer.stream(heartbeat_interval_s=_SSE_PING_S):
        if isinstance(item, SSEHeartbeat):
            yield ": heartbeat"
        else:
            yield _sse(item.name, item.data)


async def _db_replay(run: chat_repo.ChatRun) -> AsyncIterator[ServerSentEvent | str]:
    """Replay aus der DB, wenn der In-Memory-Puffer fehlt (Sidecar-Neustart).

    ANNAHME: Deltas lassen sich nach einem Neustart nicht rekonstruieren — der
    Replay enthält daher ``run_started``, die persistierten Inhalte je Message
    als je einen Delta-Event und das Terminal-Event gemäß Status. Waisen-Runs
    (Status ``running`` ohne Handle) werden im Endpunkt bereits auf ``error``
    gesetzt, bevor dieser Generator startet.
    """
    messages = await asyncio.to_thread(chat_repo.list_messages, run.id)
    yield _sse(
        "run_started",
        {
            "run_id": run.id,
            "model": run.model,
            "thinking": {"type": run.thinking_type, "effort": run.reasoning_effort},
            "warnings": [],
        },
    )
    for message in messages:
        if message.role == "assistant":
            if message.reasoning_content:
                yield _sse("reasoning_delta", {"text": message.reasoning_content})
            if message.content:
                yield _sse("content_delta", {"text": message.content})
    if run.status == "done":
        yield _sse(
            "run_finished",
            {
                "usage": {
                    "input_tokens": run.input_tokens,
                    "output_tokens": run.output_tokens,
                }
            },
        )
    elif run.status == "cancelled":
        yield _sse("run_cancelled", {})
    else:  # error — Meldung aus der DB (inkl. zuvor verwaister Läufe)
        yield _sse(
            "run_error",
            {"message": run.error or "Der Lauf wurde fehlerhaft beendet."},
        )


# ---------------------------------------------------------------------------
# Endpunkte
# ---------------------------------------------------------------------------


@router.get("/agent/models")
def get_models() -> dict[str, Any]:
    """Modellkatalog aus Settings ``ki.modelle`` inkl. Thinking-Fähigkeiten (§C.1)."""
    modelle: list[str] = list(settings_repo.get_section("ki")["modelle"])
    return {
        "modelle": [
            {
                "id": model,
                "thinking_erzwungen": llm_client.thinking_erzwungen(model),
                "erlaubte_efforts": list(llm_client.erlaubte_efforts(model)),
            }
            for model in modelle
        ]
    }


@router.post("/agent/runs", response_model=RunCreatedResponse)
async def create_run(payload: RunCreateRequest) -> RunCreatedResponse:
    """Startet einen Agent-Run → ``{run_id}`` (Defaults aus Settings ``ki``).

    Fehlender API-Key → 400 mit deutscher Meldung, **bevor** ein Client/Transport
    entsteht (DoD h).
    """
    ki = await asyncio.to_thread(settings_repo.get_section, "ki")
    api_key = await asyncio.to_thread(settings_repo.api_key_read)
    if not api_key:
        raise ApiError(
            400,
            "api_key_fehlt",
            "Kein z.ai-API-Key konfiguriert. Bitte unter Einstellungen → API-Key hinterlegen.",
            hint="PUT /api/settings/api-key",
        )
    thinking = payload.thinking
    handle = await agent_loop.start_run(
        message=payload.message,
        project_id=payload.project_id,
        model=payload.model or str(ki["standard_modell"]),
        thinking_type=thinking.type if thinking is not None else None,
        reasoning_effort=thinking.effort if thinking is not None else None,
        client=_standard_client(api_key),
        registry=GLOBAL_REGISTRY,
        ki_settings=ki,
    )
    return RunCreatedResponse(run_id=handle.run_id, status="running")


@router.get("/agent/runs")
def list_runs(project_id: int | None = None) -> dict[str, Any]:
    """Run-Historie, neueste zuerst; optional auf ein Projekt gefiltert (§C.1)."""
    runs = chat_repo.list_runs(project_id)
    return {"runs": [_run_als_dict(run) for run in runs]}


@router.get("/agent/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """Ein Run samt Messages (inkl. reasoning/tool_calls, §C.1)."""
    run = chat_repo.get_run(run_id)
    if run is None:
        raise ApiError(
            404, "lauf_nicht_gefunden", f"Kein Lauf mit der ID „{run_id}“ gefunden."
        )
    messages = chat_repo.list_messages(run_id)
    return {"run": _run_als_dict(run), "messages": [_message_als_dict(m) for m in messages]}


@router.post("/agent/runs/{run_id}/cancel", response_model=RunCancelResponse)
async def cancel_run(run_id: str) -> RunCancelResponse:
    """Kooperativer Abbruch (§C.1): wartet kurz auf das Run-Ende, liefert dann den Status.

    Ein Waisen-Run (Status ``running`` in der DB, aber kein In-Memory-Handle —
    Sidecar-Neustart) hat keinen Task mehr, der sich abbrechen ließe: Er wird
    hier endgültig als ``error`` abgeschlossen (REV/T12).
    """
    run = await asyncio.to_thread(chat_repo.get_run, run_id)
    if run is None:
        raise ApiError(
            404, "lauf_nicht_gefunden", f"Kein Lauf mit der ID „{run_id}“ gefunden."
        )
    handle = agent_loop.get_run_handle(run_id)
    if handle is None or handle.task is None:
        if run.status == "running":
            await asyncio.to_thread(
                chat_repo.set_run_status, run_id, "error", error=_WAISEN_FEHLER
            )
            return RunCancelResponse(run_id=run_id, status="error")
        return RunCancelResponse(run_id=run_id, status=run.status)
    if not handle.task.done():
        agent_loop.cancel_run(run_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CANCEL_WARTEZEIT_S
        while not handle.task.done() and loop.time() < deadline:
            await asyncio.sleep(0.01)
    aktueller = await asyncio.to_thread(chat_repo.get_run, run_id)
    if (
        aktueller is not None
        and aktueller.status == "running"
        and handle.task.done()
        and not handle.cancel_event.is_set()
    ):
        # Task tot, Status aber noch 'running' (harter Fremd-Cancel/Sidecar-Störung)
        # → gleicher Waisen-Zustand: endgültig als Fehler abschließen (REV/T12).
        await asyncio.to_thread(
            chat_repo.set_run_status, run_id, "error", error=_WAISEN_FEHLER
        )
        geladen = await asyncio.to_thread(chat_repo.get_run, run_id)
        aktueller = geladen if geladen is not None else aktueller
    return RunCancelResponse(run_id=run_id, status=aktueller.status if aktueller else run.status)


@router.get("/agent/runs/{run_id}/stream")
async def stream_run(run_id: str) -> EventSourceResponse:
    """Run-SSE (§C.2) mit vollem Replay aus dem Event-Buffer (EventSource-Reconnects)."""
    handle = agent_loop.get_run_handle(run_id)
    if handle is not None:
        generator = _sse_stream(handle.buffer)
    else:
        run = await asyncio.to_thread(chat_repo.get_run, run_id)
        if run is None:
            raise ApiError(
                404, "lauf_nicht_gefunden", f"Kein Lauf mit der ID „{run_id}“ gefunden."
            )
        if run.status == "running":
            # Waise (Sidecar-Neustart): endgültig als Fehler abschließen, statt
            # den Status dauerhaft auf „running“ einzufrieren (REV/T12).
            await asyncio.to_thread(
                chat_repo.set_run_status, run_id, "error", error=_WAISEN_FEHLER
            )
            geladen = await asyncio.to_thread(chat_repo.get_run, run_id)
            run = geladen if geladen is not None else run
        logger.info("Run %s: Replay aus DB (In-Memory-Puffer nicht vorhanden).", run_id)
        generator = _db_replay(run)
    return EventSourceResponse(generator, ping=_SSE_PING_S)
