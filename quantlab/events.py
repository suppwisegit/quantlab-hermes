"""Globaler SSE-Bus (PLAN §C.2) — App-weite Events außerhalb von Agent-Runs.

Events (C.2, Payloads exakt):

- ``download_progress {jobId, symbol, done, total, status, chunk_done?,
  chunk_total?}`` — die Chunk-Felder (T-F10) nur bei 1m-Chunking gesetzt
- ``download_finished {jobId, ok, error?}``
- ``data_changed {symbol, interval, reason}`` — Frontend invalidiert
  TanStack-Query-Caches daraufhin.

Mechanik: Publisher (der Download-Job-Manager, später weitere Tasks) rufen
:meth:`publizieren` synchron aus ihrer Coroutine heraus — ein ``put_nowait``
weckt alle Abonnenten ihres Event-Loops. Der SSE-Endpunkt
(:mod:`quantlab.api.routers.events`) konsumiert :meth:`stream` und formatiert
``event:``/``data:``-Zeilen; der Heartbeat (15 s) ist Sache des Routers
(``EventSourceResponse(ping=…)``). Es gibt bewusst **keinen** Replay-Puffer:
globale Events sind Wegwechsel-Melder („Daten haben sich geändert“), kein
Protokoll — frisch verbundene Clients holen den Zustand per REST.

Testfreundlichkeit: Abonnenten liegen in einem :class:`weakref.WeakSet` —
Warteschlangen aus beendeten Event-Loops (z. B. Tests) werden beim nächsten
``publizieren`` automatisch verworfen, statt den Publisher zu stören.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncIterator
from typing import Any

#: Offene Abonnenten-Warteschlangen (schwach gehalten, s. Modul-Docstring).
_ABONNENTEN: weakref.WeakSet[asyncio.Queue[dict[str, Any]]] = weakref.WeakSet()

#: Ereignisnamen (C.2 + C.5) — als Konstanten für Publisher und Tests.
DOWNLOAD_PROGRESS = "download_progress"
DOWNLOAD_FINISHED = "download_finished"
DATA_CHANGED = "data_changed"
QUOTES = "quotes"  # T22/E024: Live-Quotes-Push (nur geänderte Werte)

#: Gründe für ``data_changed`` (E020: ASCII-Literale, als Konstanten exportiert).
REASON_DOWNLOAD = "download"
REASON_GELOESCHT = "geloescht"


def abonnieren() -> asyncio.Queue[dict[str, Any]]:
    """Legt eine Abonnenten-Warteschlange an und meldet sie am Bus an.

    Muss im Event-Loop des Konsumenten aufgerufen werden (SSE-Endpunkt, Tests).
    """
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _ABONNENTEN.add(queue)
    return queue


def abmelden(queue: asyncio.Queue[dict[str, Any]]) -> None:
    """Meldet eine Warteschlange ab (idempotent; macht auch :meth:`stream` im Finally)."""
    _ABONNENTEN.discard(queue)


def publizieren(name: str, daten: dict[str, Any] | None = None) -> None:
    """Verteilt ein Event an alle aktuell verbundenen Abonnenten (niemals blockierend).

    Warteschlangen aus fremden/beendeten Loops werden still verworfen; ein
    Publisher soll durch einen toten Konsumenten nie blockiert werden.
    """
    nachricht = {"name": name, "daten": dict(daten or {})}
    for queue in list(_ABONNENTEN):
        try:
            queue.put_nowait(nachricht)
        except RuntimeError:
            _ABONNENTEN.discard(queue)  # fremder Loop — Abonnent ist weg
        except asyncio.QueueFull:  # defensiv (Queue ist unbegrenzt)
            _ABONNENTEN.discard(queue)


# ---------------------------------------------------------------------------
# Fachliche Publish-Helfer (C.2-Payloads an genau einer Stelle definiert)
# ---------------------------------------------------------------------------


def download_progress(
    job_id: str,
    symbol: str,
    done: int,
    total: int,
    status: str = "running",
    *,
    chunk_done: int | None = None,
    chunk_total: int | None = None,
) -> None:
    """Fortschritt eines Download-Jobs (nach jedem bearbeiteten Symbol).

    T-F10: ``chunk_done``/``chunk_total`` melden den Fortschritt **innerhalb**
    eines Symbols (1m-Chunking des YahooDirect-Beins — je Chunk ein zusätzliches
    Event mit unveränderten Symbol-Zählern ``done``/``total``). Beide Felder
    werden nur mitgesendet, wenn gesetzt — Alt-Payloads (Downloads ohne
    Chunking) bleiben byte-identisch.
    """
    daten: dict[str, Any] = {
        "jobId": job_id,
        "symbol": symbol,
        "done": done,
        "total": total,
        "status": status,
    }
    if chunk_done is not None:
        daten["chunk_done"] = chunk_done
    if chunk_total is not None:
        daten["chunk_total"] = chunk_total
    publizieren(DOWNLOAD_PROGRESS, daten)


def download_finished(job_id: str, ok: bool, error: str | None = None) -> None:
    """Terminal-Event eines Download-Jobs; ``error`` nur bei Fehlschlag gesetzt."""
    daten: dict[str, Any] = {"jobId": job_id, "ok": ok}
    if error:
        daten["error"] = error
    publizieren(DOWNLOAD_FINISHED, daten)


def data_changed(symbol: str, interval: str, reason: str) -> None:
    """Marktdaten haben sich geändert — Frontend invalidiert Caches (``interval="*"`` = alle).

    ``reason`` nutzt die exportierten Konstanten :data:`REASON_DOWNLOAD` /
    :data:`REASON_GELOESCHT` (E020).
    """
    publizieren(DATA_CHANGED, {"symbol": symbol, "interval": interval, "reason": reason})


def quotes(quote_eintraege: list[dict], fetched_at: int) -> None:
    """Live-Quotes-Push (T22, §C.5/E024): nur Einträge mit **geänderten** Werten.

    Payload ``{quotes: [...], fetched_at}`` — die Einträge folgen dem E024-
    Quote-Objekt (``{symbol, price, previous_close, change_pct, currency,
    quote_ts}``); ``fetched_at`` int64 µs UTC. Der Publisher-Helfer ist bewusst
    dumm: die Änderungserkennung macht der Quotes-Service (Cache-Vertrag).
    """
    publizieren(
        QUOTES,
        {"quotes": list(quote_eintraege), "fetched_at": int(fetched_at)},
    )


async def stream() -> AsyncIterator[dict[str, Any]]:
    """Abonnieren und live mitlesen; endet erst mit Abbruch (Client-Trennung).

    Liefert ``{"name": …, "daten": {…}}`` — der Router formatiert SSE daraus.
    """
    queue = abonnieren()
    try:
        while True:
            nachricht = await queue.get()
            yield nachricht
    finally:
        abmelden(queue)
