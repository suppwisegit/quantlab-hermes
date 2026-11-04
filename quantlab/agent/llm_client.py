"""LLM-Client für z.ai (GLM) über das OpenAI-SDK (T12 — Konventionen §F, RECHERCHE Block 1).

**Wire-Vertrag (RECHERCHE, aus der z.ai-Doku verifiziert):**

- Base ``https://api.z.ai/api/coding/paas/v4`` (Coding-Plan-Quota); das SDK ergänzt
  den Pfad ``/chat/completions``.
- ``tool_choice`` ausschließlich ``"auto"``; ``function.arguments`` ist ein
  **JSON-String** (Parsen mit Fehlerbehandlung — passiert im Dispatch,
  :mod:`quantlab.agent.registry`).
- Streaming per SSE mit ``data: [DONE]``; Reasoning kommt als
  ``delta.reasoning_content`` — ein Feld, das das OpenAI-SDK nicht typt
  (``ChoiceDelta`` erlaubt Extra-Felder → Auslesen über ``model_extra``).
- ``thinking: {"type": "enabled" | "disabled"}`` (Default ``enabled``) plus
  ``reasoning_effort`` als Top-Level-Parameter. GLM-5.3*-Modelle (inkl.
  ``[1m]``-Variante) **erzwingen** Thinking und erlauben nur ``low | high | max``.
- Preserved Thinking: ``reasoning_content`` von Assistant-Messages wird
  unverändert in Folgerequests zurückgeschickt (Aufgabe des Loops; dieses
  Modul liefert die Message-Bauer dafür).

**Testbarkeit (E013, netzfrei):** Der HTTP-Transport ist über ``http_client``
injizierbar (Tests: ``httpx.MockTransport``). Der Zugriff erfolgt zweistufig —
:meth:`LLMClient.prepare_request` baut den Request (inkl. Validierungs-Warnungen),
:meth:`LLMRequest.execute` führt ihn aus und liefert normalisierte Stream-Events;
das letzte Event ist stets ein :class:`CompletionEvent`. Die Request-Captures
der Fake-LLM-Tests sind der Vertragsbeweis.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import openai
from openai import AsyncOpenAI

#: Coding-Plan-Base (§F) — das SDK hängt ``/chat/completions`` an.
BASE_URL = "https://api.z.ai/api/coding/paas/v4"

#: Modelle mit diesem Präfix erzwingen Thinking (RECHERCHE Block 1.5; das
#: ``[1m]``-Suffix kommt ausschließlich an 5.3er-Modellen vor → Präfix-Prüfung
#: deckt beide ab).
#:
#: **E049-Live-Nachweis (2026-09-15, Probe-Request via eigenem Key):**
#: ``glm-5.3-flash`` mit ``thinking: {"type": "disabled"}`` (an
#: ``resolve_thinking`` vorbei gesendet) → HTTP 200, ``finish_reason: stop``,
#: Usage 32/11 — aber ``reasoning_content`` kam dennoch (7 Deltas). Die API
#: **ignoriert** ``disabled`` für den Flash-Ableger also Still-schweigend;
#: der Zwang ist damit live bestätigt (Ausgang A von PLAN §V5/E049), das
#: Frontend-Verhalten („Thinking: Aus" disabled) bleibt korrekt.
_THINKING_ERZWUNGEN_PRÄFIX = "glm-5.3"

#: reasoning_effort-Optionen je Modellklasse (RECHERCHE Block 1.5).
EFFORTS_ERZWUNGEN: tuple[str, ...] = ("low", "high", "max")
EFFORTS_ALLGEMEIN: tuple[str, ...] = ("max", "xhigh", "high", "medium", "low", "minimal", "none")

THINKING_TYPEN: tuple[str, ...] = ("enabled", "disabled")

#: Hartes Maximum laut z.ai-API-Referenz (RECHERCHE Block 1.7) — obere Grenze
#: für vom Nutzer konfigurierte ``max_tokens``-Werte.
MAX_TOKENS_LIMIT = 131_072


# ---------------------------------------------------------------------------
# Wertetypen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkingConfig:
    """Thinking-Wahl für einen Request (intern ``{type, effort}``).

    Übertragen wird ``thinking: {"type": …}`` plus Top-Level ``reasoning_effort``
    (nur wenn gesetzt) — siehe Moduldocstring.
    """

    type: str = "enabled"
    effort: str | None = None


@dataclass(frozen=True)
class ToolCallRequest:
    """Ein vom Modell angeforderter Tool-Call (``function.arguments`` = JSON-String)."""

    call_id: str
    name: str
    arguments_json: str

    def arguments(self) -> dict[str, Any]:
        """Parst ``arguments_json`` (``ValueError`` bei ungültigem JSON — Dispatch fängt ab)."""
        try:
            args = json.loads(self.arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ungültiges JSON in function.arguments: {exc}") from exc
        if not isinstance(args, dict):
            raise ValueError("function.arguments muss ein JSON-Objekt sein.")
        return args


@dataclass(frozen=True)
class Usage:
    """Token-Verbrauch eines Completions (normiert auf input/output)."""

    input_tokens: int = 0
    output_tokens: int = 0

    def plus(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


# --- normalisierte Stream-Events (Union; letztes Event ist immer CompletionEvent) ---


@dataclass(frozen=True)
class ReasoningDeltaEvent:
    """Fragment aus ``delta.reasoning_content`` („Denken“)."""

    text: str


@dataclass(frozen=True)
class ContentDeltaEvent:
    """Fragment aus ``delta.content`` (sichtbare Antwort)."""

    text: str


@dataclass(frozen=True)
class ToolCallDeltaEvent:
    """Fragment eines gestreamten Tool-Calls (Akkumulierung erfolgt clientseitig)."""

    index: int
    call_id: str | None
    name: str | None
    arguments_part: str


@dataclass(frozen=True)
class CompletionEvent:
    """Abschluss des Streams: akkumuliertes Ergebnis + finish_reason + Usage."""

    content: str
    reasoning_content: str | None
    tool_calls: tuple[ToolCallRequest, ...]
    finish_reason: str | None
    usage: Usage | None


LLMStreamEvent = (
    ReasoningDeltaEvent | ContentDeltaEvent | ToolCallDeltaEvent | CompletionEvent
)


# ---------------------------------------------------------------------------
# Thinking-Validierung (modellabhängig, §F)
# ---------------------------------------------------------------------------


def thinking_erzwungen(model: str) -> bool:
    """Erzwingt das Modell Thinking? (GLM-5.3*; ``[1m]``-Suffix via Präfix abgedeckt).

    Für ``glm-5.3-flash`` live verifiziert (E049, 2026-09-15 — siehe Kommentar
    an :data:`_THINKING_ERZWUNGEN_PRÄFIX`): ``disabled`` wird von der API
    ignoriert, ``reasoning_content`` kommt trotzdem → Zwang bleibt bestehen.
    """
    basis = model.strip().lower().split("[", 1)[0]
    return basis.startswith(_THINKING_ERZWUNGEN_PRÄFIX)


def erlaubte_efforts(model: str) -> tuple[str, ...]:
    """Zulässige ``reasoning_effort``-Werte je Modellklasse (RECHERCHE Block 1.5)."""
    return EFFORTS_ERZWUNGEN if thinking_erzwungen(model) else EFFORTS_ALLGEMEIN


def resolve_thinking(
    model: str, thinking: ThinkingConfig | None = None
) -> tuple[ThinkingConfig, list[str]]:
    """Validiert/normalisiert die Thinking-Wahl modellabhängig.

    Rückgabe: ``(auflösung, warnungen)`` — Warnungen sind deutsche Texte für das
    Run-Log. Regeln (RECHERCHE Block 1.5):

    - glm-5.3* erzwingt Thinking: ``disabled`` → ``enabled`` (Warnung).
    - Nicht unterstützte ``effort``-Werte werden weggelassen (Warnung), statt
      einen 400 der API zu riskieren.
    - Unbekannter ``type`` → ``ValueError`` (Programmfehler; der Router validiert
      bereits per Pydantic-Literal).
    """
    typ = thinking.type if thinking is not None else "enabled"
    effort = thinking.effort if thinking is not None else None
    if typ not in THINKING_TYPEN:
        raise ValueError(
            f"Unbekannter thinking.type: {typ!r} (erlaubt: {' | '.join(THINKING_TYPEN)})"
        )
    warnungen: list[str] = []
    if thinking_erzwungen(model):
        if typ == "disabled":
            warnungen.append(
                f"Modell „{model}“ erzwingt Thinking — „disabled“ wurde zu „enabled“ geändert."
            )
            typ = "enabled"
    if effort is not None and effort not in erlaubte_efforts(model):
        warnungen.append(
            f"reasoning_effort „{effort}“ wird von „{model}“ nicht unterstützt "
            f"(erlaubt: {' | '.join(erlaubte_efforts(model))}) — Parameter wird weggelassen."
        )
        effort = None
    return ThinkingConfig(type=typ, effort=effort), warnungen


# ---------------------------------------------------------------------------
# Message-Bauer (Wire-Format; Preserved Thinking §F)
# ---------------------------------------------------------------------------


def system_message(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def user_message(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def assistant_message(
    content: str | None,
    *,
    reasoning_content: str | None = None,
    tool_calls: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assistant-Message im z.ai-Format.

    Preserved Thinking (RECHERCHE Block 1.8): ``reasoning_content`` wird
    **unverändert** übernommen — nicht kürzen, nicht umsortieren.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        message["tool_calls"] = [dict(call) for call in tool_calls]
    return message


def tool_message(tool_call_id: str, content: str, *, name: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if name:
        message["name"] = name
    return message


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _kurz(text: str, max_zeichen: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_zeichen else text[: max_zeichen - 1] + "…"


def deutsche_fehlermeldung(exc: BaseException) -> str:
    """Mappt SDK-/API-Fehler auf deutsche Nutzermeldungen (§C.1: message de)."""
    if isinstance(exc, openai.AuthenticationError):
        return (
            "Die z.ai-API hat den API-Key abgelehnt (401). "
            "Bitte prüfe den Key in den Einstellungen."
        )
    if isinstance(exc, openai.PermissionDeniedError):
        return (
            "Kein Zugriff auf die z.ai-API (403). "
            "Bitte prüfe Key und Berechtigung deines Coding Plans."
        )
    if isinstance(exc, openai.RateLimitError):
        return (
            "Das Limit der z.ai-API ist erreicht (429). "
            "Bitte kurz warten und es erneut versuchen."
        )
    if isinstance(exc, openai.NotFoundError):
        return "Modell oder Endpunkt bei z.ai nicht gefunden (404). Bitte Modellwahl prüfen."
    if isinstance(exc, openai.BadRequestError):
        return f"Die z.ai-API hat die Anfrage abgelehnt (400): {_kurz(str(exc), 300)}"
    if isinstance(exc, openai.APIConnectionError):
        return (
            "Keine Verbindung zur z.ai-API. "
            "Bitte prüfe die Netzwerk-Verbindung und versuche es erneut."
        )
    if isinstance(exc, openai.APIStatusError):
        return f"Fehler der z.ai-API (HTTP {exc.status_code}): {_kurz(str(exc), 300)}"
    return f"Unerwarteter Fehler: {_kurz(str(exc), 300)}"


def _usage_aus(chunk_usage: Any) -> Usage | None:
    """Normalisiert ``chunk.usage``; ``None`` wenn der Chunk keinen Verbrauch meldet."""
    if chunk_usage is None:
        return None
    return Usage(
        input_tokens=int(getattr(chunk_usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(chunk_usage, "completion_tokens", 0) or 0),
    )


@dataclass(frozen=True)
class LLMRequest:
    """Vorbereiteter Request (Parameter fix, Validierungs-Warnungen bekannt)."""

    client: LLMClient
    params: dict[str, Any]
    warnings: tuple[str, ...]

    def execute(self) -> AsyncIterator[LLMStreamEvent]:
        """Führt den Streaming-Request aus (Async-Generator normalisierter Events).

        Das letzte Event ist immer ein :class:`CompletionEvent` (akkumulierte
        Inhalte, ``finish_reason`` durchgereicht — auch z.ai-spezifische Werte wie
        ``sensitive`` —, Usage aus dem letzten Chunk, der Usage meldet; ANNAHME
        gemäß RECHERCHE: z.ai schickt Usage im Final-Chunk des Streams).
        Gestreamte Tool-Calls werden per Index akkumuliert:
        ``function.arguments``-Fragments werden in der Reihenfolge ihres
        Eintreffens zusammengesetzt (Ergebnis bleibt ein JSON-String).
        """
        return self._iter_chunks()

    async def _iter_chunks(self) -> AsyncIterator[LLMStreamEvent]:
        stream = await self.client.raw.chat.completions.create(**self.params)
        content_teile: list[str] = []
        reasoning_teile: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: Usage | None = None
        finish_reason: str | None = None
        try:
            async for chunk in stream:
                chunk_usage = _usage_aus(getattr(chunk, "usage", None))
                if chunk_usage is not None:
                    usage = chunk_usage  # letzter Usage-Chunk gewinnt
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = str(choice.finish_reason)
                delta = choice.delta
                if delta is None:
                    continue
                reasoning = (delta.model_extra or {}).get("reasoning_content")
                if reasoning:
                    reasoning_teile.append(str(reasoning))
                    yield ReasoningDeltaEvent(text=str(reasoning))
                if delta.content:
                    content_teile.append(str(delta.content))
                    yield ContentDeltaEvent(text=str(delta.content))
                for tc in delta.tool_calls or ():
                    index = int(tc.index or 0)
                    eintrag = calls.setdefault(index, {"id": None, "name": None, "parts": []})
                    if tc.id:
                        eintrag["id"] = str(tc.id)
                    funktion = tc.function
                    fragment = ""
                    if funktion is not None:
                        if funktion.name:
                            eintrag["name"] = str(funktion.name)
                        if funktion.arguments:
                            fragment = str(funktion.arguments)
                            eintrag["parts"].append(fragment)
                    yield ToolCallDeltaEvent(
                        index=index,
                        call_id=eintrag["id"],
                        name=eintrag["name"],
                        arguments_part=fragment,
                    )
        finally:
            # Stream deterministisch schließen (auch bei Cancel/GeneratorExit), um
            # die HTTP-Verbindung zeitnah freizugeben. Aufräum-Fehler dürfen den
            # ursprünglichen Ablauf (normales Ende, Cancel, Fehler) nie kippen.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 — bewusst breit: reines Aufräumen
                    pass
        tool_calls = tuple(
            ToolCallRequest(
                call_id=eintrag["id"] or f"call_{index}",
                name=eintrag["name"] or "",
                arguments_json="".join(eintrag["parts"]) or "{}",
            )
            for index, eintrag in sorted(calls.items())
        )
        yield CompletionEvent(
            content="".join(content_teile),
            reasoning_content="".join(reasoning_teile) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )


class LLMClient:
    """Dünner Wrapper um :class:`openai.AsyncOpenAI` mit z.ai-Vertrag.

    ``http_client`` erlaubt den injizierten Transport (Tests: ``httpx.MockTransport``).
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        http_client: Any | None = None,
        timeout: float = 120.0,
        max_retries: int = 1,
    ) -> None:
        self.raw = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            timeout=timeout,
            max_retries=max_retries,
        )

    def prepare_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: ThinkingConfig | None = None,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMRequest:
        """Baut den Request-Body (modellabhängige Thinking-Validierung inklusive).

        Die Warnungen aus :func:`resolve_thinking` liegen anschließend auf
        :attr:`LLMRequest.warnings`. Ohne registrierte Tools (``tools`` leer/None)
        bleiben ``tools``/``tool_choice`` außen vor (z.ai erwartet keine leere
        Tool-Liste).

        ANNAHME (RECHERCHE Block 1.5): ``reasoning_effort`` wird als Top-Level-
        Parameter übertragen, ``thinking`` enthält nur ``type``.
        """
        aufgelöst, warnungen = resolve_thinking(model, thinking)
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": min(max(int(max_tokens), 1), MAX_TOKENS_LIMIT),
            "extra_body": {"thinking": {"type": aufgelöst.type}},
        }
        if aufgelöst.effort is not None:
            params["extra_body"]["reasoning_effort"] = aufgelöst.effort
        if tools:
            params["tools"] = list(tools)
            # z.ai unterstützt ausschließlich tool_choice "auto" (RECHERCHE Block 1.3).
            params["tool_choice"] = "auto"
        return LLMRequest(client=self, params=params, warnings=tuple(warnungen))

    async def close(self) -> None:
        """Schließt den SDK-/HTTP-Client und gibt den Connection-Pool frei.

        Idempotent aufrufbar. Fehler beim Schließen werden bewusst verschluckt —
        Aufräumen darf den Run-Ende-Pfad niemals kippen (der Loop ruft dies im
        finally; Ownership: :func:`quantlab.agent.loop.start_run` übernimmt den
        Client und schließt ihn am Run-Ende, siehe dort).
        """
        close = getattr(self.raw, "close", None)
        if close is None:  # pragma: no cover — AsyncOpenAI hat close() immer
            return
        try:
            await close()
        except Exception:  # noqa: BLE001 — reines Aufräumen
            pass
