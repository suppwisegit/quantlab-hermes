"""Agent-Paket (T12): LLM-Client, Run-Loop, Tool-Registry-Mechanik, SSE-Puffer.

Module:
- :mod:`quantlab.agent.llm_client` — z.ai (GLM) über das OpenAI-SDK, Streaming-Normalisierung.
- :mod:`quantlab.agent.loop` — Run = asyncio.Task; Historie, Tool-Dispatch, Cancel, Persistenz.
- :mod:`quantlab.agent.registry` — ``ToolSpec`` + Registry + Dispatch (T12-Mechanik; die 15
  fachlichen Tools aus PLAN §E registriert T14).
- :mod:`quantlab.agent.sse` — Run-Event-Buffer mit Replay (C.2).
"""
