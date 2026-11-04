"""Fehlerbehandlung: einheitliches Format ``{"detail": {"code", "message", "hint?"}}``.

``message`` ist immer deutsch (UI-Sprache, E001), ``code`` stabil und
maschinenlesbar für das Frontend (siehe ``src/renderer/src/lib/backend.ts``).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Anzahl der Feldfehler, die in die deutsche Zusammenfassung aufgenommen werden.
_MAX_SHOWN_VALIDATION_ERRORS = 5


class ApiError(Exception):
    """Fachlicher Fehler mit deutscher Nutzermeldung."""

    def __init__(self, status_code: int, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.hint = hint

    def to_payload(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            detail["hint"] = self.hint
        return {"detail": detail}


async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


def _validation_summary(exc: RequestValidationError) -> str:
    """Deutsche Zusammenfassung der Pydantic-Feldfehler (Vertrag C.1).

    Struktur/Sprache deutsch; die Fachdetails pro Feld liefert Pydantic
    („Fachbegriffe englisch", E001).
    """
    errors = exc.errors()
    parts: list[str] = []
    for err in errors[:_MAX_SHOWN_VALIDATION_ERRORS]:
        loc = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        msg = str(err.get("msg", "ungültiger Wert"))
        parts.append(f"{loc or 'Body'}: {msg}")
    extra = len(errors) - _MAX_SHOWN_VALIDATION_ERRORS
    suffix = f" (+{extra} weitere)" if extra > 0 else ""
    return f"Eingabe ungültig: {'; '.join(parts)}{suffix}"


async def _request_validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "validierung_fehler",
                "message": _validation_summary(exc),
            }
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """Registriert die Fehler-Handler an der App (idempotent je App-Instanz)."""
    app.add_exception_handler(ApiError, _api_error_handler)
    # Überschreibt FastAPIs Default ({detail: [ {loc, msg, …}, … ]}) durch den
    # einheitlichen deutschen Vertrag {detail: {code, message}} (C.1).
    app.add_exception_handler(RequestValidationError, _request_validation_error_handler)
