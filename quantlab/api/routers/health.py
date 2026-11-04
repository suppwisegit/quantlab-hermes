"""Health-Endpunkt — erste Anlaufstelle des Health-Polls im Electron-Main (E004)."""

from __future__ import annotations

import platform

from fastapi import APIRouter
from pydantic import BaseModel

from quantlab import APP_VERSION
from quantlab.config import get_data_dir

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app_version: str
    python_version: str
    data_dir: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + App-Version, Python-Version und aufgelöstes Datenverzeichnis."""
    return HealthResponse(
        status="ok",
        app_version=APP_VERSION,
        python_version=platform.python_version(),
        data_dir=str(get_data_dir()),
    )
