"""Screener-Tool (T34, PLAN §T34): Regel-Scan über das lokale Universum.

Konsumiert ausschließlich :func:`quantlab.services.screener.screen` (T32) —
synchron, nur lokale 1d-Bestände (E027), Fehler als ``ProviderError`` 422
(deutsch) → :class:`ToolFehler`.

Bewusst **kein** Tool für den S&P-500-Bulk-Import: Der schützt seinen opt-in-
Charakter (Netzlast, ~20-s-Job, R10) — der Agent verweist dafür auf das
Screener-UI (Hinweis im Tool-Ergebnis).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, pruefe_args, schema_von
from quantlab.providers.base import ProviderError
from quantlab.services import screener


class ScreenerRegelArgs(BaseModel):
    """Eine Screener-Regel ``{feld, op, wert}`` (AND-verknüpft, T32-Vertrag)."""

    feld: str = Field(
        min_length=1,
        description=(
            'Feld: perf_1w/perf_1m/perf_3m/perf_1y, rsi_14, ann_vol, sma50_ratio, '
            'last_price, change_pct oder fundamental pe/market_cap/div_yield/'
            'netto_marge/roe/umsatzwachstum_yoy.'
        ),
    )
    op: Literal["gt", "lt", "between"] = Field(
        description='"gt" (größer), "lt" (kleiner) oder "between" (Intervall inklusive).'
    )
    wert: float | list[float] = Field(
        description='Schwellwert als Zahl; bei "between" als [min, max] (Bruchzahlen: '
        '0.05 = +5 %). Beispiel: {"feld": "rsi_14", "op": "lt", "wert": 30}.'
    )


class RunScreenerArgs(BaseModel):
    """Argumente von ``run_screener``."""

    regeln: list[ScreenerRegelArgs] = Field(
        min_length=1,
        max_length=10,
        description="Mindestens eine Regel; mehrere Regeln sind AND-verknüpft.",
    )
    sort_feld: str | None = Field(
        default=None,
        description="Optionales Sortierfeld (gleiches Vokabular wie feld).",
    )
    richtung: Literal["auf", "ab"] = Field(
        default="ab", description='Sortierrichtung: "ab" (Default, absteigend) oder "auf".'
    )
    limit: int = Field(
        default=25,
        ge=1,
        le=50,
        description="Maximale Trefferzahl, Default 25 (max. 50 — Ergebnis wird "
        "ansonsten gekappt).",
    )


async def run_screener(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Screened das lokale 1d-Universum gegen die Regeln (nur lokale Daten)."""
    args = pruefe_args(RunScreenerArgs, roh_args, "run_screener")
    regeln = [regel.model_dump() for regel in args.regeln]

    def _screenen() -> dict[str, Any]:
        return screener.screen(
            regeln,
            sort_feld=args.sort_feld,
            richtung=args.richtung,
            limit=args.limit,
        )

    try:
        ergebnis = await asyncio.to_thread(_screenen)
    except ProviderError as exc:
        # T32-Fehler tragen den Gültig-Bereich im ``hint`` (z. B. Feldliste) —
        # für den Agenten entscheidend, daher in die Tool-Meldung aufnehmen.
        meldung = exc.message
        if exc.hint:
            meldung = f"{meldung} {exc.hint}"
        raise ToolFehler(meldung) from exc
    return {
        **ergebnis,
        "hinweis": (
            "Universum = alle lokal gespeicherten 1d-Symbole; fehlende Feldwerte "
            "lassen eine Zeile herausfallen. Das S&P-500-Bulk-Universum legst du "
            "bewusst im Screener-UI an (opt-in) — nicht über den Agenten."
        ),
    }


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="run_screener",
            description=(
                "Findet Kandidaten im lokalen Universum nach Regeln, z. B. RSI(14) "
                "unter 30 oder Performance-Schaufenster — AND-verknüpft, sortierbar, "
                "rechnet nur auf gespeicherten Tagesdaten (kein Netz). "
                'Beispiel: {"regeln": [{"feld": "rsi_14", "op": "lt", "wert": 30}], '
                '"sort_feld": "rsi_14", "richtung": "auf", "limit": 10}.'
            ),
            parameters=schema_von(RunScreenerArgs),
            handler=run_screener,
        )
    )


_registrieren()
