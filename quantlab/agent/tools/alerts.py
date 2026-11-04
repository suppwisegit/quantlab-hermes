"""Alerts-Tool (T34, PLAN §T34): Alerts anlegen, listen, pausieren, prüfen.

Konsumiert :mod:`quantlab.storage.repos.alerts_repo` (T18) und
:func:`quantlab.services.alerts.validiere_params`/``evaluate_now`` (T33) —
keine Router-Imports. Der §D.2-Vertrag (fünf Bedingungsarten, Zustandsmaschine
mit Cooldown/Re-Arm) bleibt unangetastet.

Bewusst **ohne** ``delete``: Das Löschen (inkl. Historie) bleibt dem Alert-
Center der UI vorbehalten (§T34: „keine delete ohne Bestätigung“) — der Agent
bietet Pausieren an und verweist fürs Löschen auf die UI.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, pruefe_args, schema_von
from quantlab.services import alerts as alerts_service
from quantlab.storage.repos import alerts_repo


class ManageAlertsArgs(BaseModel):
    """Argumente von ``manage_alerts``."""

    action: Literal["list", "create", "pause", "activate", "evaluate"] = Field(
        description='Aktion: "list" (Alerts + Zustände), "create" (braucht symbol+kind '
        'und price bzw. pct+bars), "pause"/"activate" (braucht alert_id), '
        '"evaluate" (alert_id sofort gegen den aktuellen Kurs prüfen).'
    )
    alert_id: int | None = Field(
        default=None, ge=1, description="Alert-ID (pause/activate/evaluate), z. B. 3."
    )
    symbol: str | None = Field(
        default=None, min_length=1, max_length=24, description='Symbol (create), z. B. "AAPL".'
    )
    kind: Literal[
        "price_above",
        "price_below",
        "price_crossing_up",
        "price_crossing_down",
        "percent_move",
    ] | None = Field(
        default=None,
        description='Bedingungsart (create): price_above/below, price_crossing_up/down '
        "oder percent_move.",
    )
    price: float | None = Field(
        default=None, gt=0, description="Schwelle > 0 (Preis-Arten), z. B. 240."
    )
    pct: float | None = Field(
        default=None, gt=0, description="Schwellbewegung in Prozent > 0 (percent_move), z. B. 5."
    )
    bars: int | None = Field(
        default=None, ge=1, description="Balken-Anzahl ≥ 1 (percent_move), z. B. 5."
    )
    cooldown_min: int = Field(
        default=60,
        ge=1,
        le=10_080,
        description="Mindestabstand zwischen zwei Auslösungen in Minuten, Default 60.",
    )
    note: str | None = Field(
        default=None, max_length=500, description="Optionale Notiz (create)."
    )
    state: Literal["active", "paused", "triggered"] | None = Field(
        default=None, description='Zustands-Filter (list): "active", "paused", "triggered".'
    )


def _alert_als_dict(alert: alerts_repo.Alert) -> dict[str, Any]:
    return {
        "id": alert.id,
        "symbol": alert.symbol,
        "kind": alert.kind,
        "params": alert.params,
        "state": alert.state,
        "note": alert.note,
        "cooldown_min": alert.cooldown_min,
    }


def _params_bauen(args: ManageAlertsArgs) -> dict[str, Any]:
    """Alert-Parameter je Art aus den flachen Argumenten (Validierung im Service)."""
    if args.kind is None:
        raise ToolFehler('„create“ benötigt eine "kind"-Angabe.')
    if args.kind == "percent_move":
        if args.pct is None or args.bars is None:
            raise ToolFehler(
                '„percent_move“ benötigt "pct" > 0 und "bars" ≥ 1 '
                '(z. B. pct=5, bars=3 → ±5 % über 3 Tagesbalken).'
            )
        return {"pct": args.pct, "bars": args.bars}
    if args.price is None:
        raise ToolFehler(
            f'„{args.kind}“ benötigt eine "price"-Schwelle > 0 '
            "(z. B. price=240 für „warne mich unter 240“)."
        )
    return {"price": args.price}


def _sync_aktion(args: ManageAlertsArgs) -> dict[str, Any]:
    """DB-seitige Aktionen (list/create/pause/activate) — laufen im Threadpool."""
    if args.action == "list":
        alerts = alerts_repo.list_alerts(state=args.state, symbol=args.symbol)
        return {
            "action": "list",
            "anzahl": len(alerts),
            "alerts": [_alert_als_dict(alert) for alert in alerts],
            "hinweis": "Auslöse-Historie und Löschen findest du im Alert-Center "
            "der UI; pausierte Alerts werden nicht geprüft.",
        }
    if args.action == "create":
        if args.symbol is None or not args.symbol.strip():
            raise ToolFehler('„create“ benötigt ein "symbol".')
        params = _params_bauen(args)
        try:
            alerts_service.validiere_params(args.kind, params)  # 422-Spiegel, deutsch
            alert = alerts_repo.create_alert(
                args.symbol.strip(),
                args.kind,
                params,
                note=args.note,
                cooldown_min=args.cooldown_min,
            )
        except ValueError as exc:
            raise ToolFehler(str(exc)) from exc
        return {
            "action": "create",
            "alert": _alert_als_dict(alert),
            "hinweis": "Der Alert ist aktiv und wird vom Hintergrund-Job geprüft; "
            "beim Auslösen gibt es eine Benachrichtigung und einen Historien-"
            "eintrag (testbar via action=\"evaluate\").",
        }
    if args.alert_id is None:
        raise ToolFehler(f'„{args.action}“ benötigt eine "alert_id".')
    ziel = "paused" if args.action == "pause" else "active"
    alert = alerts_repo.update_alert(args.alert_id, state=ziel)
    if alert is None:
        raise ToolFehler(f"Alert {args.alert_id} existiert nicht.")
    return {
        "action": args.action,
        "alert": _alert_als_dict(alert),
        "hinweis": "Pausierte Alerts verlieren ihr Crossing-Gedächtnis und "
        "werden nicht evaluiert.",
    }


async def manage_alerts(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Alerts verwalten: Auflistung, Anlage (aktiv), Pause/Aktivierung, Sofort-Check."""
    args = pruefe_args(ManageAlertsArgs, roh_args, "manage_alerts")

    if args.action == "evaluate":
        if args.alert_id is None:
            raise ToolFehler('„evaluate“ benötigt eine "alert_id".')
        try:
            return {
                "action": "evaluate",
                "ergebnis": await alerts_service.evaluate_now(args.alert_id),
                "hinweis": "Ein manuelles Auslösen startet den Cooldown wie ein "
                "automatisches (kein Doppel-Feuern).",
            }
        except LookupError as exc:
            raise ToolFehler(str(exc)) from exc

    try:
        return await asyncio.to_thread(_sync_aktion, args)
    except ValueError as exc:
        raise ToolFehler(str(exc)) from exc


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="manage_alerts",
            description=(
                "Verwaltet Kurs-Alerts: anlegen (action=\"create\", z. B. „warne mich, "
                "wenn AAPL unter 240 fällt“ → kind=price_below, price=240), auflisten "
                "(action=\"list\"), pausieren/reaktivieren (action=\"pause\"|"
                "\"activate\", alert_id=…) oder sofort prüfen (action=\"evaluate\"). "
                "Löschen passiert bewusst nur im Alert-Center der UI. "
                'Beispiel: {"action": "create", "symbol": "AAPL", "kind": '
                '"price_below", "price": 240}.'
            ),
            parameters=schema_von(ManageAlertsArgs),
            handler=manage_alerts,
        )
    )


_registrieren()
