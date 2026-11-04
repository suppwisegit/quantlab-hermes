"""Kostenmodell für Backtests (§F Backtest, E008) — bewusst schmal und testbar.

Kosten fallen ausschließlich bei Fills an und sind prozentual (bps) auf das
**Trade-Volumen** (Notional der getauschten Wertseinheiten) zu entrichten:

- ``commission_bps`` — Broker-Gebühr (Default 10 bps = 0,10 %),
- ``slippage_bps``   — Ausführungs-Slippage (Default 5 bps = 0,05 %).

Beide Anteile werden als eine Summe vom Cash abgezogen (keine adverse
Kurs-Verrechnung) — die Gebühr ist damit in der Equity-Kurve direkt
nachvollziehbar und im Test exakt berechenbar.
"""

from __future__ import annotations

from dataclasses import dataclass

#: 1 bp = 0,01 % → Faktor 1/10_000.
BPS_DIVISOR: float = 10_000.0


@dataclass(frozen=True)
class CostModel:
    """bps-Kosten auf Trade-Volumen (§F: commission_bps + slippage_bps).

    Welle H (E100, additiv): ``min_fee_eur`` — Mindestgebühr je Fill in Euro
    (0,0 = aus, Default ⇒ Verhalten byte-identisch zur vorigen Form). Gebraucht
    vom PSB-Kostenmodell ``gebuehren_modell='psb'`` (0,3 % min 15 €,
    :func:`quantlab.services.planspiel.regeln.psb_kostenmodell`).
    """

    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    min_fee_eur: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_bps", "slippage_bps", "min_fee_eur"):
            value = float(getattr(self, name))
            if value < 0 or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} muss eine endliche Zahl >= 0 sein.")

    @property
    def total_bps(self) -> float:
        """Gesamtkostensatz in Basispunkten (Commission + Slippage)."""
        return float(self.commission_bps) + float(self.slippage_bps)

    def cost(self, notional: float) -> float:
        """Kosten für ein Trade-Volumen (absoluter Notional) — immer >= 0.

        Ohne ``min_fee_eur`` exakt der bps-Anteil (alter Vertrag). Mit Minimum
        gilt ``max(bps-Anteil, min_fee_eur)``; ein Trade-Volumen von 0 bleibt
        gebührenfrei (kein Fill — kein Mindestbetrag).
        """
        bps_anteil = abs(float(notional)) * self.total_bps / BPS_DIVISOR
        if bps_anteil <= 0.0:
            return 0.0
        return max(bps_anteil, float(self.min_fee_eur))
