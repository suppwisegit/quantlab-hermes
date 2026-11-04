"""Backtesting-Domäne (T10): Engine, Kostenmodell, Referenz-Strategien.

Aufbau (§F Backtest, E008):

- :mod:`quantlab.backtest.costs` — bps-basiertes Kostenmodell (Commission +
  Slippage auf Trade-Volumen).
- :mod:`quantlab.backtest.engine` — Bar-Loop über kanonische OHLC-Daten;
  Strategie-Gewichte werden intern um 1 Bar geshiftet (Fill t+1 Open,
  optional t+1 Close) — Look-Ahead ist strukturell ausgeschlossen.
- :mod:`quantlab.backtest.strategies` — Referenz-Strategien mit
  Parameter-Schema (``STRATEGIES``-Registry für UI/Agent).
"""
