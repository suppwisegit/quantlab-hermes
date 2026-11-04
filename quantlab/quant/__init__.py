"""Quant-Engine (T8): reine Berechnungsfunktionen — bewusst ohne Datenbank-Zugriff.

Module:

- :mod:`quantlab.quant.returns` — Renditen aus Close-Kursen, Alignment, Annualisierung
- :mod:`quantlab.quant.metrics` — Kennzahlen (Total/CAGR/AnnReturn/AnnVol/Sharpe/
  Sortino/Calmar/MaxDD)
- :mod:`quantlab.quant.covariance` — Sample-/LedoitWolf-Kovarianz + Korrelation
- :mod:`quantlab.quant.pca` — Hauptkomponenten-Analyse auf standardisierten Renditen
- :mod:`quantlab.quant.risk` — VaR/CVaR, Beta/Alpha (OLS), Information Ratio
- :mod:`quantlab.quant.portfolio` — Gewichtete Portfolio-Renditen + kombinierte Analyse

Konventionen (PLAN §F, bindend): 252 bzw. √252, CAGR aus der Equity-Kurve,
Sortino mit Downside-Deviation, Calmar = CAGR/|MaxDD|, MaxDD pfadbasiert,
VaR historisch als Default (95 %), Beta/Alpha via OLS (Alpha × 252),
Kovarianz annualisiert × 252 (Default LedoitWolf), PCA auf standardisierten
Renditen, rf = 0 (konfigurierbar).

Alle Funktionen sind **reine Funktionen** auf pandas-/numpy-Eingaben: Der
Daten-Zugriff (Parquet über :mod:`quantlab.services.data_service`) passiert
ausschließlich in der Routerschicht (:mod:`quantlab.api.routers.quant`) —
dadurch bleiben die Berechnungen ohne Datenbank/Dateisystem testbar.
"""
