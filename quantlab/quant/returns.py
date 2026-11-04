"""Renditen aus Close-Kursen, Alignment auf gemeinsamen Index, Annualisierung (T8).

**Entscheidung — simple Renditen als Default** (dokumentiert gem. Task):
``method="simple"`` ist der Default, ``method="log"`` steht als Option bereit.
Begründung:

1. **Portfolio-Aggregation:** Die gewichtete Portfolio-Rendite
   ``R_p = Σ w_i · r_i`` ist für simple Renditen exakt (linear). Log-Renditen
   sind nur zeitadditiv, nicht über Assets gewichtsmittelbar — Portfolio-,
   Kovarianz-, PCA- und Beta-Berechnungen blieben inkonsistent.
2. **Durchgängige Kette:** Simple Renditen erlauben die exakte Equity-Kurve
   ``equity_t = Π (1 + r)`` (für CAGR/MaxDD) und zugleich lineare Algebra in
   Kovarianz/PCA/Beta — eine Rendite-Matrix speist alle Kennzahlen.
3. Log-Renditen bleiben für Zeitradius-/Vol-Cones-Analysen als Option
   verfügbar (``close_to_close_returns(..., method="log")``).

**Alignment (bindend):** Mehrere Symbole werden auf die **Schnittmenge**
 ihrer Handelszeitpunkte gebracht (inner join auf dem Index, Zeilen mit
 fehlenden Werten verworfen). Der verwendete Zeitraum (Start/Ende,
 Beobachtungszahl) wird von den Endpunkten im Response ausgewiesen —
 Beobachtungen außerhalb der gemeinsamen Historie fließen bewusst nicht ein.

Annualisierung nach §F: ``periods_per_year`` konfigurierbar, Default 252
(Handelstage); Ann. Rendite = Ø Periodenrendite × periods_per_year, ann. Vol =
std × √periods_per_year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Default-Annualisierungsfaktor: Handelstage pro Jahr (§F, bindend).
PERIODS_PER_YEAR: int = 252

#: Unterstützte Rendite-Methoden (Default: ``simple`` — siehe Moduldocstring).
RETURNS_METHODS: tuple[str, ...] = ("simple", "log")

#: Mindestanzahl Rendite-Beobachtungen für aussagekräftige Kennzahlen.
#: 2 wegen ddof=1 (Stichproben-Standardabweichung) — weniger → ``ValueError``.
MIN_OBSERVATIONS: int = 2


def require_observations(
    returns: pd.DataFrame | pd.Series, min_obs: int = MIN_OBSERVATIONS
) -> None:
    """Wirft :class:`ValueError` (deutsch), wenn zu wenig Beobachtungen vorliegen.

    Reine Prüf-Funktion für die Quant-Module; die Routerschicht übersetzt den
    ``ValueError`` in HTTP 422 (Code ``zu_wenig_daten``).
    """
    count = int(len(returns))
    if count < min_obs:
        raise ValueError(
            f"Zu wenig Daten: {count} Rendite-Beobachtungen, mindestens {min_obs} "
            "erforderlich. Zeitraum oder Symbolauswahl erweitern."
        )


def annualization_factor(periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualisierungsfaktor für Volatilitäten: ``√periods_per_year`` (§F)."""
    if periods_per_year < 1:
        raise ValueError("periods_per_year muss mindestens 1 sein.")
    return float(np.sqrt(float(periods_per_year)))


def simple_returns(close: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Close-to-close Simple-Renditen (prozentuale Veränderung, ohne Füllung).

    ``fill_method=None``: fehlende Kurse erzeugen keine vorgetäuschte 0-%-Rendite;
    resultierende NaN-Zeilen fallen weg.
    """
    return close.pct_change(fill_method=None).dropna(how="any")


def log_returns(close: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Close-to-close Log-Renditen ``ln(P_t / P_{t-1})`` (Option zu ``simple``)."""
    return np.log(close).diff().dropna(how="any")


def close_to_close_returns(
    close: pd.DataFrame | pd.Series, method: str = "simple"
) -> pd.DataFrame | pd.Series:
    """Renditen aus Close-Kursen; ``method`` ∈ ``{"simple", "log"}`` (Default simple).

    Ungültige Methode → :class:`ValueError` (deutsch). Die erste Zeile (ohne
    Vorgänger) fällt jeweils weg.
    """
    if method == "simple":
        return simple_returns(close)
    if method == "log":
        return log_returns(close)
    raise ValueError(
        f"Unbekannte Rendite-Methode „{method}“. Gültig: {', '.join(RETURNS_METHODS)}."
    )


def align_close(close: pd.DataFrame) -> pd.DataFrame:
    """Schneidet mehrere Close-Serien auf die gemeinsamen Zeitpunkte (Schnittmenge).

    Verwirft Zeilen mit irgendeinem fehlenden Kurs („inner join“-Semantik,
    Dokumentations-Konvention im Modulkopf) sowie Duplikate im Index. Der
    übliche Weg zur gemeinsamen Matrix ist ``pd.concat(series_dict, axis=1)``
    (inner) — danach macht :func:`align_close` die Zeilen hart NaN-frei.
    Ein leeres Ergebnis ist zulässig; die Prüfung „zu wenig Daten“ übernimmt
    :func:`require_observations`.
    """
    joined = close.dropna(how="any")
    return joined[~joined.index.duplicated(keep="first")]


def align_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Wie :func:`align_close`, aber für bereits berechnete Rendite-Matrizen."""
    joined = returns.dropna(how="any")
    return joined[~joined.index.duplicated(keep="first")]


def equity_curve(
    returns: pd.Series | pd.DataFrame, initial: float = 1.0
) -> pd.Series | pd.DataFrame:
    """Aufkumulierte Equity aus Renditen: ``initial · Π (1 + r)`` (inkl. Startwert).

    Der implizite Startwert (Default 1.0) geht in die Pfadkennzahlen ein:
    Total Return = Ende/initial − 1, CAGR über ``len(returns)`` Perioden,
    MaxDD pfadbasiert über die gesamte Kurve (§F).
    """
    return (1.0 + returns).cumprod() * initial


# ---------------------------------------------------------------------------
# Annualisierungs-Helfer (§F)
# ---------------------------------------------------------------------------


def annualize_return(
    mean_period_return: float, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Ø Periodenrendite → annualisierte (arithmetische) Rendite: ``mean × ppy`` (§F)."""
    return float(mean_period_return) * float(periods_per_year)


def annualize_volatility(
    period_volatility: float, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Perioden-Volatilität → annualisierte Volatilität: ``σ × √ppy`` (§F)."""
    return float(period_volatility) * annualization_factor(periods_per_year)
