"""Portfolio-Performance-Kern (T23) — reine Berechnungen, kein I/O.

Fachliche Grundlage E025/E031 (RECHERCHE Block 6):

- **TWR (time-weighted return):** geometrisches Chain-Linking täglicher
  Sub-Perioden-Renditen ``r_TWR = ∏(1 + r_i) − 1``. Bewertungsgrid = tägliche
  Balken (Union aller beteiligten Symbole) plus Buchungs-/Randtage; Cashflow-
  Tage werden **nicht** in zwei Sub-Perioden aufgeteilt (kein intraday
  Repricing verfügbar), sondern über **Modified Dietz** behandelt:
  ``r = (V_end − V_start − CF) / (V_start + Σ w_i · CF_i)`` mit Dietz-Gewicht
  ``w = 1 − Tageszeit-Fraktion`` (Buchung um Mitternacht → w = 1, „der gesamte
  Tag investiert"; mittags → 0,5). Mehrtages-Lücken im Grid (Wochenenden bei
  gemischtem Grid) werden näherungsweise gleichbehandelt — dokumentierte
  Vereinfachung, exakt für 1-Tag-Sub-Perioden (der Normalfall).
- **XIRR (money-weighted):** löst ``Σ CF_i/(1+r)^{t_i} = 0`` mit
  ``scipy.optimize.brentq`` und Bracket ``(−0.9999, 10)``. Kein Vorzeichen-
  wechsel der Cashflows → :class:`ValueError` (Router → 422 deutsch);
  Bracket-Versagen → ``None`` (Router → ``xirr: null`` + Hinweisfeld).
- **Performance-Wirkung (E031):** Kauf/Verkauf sind performance-neutral (bis
  auf ihre Gebühren/Steuern), Dividenden erhöhen den Tageswert, ``cash_in``/
  ``cash_out``/``transfer`` sind TWR-neutrale externe Cashflows (gehen in
  XIRR ein) und Gebühren/Steuern senken den Wert als eigene Posten.
- **Einstandsverfahren:** Ø-EK (Default) oder FIFO (Option) mit
  realisiertem/unrealisiertem P&L. Gebühren/Steuern sind **nicht** im
  Einstandspreis versteckt (Ø-EK von 10@100 + 10@120 bleibt 110), sondern
  laufen als eigener Posten ``gebuehren_steuern`` mit;
  ``realisiert_netto = realisiert_preis_pnl − gebuehren_steuern``.

Engine-Typen: ``transfer`` wird vom Service in ``transfer_out`` (eigenes
Ledger) und ``transfer_in`` (Gegen-Depot bucht auf dieses Depot) aufgelöst —
eine Umbuchung ist damit TWR-neutral und XIRR-wirksam **als Paar** (im
Abgeber-Depot +amount, im Empfänger-Depot −amount aus Investorensicht).

Bewertung: Daily-Close **raw** (E022/E031); fehlender Kurs → letzter
bekannter Kurs (Carry-Forward, gezählt), kein Kurs vor dem ersten Bar →
Backfill des ersten Kurses (ebenfalls als Carry markiert), gar keine Kurse →
Symbol trägt 0 zum Wert bei und wird als ``symbole_ohne_kurs`` gemeldet.

Verkauf über den Bestand hinaus ist keine Struktur-Validierung des Ledgers:
Die Übermaße öffnet eine Short-Position (Einstand = Verkaufspreis), ein
folgender Kauf stellt glatt. Alle Funktionen sind rein; fachlich nicht
definierbare Werte → ``None``, zu wenig Input → ``ValueError`` (deutsch).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

#: int64-µs pro Kalendertag (E006).
US_PRO_TAG = 86_400_000_000

#: XIRR-Bracket (E025): (−0.9999, 10) — brentq braucht Vorzeichenwechsel.
XIRR_BRACKET: tuple[float, float] = (-0.9999, 10.0)

#: Toleranz gegen Float-Artefakte („theoretisch 0“-Bestände gelten als 0).
_EPS = 1e-12

#: Buchungstypen der Engine (``transfer`` ist vorab in out/in aufgelöst).
BUCHUNGSTYPEN: frozenset[str] = frozenset(
    {"buy", "sell", "dividend", "cash_in", "cash_out", "transfer_out", "transfer_in"}
)

#: Externe Cashflows: TWR-neutral (Dietz-Behandlung), XIRR-wirksam (E031).
EXTERNE_TYPEN: frozenset[str] = frozenset({"cash_in", "cash_out", "transfer_out", "transfer_in"})

#: Zulässige Einstandsverfahren (E031: Ø-EK Default, FIFO Option).
METHODEN: tuple[str, ...] = ("avg_cost", "fifo")

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Buchung:
    """Eine Ledger-Buchung in Engine-Begriffen (aus dem Repo abgeleitet).

    ``typ`` nutzt ``transfer_out``/``transfer_in`` statt ``transfer``; die
    Gebühren/Steuern einer Umbuchung zahlt ausschließlich die **sendende**
    Seite (der Service setzt sie beim Empfänger auf 0 — der Betrag kommt
    netto an).
    """

    ts: str
    typ: str
    symbol: str | None = None
    quantity: float | None = None
    price: float | None = None
    amount: float | None = None
    fees: float = 0.0
    taxes: float = 0.0
    ts_us: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.typ not in BUCHUNGSTYPEN:
            raise ValueError(f"Unbekannter Buchungstyp: {self.typ!r}")
        object.__setattr__(self, "ts_us", ts_zu_us(self.ts))

    @property
    def tag(self) -> int:
        """Kalendertag-Index (UTC) der Buchung."""
        return self.ts_us // US_PRO_TAG


def ts_zu_us(text: str) -> int:
    """ISO-8601-Text (naive Angaben gelten als UTC) → int64 µs UTC.

    ``ValueError`` mit deutscher Meldung bei unparsebarem Text.
    """
    try:
        moment = datetime.fromisoformat(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"Ungültiger Buchungszeitpunkt „{text}“.") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment.astimezone(UTC) - _EPOCH) // timedelta(microseconds=1)


def tag_zu_us(tag: int) -> int:
    """Kalendertag-Index → Mitternacht-µs UTC (Grid-Raster, E006)."""
    return int(tag) * US_PRO_TAG


def heute_tag() -> int:
    """Kalendertag-Index „heute“ (UTC) — Default-Ende des Bewertungsgrids."""
    return (datetime.now(UTC) - _EPOCH).days


def dietz_gewicht(ts_us: int) -> float:
    """Modified-Dietz-Gewicht einer Buchung: 1 − Tageszeit-Fraktion.

    Mitternacht → 1,0 (ganzer Tag investiert), 12:00 UTC → 0,5, 23:59 → ≈ 0.
    """
    return 1.0 - (ts_us % US_PRO_TAG) / US_PRO_TAG


# ---------------------------------------------------------------------------
# Positionen: Einstandsverfahren Ø-EK (Default) / FIFO (E031)
# ---------------------------------------------------------------------------


@dataclass
class _WalkPosition:
    """Mutabler Walk-Zustand je Symbol (nicht Teil der öffentlichen API)."""

    symbol: str
    menge: float = 0.0
    avg_einstand: float = 0.0  # Ø-EK-Verfahren (Stückpreis, immer > 0)
    lots: list[list[float]] = field(default_factory=list)  # FIFO: [menge, preis]
    realisiert_preis: float = 0.0
    gebuehren_steuern: float = 0.0
    dividenden_total: float = 0.0
    dividenden_count: int = 0
    dividenden_letzte_ts: str | None = None
    erste_ts: str | None = None
    letzte_ts: str | None = None


def _avg_kauf(pos: _WalkPosition, menge: float, preis: float) -> None:
    if pos.menge < -_EPS:  # Short teilweise glattstellen
        glatt = min(menge, -pos.menge)
        pos.realisiert_preis += glatt * (pos.avg_einstand - preis)
        rest = menge - glatt
        pos.menge += menge
        if rest > _EPS:
            pos.avg_einstand = preis
    elif pos.menge > _EPS:
        pos.avg_einstand = (pos.menge * pos.avg_einstand + menge * preis) / (
            pos.menge + menge
        )
        pos.menge += menge
    else:
        pos.avg_einstand = preis
        pos.menge = menge


def _avg_verkauf(pos: _WalkPosition, menge: float, preis: float) -> None:
    if pos.menge > _EPS:
        glatt = min(menge, pos.menge)
        pos.realisiert_preis += glatt * (preis - pos.avg_einstand)
        rest = menge - glatt
        pos.menge -= menge
        if rest > _EPS:  # Short eröffnet
            pos.avg_einstand = preis
    elif pos.menge < -_EPS:  # Short ausbauen
        pos.avg_einstand = (-pos.menge * pos.avg_einstand + menge * preis) / (
            -pos.menge + menge
        )
        pos.menge -= menge
    else:  # neue Short-Position
        pos.avg_einstand = preis
        pos.menge = -menge


def _fifo_verbrauche(pos: _WalkPosition, menge: float, preis: float, *, kurz: bool) -> None:
    """FIFO-Konsum: Verkauf isst Long-Lots von vorn, Kauf isst Short-Lots.

    ``kurz=True`` heißt „Kauf“ — dann werden negative (Short-)Lots zuerst
    glattgestellt (realisiert = Short-Einstand − Rückkaufpreis).
    """
    rest = menge
    while rest > _EPS and pos.lots:
        lot_menge = pos.lots[0][0]
        passt = lot_menge < -_EPS if kurz else lot_menge > _EPS
        if not passt:
            break
        lot_preis = pos.lots[0][1]
        stueck = min(rest, abs(lot_menge))
        pos.realisiert_preis += (
            stueck * (lot_preis - preis) if kurz else stueck * (preis - lot_preis)
        )
        neu = lot_menge + (stueck if kurz else -stueck)
        if abs(neu) < _EPS:
            pos.lots.pop(0)
        else:
            pos.lots[0] = [neu, lot_preis]
        rest -= stueck
    if rest > _EPS:  # Rest: Kauf-Rest ist Long, Verkaufs-Rest ist Short
        pos.lots.append([rest, preis] if kurz else [-rest, preis])


@dataclass(frozen=True)
class PositionErgebnis:
    """Abgeleitete Position nach Abschluss des Walks (E031)."""

    symbol: str
    menge: float
    einstand_stueck: float | None  # None bei Menge ≈ 0
    einstand_wert: float  # menge × einstand_stueck (0 bei Menge ≈ 0)
    realisiert_preis_pnl: float  # rein kursbedingt (matched lots)
    gebuehren_steuern: float  # Gebühren + Steuern aller Buchungen des Symbols
    realisiert_netto: float  # realisiert_preis_pnl − gebuehren_steuern
    dividenden_total: float
    dividenden_count: int
    dividenden_letzte_ts: str | None
    erste_ts: str | None
    letzte_ts: str | None


@dataclass(frozen=True)
class PositionsErgebnis:
    """Ergebnis von :func:`positionen_ableiten`: Positionen + Cash-Stand."""

    cash: float
    positionen: dict[str, PositionErgebnis]


def positionen_ableiten(buchungen: list[Buchung], method: str = "avg_cost") -> PositionsErgebnis:
    """Leitet Bestand, Einstand und realisierten P&L aus dem Ledger ab.

    ``method`` ∈ ``{"avg_cost", "fifo"}`` (E031: Ø-EK Default, FIFO Option).
    Chronologischer Walk (stabil nach Eingabereihenfolge bei gleichem ts);
    ``cash`` ist der Saldo aller Geldbewegungen (Käufe zahlen inkl. Gebühren/
    Steuern, Verkäufe erhalten netto, Dividenden brutto minus Steuern — E031).
    """
    if method not in METHODEN:
        raise ValueError(
            f"Unbekanntes Einstandsverfahren „{method}“. Gültig: {', '.join(METHODEN)}."
        )
    sortiert = sorted(buchungen, key=lambda b: b.ts_us)
    walk: dict[str, _WalkPosition] = {}
    cash = 0.0

    def _symbol_pos(symbol: str | None) -> _WalkPosition | None:
        if symbol is None:
            return None
        if symbol not in walk:
            walk[symbol] = _WalkPosition(symbol=symbol)
        return walk[symbol]

    for b in sortiert:
        pos = _symbol_pos(b.symbol) if b.typ in ("buy", "sell", "dividend") else None
        if b.typ == "buy":
            assert pos is not None and b.quantity and b.price is not None
            cash -= b.quantity * b.price + b.fees + b.taxes
            pos.gebuehren_steuern += b.fees + b.taxes
            if method == "avg_cost":
                _avg_kauf(pos, b.quantity, b.price)
            else:
                pos.menge += b.quantity
                _fifo_verbrauche(pos, b.quantity, b.price, kurz=True)
        elif b.typ == "sell":
            assert pos is not None and b.quantity and b.price is not None
            cash += b.quantity * b.price - b.fees - b.taxes
            pos.gebuehren_steuern += b.fees + b.taxes
            if method == "avg_cost":
                _avg_verkauf(pos, b.quantity, b.price)
            else:
                pos.menge -= b.quantity
                _fifo_verbrauche(pos, b.quantity, b.price, kurz=False)
        elif b.typ == "dividend":
            cash += (b.amount or 0.0) - b.fees - b.taxes
            if pos is not None:
                pos.gebuehren_steuern += b.fees + b.taxes
                pos.dividenden_total += b.amount or 0.0
                pos.dividenden_count += 1
                pos.dividenden_letzte_ts = b.ts
        elif b.typ == "cash_in":
            cash += (b.amount or 0.0) - b.fees - b.taxes
        elif b.typ == "cash_out":
            cash -= (b.amount or 0.0) + b.fees + b.taxes
        elif b.typ == "transfer_out":
            cash -= (b.amount or 0.0) + b.fees + b.taxes
        elif b.typ == "transfer_in":
            cash += b.amount or 0.0  # netto — Gebühren zahlt die Sendeseite
        else:  # pragma: no cover — Buchung.__post_init__ fängt unbekannte Typen
            raise ValueError(f"Unbekannter Buchungstyp: {b.typ!r}")
        if pos is not None:
            pos.letzte_ts = b.ts
            if pos.erste_ts is None:
                pos.erste_ts = b.ts

    positionen: dict[str, PositionErgebnis] = {}
    for symbol, pos in walk.items():
        if method == "avg_cost":
            einstand = pos.avg_einstand if abs(pos.menge) > _EPS else None
            einstand_wert = pos.menge * einstand if einstand is not None else 0.0
        else:
            menge_netto = sum(lot[0] for lot in pos.lots)
            einstand_wert = sum(abs(lot[0]) * lot[1] for lot in pos.lots)
            einstand = (
                einstand_wert / sum(abs(lot[0]) for lot in pos.lots)
                if abs(menge_netto) > _EPS and pos.lots
                else None
            )
            if einstand is None:
                einstand_wert = 0.0
            else:
                einstand_wert = menge_netto * einstand
        positionen[symbol] = PositionErgebnis(
            symbol=symbol,
            menge=0.0 if abs(pos.menge) < _EPS else float(pos.menge),
            einstand_stueck=einstand,
            einstand_wert=float(einstand_wert),
            realisiert_preis_pnl=float(pos.realisiert_preis),
            gebuehren_steuern=float(pos.gebuehren_steuern),
            realisiert_netto=float(pos.realisiert_preis - pos.gebuehren_steuern),
            dividenden_total=float(pos.dividenden_total),
            dividenden_count=pos.dividenden_count,
            dividenden_letzte_ts=pos.dividenden_letzte_ts,
            erste_ts=pos.erste_ts,
            letzte_ts=pos.letzte_ts,
        )
    return PositionsErgebnis(cash=float(cash), positionen=positionen)


# ---------------------------------------------------------------------------
# Bewertungsreihe: tägliche Grid-Werte mit Carry-Forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BewertungErgebnis:
    """Tägliche Gesamtwerte (Cash + Positionen zum Daily-Close, E031)."""

    grid_us: list[int]  # Bewertungszeitpunkte (Mitternacht-µs UTC, E006)
    werte: list[float]  # Gesamtwert je Grid-Punkt (Cash + Kurse)
    carry_tage: int  # Symbol-Tage mit getragenem Kurs (kein Bar am Tag)
    symbole_ohne_kurs: list[str]  # gehaltene Symbole ohne jegliche Kurse
    letzte_kurse: dict[str, tuple[float, int]]  # symbol → (kurs, bar_ts_us)


def _kurs_lookup(
    kurse: pd.Series | None, g_us: int
) -> tuple[float | None, int | None, bool]:
    """Kurs zum Grid-Tag: exakt > Carry-Forward (ffill) > Backfill > None."""
    if kurse is None or kurse.empty:
        return None, None, True
    index = kurse.index.to_numpy()
    werte = kurse.to_numpy(dtype=float)
    pos = bisect_left(index, g_us)
    if pos < len(index) and int(index[pos]) == g_us:
        return float(werte[pos]), int(index[pos]), False
    if pos > 0:  # letzter bekannter Kurs vor dem Tag (Carry-Forward, E031)
        return float(werte[pos - 1]), int(index[pos - 1]), True
    if pos < len(index):  # vor der ersten Notierung: Backfill des Erstkurses
        return float(werte[0]), int(index[0]), True
    return None, None, True


def bewertungsreihe(
    buchungen: list[Buchung],
    kurse: dict[str, pd.Series | None],
    start_tag: int | None = None,
    end_tag: int | None = None,
) -> BewertungErgebnis:
    """Tägliche Gesamtwerte des Depots auf dem Bewertungsgrid.

    Grid = Union der Daily-Bar-Tage aller Kurse ∪ Buchungstage ∪ Randtage
    (``start_tag`` Default: erster Buchungstag; ``end_tag`` Default: heute),
    begrenzt auf ``[start_tag, end_tag]``. Damit liegen externe Cashflows
    immer auf Grid-Tagen (Dietz-Gewicht sauber) und reente Buchungen nach dem
    letzten Bar werden bis ``end_tag`` flach weitergeführt (XIRR-Endwert
    korrekt). Nur-Cash-Depots bekommen das Buchungs-/Randtage-Grid.
    """
    if not buchungen:
        return BewertungErgebnis([], [], 0, [], {})
    start = min(b.tag for b in buchungen) if start_tag is None else int(start_tag)
    ende = heute_tag() if end_tag is None else int(end_tag)
    if ende < start:
        raise ValueError(
            "Zeitraum ungültig: Das Ende liegt vor dem Start der Buchungen."
        )

    tage: set[int] = set()
    for series in kurse.values():
        if series is not None and not series.empty:
            for ts in series.index.to_numpy():
                tag = int(ts) // US_PRO_TAG
                if start <= tag <= ende:
                    tage.add(tag)
    for b in buchungen:
        if start <= b.tag <= ende:
            tage.add(b.tag)
    tage.add(start)
    tage.add(ende)
    grid_us = [tag_zu_us(tag) for tag in sorted(tage)]

    sortiert = sorted(buchungen, key=lambda b: b.ts_us)
    zeiger = 0
    cash = 0.0
    mengen: dict[str, float] = {symbol: 0.0 for symbol in kurse}
    werte: list[float] = []
    carry = 0
    ohne_kurs: set[str] = set()
    letzte_kurse: dict[str, tuple[float, int]] = {}

    for g_us in grid_us:
        g_tag = g_us // US_PRO_TAG
        while zeiger < len(sortiert) and sortiert[zeiger].tag <= g_tag:
            b = sortiert[zeiger]
            if b.typ == "buy":
                cash -= (b.quantity or 0.0) * (b.price or 0.0) + b.fees + b.taxes
                if b.symbol:
                    mengen[b.symbol] = mengen.get(b.symbol, 0.0) + (b.quantity or 0.0)
            elif b.typ == "sell":
                cash += (b.quantity or 0.0) * (b.price or 0.0) - b.fees - b.taxes
                if b.symbol:
                    mengen[b.symbol] = mengen.get(b.symbol, 0.0) - (b.quantity or 0.0)
            elif b.typ == "dividend":
                cash += (b.amount or 0.0) - b.fees - b.taxes
            elif b.typ == "cash_in":
                cash += (b.amount or 0.0) - b.fees - b.taxes
            elif b.typ in ("cash_out", "transfer_out"):
                cash -= (b.amount or 0.0) + b.fees + b.taxes
            else:  # transfer_in
                cash += b.amount or 0.0
            zeiger += 1
        total = cash
        for symbol, menge in mengen.items():
            if abs(menge) < _EPS:
                continue
            preis, bar_ts, is_carry = _kurs_lookup(kurse.get(symbol), g_us)
            if preis is None or bar_ts is None:
                ohne_kurs.add(symbol)  # trägt 0 bei — bewusst, gemeldet (E031)
                continue
            if is_carry:
                carry += 1
            total += menge * preis
            letzte_kurse[symbol] = (preis, bar_ts)
        werte.append(float(total))
    return BewertungErgebnis(
        grid_us=grid_us,
        werte=werte,
        carry_tage=carry,
        symbole_ohne_kurs=sorted(ohne_kurs),
        letzte_kurse=letzte_kurse,
    )


# ---------------------------------------------------------------------------
# TWR: tägliches Chain-Linking mit Modified Dietz an Cashflow-Tagen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwrErgebnis:
    """TWR über das Bewertungsgrid (E025/E031)."""

    total: float | None  # None bei weniger als 2 Bewertungspunkten
    taeglich: list[float]  # Sub-Perioden-Renditen (len(grid) − 1)
    neutrale_tage: int  # Sub-Perioden mit Basis ≤ 0 (neutral behandelt)


def twr(bewertung: BewertungErgebnis, buchungen: list[Buchung]) -> TwrErgebnis:
    """Chain-Linking der Sub-Perioden-Renditen; externe Cashflows via Dietz.

    Sub-Perioden laufen zwischen aufeinanderfolgenden Grid-Punkten; ein
    Cashflow zählt zur Sub-Periode, die mit dem nächsten Grid-Tag **nach**
    seinem Kalendertag endet (erster Grid-Index via ``bisect_left``; Flows
    vor dem ersten Bewertungspunkt sind bereits in der Basis). Numerisch
    unheilbare Sub-Perioden (Basis ≤ 0) werden neutral (0) verbucht und
    gezählt — kein ``inf``/``nan`` (JSON-Sicherheit).
    """
    grid = bewertung.grid_us
    werte = bewertung.werte
    if len(grid) < 2:
        return TwrErgebnis(None, [], 0)
    grid_tage = [ts // US_PRO_TAG for ts in grid]

    # Externe Cashflows je Sub-Perioden-Endindex: (Basis-Cashflow, Dietz-Summe)
    fluesse: dict[int, list[tuple[float, float]]] = {}
    for b in buchungen:
        if b.typ not in EXTERNE_TYPEN:
            continue
        idx = bisect_left(grid_tage, b.tag)
        if idx <= 0 or idx >= len(grid):
            continue  # vor dem ersten Bewertungspunkt oder hinter dem Fenster
        betrag = {
            "cash_in": b.amount or 0.0,
            "cash_out": -(b.amount or 0.0),
            "transfer_in": b.amount or 0.0,
            "transfer_out": -(b.amount or 0.0),
        }[b.typ]
        fluesse.setdefault(idx, []).append((betrag, dietz_gewicht(b.ts_us)))

    taeglich: list[float] = []
    neutral = 0
    for i in range(1, len(grid)):
        v_start, v_ende = werte[i - 1], werte[i]
        cf = 0.0
        basis_fluss = 0.0
        for betrag, gewicht in fluesse.get(i, []):
            cf += betrag
            basis_fluss += gewicht * betrag
        basis = v_start + basis_fluss
        if basis <= 1e-9:
            taeglich.append(0.0)
            neutral += 1
            continue
        taeglich.append((v_ende - v_start - cf) / basis)

    produkt = 1.0
    for r in taeglich:
        produkt *= 1.0 + r
    return TwrErgebnis(float(produkt - 1.0), taeglich, neutral)


# ---------------------------------------------------------------------------
# XIRR: brentq mit Bracket (−0.9999, 10)
# ---------------------------------------------------------------------------


def xirr_cashflows(
    buchungen: list[Buchung], endwert: float, end_tag: int
) -> list[tuple[int, float]]:
    """Externe Cashflows aus Investorensicht plus Endwert-Position.

    Einzahlung/Transfer-Eingang = negativ (Investor zahlt ein), Auszahlung/
    Transfer-Abgang = positiv; der Endwert des Depots ist die letzte positive
    Position (fiktive Auflösung). Rückgabe: ``(tag_index, betrag)``.
    """
    cashflows: dict[int, float] = {}
    for b in buchungen:
        if b.typ not in EXTERNE_TYPEN:
            continue
        betrag = {
            "cash_in": -(b.amount or 0.0),
            "cash_out": b.amount or 0.0,
            "transfer_in": -(b.amount or 0.0),
            "transfer_out": b.amount or 0.0,
        }[b.typ]
        cashflows[b.tag] = cashflows.get(b.tag, 0.0) + betrag
    if abs(endwert) > _EPS:
        cashflows[end_tag] = cashflows.get(end_tag, 0.0) + float(endwert)
    return sorted(cashflows.items())


def xirr(cashflows: list[tuple[int, float]]) -> float | None:
    """Löst ``Σ CF_i/(1+r)^{t_i} = 0`` (t in Jahren, 365-Tage-Basis).

    ``ValueError`` ohne Vorzeichenwechsel (Router → 422 deutsch);
    ``None``, wenn brentq im Bracket ``(−0.9999, 10)`` keine Wurzel findet
    (Router → ``xirr: null`` + Hinweisfeld, R14).
    """
    if not cashflows:
        raise ValueError(
            "XIRR undefiniert: Es liegen keine Cashflows vor. "
            "Buche mindestens eine Einzahlung (cash_in)."
        )
    zusammen: dict[int, float] = {}
    for tag, betrag in cashflows:
        if abs(betrag) > _EPS:
            zusammen[tag] = zusammen.get(tag, 0.0) + betrag
    werte = sorted(zusammen.items())
    if not any(b > _EPS for _, b in werte) or not any(b < -_EPS for _, b in werte):
        raise ValueError(
            "XIRR undefiniert: Die Cashflows haben keinen Vorzeichenwechsel "
            "(nur Zu- oder nur Abflüsse). Buche Ein- und Auszahlungen, damit "
            "eine Rendite existiert."
        )
    t0 = werte[0][0]
    vorbereitet = [((tag - t0) / 365.0, betrag) for tag, betrag in werte]

    def npv(rate: float) -> float:
        summe = 0.0
        for t, betrag in vorbereitet:
            try:
                summe += betrag / (1.0 + rate) ** t
            except OverflowError:
                # (1+r)^t läuft gegen ∞ bzw. 0 — der Term kollabiert mit Vorzeichen.
                summe += float("inf") if betrag > 0 else float("-inf")
        return summe

    try:
        # M2a/E-M08: scipy lazy in der Funktion — Android/Chaquopy hat kein
        # scipy (Desktop-Verhalten unverändert).
        from scipy.optimize import brentq

        return float(brentq(npv, XIRR_BRACKET[0], XIRR_BRACKET[1], xtol=1e-12))
    except ValueError:
        return None  # keine Wurzel im Bracket — definiertes None (E025)


# ---------------------------------------------------------------------------
# Kennzahlen auf den TWR-Tagesrenditen (§F über bestehende Metriken)
# ---------------------------------------------------------------------------


def taegliche_renditen(twr_ergebnis: TwrErgebnis, grid_us: list[int]) -> pd.Series:
    """TWR-Sub-Perioden-Renditen als pandas-Serie (Index = Grid-µs, §F-Basis)."""
    index = pd.Index(grid_us[1:], dtype="int64")
    return pd.Series(twr_ergebnis.taeglich, index=index, dtype=float)


def produkt_der_renditen(taeglich: list[float]) -> float:
    """``∏(1 + r) − 1`` — Kettenprodukt (Referenz für Total-Return-Tests)."""
    produkt = 1.0
    for r in taeglich:
        produkt *= 1.0 + r
    return float(produkt - 1.0)


def sample_equity(
    grid_us: list[int], werte: list[float], max_punkte: int = 2000
) -> list[dict[str, Any]]:
    """Gleichmäßige Stichprobe der Equity-Kurve (Endpunkt bleibt, Muster T8)."""
    if not grid_us:
        return []
    if len(grid_us) > max_punkte:
        indices = np.unique(np.linspace(0, len(grid_us) - 1, max_punkte).round().astype(int))
    else:
        indices = np.arange(len(grid_us))
    return [
        {"ts": int(grid_us[i]), "value": float(werte[i])}
        for i in indices
    ]
