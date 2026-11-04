"""PSB-Live-Recorder (Welle H4, E096): Tab-Sweeps + Fokus-Quotes → 1m-Bars.

Ein **Sweep** (Muster Referenz ``planspiel_data_tools/.../daten/recorder.py``,
``_sweep`` Z. 119–161 / ``_fokus_quotes`` Z. 163–213):

- **Breite:** ``tabs_lesen()`` (~18 Marktgruppen) → je Gruppe ``markt_tab``
  (Kurs/Performance/Volumen je Instrument). Einzelne Gruppen dürfen
  scheitern — sie werden gezählt, der Sweep läuft weiter (Referenz Z. 135).
- **Fokus:** Depot-Positionen (``standard_depot_id`` → ``positionen_holen``)
  und Watchlist-Symbole mit ``symbols.isin`` (E091-Brücke) werden dedupliert
  und auf :data:`FOKUS_LIMIT` (40) gekappt; je ISIN liefert
  ``instrument_mit_kursen`` die Detail-Quotes (bid/ask/high/low/totalVolume).
  Fehlender Kurs → ``(bid + ask) / 2`` (Referenz Z. 200–203), danach bid,
  danach ask; ohne jeden Kurs fällt das Instrument aus.
- **Speicherung:** Close-only-Bars konsistent zum Byteworx-Bein (E094) —
  ``open = high = low = close = Kurs`` (keine erfundenen Hoch/Tiefs),
  ``volume`` = geliefertes (Gesamt-)Volumen, sonst **0.0** (ANNAHME,
  konsistent zum Bestand: Byteworx-Bein und
  :func:`parquet_store.normalize_ohlcv` füllen fehlendes Volumen neutral mit
  0.0). Geschrieben wird je ISIN als QuantLab-Symbol (E093: die ISIN ist ein
  gültiges Symbol) in den Parquet-Store, Intervall ``1m``; ``bar_meta.source``
  wird auf ``"psb"`` gesetzt (Muster ``downloads._bar_meta_source_setzen`` —
  hier bewusst eigener SQL-Write, weil der Recorder nicht über den
  Download-Job-Manager läuft).
- **Dedup:** Der ``ts`` eines Sweep-Balkens ist die auf die Minute gefloorte
  Sweep-Zeit — mehrere Sweeps in derselben Minute erzeugen denselben ``ts``
  und das Upsert-Protokoll (last-write-wins auf ``ts``) überschreibt den
  Balken; Duplikatzeilen kann es dadurch nie geben (E005).
- **Adaptiv/Backoff:** Intervall aus der Settings-Sektion ``planspiel``
  (``recorder_intervall_s``, 30–3600 s, Default 60) — je Aufruf neu gelesen.
  Nach fehlgeschlagenen Sweeps verdoppelt sich das Intervall exponentiell
  (Muster Referenz Z. 106: ``intervall * 2 ** min(fehler, 4)``) mit der
  Obergrenze :data:`MAX_INTERVALL_S`; ein erfolgreicher Sweep setzt den
  Fehler-Zähler zurück. Der Zustand lebt prozessweit unter Lock (der
  Scheduler sequenziert die Job-Art ohnehin per Single-Flight).

Alles synchron mit injizierbarem Client — netzfreie Tests fahren Fakes
(Muster ``universum.importiere_universum``). Konsumiert ausschließlich
:mod:`quantlab.services.planspiel` + Repos/Store (kein Router-Import,
kein eigenes httpx — Schichtungsvertrag der Tools).
"""

from __future__ import annotations

import math
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from quantlab import db
from quantlab.services.planspiel.fehler import PlanspielFehler
from quantlab.storage import parquet_store
from quantlab.storage.repos import (
    psb_repo,
    settings_repo,
    symbols_repo,
    watchlists_repo,
)

#: ``bar_meta.source``-Marke der Recorder-Balken (E096: „quelle ``psb``“).
QUELLE = "psb"

#: Parquet-Intervall der Recorder-Balken (E096: 1m-Bars).
INTERVALL = "1m"

#: Maximalzahl Fokus-ISINs je Sweep (Referenz ``FOKUS_LIMIT`` = 40).
FOKUS_LIMIT = 40

#: Recorder-Intervall: Default und erlaubtes Fenster in Sekunden (E096 —
#: Validierung deutsch in :func:`einstellungen_setzen`, Klemmen in
#: :func:`einstellungen`). Der Client Pacingt ≥ 1 s je Request, ein voller
#: Tab-Sweep dauert real ~20 s — Untergrenze 30 s bleibt daher netzfair.
STANDARD_INTERVALL_S = 60
MIN_INTERVALL_S = 30
MAX_INTERVALL_S = 3600

#: Höchster Backoff-Exponent (Muster Referenz Z. 106: ``min(fehler, 4)``).
_MAX_BACKOFF_POTENZ = 4

_SPERR_ZUSTAND = threading.Lock()
_fehler_serie: int = 0  # aufeinanderfolgende fehlgeschlagene Sweeps


# ---------------------------------------------------------------------------
# Einstellungen (je Aufruf neu gelesen — Änderungen greifen ohne Neustart)
# ---------------------------------------------------------------------------


def einstellungen() -> dict[str, Any]:
    """Settings-Sektion ``planspiel`` (Recorder-Keys), defensiv getypt.

    Defekte Werte (falscher Typ) fallen auf die Defaults, außerhalb des
    Fensters liegende Intervalle werden auf 30–3600 geklemmt — der Recorder-
    Tick darf an einer verstümmelten Einstellung nie sterben (Muster
    ``scheduler.scheduler_einstellungen``).
    """
    try:
        sektion = settings_repo.get_section("planspiel")
    except ValueError:  # pragma: no cover — Defaults sind ab H4 vorhanden
        sektion = {}

    aktiv = sektion.get("recorder_aktiv", False)
    intervall = sektion.get("recorder_intervall_s", STANDARD_INTERVALL_S)
    if isinstance(intervall, bool) or not isinstance(intervall, int | float):
        intervall = STANDARD_INTERVALL_S
    return {
        "aktiv": bool(aktiv) if isinstance(aktiv, bool) else False,
        "intervall_s": int(min(max(int(intervall), MIN_INTERVALL_S), MAX_INTERVALL_S)),
    }


def einstellungen_setzen(
    *, aktiv: bool | None = None, intervall_s: int | None = None
) -> dict[str, Any]:
    """Setzt Recorder-Einstellungen mit deutscher Bereichsvalidierung (E096).

    ``intervall_s`` muss ganzzahlig zwischen 30 und 3600 liegen (sonst
    ``ValueError`` mit deutscher Meldung — Muster Referenz
    ``einstellungen_setzen`` Z. 61–69). Liefert die effektiven Einstellungen.
    """
    updates: dict[str, Any] = {}
    if intervall_s is not None:
        if isinstance(intervall_s, bool) or not isinstance(intervall_s, int):
            raise ValueError(
                "Das Recorder-Intervall muss eine ganze Zahl in Sekunden sein."
            )
        if not MIN_INTERVALL_S <= intervall_s <= MAX_INTERVALL_S:
            raise ValueError(
                f"Das Recorder-Intervall muss zwischen {MIN_INTERVALL_S} und "
                f"{MAX_INTERVALL_S} Sekunden liegen."
            )
        updates["recorder_intervall_s"] = int(intervall_s)
    if aktiv is not None:
        if not isinstance(aktiv, bool):
            raise ValueError("Der Recorder-Schalter „recorder_aktiv“ muss ein Boolean sein.")
        updates["recorder_aktiv"] = aktiv
    if updates:
        settings_repo.update_section("planspiel", updates)
    return einstellungen()


# ---------------------------------------------------------------------------
# Backoff (prozentweite Fehler-Serie, Muster Referenz Z. 106)
# ---------------------------------------------------------------------------


def fehler_registrieren() -> int:
    """Registriert einen fehlgeschlagenen Sweep; liefert die Serie zurück."""
    global _fehler_serie
    with _SPERR_ZUSTAND:
        _fehler_serie += 1
        return _fehler_serie


def fehler_zuruecksetzen() -> None:
    """Setzt die Fehler-Serie nach einem erfolgreichen Sweep zurück."""
    global _fehler_serie
    with _SPERR_ZUSTAND:
        _fehler_serie = 0


def naechstes_intervall_s() -> int:
    """Intervall des nächsten Sweep-Takts (Basis × Backoff, gekappt).

    Ohne Fehler-Serie die geklemmte Basiseinstellung; danach
    ``basis × 2 ** min(serie, 4)`` mit Obergrenze :data:`MAX_INTERVALL_S`
    (Referenz Z. 106 — dort 900 s, hier das Settings-Maximum).
    """
    basis = einstellungen()["intervall_s"]
    with _SPERR_ZUSTAND:
        serie = _fehler_serie
    if serie <= 0:
        return basis
    return min(basis * (2 ** min(serie, _MAX_BACKOFF_POTENZ)), MAX_INTERVALL_S)


# ---------------------------------------------------------------------------
# Fokus-ISINs (Depot + Watchlist, Dedup, Cap 40) — degradationstolerant
# ---------------------------------------------------------------------------


def fokus_isins(client: Any) -> tuple[list[str], list[str]]:
    """Ermittelt die Fokus-ISINs des Sweeps (Referenz ``_fokus_quotes``).

    Quellen: Depot-Positionen der ``planspiel.standard_depot_id`` (Client)
    und Watchlist-Symbole mit gepflegter ``symbols.isin`` (E091-Brücke).
    Dedup (sortiert, deterministisch), Cap :data:`FOKUS_LIMIT`.

    Liefert ``(isins, hinweise)`` — fehlende/scheiternde Quellen degradieren
    elegant (deutsche Hinweise im Ergebnis; das planspiel-Paket schreibt
    bewusst keine Log-Ausgabe, QA-Pin test_p1_qa — der Sweep läuft als reiner
    Tab-Sweep weiter), sie werfen nie.
    """
    hinweise: list[str] = []
    mengen: set[str] = set()

    # Depot-Positionen (bestehender Client-Zugriff, Muster Referenz Z. 172–181).
    try:
        depot_id = int(settings_repo.get_section("planspiel").get("standard_depot_id") or 0)
    except ValueError:  # pragma: no cover — Defaults sind ab H4 vorhanden
        depot_id = 0
    if depot_id > 0:
        try:
            positionen = client.positionen_holen(depot_id)
        except PlanspielFehler as exc:
            hinweise.append(
                f"Depot-Positionen konnten nicht gelesen werden: {exc.message}"
            )
        else:
            for item in positionen.get("items") or []:
                instrument = item.get("instrument") if isinstance(item, dict) else None
                isin = (instrument or {}).get("isin") if isinstance(instrument, dict) else None
                if isinstance(isin, str) and isin.strip():
                    mengen.add(isin.strip().upper())
    else:
        hinweise.append("Kein Standard-Depot hinterlegt — Fokus nur aus der Watchlist.")

    # Watchlist-Symbole → Registry-ISIN (E091-Brücke über symbols.isin).
    watchlist = watchlists_repo.list_symbols(watchlists_repo.DEFAULT_WATCHLIST_ID)
    ohne_isin = 0
    for eintrag in watchlist:
        symbol = symbols_repo.get_symbol(eintrag.symbol)
        if symbol is not None and symbol.isin:
            mengen.add(symbol.isin.strip().upper())
        else:
            ohne_isin += 1
    if ohne_isin:
        hinweise.append(
            f"{ohne_isin} Watchlist-Symbole ohne gepflegte ISIN — für sie gibt "
            "es keinen Fokus-Quote."
        )

    if not mengen:
        hinweise.append("Keine Fokus-ISINs ermittelbar — nur Tab-Sweeps.")

    sortiert = sorted(mengen)
    if len(sortiert) > FOKUS_LIMIT:
        hinweise.append(
            f"Die Fokus-ISINs wurden von {len(sortiert)} auf das Limit von "
            f"{FOKUS_LIMIT} gekappt (sortiert)."
        )
        sortiert = sortiert[:FOKUS_LIMIT]
    return sortiert, hinweise


# ---------------------------------------------------------------------------
# Sweep: Tabs + Fokus → 1m-Bars (Parquet, quelle „psb“)
# ---------------------------------------------------------------------------


def _kurs(wert: Any) -> float | None:
    """API-Kurswert → float oder ``None`` (defekt/leer/nicht-finit = „fehlt“)."""
    if isinstance(wert, bool) or wert is None:
        return None
    if isinstance(wert, int | float):
        zahl = float(wert)
    elif isinstance(wert, str):
        try:
            zahl = float(wert.strip().replace(",", "."))
        except ValueError:
            return None
    else:
        return None
    return zahl if math.isfinite(zahl) and zahl > 0 else None


def _volumen(wert: Any) -> float:
    """Geliefertes (Gesamt-)Volumen → float, fehlt/defekt → 0.0 (E094-konsistent)."""
    zahl = _kurs(wert)
    return zahl if zahl is not None else 0.0


def _minute_us(jetzt: datetime) -> int:
    """Sweep-Zeit auf die Minute flooren → int64 µs UTC (E006-Kanonik)."""
    moment = jetzt.astimezone(UTC)
    minute = moment.replace(second=0, microsecond=0)
    return int((minute - datetime(1970, 1, 1, tzinfo=UTC)) / timedelta(microseconds=1))


def _bar_zeile(kurs: float, volumen: float, ts_us: int) -> dict[str, float | int]:
    """Close-only-Balken konsistent zum Byteworx-Bein (E094: o=h=l=c)."""
    return {
        "ts": ts_us,
        "open": kurs,
        "high": kurs,
        "low": kurs,
        "close": kurs,
        "volume": volumen,
    }


def _quelle_setzen(symbol: str, interval: str) -> None:
    """Setzt ``bar_meta.source`` auf :data:`QUELLE` (letzer Schreiber gewinnt).

    Bewusster SQL-Zeilen-Write statt Store-/Repo-Änderung (Muster
    ``downloads._bar_meta_source_setzen``): Der Recorder läuft nicht über den
    Download-Job-Manager, markiert seine Balken aber mit derselben groben
    Herkunftsmarke (E096: „quelle ``psb``“).
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE bar_meta SET source = ? WHERE symbol = ? AND interval = ?",
            (QUELLE, symbol, interval),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _schreibe_bars(bars: dict[str, list[dict[str, float | int]]]) -> int:
    """Upsertet gesammelte Balken je ISIN in den Parquet-Store; Balken-Zähler."""
    gesamt = 0
    for isin in sorted(bars):
        frame = pd.DataFrame(bars[isin])
        parquet_store.upsert_bars(isin, INTERVALL, frame)
        _quelle_setzen(isin, INTERVALL)
        gesamt += len(bars[isin])
    return gesamt


def sweep(client: Any, *, jetzt: datetime) -> dict[str, Any]:
    """Ein vollständiger Durchlauf: alle Gruppen + Fokus-Quotes (E096).

    ``tabs_lesen()`` ist der harte Kern — schlägt er fehl, wird der
    :class:`PlanspielFehler` weitergereicht (der Tick registriert Backoff).
    Einzelne Gruppen-/Fokus-Requests dürfen scheitern (Zähler ``fehler``,
    ``rate_limit`` setzt das ``gedrosselt``-Flag). Rückgabe (kompakt, für
    Log/Settings-Anzeige): ``sweeps`` (bediente Marktgruppen), ``instrumente``
    (geschriebene Kurs-Balken aus Tabs), ``fokussiert`` (geschriebene
    Fokus-Balken), ``fehler``, ``gedrosselt``, optional ``hinweise``. Ein
    erfolgreicher Sweep setzt die Backoff-Fehler-Serie zurück.
    """
    ts_us = _minute_us(jetzt)
    tabs = client.tabs_lesen()
    gruppen = list(
        dict.fromkeys(
            str(tab["key"]) for tab in tabs if isinstance(tab, dict) and tab.get("key")
        )
    )

    fehler = 0
    gedrosselt = False
    sweeps = 0
    gescheiterte_gruppen: list[str] = []
    bars: dict[str, list[dict[str, float | int]]] = {}
    for gruppe in gruppen:
        try:
            instrumente = client.markt_tab(gruppe)
        except PlanspielFehler as exc:
            fehler += 1
            gedrosselt = gedrosselt or exc.code == "rate_limit"
            gescheiterte_gruppen.append(gruppe)
            continue
        sweeps += 1
        for eintrag in instrumente:
            kurs = _kurs(eintrag.get("price"))
            if kurs is None:
                continue  # ohne Kurs kein Balken (Referenz Z. 140–141)
            isin = str(eintrag.get("isin") or "").strip().upper()
            if not isin:
                continue  # markt_tab filtert bereits; Doppelsicherung
            bars.setdefault(isin, []).append(
                _bar_zeile(kurs, _volumen(eintrag.get("volume")), ts_us)
            )
    instrumente_geschrieben = _schreibe_bars(bars)

    fokus_isin_liste, hinweise = fokus_isins(client)
    stamm = {instr.isin: instr for instr in psb_repo.liste()}
    fokus_bars: dict[str, list[dict[str, float | int]]] = {}
    for isin in fokus_isin_liste:
        instrument = stamm.get(isin)
        if instrument is None or not instrument.id_external:
            continue  # ohne idExternal kein Detail-Quote (Referenz Z. 187–189)
        try:
            detail = client.instrument_mit_kursen(isin, instrument.id_external)
        except PlanspielFehler as exc:
            fehler += 1
            gedrosselt = gedrosselt or exc.code == "rate_limit"
            continue
        quotes = detail.get("quoteList") if isinstance(detail, dict) else None
        quote = (
            quotes[0]
            if isinstance(quotes, list) and quotes and isinstance(quotes[0], dict)
            else None
        )
        if quote is None:
            continue
        kurs = _kurs(quote.get("price"))
        if kurs is None:
            bid, ask = _kurs(quote.get("bid")), _kurs(quote.get("ask"))
            if bid is not None and ask is not None:
                kurs = round((bid + ask) / 2, 4)  # Referenz Z. 200–201
            else:
                kurs = bid if bid is not None else ask
        if kurs is None:
            continue  # ohne jeden Kurs kein Balken (Referenz Z. 203)
        fokus_bars.setdefault(isin, []).append(
            _bar_zeile(kurs, _volumen(quote.get("totalVolume")), ts_us)
        )
    fokussiert = _schreibe_bars(fokus_bars)

    fehler_zuruecksetzen()
    if gescheiterte_gruppen:
        hinweise.append(
            "Marktgruppen fehlgeschlagen: " + ", ".join(gescheiterte_gruppen) + "."
        )
    ergebnis: dict[str, Any] = {
        "sweeps": sweeps,
        "instrumente": instrumente_geschrieben,
        "fokussiert": fokussiert,
        "fehler": fehler,
        "gedrosselt": gedrosselt,
    }
    if hinweise:
        ergebnis["hinweise"] = hinweise
    return ergebnis
