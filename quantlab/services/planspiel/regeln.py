"""PSB-Regelmodul (Welle H, E100) — **Single Source of Truth** der Spielregeln
der Planspiel-Börse 2026/27 (RECHERCHE H.6, Live-Abgleich getFeeModel/
getTradingHours).

Konsumenten:

- ``agent/tools/planspiel.py`` — ``platziere_order``-Prechain
  (:func:`pruefe_order_vorab`) und die statische Kurzreferenz
  (``_REGELN_KURZREFERENZ`` baut ihre Zahlen aus den Konstanten hier).
- Backtest-Kostenmodell ``gebuehren_modell='psb'`` (:func:`psb_kostenmodell`).
- Wettbewerb-Dashboard (:mod:`quantlab.services.planspiel.wettbewerb`) —
  20 %-Gewichtswarnung und Aktivitätszähler gegen :data:`MIN_KAEUFE`.

Modelliert werden die harten Spielregeln (0,3 % min 15 € Gebühr, 20 %-Depot-
gewicht, 5 Orders je Tag und Wertpapier, Penny-Sperre < 1 €, 3-Käufe-
Mindestaktivität, Handelszeiten 08:00–22:00 Europe/Berlin). Die Börse erzwingt
Gewichts-/Orderlimits serverseitig — :func:`pruefe_order_vorab` prüft **vorab
ehrlich** und liefert strukturierte Hinweise, erzwingt aber nichts doppelt:
der bestehende Kapital-Check (``order/check/capital`` + ``getFeeModel``) bleibt
in der Werkzeugkette unangetastet (E070/E074).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

#: Handelszeit-Gebühr: 0,3 % vom Kurswert (Aktien/ETFs/Fonds/Anleihen).
GEBUEHR_PROZENT = 0.3
#: Mindestgebühr je Order in Euro.
MIN_GEBUEHR_EUR = 15.0
#: Krypto-Gebühr (nur Trainingsdepot): 0,35 %.
KRYPTO_PROZENT = 0.35
#: Krypto-Minimum: 0,01 €.
KRYPTO_MIN_EUR = 0.01
#: Order-Storno-Gebühr: 2,50 €.
STORNO_EUR = 2.50
#: Max. Depotgewicht je Wertpapier (Anteil am Depotgesamtwert): 20 %.
MAX_DEPOTGEWICHT = 0.20
#: Max. Orders je Tag und Wertpapier.
MAX_ORDERS_JE_TAG = 5
#: Penny-Sperre: Kurse unter 1,00 € sind nicht kaufbar.
PENNY_GRENZE_EUR = 1.0
#: Mindestaktivität: abgerechnete Käufe je Saisonmonat (Task-Konvention H7),
#: sonst Ausschluss wegen Passivität (saisonweise bis 22.01.2027).
MIN_KAEUFE = 3
#: Handelszeiten (Ausführung nur in diesem Fenster; Orders außerhalb bleiben
#: platziert und werden zur Marktöffnung ausgeführt).
HANDELSZEIT_START_STUNDE = 8
HANDELSZEIT_ENDE_STUNDE = 22
HANDELSZEIT_ZONE = "Europe/Berlin"

#: Warnschwelle „nah dran“ am Depotgewichts-Limit (ab 90 % des Limits, d. h.
#: ab 18 % Depotgewicht) — unterhalb der harten 20-%-Grenze.
_NAHE_DRAN_FAKTOR = 0.9

#: Toleranz für Float-Grenzfälle (20,0 % exakt bleibt erlaubt).
_EPSILON = 1e-9

#: ISIN-Form (12 Zeichen: 2 Länderbuchstaben + 9 alphanumerisch + 1 Ziffer) —
#: bewusst hier statt Import aus dem Agent-Layer (Service bleibt agent-frei).
_ISIN_MUSTER = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")

#: Order-Status „abgerechnet“ (RECHERCHE P.3, verifiziert) — derselbe Wert wie
#: in ``agent/tools/planspiel.py``/``api/routers/planspiel.py``.
STATUS_AUSGEFUEHRT = "EXECUTED"

_STATUS_AUSGEFUEHRT = STATUS_AUSGEFUEHRT


def ist_isin_form(text: str) -> bool:
    """True, wenn ``text`` der ISIN-Form entspricht (12 Zeichen, Prüfziffer)."""
    return bool(_ISIN_MUSTER.fullmatch(text))


def gebuehr(orderwert: float, *, krypto: bool = False) -> float:
    """PSB-Gebühr für einen Kurswert: max(Prozentsatz × Orderwert, Minimum).

    Standard 0,3 % min 15 €; Krypto (nur Trainingsdepot) 0,35 % min 0,01 €.
    Negativer/kein Orderwert → 0,0 (keine Order — keine Gebühr).
    """
    wert = float(orderwert)
    if wert <= 0.0:
        return 0.0
    prozent, minimum = (
        (KRYPTO_PROZENT, KRYPTO_MIN_EUR) if krypto else (GEBUEHR_PROZENT, MIN_GEBUEHR_EUR)
    )
    return max(wert * prozent / 100.0, minimum)


def psb_kostenmodell():
    """CostModel des Backtests im PSB-Modus (``gebuehren_modell='psb'``).

    0,3 % je Order (30 bps), kein Slippage-Anteil, Mindestgebühr 15 € je Fill
    (:data:`MIN_GEBUEHR_EUR`). Die Krypto-Sondergebühr ist im Backtest bewusst
    NICHT modelliert — der Instrumententyp ist dort nicht erkennbar (ANNAHME:
    Aktien/ETFs, der dominante Wettbewerbs-Fall).
    """
    # Lazy (Modul-Import-Kette: services → backtest ist zulässig und zyklusfrei,
    # aber der Backtest-Import gehört an den Nutzungsort).
    from quantlab.backtest.costs import CostModel

    return CostModel(
        commission_bps=GEBUEHR_PROZENT * 100.0,
        slippage_bps=0.0,
        min_fee_eur=MIN_GEBUEHR_EUR,
    )


# ---------------------------------------------------------------------------
# Vorab-Check (E100): ehrliche Prechain-Hinweise, best-effort ohne Kippen
# ---------------------------------------------------------------------------


def _zahl_de(wert: float, stellen: int = 2) -> str:
    """Deutsche Zahlenschreibweise („0,80“, „20,1“) für Regel-Meldungen."""
    return f"{wert:.{stellen}f}".replace(".", ",")


def _hinweis(code: str, schwere: str, meldung: str) -> dict[str, str]:
    """Ein strukturiertes Regel-Hinweis-Objekt (Vertrag: code/schwere/meldung)."""
    return {"code": code, "schwere": schwere, "meldung": meldung}


def _berlin_datum(iso: str | None, jetzt: datetime) -> Any | None:
    """Datum (Europe/Berlin) eines ISO-Zeitstempels — ``None`` bei Unlesbarkeit."""
    if not isinstance(iso, str) or not iso.strip():
        return None
    try:
        moment = datetime.fromisoformat(iso.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(ZoneInfo(HANDELSZEIT_ZONE)).date()


def _ordner_datum(order: dict[str, Any], jetzt: datetime) -> Any | None:
    """Ausführungsdatum einer Order (Berlin-Tag) — ``executionDate`` vor
    ``creationDate`` (ANNAHME H7: Feldname nach RECHERCHE P.3-Antwortform;
    ohne Ausführungsdatum gilt das Erstellungsdatum als bester Schätzer)."""
    datum = _berlin_datum(order.get("executionDate"), jetzt)
    if datum is None:
        datum = _berlin_datum(order.get("creationDate"), jetzt)
    return datum


def _positionswert_je_isin(items: Any) -> dict[str, float]:
    """``items`` der getPortfolioWithItems-Antwort → ``{isin: currentValue}``."""
    werte: dict[str, float] = {}
    if not isinstance(items, list):
        return werte
    for item in items:
        if not isinstance(item, dict):
            continue
        instrument = item.get("instrument") or {}
        isin = instrument.get("isin") if isinstance(instrument, dict) else None
        wert = item.get("currentValue")
        if isinstance(isin, str) and isinstance(wert, int | float) and not isinstance(wert, bool):
            werte[isin] = werte.get(isin, 0.0) + float(wert)
    return werte


def abgerechnete_orders_heute(orders: Any, isin: str, jetzt: datetime) -> int:
    """Abgerechnete (EXECUTED) Orders einer ISIN am heutigen Berlin-Tag.

    ``orders`` ist die ``getOrders status=EXECUTED``-Antwort (Liste von Dicts);
    unlesbare Einträge/Daten zählen bewusst nicht (best-effort-Zähler).
    """
    heutiger_tag = jetzt.astimezone(ZoneInfo(HANDELSZEIT_ZONE)).date()
    heute = 0
    if not isinstance(orders, list):
        return 0
    for order in orders:
        if not isinstance(order, dict) or order.get("orderStatus") != _STATUS_AUSGEFUEHRT:
            continue
        datum = _ordner_datum(order, jetzt)
        if datum is not None and datum == heutiger_tag and str(order.get("isin") or "") == isin:
            heute += 1
    return heute


def abgerechnete_kaeufe(orders: Any, jetzt: datetime) -> int:
    """Abgerechnete Käufe (EXECUTED + BUY) im Saisonmonat des ``jetzt``.

    ANNAHME (Task-Konvention H7): „Saisonmonat“ = laufender Kalendermonat
    (Europe/Berlin) — Monats-Fortschrittsindikator der Mindestaktivität
    (:data:`MIN_KAEUFE`); die saisonale Bewertungsregel der Börse zählt die
    ganze Saison (bis 22.01.2027). Unlesbare Einträge zählen nicht.
    """
    if not isinstance(orders, list):
        return 0
    heutiger_tag = jetzt.astimezone(ZoneInfo(HANDELSZEIT_ZONE)).date()
    kaeufe = 0
    for order in orders:
        if not isinstance(order, dict) or order.get("orderStatus") != _STATUS_AUSGEFUEHRT:
            continue
        datum = _ordner_datum(order, jetzt)
        if (
            datum is not None
            and datum.year == heutiger_tag.year
            and datum.month == heutiger_tag.month
            and str(order.get("orderType") or "") == "BUY"
        ):
            kaeufe += 1
    return kaeufe


def pruefe_order_vorab(
    client: Any,
    *,
    depot_id: int,
    seite: str,
    isin: str,
    menge: int,
    preis: float | None,
    jetzt: datetime | None = None,
) -> list[dict[str, str]]:
    """Ehrlicher Vorab-Check der PSB-Spielregeln (E100) — Liste von Hinweisen.

    Beschafft die Basis **best-effort selbst** über den (Fake-fähigen) Client:
    Depotpositionen + Depotwert (20 %-Gewicht), ``getOrders status=EXECUTED``
    (Orders je Tag/WP, Käufe im Saisonmonat). Checks ohne ermittelbare Basis
    werden übersprungen — mit Warn-Hinweis, wenn die Client-Methoden existieren
    aber keine brauchbaren Daten liefern; fehlende Methoden (schlanker Fake/
    älterer Vertrag) werden still übersprungen, damit bestehende Verträge
    unberührt bleiben. Exceptions der Lesezugriffe kippen die Order nie —
    die Börse erzwingt die Limits ohnehin serverseitig.

    Rückgabe: Liste von ``{code, schwere: "hart"|"warnung", meldung}``.
    Harte Einträge sind Verweigerungsgründe (Kettenaufrufer entscheidet).
    """
    moment = jetzt or datetime.now(UTC)
    hinweise: list[dict[str, str]] = []

    # --- Depotbasis (Positionswert je ISIN + Depotgesamtwert) --------------
    positions_werte: dict[str, float] | None = None
    depotwert: float | None = None
    depot_basis_ok = False
    if callable(getattr(client, "positionen_holen", None)) and callable(
        getattr(client, "portfolio_holen", None)
    ):
        try:
            positions_werte = _positionswert_je_isin(
                client.positionen_holen(depot_id).get("items")
            )
            roh_wert = client.portfolio_holen(depot_id).get("value")
            depotwert = (
                float(roh_wert)
                if isinstance(roh_wert, int | float) and not isinstance(roh_wert, bool)
                else None
            )
            depot_basis_ok = True
        except Exception:  # noqa: BLE001 — best-effort: Depot-Lesestörung kippt nie
            positions_werte = None
            depotwert = None
            depot_basis_ok = False

    # --- Orders (EXECUTED): Zähler je Tag/Monat ----------------------------
    orders_heute: int | None = None
    kaeufe_monat: int | None = None
    if callable(getattr(client, "orders_holen", None)):
        try:
            roh = client.orders_holen(depot_id, _STATUS_AUSGEFUEHRT)
        except Exception:  # noqa: BLE001 — best-effort: Lese-Störung kippt nie
            roh = None
        if isinstance(roh, list):
            orders_heute = abgerechnete_orders_heute(roh, isin, moment)
            kaeufe_monat = abgerechnete_kaeufe(roh, moment)

    # --- Penny-Sperre (nur Kauf mit erkennbarem Preis) ---------------------
    if seite == "kaufen" and preis is not None and preis < PENNY_GRENZE_EUR:
        hinweise.append(
            _hinweis(
                "penny_kauf",
                "hart",
                f"Penny-Sperre: {isin} kostet {_zahl_de(preis)} € und liegt damit "
                f"unter der Kaufgrenze von {_zahl_de(PENNY_GRENZE_EUR)} € — der Kauf "
                "ist an der Planspiel-Börse nicht möglich (Verkauf wäre erlaubt).",
            )
        )

    # --- 20 %-Depotgewicht (nur Kauf, nur mit erkennbarem Orderwert) -------
    if seite == "kaufen" and preis is not None:
        if depot_basis_ok and depotwert is not None and depotwert > 0.0:
            bestand = positions_werte.get(isin, 0.0) if positions_werte is not None else 0.0
            orderwert = menge * float(preis)
            summe = bestand + orderwert
            anteil = summe / depotwert
            if anteil > MAX_DEPOTGEWICHT + _EPSILON:
                hinweise.append(
                    _hinweis(
                        "depotgewicht",
                        "hart",
                        f"20-%-Limit verletzt: {isin} läge mit Bestand "
                        f"({_zahl_de(bestand)} €) + Orderwert ({_zahl_de(orderwert)} €) "
                        f"bei {_zahl_de(anteil * 100, 1)} % des Depotwerts "
                        f"({_zahl_de(depotwert)} €) — erlaubt sind höchstens "
                        f"{int(MAX_DEPOTGEWICHT * 100)} % je Wertpapier. "
                        "Bitte Menge reduzieren.",
                    )
                )
            elif anteil > _NAHE_DRAN_FAKTOR * MAX_DEPOTGEWICHT - _EPSILON:
                hinweise.append(
                    _hinweis(
                        "depotgewicht_nahe",
                        "warnung",
                        f"Nahe am 20-%-Limit: {isin} läge bei "
                        f"{_zahl_de(anteil * 100, 1)} % des Depotwerts "
                        f"(Limit {int(MAX_DEPOTGEWICHT * 100)} %, "
                        "inkl. offener Käufe gerechnet) — wenig Spielraum nach oben.",
                    )
                )
        elif depot_basis_ok:
            hinweise.append(
                _hinweis(
                    "depotgewicht_unbekannt",
                    "warnung",
                    "Der Depotwert ließ sich nicht ermitteln — der Vorab-Check des "
                    f"20-%-Limits für {isin} wurde übersprungen (die Börse prüft "
                    "serverseitig weiter).",
                )
            )

    # --- Max. 5 Orders je Tag und Wertpapier -------------------------------
    if orders_heute is not None:
        if orders_heute >= MAX_ORDERS_JE_TAG:
            hinweise.append(
                _hinweis(
                    "orders_pro_tag",
                    "hart",
                    f"Order-Limit verletzt: {isin} hat heute bereits "
                    f"{orders_heute} abgerechnete Orders — erlaubt sind höchstens "
                    f"{MAX_ORDERS_JE_TAG} je Tag und Wertpapier. Bitte erst am "
                    "nächsten Handelstag wieder bestellen.",
                )
            )
    elif callable(getattr(client, "orders_holen", None)):
        hinweise.append(
            _hinweis(
                "orders_pro_tag_unbekannt",
                "warnung",
                "Die heutigen Orders ließen sich nicht ermitteln — der Vorab-Check "
                f"des 5-Orders-Limits für {isin} wurde übersprungen (die Börse "
                "zählt serverseitig weiter).",
            )
        )

    # --- Mindestaktivität (Warnung, best-effort, nur bei Käufen) -----------
    if seite == "kaufen" and kaeufe_monat is not None and kaeufe_monat < MIN_KAEUFE:
        hinweise.append(
            _hinweis(
                "mindestaktivitaet",
                "warnung",
                f"Mindestaktivität: erst {kaeufe_monat} abgerechnete Käufe in diesem "
                f"Saisonmonat — die Börse verlangt mindestens {MIN_KAEUFE} "
                "(sonst Ausschluss wegen Passivität).",
            )
        )

    return hinweise
