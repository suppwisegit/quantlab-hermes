"""Planspiel-Börse-Tools (Welle P, E073/E074; H1: + planspiel_universum;
H3: + planspiel_kennzahlen).

Konsumiert ausschließlich :mod:`quantlab.services.planspiel` (E068-Client mit
Whitelist-Guard und Pacing) — keine Router-Imports, kein eigenes httpx. Die
Handler laufen über ``asyncio.to_thread`` (Muster portfolio.py, der Client ist
sync); :class:`PlanspielFehler` (E069) wird auf :class:`ToolFehler` gemappt —
fachliche Fehler sind ``ok:false``-Tool-Ergebnisse, der Run läuft weiter (§E).

**Depot-Fallback (E070):** Lesen funktioniert ohne Rollen-Markierung — ohne
``portfolio_id``-Argument gilt ``planspiel.standard_depot_id`` aus den
Einstellungen; ist dort ``0`` (nicht markiert), wird das erste Depot aus
``getUser`` genutzt.

``platziere_order`` (P2b, E074) ist der einzige **Schreib**-Pfad — Guard-first
vor jedem Senden: E070-Depot-Regeln → Preis-Kohärenz → Gültigkeit → Pre-Checks
(``kapital_check`` + ``gebuehren_modell``) → PSB-Regel-Vorabcheck (H7/E100,
``regeln.pruefe_order_vorab``; harte Verstöße verweigern, Warnungen landen im
Feld ``regelhinweise``) → ``order_platzieren``. Der
E066-Marker in der Order-Notiz wird immer handlerseitig gebaut (nie vom LLM
übernommen); die Antwort trennt „platziert (wartet auf Ausführung)“ von
„ausgeführt“.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, pruefe_args, schema_von
from quantlab.agent.tools.daten import loese_isin_zu_kuerzel
from quantlab.services.planspiel import (
    NOTIZ_MAX_ZEICHEN,
    PlanspielClient,
    PlanspielFehler,
    anmeldeinfos_aus_settings,
    hole_client,
)
from quantlab.services.planspiel import regeln as regeln_service
from quantlab.services.planspiel import vwd as vwd_service
from quantlab.services.planspiel.universum import importiere_universum
from quantlab.storage import parquet_store
from quantlab.storage.repos import psb_repo, settings_repo

#: Order-Status „abgerechnet“ (RECHERCHE P.3, [LIVE] verifiziert). Alle anderen
#: Status — auch unbekannte (ANNAHME P.7.7, Enum nur teilweise verifiziert) —
#: gelten tolerant als „offen“ (PLAN §P2a-Risiko).
_STATUS_AUSGEFUEHRT = "EXECUTED"

#: Abgelaufene Orders (zweiter [LIVE]-verifizierter Filter-Wert, P4:
#: getOrders akzeptiert EXECUTED + EXPIRED; PLACED ist nur ein Feldwert
#: der placeOrder-Antwort, KEIN gültiger Abfragefilter — HTTP 400).
_STATUS_ABGELAUFEN = "EXPIRED"

#: P4-Live-Befund (2026-09-21): ``getOrders`` verlangt ``status`` zwingend
#: (ohne → HTTP 400); die Tool-Gruppierung fragt daher je verifiziertem
#: Enum-Wert einzeln ab statt ohne Filter.

#: Statisches Label der Kurzreferenz (E073: „Stand 2026/27“-Pflicht).
_REGELN_STAND = (
    "Stand: Saison 2026/27 (Quelle: planspiel-boerse.de/regeln.html, "
    "Live-Abgleich RECHERCHE P.6)"
)

def _de_zahl(wert: float, stellen: int) -> str:
    """Deutsche Zahlenschreibweise („0,3“, „15“, „2,50“) aus dem Regelmodul."""
    return f"{wert:.{stellen}f}".replace(".", ",")


#: STATISCHE deutsche Kurzreferenz der Spielmechaniken (E073, Quelle RECHERCHE
#: P.6). Die Zahlen kommen seit H7/E100 aus dem Regelmodul
#: (:mod:`quantlab.services.planspiel.regeln`, Single Source of Truth) — der
#: Text selbst ändert sich nur zur Saisonzäsur.
_REGELN_KURZREFERENZ = (
    "Spielmechaniken der Planspiel-Börse 2026/27 (verbindlich nachlesen: "
    "planspiel-boerse.de/regeln):\n"
    "- Startkapital: 50.000 € virtuell je Depot; zwei Depots (nur das "
    "Wettbewerbsdepot ist gewertet, das Trainingsdepot dient zum Üben).\n"
    f"- Gebühren: {_de_zahl(regeln_service.GEBUEHR_PROZENT, 1)} % vom Kurswert, "
    f"mindestens {_de_zahl(regeln_service.MIN_GEBUEHR_EUR, 0)} € (Aktien/ETFs/Fonds/"
    "Anleihen); "
    f"Krypto {_de_zahl(regeln_service.KRYPTO_PROZENT, 2)} %, min. "
    f"{_de_zahl(regeln_service.KRYPTO_MIN_EUR, 2)} € (nur Trainingsdepot); "
    f"Order-Storno {_de_zahl(regeln_service.STORNO_EUR, 2)} €; "
    "Verfall eines Limits/Stops gebührenfrei.\n"
    "- Handelsplätze: Börse Stuttgart, Wiener Börse (ATX), Börse Luxemburg. "
    f"Handelszeiten {regeln_service.HANDELSZEIT_START_STUNDE:02d}:00–"
    f"{regeln_service.HANDELSZEIT_ENDE_STUNDE:02d}:00 — Ausführung nur in diesem "
    "Fenster; Orders "
    "außerhalb bleiben platziert und werden zur Marktöffnung ausgeführt; "
    "keine Teilausführungen; Feiertage/Wochenende still.\n"
    "- Ordertypen: BEST (unlimitiert), LIMIT (Laufzeit 1 Börsentag, optional "
    "bis 14 Kalendertage), STOP (Stop-Loss/-Buy, max. 14 Kalendertage).\n"
    "- Depot-Regeln: max. "
    f"{round(regeln_service.MAX_DEPOTGEWICHT * 100)} % des Depotgesamtwerts je "
    "Wertpapier (inkl. "
    f"offener Käufe); max. {regeln_service.MAX_ORDERS_JE_TAG} Orders je Tag und "
    "Wertpapier; Penny-Stocks "
    f"(Kurs < {_de_zahl(regeln_service.PENNY_GRENZE_EUR, 2)} €) sind nicht kaufbar "
    "(Verkauf möglich); keine Verzinsung, "
    "keine Überziehung.\n"
    "- Laufzeit 2026/27: erster Handelstag 01.10.2026, Spielende 25.01.2027 "
    "(Bewertung zu Schluss-/Geldkursen, keine Verkaufspflicht).\n"
    f"- Mindestaktivität: mind. {regeln_service.MIN_KAEUFE} abgerechnete "
    "Kaufaufträge bis 22.01.2027, "
    "sonst Ausschluss wegen Passivität.\n"
    "- Nur im Trainingsdepot handelbar: Kryptowährungen und Index-Zertifikate."
)


# ---------------------------------------------------------------------------
# Gemeinsame Helfer (Fehler-Mapping, Depot-Fallback, Settings-Zugriff)
# ---------------------------------------------------------------------------


def _tool_fehler_aus(exc: PlanspielFehler) -> ToolFehler:
    """E069-Fachfehler → deutscher ``ok:false``-Tool-Text (Run läuft weiter)."""
    meldung = f"Planspiel-Börse ({exc.code}): {exc.message}"
    if exc.hint:
        meldung = f"{meldung} [{exc.hint}]"
    return ToolFehler(meldung)


async def _im_thread(_ausfuehren: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Sync-Client-Aufruf im Thread (Muster portfolio.py) + Fehler-Mapping."""
    try:
        return await asyncio.to_thread(_ausfuehren)
    except PlanspielFehler as exc:
        raise _tool_fehler_aus(exc) from exc


def _planspiel_sektion() -> dict[str, Any]:
    """Settings-Sektion ``planspiel`` — leeres Dict, falls der Bestand sie fehlt."""
    try:
        return settings_repo.get_section("planspiel")
    except ValueError:  # alter Bestand ohne Migration der Defaults (P1-Muster)
        return {}


def _loese_depot_id(client: PlanspielClient, portfolio_id: int | None) -> int:
    """E070-Lesefallback: Argument → ``standard_depot_id`` → erstes Depot."""
    if portfolio_id is not None:
        return portfolio_id
    standard = int(_planspiel_sektion().get("standard_depot_id") or 0)
    if standard:
        return standard
    for depot in client.user_holen().get("userPortfolios") or []:
        if isinstance(depot, dict) and depot.get("id") is not None:
            return int(depot["id"])
    raise ToolFehler(
        "Kein Planspiel-Depot gefunden — bitte die Verbindung in den "
        "Einstellungen testen (Einstellungen → Planspiel)."
    )


def _erster_wert(*kandidaten: Any) -> Any:
    """Erster nicht-``None``-Wert (tolerantes Mapping optionaler API-Felder)."""
    for kandidat in kandidaten:
        if kandidat is not None:
            return kandidat
    return None


def _yahoo_kuerzel(isin: Any) -> str | None:
    """Best effort (E091/T-G5): ISIN → Yahoo-Kürzel-Vorschlag (kein Netz).

    Löst lokal über die Registry (``symbols.isin``) und die Seed-Map des
    Daten-Tool-Layers auf — ``None`` wenn unaufgelöst. Das Feld ist additiv;
    fehlende Auflösung ist kein Fehler (E073-Vertrag: additive Keys).
    """
    if not isinstance(isin, str):
        return None
    return loese_isin_zu_kuerzel(isin)


# ---------------------------------------------------------------------------
# planspiel_status
# ---------------------------------------------------------------------------


class PlanspielStatusArgs(BaseModel):
    """Argumente von ``planspiel_status`` (bewusst leer — keine Parameter)."""


async def planspiel_status(_roh_args: dict[str, Any]) -> dict[str, Any]:
    """Konfiguration, Login-Status und Contest-/Depot-/Team-Übersicht (getUser)."""

    def _lesen() -> dict[str, Any]:
        client = hole_client()
        sektion = _planspiel_sektion()
        user = client.user_holen()
        contest = user.get("contest") or {}
        team = user.get("teamInfo") or {}
        depots = [
            {
                "id": depot.get("id"),
                "name": depot.get("depotName"),
                "trades": depot.get("numOfTrades"),
            }
            for depot in user.get("userPortfolios") or []
            if isinstance(depot, dict)
        ]
        return {
            "konfiguriert": anmeldeinfos_aus_settings() is not None,
            "eingeloggt": client.eingeloggt,
            "email": sektion.get("email") or "",
            "standard_depot_id": sektion.get("standard_depot_id", 0),
            "wettbewerbs_depot_id": sektion.get("wettbewerbs_depot_id", 0),
            "contest": {
                "name": contest.get("publicName"),
                "phase": contest.get("contestPhase"),
                "start": contest.get("startDate"),
                "ende": contest.get("endDate"),
                "startkapital": contest.get("credit"),
            },
            "depots": depots,
            "team": (
                {"name": team.get("name")}
                if isinstance(team, dict) and team
                else None
            ),
            "hinweis": "Depot-Kennzahlen via planspiel_depot, Spielregeln via "
            "planspiel_regeln.",
        }

    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# planspiel_suche
# ---------------------------------------------------------------------------


class PlanspielSucheArgs(BaseModel):
    """Argumente von ``planspiel_suche``."""

    name: str = Field(
        min_length=1,
        max_length=100,
        description='Suchbegriff (Instrumenten-Name), z. B. "SAP" oder "DAX".',
    )
    seite: int = Field(
        default=0,
        ge=0,
        description="Seite der Trefferliste (0-basiert), Default 0.",
    )
    groesse: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Treffer pro Seite (max. 10 — Kontext-Schonung), Default 5.",
    )


def _such_zeile(treffer: dict[str, Any]) -> dict[str, Any]:
    """Suchtreffer → kompakte deutsch-flache Zeile (E073-Feldsatz)."""
    return {
        "id_external": treffer.get("idExternal"),
        "id_exchange": treffer.get("exchangeExternalId"),
        "isin": treffer.get("isin"),
        "wkn": treffer.get("wkn"),
        "name": treffer.get("name"),
        "preis": treffer.get("price"),
        "bid": treffer.get("bid"),
        "ask": treffer.get("ask"),
        "exchange": treffer.get("exchange"),
        "typ": treffer.get("instrumentType"),
        # E091/T-G5 (additiv): Yahoo-Kürzel-Vorschlag für den Datendownload;
        # ``None`` wenn die ISIN lokal nicht auflösbar ist (kein Netz-Zwang).
        "yahoo_kuerzel": _yahoo_kuerzel(treffer.get("isin")),
    }


async def planspiel_suche(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Handelbare Wertpapiere der Planspiel-Börse nach Name suchen."""

    def _lesen() -> dict[str, Any]:
        treffer = hole_client().instrument_suchen(
            args.name, seite=args.seite, groesse=args.groesse
        )
        ergebnisse = [_such_zeile(t) for t in treffer if isinstance(t, dict)]
        return {
            "suchbegriff": args.name,
            "seite": args.seite,
            "anzahl": len(ergebnisse),
            "ergebnisse": ergebnisse,
            "hinweis": "Details/Kurse je Treffer via planspiel_instrument "
            "(isin + id_external).",
        }

    args = pruefe_args(PlanspielSucheArgs, roh_args, "planspiel_suche")
    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# planspiel_instrument
# ---------------------------------------------------------------------------


class PlanspielInstrumentArgs(BaseModel):
    """Argumente von ``planspiel_instrument``."""

    isin: str = Field(
        min_length=12,
        max_length=12,
        description='ISIN des Instruments (12 Zeichen), z. B. "DE0007164600".',
    )
    # str | int: die API-Beispiele zeigen idExternal teils als Zahl — beides
    # wird als Query-Parameter transparent durchgereicht (Client: str | int).
    id_external: str | int = Field(
        description="idExternal aus planspiel_suche (Börsen-Referenz des Instruments).",
    )


def _kurs_zeile(quote: dict[str, Any]) -> dict[str, Any]:
    """Ein Kurs der ``quoteList`` → kompakte deutsch-flache Zeile."""
    return {
        "preis": quote.get("price"),
        "hoch": quote.get("high"),
        "tief": quote.get("low"),
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
        "datum": quote.get("priceDate"),
        "volumen": quote.get("totalVolume"),
        "veraenderung_abs": quote.get("diffAbs"),
        "veraenderung_rel": quote.get("diffRel"),
        "waehrung": quote.get("isoCurrency"),
    }


async def planspiel_instrument(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Instrument-Details und aktuelle Kurse (bid/ask) je ISIN holen."""

    def _lesen() -> dict[str, Any]:
        daten = hole_client().instrument_mit_kursen(args.isin, args.id_external)
        return {
            "isin": daten.get("isin"),
            "name": daten.get("name"),
            "wkn": daten.get("wkn"),
            "typ": daten.get("instrumentType"),
            "exchange": daten.get("exchange"),
            "id_exchange": daten.get("exchangeExternalId"),
            # E091/T-G5 (additiv): Yahoo-Kürzel-Vorschlag (best effort, lokal).
            "yahoo_kuerzel": _yahoo_kuerzel(daten.get("isin")),
            "kurse": [
                _kurs_zeile(q) for q in daten.get("quoteList") or [] if isinstance(q, dict)
            ],
            "hinweis": "bid/ask ist die Abrechnungs-Basis des Spiels; Handelsfenster "
            "via planspiel_regeln.",
        }

    args = pruefe_args(PlanspielInstrumentArgs, roh_args, "planspiel_instrument")
    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# planspiel_depot
# ---------------------------------------------------------------------------


class PlanspielDepotArgs(BaseModel):
    """Argumente von ``planspiel_depot``."""

    portfolio_id: int | None = Field(
        default=None,
        ge=1,
        description="Planspiel-Depot-ID (Default: standard_depot_id aus den "
        "Einstellungen, sonst das erste Depot aus getUser).",
    )


def _position_zeile(item: dict[str, Any]) -> dict[str, Any]:
    """Positions-Item der ``getPortfolioWithItems``-Antwort → flache Zeile."""
    instrument = item.get("instrument") or {}
    return {
        "isin": instrument.get("isin"),
        "wkn": instrument.get("wkn"),
        "name": instrument.get("name"),
        "menge": item.get("quantity"),
        "wert": item.get("currentValue"),
        "performance_rel": _erster_wert(
            item.get("performanceRelTotal"), item.get("performanceRel")
        ),
        "bid": instrument.get("bid"),
        "ask": instrument.get("ask"),
        "typ": instrument.get("instrumentType"),
    }


async def planspiel_depot(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Depot-Kennzahlen (Wert/Barbestand/Performance) + Positionen lesen."""

    def _lesen() -> dict[str, Any]:
        client = hole_client()
        depot_id = _loese_depot_id(client, args.portfolio_id)
        kennzahlen = client.portfolio_holen(depot_id)
        mit_positionen = client.positionen_holen(depot_id)
        positionen = [
            _position_zeile(item)
            for item in mit_positionen.get("items") or []
            if isinstance(item, dict)
        ]
        return {
            "portfolio_id": depot_id,
            "wert": kennzahlen.get("value"),
            "barbestand": kennzahlen.get("credit"),
            "verfuegbar": kennzahlen.get("availableCredit"),
            "instrumentswert": kennzahlen.get("valueOfInstruments"),
            "performance_abs": kennzahlen.get("performanceAbs"),
            "performance_rel": kennzahlen.get("performanceRel"),
            "anzahl_trades": kennzahlen.get("numTrades"),
            "anzahl_positionen": len(positionen),
            "positionen": positionen,
            "hinweis": "Barbestand = Guthaben; Orders/Trade-Historie via "
            "planspiel_orders.",
        }

    args = pruefe_args(PlanspielDepotArgs, roh_args, "planspiel_depot")
    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# planspiel_orders
# ---------------------------------------------------------------------------


class PlanspielOrdersArgs(BaseModel):
    """Argumente von ``planspiel_orders``."""

    portfolio_id: int | None = Field(
        default=None,
        ge=1,
        description="Planspiel-Depot-ID (Default: standard_depot_id aus den "
        "Einstellungen, sonst das erste Depot aus getUser).",
    )
    status: Literal["alle", "offen", "ausgefuehrt"] = Field(
        default="alle",
        description='Filter: "alle" (Default), "offen" (platziert, wartet) oder '
        '"ausgefuehrt" (abgerechnete Trades).',
    )


def _order_zeile(order: dict[str, Any]) -> dict[str, Any]:
    """Order-Objekt der ``getOrders``-Antwort → deutsch-flache Zeile (E072-Satz)."""
    instrument = order.get("instrument") or {}
    return {
        "id": order.get("id"),
        "typ": order.get("orderType"),
        "ordertyp": order.get("orderSubType"),
        "menge": order.get("quantity"),
        "instrument_name": instrument.get("name"),
        "isin": instrument.get("isin"),
        "limit": order.get("priceLimit"),
        "stop": order.get("stopLimit"),
        "status": order.get("orderStatus"),
        "erstellt": order.get("creationDate"),
        "gueltig_bis": order.get("expirationDate"),
        "ausfuehrungspreis": order.get("executionPrice"),
        "gebuehren": order.get("transactionCosts"),
        "fehler_code": order.get("validationErrorCode") or None,
    }


def _notizen_nach_referenz(client: PlanspielClient) -> dict[str, str]:
    """``note/list``-Notizen je Order-ID (``reference``, E066-Verifikation).

    Ein Notiz-Fehler kippt bewusst NICHT die Order-Liste — Notizen sind
    Zusatzinformation; eine Kernantwort (Orders) bleibt davon unabhängig.
    """
    try:
        notizen = client.notizen_lesen()
    except PlanspielFehler:
        return {}
    zuordnung: dict[str, str] = {}
    for notiz in notizen:
        if isinstance(notiz, dict) and notiz.get("reference") is not None:
            zuordnung[str(notiz["reference"])] = str(notiz.get("text") or "")
    return zuordnung


async def planspiel_orders(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Orders eines Depots lesen — offen (platziert) vs. ausgeführt getrennt."""

    def _lesen() -> dict[str, Any]:
        client = hole_client()
        depot_id = _loese_depot_id(client, args.portfolio_id)
        if args.status == "ausgefuehrt":
            roh = client.orders_holen(depot_id, _STATUS_AUSGEFUEHRT)
        elif args.status == "offen":
            # P4-Live: ohne status-Parameter gibt es keinen Abruf; „offen“
            # fragt EXPIRED ab (einziger weiterer gültiger Filter-Wert).
            roh = client.orders_holen(depot_id, _STATUS_ABGELAUFEN)
        else:
            roh = client.orders_holen(
                depot_id, _STATUS_AUSGEFUEHRT
            ) + client.orders_holen(depot_id, _STATUS_ABGELAUFEN)
        notizen = {} if args.status == "offen" else _notizen_nach_referenz(client)
        offen: list[dict[str, Any]] = []
        ausgefuehrt: list[dict[str, Any]] = []
        for order in roh:
            if not isinstance(order, dict):
                continue
            zeile = _order_zeile(order)
            if zeile["status"] == _STATUS_AUSGEFUEHRT:
                zeile["notiz"] = notizen.get(str(zeile["id"]))
                ausgefuehrt.append(zeile)
            else:
                # EXPIRED u. a. zählen zur Offen-Sicht (Row-Status bleibt sichtbar).
                offen.append(zeile)
        ergebnis: dict[str, Any] = {
            "portfolio_id": depot_id,
            "offen": offen,
            "ausgefuehrt": ausgefuehrt,
            "hinweis": "„offen“ = platziert (wartet auf Ausführung), "
            "„ausgeführt“ = abgerechnet; die Marker-Notiz („[QuantLab-Agent] …“) "
            "hängt mit reference = Order-ID an den ausgeführten Orders.",
        }
        if args.status == "offen":
            ergebnis["ausgefuehrt"] = []
        elif args.status == "ausgefuehrt":
            ergebnis["offen"] = []
        return ergebnis

    args = pruefe_args(PlanspielOrdersArgs, roh_args, "planspiel_orders")
    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# planspiel_depotwerte
# ---------------------------------------------------------------------------


class PlanspielDepotwerteArgs(BaseModel):
    """Argumente von ``planspiel_depotwerte``."""

    portfolio_id: int | None = Field(
        default=None,
        ge=1,
        description="Planspiel-Depot-ID (Default: standard_depot_id aus den "
        "Einstellungen, sonst das erste Depot aus getUser).",
    )


async def planspiel_depotwerte(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Depotwert-Historie (Zeitreihe des Dashboard-Charts) lesen."""

    def _lesen() -> dict[str, Any]:
        client = hole_client()
        depot_id = _loese_depot_id(client, args.portfolio_id)
        werte = client.depotwerte_holen(depot_id)
        return {
            "portfolio_id": depot_id,
            "anzahl": len(werte),
            "zeitreihe": werte,
            "hinweis": "Vor dem ersten Handelstag (01.10.2026) kann die Reihe "
            "leer sein — das ist kein Fehler.",
        }

    args = pruefe_args(PlanspielDepotwerteArgs, roh_args, "planspiel_depotwerte")
    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# planspiel_regeln
# ---------------------------------------------------------------------------


class PlanspielRegelnArgs(BaseModel):
    """Argumente von ``planspiel_regeln``."""

    isin: str | None = Field(
        default=None,
        min_length=12,
        max_length=12,
        description='Optionale ISIN — mit ISIN kommen zusätzlich Gebührenmodell '
        '(getFeeModel) und Kapital-Check des Depots, z. B. "DE0007164600".',
    )


def _handelszeiten_zeile(zeile: dict[str, Any]) -> dict[str, Any]:
    """Handelsfenster-Eintrag der ``getTradingHours``-Antwort → flache Zeile."""
    return {
        "typ": zeile.get("instrumentType"),
        "boerse": zeile.get("exchangeExternalId"),
        "von": zeile.get("tradingHoursFrom"),
        "bis": zeile.get("tradingHoursTo"),
    }


async def planspiel_regeln(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Spielmechaniken 2026/27 (statisch) + lebende Regeln der Börse holen."""

    def _lesen() -> dict[str, Any]:
        client = hole_client()
        ergebnis: dict[str, Any] = {
            "stand": _REGELN_STAND,
            "regeln_kurzreferenz": _REGELN_KURZREFERENZ,
            "ordertypen": client.ordertypen_holen(),
            "handelszeiten": [
                _handelszeiten_zeile(z)
                for z in client.handelszeiten_holen()
                if isinstance(z, dict)
            ],
        }
        feiertage = client.feiertage_holen()
        ergebnis["feiertage"] = {
            "liste": feiertage.get("publicHolidays") or [],
            "feiertag_heute": feiertage.get("publicHoliday"),
            "wochenende_heute": feiertage.get("weekend"),
        }
        if args.isin:
            ergebnis["gebuehren_modell"] = client.gebuehren_modell(args.isin)
            depot_id = _loese_depot_id(client, None)
            kapital = client.kapital_check(depot_id)
            ergebnis["kapital_check"] = {
                "portfolio_id": depot_id,
                "betrag": kapital.get("amount"),
                "waehrung": kapital.get("isoCurrency"),
            }
        ergebnis["hinweis"] = (
            "Diese Regeln vor jeder Order-Empfehlung beachten (20 %-Limit, "
            "max. 5 Orders je Tag und Wertpapier, Gebühren)."
        )
        return ergebnis

    args = pruefe_args(PlanspielRegelnArgs, roh_args, "planspiel_regeln")
    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# platziere_order (P2b, E074) — der einzige Schreib-Pfad
# ---------------------------------------------------------------------------

#: E066/E074-Marker-Prefix — wird IMMER handlerseitig gebaut (Vertrauensgrenze:
#: das LLM liefert nur die Fachnotiz, niemals den Prefix).
_MARKER_PREFIX = "[QuantLab-Agent] "

#: Dieselbe Markierung OHNE folgendes Leerzeichen — der Doppel-Präfix-Schutz
#: prüft dagegen (QA-Fund F-P2B-1): auch „[QuantLab-Agent]x …“ gilt als schon
#: markiert und wird nicht ein zweites Mal präfixiert.
_MARKER_ZEICHEN = "[QuantLab-Agent]"

#: Default-Fachnotiz, wenn der Agent keine Notiz übergibt (E074).
_NOTIZ_OHNE = "manuell durch Agent-Run ausgelöst"

#: Handelsende der Börse (RECHERCHE P.6.4: Ausführung 08:00–22:00) — Default-
#: Gültigkeitsende einer Order (E074: „nächstes 22:00 Europe/Berlin“).
_HANDELSSCHLUSS_STUNDE = 22

#: seite (deutsch) → ``orderType`` der Börse (RECHERCHE P.5).
_ORDER_TYPE_NACH_SEITE = {"kaufen": "BUY", "verkaufen": "SELL"}


def _jetzt() -> datetime:
    """Uhr für die Gültigkeitsberechnung — Injektionspunkt für netzfreie Tests."""
    return datetime.now(UTC)


def _valid_until_iso(jetzt: datetime, tage: int) -> str:
    """Gültigkeitsende nach E074: das ``tage``-te 22:00 Europe/Berlin ab jetzt.

    ``tage=1`` (auch der Default) ist das **nächste** 22:00 — bei einem Submit
    nach 22:00 also der Folgetag (RECHERCHE P.6.5: LIMIT läuft 1 Börsentag,
    optional bis 14 Kalendertage). Format wie der Web-Client (RECHERCHE P.5:
    ``new Date().toISOString()``): UTC mit Millisekunden und „Z“-Endung.
    """
    lokal = jetzt.astimezone(ZoneInfo("Europe/Berlin"))
    ende = lokal.replace(hour=_HANDELSSCHLUSS_STUNDE, minute=0, second=0, microsecond=0)
    if ende <= lokal:  # Submit nach (oder genau um) 22:00 → nächster Handelstag
        ende += timedelta(days=1)
    return (ende + timedelta(days=tage - 1)).astimezone(UTC).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )


def _baue_marker_notiz(notiz: str | None) -> str:
    """E066/E074: ``noteText`` IMMER handlerseitig — Prefix nie vom LLM-Argument.

    Eine Fachnotiz, die selbst mit der Markierung beginnt, wird nicht doppelt
    markiert — geprüft wird gegen ``_MARKER_ZEICHEN`` ohne Leerzeichen, damit
    auch „[QuantLab-Agent]x …“-Formen (QA-Fund F-P2B-1) nicht erneut präfixiert
    werden; der Nutzer-Text bleibt dabei unverändert. Ohne Notiz setzt der
    Handler den Default-Text (E074). Final wird hart auf
    :data:`NOTIZ_MAX_ZEICHEN` gekappt (RECHERCHE P.5-Formvalidator).
    """
    fachnotiz = (notiz or "").strip() or _NOTIZ_OHNE
    if fachnotiz.startswith(_MARKER_ZEICHEN):
        return fachnotiz[:NOTIZ_MAX_ZEICHEN]
    return (_MARKER_PREFIX + fachnotiz)[:NOTIZ_MAX_ZEICHEN]


def _loese_order_depot(args: PlatziereOrderArgs) -> tuple[int, str]:
    """E070-Schreibkette (Guard-first): Ziel-Depot + Rolle — sonst Ablehnung.

    Reihenfolge (Fail-safe bis Markierung, E070): (1) weder ``portfolio_id``-
    Arg noch ``standard_depot_id ≠ 0`` → Schreib-Sperre mit Einstellungen-
    Verweis (der Lesefallback „erstes Depot“ gilt bewusst NICHT fürs Schreiben);
    (2) ``wettbewerbs_depot_id == 0`` → Sicherheitssperre — ohne beide Rollen-
    Markierungen ist nicht prüfbar, ob ein Ziel-Depot in der realen Wertung
    steht; (3) Ziel == Wettbewerbsdepot → nur mit ``bestaetigung_wettbewerb``.
    Alle drei Prüfungen laufen OHNE Netz-Call.
    """
    sektion = _planspiel_sektion()
    standard = int(sektion.get("standard_depot_id") or 0)
    wettbewerb = int(sektion.get("wettbewerbs_depot_id") or 0)
    if args.portfolio_id is not None:
        ziel = args.portfolio_id
    elif standard:
        ziel = standard
    else:
        raise ToolFehler(
            "Order abgelehnt — Schreib-Sperre: Es ist kein Standard-Depot "
            "markiert. Bitte zuerst in den Spieleinstellungen ein Standard-Depot "
            "festlegen (Einstellungen → Planspiel); ohne diese Markierung werden "
            "keine Orders platziert."
        )
    if not wettbewerb:
        raise ToolFehler(
            "Order abgelehnt (Sicherheitssperre): Das Wettbewerbsdepot ist nicht "
            "markiert — bitte einmal in den Einstellungen festlegen (Einstellungen "
            "→ Planspiel). Ohne beide Depot-Markierungen kann das Tool nicht "
            "prüfen, ob ein Depot in der realen Wertung steht."
        )
    if ziel == wettbewerb and not args.bestaetigung_wettbewerb:
        raise ToolFehler(
            f"Das Ziel-Depot {ziel} ist als Wettbewerbsdepot markiert — Orders "
            "dort gehen in die reale Wertung (Team-Ranking). Bitte den Nutzer im "
            "Chat ausdrücklich um Bestätigung bitten; nach seiner Zustimmung den "
            "Aufruf mit bestaetigung_wettbewerb=true wiederholen."
        )
    rolle = "Wettbewerbsdepot" if ziel == wettbewerb else "Trainingsdepot (Standard)"
    return ziel, rolle


def _preise_je_ordertyp(
    args: PlatziereOrderArgs,
) -> tuple[float | None, float | None]:
    """Ordertyp-/Preis-Kohärenz (E074): limit nur bei LIMIT, stop nur bei STOP.

    Werte werden auf 2 Nachkommastellen gerundet (RECHERCHE P.5); Fehler
    entstehen hier ohne jeden Netz-Call (DoD: Log bleibt leer).
    """
    limit = round(float(args.limit), 2) if args.limit is not None else None
    stop = round(float(args.stop), 2) if args.stop is not None else None
    if args.ordertyp != "limit" and limit is not None:
        raise ToolFehler(
            "„limit“ gehört nur zum Ordertyp „limit“ — bitte die Preisangabe "
            "entfernen oder ordertyp „limit“ wählen."
        )
    if args.ordertyp != "stop" and stop is not None:
        raise ToolFehler(
            "„stop“ gehört nur zum Ordertyp „stop“ — bitte die Preisangabe "
            "entfernen oder ordertyp „stop“ wählen."
        )
    if args.ordertyp == "limit" and limit is None:
        raise ToolFehler(
            'Ordertyp „limit“ benötigt einen Limit-Preis — bitte „limit“ angeben.'
        )
    if args.ordertyp == "stop" and stop is None:
        raise ToolFehler(
            'Ordertyp „stop“ benötigt eine Stop-Schwelle — bitte „stop“ angeben.'
        )
    return limit, stop


def _kapital_reicht(
    args: PlatziereOrderArgs,
    *,
    limit: float | None,
    stop: float | None,
    kapital: dict[str, Any],
    gebuehren: dict[str, Any],
) -> float:
    """Vorab-Kapitalprüfung (E074-Pre-Check) — wirft bei Mangel, liefert Betrag.

    Kalkuliert wird nur, wo ein Preis im Tool-Vertrag existiert (Kauf mit
    LIMIT/STOP): Menge × Preis + geschätzte Gebühr (max. Minimum, Prozentsatz
    des Kurswerts). Bei BEST kennt der Tool-Vertrag keinen Preis — die Prüfung
    macht die Börse (``validationErrorCode`` → ``order_abgelehnt``, E069).
    """
    verfuegbar = float(kapital.get("amount") or 0.0)
    preis = (
        limit if args.ordertyp == "limit" else stop if args.ordertyp == "stop" else None
    )
    if args.seite == "verkaufen" or preis is None:
        return verfuegbar
    volumen = args.menge * preis
    satz = float(gebuehren.get("percentage") or 0.0)
    minimum = float(gebuehren.get("minimum") or 0.0)
    bedarf = volumen + max(minimum, volumen * satz / 100.0)
    if verfuegbar < bedarf:
        raise ToolFehler(
            f"Order vorab abgelehnt: Der Kapital-Check des Depots zeigt "
            f"{verfuegbar:.2f} {kapital.get('isoCurrency') or 'EUR'}, für "
            f"{args.menge} Stück × {args.isin} zu {preis:.2f} sind aber rund "
            f"{bedarf:.2f} (inkl. geschätzter Gebühr) nötig. Bitte Menge oder "
            "Preis anpassen."
        )
    return verfuegbar


class PlatziereOrderArgs(BaseModel):
    """Argumente von ``platziere_order`` (E074) — dem einzigen Schreib-Tool."""

    isin: str = Field(
        min_length=12,
        max_length=12,
        description='ISIN des Instruments (12 Zeichen), z. B. "DE0007164600".',
    )
    seite: Literal["kaufen", "verkaufen"] = Field(
        description='"kaufen" (BUY) oder "verkaufen" (SELL).',
    )
    menge: int = Field(
        ge=1,
        description="Anzahl Stück (ganze Zahl ≥ 1).",
    )
    ordertyp: Literal["best", "limit", "stop"] = Field(
        description='"best" (unlimitiert zum Kurs), "limit" (mit limit) oder '
        '"stop" (mit stop).',
    )
    id_exchange: str | int = Field(
        description="Börsen-ID des Instruments (id_exchange aus planspiel_suche "
        'bzw. planspiel_instrument), z. B. "1".',
    )
    limit: float | None = Field(
        default=None,
        ge=0.01,
        description='Limit-Preis — nur bei ordertyp "limit" (2 Nachkommastellen).',
    )
    stop: float | None = Field(
        default=None,
        ge=0.01,
        description='Stop-Schwelle — nur bei ordertyp "stop" (2 Nachkommastellen).',
    )
    portfolio_id: int | None = Field(
        default=None,
        ge=1,
        description="Ziel-Depot (Default: standard_depot_id aus den Einstellungen). "
        "Das Wettbewerbsdepot nur mit bestaetigung_wettbewerb=true.",
    )
    notiz: str | None = Field(
        default=None,
        description="Fachnotiz zur Order-Idee — der Marker „[QuantLab-Agent]“ "
        "wird automatisch gesetzt (Gesamttext max. 1000 Zeichen).",
    )
    bestaetigung_wettbewerb: bool = Field(
        default=False,
        description="Nur true, wenn der Nutzer im Chat ausdrücklich eine Order "
        "im Wettbewerbsdepot bestätigt hat.",
    )
    valid_until_tage: int | None = Field(
        default=None,
        ge=1,
        le=14,
        description="Gültigkeit in Tagen (1–14; Default 1 = nächstes 22:00 "
        "Europe/Berlin).",
    )
    regel_pruefung: bool = Field(
        default=True,
        description="Vorab-Check der Spielregeln (Penny-Sperre, 20 %-Limit, "
        "5 Orders/Tag, Mindestaktivität; Default an). false nur als Notaus — "
        "die Börse erzwingt die Limits ohnehin serverseitig.",
    )


def _order_zusammenfassung(
    args: PlatziereOrderArgs,
    *,
    depot_id: int,
    rolle: str,
    limit: float | None,
    stop: float | None,
    valid_until: str,
) -> str:
    """Deutsche Ein-Zeilen-Zusammenfassung (Muster ``_zusammenfassung``/portfolio)."""
    preis = ""
    if args.ordertyp == "limit" and limit is not None:
        preis = f" Limit {limit:g}"
    elif args.ordertyp == "stop" and stop is not None:
        preis = f" Stop {stop:g}"
    return (
        f"{args.menge} × {args.isin} — {args.seite} ({args.ordertyp.upper()}"
        f"{preis}) in Depot {depot_id} ({rolle}), platziert, gültig bis "
        f"{valid_until}."
    )


async def platziere_order(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Platziert eine Order (placeOrder) — Marker, Guards und Antwortvertrag E074."""

    def _schreiben() -> dict[str, Any]:
        client = hole_client()
        # Guard-first (E074): alles ohne Netz zuerst, Pre-Checks erst danach.
        notiz_text = _baue_marker_notiz(args.notiz)
        depot_id, rolle = _loese_order_depot(args)  # (1) E070-Depot-Regeln
        limit, stop = _preise_je_ordertyp(args)  # (2) limit/stop-Kohärenz
        valid_until = _valid_until_iso(_jetzt(), args.valid_until_tage or 1)  # (3)
        kapital = client.kapital_check(depot_id)  # (4) Pre-Check 1 (Netz)
        gebuehren = client.gebuehren_modell(args.isin)  # (4) Pre-Check 2 (Netz)
        verfuegbar = _kapital_reicht(
            args, limit=limit, stop=stop, kapital=kapital, gebuehren=gebuehren
        )
        # (4b) E100-Regel-Vorabcheck (nach Kapital-Check; best-effort, kippt
        # nur bei HARTEN Verstößen — die Börse erzwingt die Limits ohnehin).
        # Preis-Basis wie _kapital_reicht: nur wo der Tool-Vertrag einen hat.
        regelhinweise: list[dict[str, str]] = []
        if args.regel_pruefung:
            regelhinweise = regeln_service.pruefe_order_vorab(
                client,
                depot_id=depot_id,
                seite=args.seite,
                isin=args.isin,
                menge=args.menge,
                preis=(
                    limit
                    if args.ordertyp == "limit"
                    else stop
                    if args.ordertyp == "stop"
                    else None
                ),
                jetzt=_jetzt(),
            )
            harte = [h for h in regelhinweise if h["schwere"] == "hart"]
            if harte:
                raise ToolFehler(
                    "Order vorab abgelehnt (Spielregeln der Planspiel-Börse): "
                    + " ".join(h["meldung"] for h in harte)
                )
        order = client.order_platzieren(  # (5) erst jetzt der Schreib-Call
            isin=args.isin,
            order_type=_ORDER_TYPE_NACH_SEITE[args.seite],
            menge=args.menge,
            ordertyp=args.ordertyp.upper(),
            id_exchange=args.id_exchange,
            portfolio_id=depot_id,
            valid_until=valid_until,
            notiz=notiz_text,
            limit=limit,
            stop=stop,
        )
        order_status = order.get("orderStatus")
        ausgefuehrt = bool(
            order_status == _STATUS_AUSGEFUEHRT or order.get("executionPrice") is not None
        )
        hinweis = (
            "Die Order ist platziert, aber noch NICHT ausgeführt — die Ausführung "
            "erfolgt zu Handelszeiten (08:00–22:00). Den Status prüfst du mit "
            "planspiel_orders („offen“ vs. „ausgeführt“)."
        )
        if rolle == "Wettbewerbsdepot":
            hinweis = (
                f"Nutzer-Bestätigung lag vor: Order ins Wettbewerbsdepot (Depot "
                f"{depot_id}) gesendet. {hinweis}"
            )
        if ausgefuehrt:
            hinweis = (
                "Die Order wurde bereits AUSGEFÜHRT (abgerechnet) — Details via "
                "planspiel_orders."
            )
        return {
            "status": "ausgeführt" if ausgefuehrt else "platziert",
            "order_id": order.get("id"),
            "order_status": order_status,
            "portfolio_id": depot_id,
            "depot_rolle": rolle,
            "zusammenfassung": _order_zusammenfassung(
                args,
                depot_id=depot_id,
                rolle=rolle,
                limit=limit,
                stop=stop,
                valid_until=valid_until,
            ),
            "gebuehr": {
                "satz": float(gebuehren.get("percentage") or 0.0),
                "minimum": float(gebuehren.get("minimum") or 0.0),
            },
            "kapital_vor_order": verfuegbar,
            "regelhinweise": regelhinweise,
            "valid_until": valid_until,
            "notiz": notiz_text,
            "hinweis_ausfuehrung": hinweis,
        }

    args = pruefe_args(PlatziereOrderArgs, roh_args, "platziere_order")
    return await _im_thread(_schreiben)


# ---------------------------------------------------------------------------
# planspiel_universum (Welle H1, E093)
# ---------------------------------------------------------------------------


class PlanspielUniversumArgs(BaseModel):
    """Argumente von ``planspiel_universum``."""

    aktion: Literal["liste", "importiere"] = Field(
        default="liste",
        description='"liste" liest das lokal gespeicherte Universum (netzfrei), '
        '"importiere" lädt das vollständige Universum neu von der Börse '
        "(~18 Gruppen, dauert wegen Pacing rund 20 Sekunden).",
    )
    gruppe: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description='Optionaler Gruppen-Filter (Marktgruppe, z. B. "DAX" oder '
        '"ETC") — nur bei aktion=liste.',
    )


def _universum_zeile(instrument: psb_repo.PsbInstrument) -> dict[str, Any]:
    """Universum-Zeile → kompakte deutsch-flache Zeile (E073-Feldsatz)."""
    return {
        "isin": instrument.isin,
        "wkn": instrument.wkn,
        "name": instrument.name,
        "typ": instrument.typ,
        "gruppen": instrument.gruppen_liste(),
        "yahoo_kuerzel": instrument.yahoo_kuerzel,
    }


async def planspiel_universum(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Aktien-Universum der Planspiel-Börse lokal listen oder importieren."""

    args = pruefe_args(PlanspielUniversumArgs, roh_args, "planspiel_universum")

    if args.aktion == "importiere":

        def _importieren() -> dict[str, Any]:
            ergebnis = importiere_universum(
                hole_client(), kuerzel_aufloeser=loese_isin_zu_kuerzel
            )
            return {
                "aktion": "importiere",
                **ergebnis,
                "hinweis": "Import abgeschlossen — Einzelnachweis via aktion=liste "
                "(Gruppen-Filter optional).",
            }

        return await _im_thread(_importieren)

    def _lesen() -> dict[str, Any]:
        filter_gruppe = (args.gruppe or "").strip() or None
        instrumente = psb_repo.liste(gruppe=filter_gruppe)
        return {
            "aktion": "liste",
            "gruppe": filter_gruppe,
            "anzahl": len(instrumente),
            "instrumente": [_universum_zeile(i) for i in instrumente],
            "hinweis": "Leeres Ergebnis? aktion=importiere lädt das Universum "
            "von der Börse (dauert wegen Pacing rund 20 Sekunden).",
        }

    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# planspiel_kennzahlen (Welle H3, E095)
# ---------------------------------------------------------------------------


#: Sektion (API-Name, wie REST ``GET /planspiel/kennzahlen?sektion=``) →
#: deutsch-flacher Antwort-Schlüssel (E073-Feldsatz).
_KENNZAHLEN_SEKTIONEN: dict[str, str] = {
    "getKennzahlen": "kennzahlen",
    "getKursdaten": "kursdaten",
    "getPerformance": "performance",
    "getAnalystenempfehlungen": "analysten",
    "getUnternehmensprofil": "profil",
}


class PlanspielKennzahlenArgs(BaseModel):
    """Argumente von ``planspiel_kennzahlen``."""

    isin: str = Field(
        min_length=12,
        max_length=12,
        description='ISIN des Instruments (12 Zeichen), z. B. "DE0007164600".',
    )
    sektionen: list[Literal[
        "getKennzahlen",
        "getKursdaten",
        "getPerformance",
        "getAnalystenempfehlungen",
        "getUnternehmensprofil",
    ]] | None = Field(
        default=None,
        description="Optionale Einschränkung der Sektionen (Default: alle "
        "fünf — Kennzahlen, Kursdaten, Performance, Analysten, Profil).",
    )
    frisch: bool = Field(
        default=False,
        description="true = VWD-Snippets live laden und den Cache erneuern "
        "(Default: aus dem lokalen Cache, bei leerem Cache wird live geladen).",
    )


def _letzter_kurs_von_isin(isin: str) -> float | None:
    """Letzter Schlusskurs der ISIN aus dem Parquet-Store (1d, best effort).

    Dieselbe Basis wie das Screener-Kriterium ``analysten_upside`` (E095):
    ``1d``-Balken des ISIN-Symbols, egal welche Quelle; ohne Bestand →
    ``None`` (kein Fehler — Upside entfällt dann nur).
    """
    try:
        balken = parquet_store.read_bars(isin, "1d", limit=1)
    except Exception:  # noqa: BLE001 — best effort: fehlender/defekter Bestand
        return None
    if balken.empty:
        return None
    kurs = float(balken["close"].iloc[-1])
    return kurs if kurs > 0 else None


def _kennzahlen_upside(analysten: dict[str, Any], isin: str) -> float | None:
    """Kursziel-Upside = (Kursziel − letzter Kurs) / letzter Kurs (Bruchzahl).

    Best-effort: fehlt das Kursziel oder der letzte Kurs (Parquet-Bestand der
    ISIN, s. :func:`_letzter_kurs_von_isin`), bleibt das Feld weg — kein
    Fehler (additives Feld, E073-Vertrag).
    """
    kursziel = analysten.get("kursziel")
    if not isinstance(kursziel, (int, float)) or isinstance(kursziel, bool):
        return None
    kurs = _letzter_kurs_von_isin(isin)
    if kurs is None:
        return None
    return (float(kursziel) - kurs) / kurs


def _hole_oder_lade_kennzahlen(
    isin: str, gefordert: frozenset[str], frisch: bool
) -> tuple[dict[str, Any], dict[str, str], bool]:
    """Cache-Pfad → Live-Pfad (Referenz ``markt.kennzahlen``) — gemeinsame
    Logik des Tools. Liefert ``(daten, aktualisiert, aus_cache)``.

    ``frisch=false`` bedient aus dem Cache, sobald **alle geforderten**
    Sektionen liegen; sonst (und bei ``frisch=true``) lädt
    :func:`vwd.kennzahlen_laden` live über die idNotation-Brücke und upsertet
    sämtliche Sektionen (vollständiger Cache — auch für Screener/Dashboard).
    PlanspielFehler propagieren (Mapping über ``_im_thread``).
    """
    normalisiert = isin.strip().upper()
    if not frisch:
        gecacht = psb_repo.kennzahlen_lesen(normalisiert)
        if gefordert <= set(gecacht):
            aktualisiert = psb_repo.kennzahlen_aktualisiert(normalisiert)
            return gecacht, aktualisiert, True
    instrument = psb_repo.instrument(normalisiert)
    if instrument is None or not instrument.id_external:
        raise ToolFehler(
            f"Für {normalisiert} liegen keine VWD-Kennzahlen vor: Das "
            "Instrument fehlt im lokalen Universum (oder ohne Börsen-"
            "Referenz idExternal). Bitte zuerst das Universum importieren "
            "(planspiel_universum, aktion=importiere)."
        )
    detail = hole_client().instrument_mit_kursen(normalisiert, instrument.id_external)
    quotes = detail.get("quoteList") or []
    id_notation = (
        quotes[0].get("idNotation") if quotes and isinstance(quotes[0], dict) else None
    )
    if not id_notation:
        raise ToolFehler(
            f"Die Planspiel-Börse liefert für {normalisiert} keine idNotation — "
            "VWD-Kennzahlen sind ohne sie nicht abrufbar. Bitte das Universum "
            "importieren (planspiel_universum, aktion=importiere) und es später "
            "erneut versuchen."
        )
    daten = vwd_service.kennzahlen_laden(str(id_notation))
    aktualisiert = {
        sektion: psb_repo.kennzahlen_speichern(normalisiert, sektion, inhalt)
        for sektion, inhalt in daten.items()
    }
    return daten, aktualisiert, False


async def planspiel_kennzahlen(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Fundamentaldaten/Analysten der Planspiel-Börse (VWD) je ISIN holen."""

    args = pruefe_args(PlanspielKennzahlenArgs, roh_args, "planspiel_kennzahlen")
    normalisiert = args.isin.strip().upper()
    gefordert = (
        frozenset(args.sektionen)
        if args.sektionen
        else frozenset(_KENNZAHLEN_SEKTIONEN)
    )

    def _lesen() -> dict[str, Any]:
        daten, aktualisiert, aus_cache = _hole_oder_lade_kennzahlen(
            normalisiert, gefordert, args.frisch
        )
        erg: dict[str, Any] = {"isin": normalisiert, "aus_cache": aus_cache}
        # Alle fünf Antwort-Felder bleiben konstant gesetzt (E073-Vertrag) —
        # nicht geforderte Sektionen stehen bewusst auf None.
        for api_name, feld in _KENNZAHLEN_SEKTIONEN.items():
            erg[feld] = daten.get(api_name) if api_name in gefordert else None
        if "getAnalystenempfehlungen" in gefordert:
            analysten = daten.get("getAnalystenempfehlungen") or {}
            upside = _kennzahlen_upside(analysten, normalisiert)
            if upside is not None:
                erg["kursziel_upside"] = upside
        erg["aktualisiert"] = {
            _KENNZAHLEN_SEKTIONEN.get(sektion, sektion): zeit
            for sektion, zeit in aktualisiert.items()
            if sektion in gefordert
        }
        erg["hinweis"] = (
            "Kursziel-Upside (falls vorhanden) = (Kursziel − letzter Kurs) / "
            "letzter Kurs als Bruchzahl; letzter Kurs aus dem lokalen "
            "1d-Bestand der ISIN. Kennzahlen via planspiel_instrument "
            "aktualisieren sich mit frisch=true."
        )
        return erg

    return await _im_thread(_lesen)


# ---------------------------------------------------------------------------
# Registrierung (E073/E074 — 7 lesende Tools + platziere_order)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_status",
            description=(
                "Status der Planspiel-Börse: Konfiguration/Einloggen, Contest "
                "(Phase, Zeitraum, 50.000 €-Startkapital), Depots und Team. "
                "Keine Parameter: {}."
            ),
            parameters=schema_von(PlanspielStatusArgs),
            handler=planspiel_status,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_suche",
            description=(
                "Handelbare Wertpapiere der Planspiel-Börse nach Name suchen "
                "(Aktien, ETFs, Fonds, Anleihen; Krypto nur Trainingsdepot). "
                "Liefert id_external/isin/wkn/name/preis/exchange — Details via "
                "planspiel_instrument. "
                'Beispiel: {"name": "SAP", "groesse": 5}.'
            ),
            parameters=schema_von(PlanspielSucheArgs),
            handler=planspiel_suche,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_instrument",
            description=(
                "Instrument-Details und aktuelle Kurse (bid/ask, Tageshoch/-tief) "
                "für ein Instrument je ISIN; id_external kommt aus planspiel_suche. "
                'Beispiel: {"isin": "DE0007164600", "id_external": "1"}.'
            ),
            parameters=schema_von(PlanspielInstrumentArgs),
            handler=planspiel_instrument,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_depot",
            description=(
                "Planspiel-Depot lesen: Depotwert, Barbestand, verfügbares "
                "Kapital, Performance und die Positionen (je ISIN/WKN, Menge, "
                "Wert, bid/ask). "
                'Beispiel: {} oder {"portfolio_id": 111}.'
            ),
            parameters=schema_von(PlanspielDepotArgs),
            handler=planspiel_depot,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_orders",
            description=(
                "Orders eines Planspiel-Depots lesen — „offen“ (platziert, "
                "wartet auf Ausführung) und „ausgeführt“ (abgerechnete Trades "
                "inkl. Marker-Notiz) getrennt. "
                'Beispiel: {"status": "ausgefuehrt"}.'
            ),
            parameters=schema_von(PlanspielOrdersArgs),
            handler=planspiel_orders,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_depotwerte",
            description=(
                "Depotwert-Historie (Zeitreihe des Planspiel-Dashboards) lesen; "
                "vor dem ersten Handelstag 01.10.2026 kann sie leer sein. "
                "Beispiel: {} oder {\"portfolio_id\": 111}."
            ),
            parameters=schema_von(PlanspielDepotwerteArgs),
            handler=planspiel_depotwerte,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_regeln",
            description=(
                "Spielmechaniken der Planspiel-Börse: statische Kurzreferenz "
                "2026/27 (50.000 €, 0,3 % min. 15 €, 20 %-Limit, max. 5 Orders/"
                "Tag/WP, 08:00–22:00, Laufzeit bis 25.01.2027) plus lebende "
                "Daten (Ordertypen, Handelszeiten, Feiertage; mit isin auch "
                "Gebührenmodell + Kapital-Check). "
                'Beispiel: {} oder {"isin": "DE0007164600"}.'
            ),
            parameters=schema_von(PlanspielRegelnArgs),
            handler=planspiel_regeln,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_universum",
            description=(
                "Aktien-Universum der Planspiel-Börse (alle handelbaren "
                "Instrumente je Marktgruppe, z. B. DAX, MDAX, ETC): lokal "
                "listen (netzfrei, optional nach Gruppe gefiltert) oder von "
                "der Börse importieren (Hintergrunddaten, dauert wegen "
                "Pacing rund 20 Sekunden — vor Backtests/Screens des "
                "Universums einmal ausführen). "
                'Beispiel: {"aktion": "liste"} oder {"aktion": "liste", '
                '"gruppe": "DAX"} bzw. {"aktion": "importiere"}.'
            ),
            parameters=schema_von(PlanspielUniversumArgs),
            handler=planspiel_universum,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="planspiel_kennzahlen",
            description=(
                "Fundamentaldaten der Planspiel-Börse je ISIN (VWD-Snippets): "
                "Kennzahlen (KGV, Dividendenrendite, Marktkapitalisierung …), "
                "Kursdaten, Performance je Zeitraum, Analystenempfehlungen "
                "(kauf/neutral/verkauf, Kursziel) und Unternehmensprofil; "
                "liest den lokalen Cache, mit frisch=true wird live geladen. "
                "Beispiel: {\"isin\": \"DE0007164600\"} oder {\"isin\": "
                "\"DE0007164600\", \"frisch\": true}."
            ),
            parameters=schema_von(PlanspielKennzahlenArgs),
            handler=planspiel_kennzahlen,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="platziere_order",
            description=(
                "Platziert eine Order an der Planspiel-Börse (kaufen/verkaufen, "
                "best/limit/stop) — nur bei klarer Nutzer-Beauftragung im Chat. "
                "Setzt automatisch die Marker-Notiz „[QuantLab-Agent]“, prüft "
                "Depot-Markierung, Gültigkeit (1–14 Tage), Kapital und Gebühren "
                "vorab und liefert „platziert (wartet auf Ausführung)“ mit "
                "Order-ID; Ausführung via planspiel_orders. Das Wettbewerbsdepot "
                "nur nach ausdrücklicher Nutzer-Bestätigung "
                "(bestaetigung_wettbewerb=true). "
                'Beispiel: {"isin": "DE0007164600", "seite": "kaufen", "menge": '
                '2, "ordertyp": "best", "id_exchange": "1"}.'
            ),
            parameters=schema_von(PlatziereOrderArgs),
            handler=platziere_order,
        )
    )


_registrieren()
