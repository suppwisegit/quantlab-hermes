"""Planspiel-Börse-Endpunkte (Welle P, E072) — EINGEFRORENER Vertrag für P3.

P3 baut mit gemocktem API-Client dagegen (Muster B4/E059 „Client gegen Spec");
der Live-Erweis erfolgt in §P4. Der Fachfehler-Client (:mod:`quantlab.services.
planspiel`, E068/E069) liefert die Daten; Fehler werden 1:1 über
``PlanspielFehler.to_api_error()`` auf die feste Mapping-Tabelle
:data:`~quantlab.services.planspiel.fehler.STATUS_NACH_CODE` abgebildet
(Haus-Fehlerformat ``{detail: {code, message, hint?}}``, C.1).

Ausnahme: ``GET /planspiel/status`` wirft bei Planspiel-Fehlern **bewusst
nicht** — die Antwort bleibt HTTP 200 mit strukturiertem ``fehler``-Feld, damit
die UI bei Ausfällen bedienbar bleibt (E072 „niemals 5xx bei Planspiel-
Ausfällen“).

Die Endpunkte sind bewusst normale ``def`` (nicht ``async``): der Client ist
sync (Pacing ≥ 1 s) und wird von FastAPI im Threadpool ausgeführt — der
Event-Loop blockiert nicht (Muster: routers/settings.py).

Depot-Fallback (E070): ohne ``portfolio_id`` gilt ``standard_depot_id`` aus
der Settings-Sektion ``planspiel``; ist dort ``0`` (nicht markiert), das erste
Depot aus ``getUser``.

Universum (H1/E093): ``GET /planspiel/universum`` liest den lokalen Spiegel
(``psb_instrumente``) netzfrei; ``POST /planspiel/universum/import`` startet
den Tab-Download als **Hintergrundjob** (prozessweit genau EIN Lauf — ein
zweiter POST während eines Laufs wird mit 409 ``import_laeuft`` abgewiesen),
``GET /planspiel/universum/import`` liefert den Fortschritt für die UI. Der
Lauf selbst konsumiert :mod:`quantlab.services.planspiel.universum` (Client
sync + Pacing → ``asyncio.to_thread``, Muster ``import_sp500``).

Kennzahlen (H3/E095): ``GET /planspiel/kennzahlen`` liest den VWD-Cache
(``psb_kennzahlen``); ``frisch=true`` (oder Cache-Miss) lädt live über
:mod:`quantlab.services.planspiel.vwd` und upsertet. Der Endpunkt folgt dem
Referenz-Muster ``markt.kennzahlen``: **Fachfehler sind ein strukturiertes
``fehler``-Objekt in einer 200-Antwort** (``ok:false``), niemals 5xx — nur
Argument-Verstöße (ISIN/Sektion) bleiben ApiError 400 (Haus-Format).

Historien-Backfill (H5/E097): ``POST /planspiel/universum/historie``
(``{aktion: "starten"|"pausieren"}``) steuert den resumablen Bulk-Lauf
(5Y/1d fürs gesamte Universum via Byteworx) als asyncio-Hintergrundjob;
``GET /planspiel/universum/historie`` liefert den Fortschritts-Snapshot mit
je-ISIN-Kompaktstatistik. Dedup: läuft bereits ein Lauf, antwortet der Start
mit 409 ``backfill_laeuft``. Zustand/Mechanik: :mod:`quantlab.services.
planspiel.backfill`.

Altbestand-Import (H6/E098): ``POST /planspiel/altbestand_import``
(``{pfad}``) überführt den SQLite-Altbestand der Referenz-App (instrumente,
kurse, historie, kennzahlen) idempotent nach QuantLab — Quelle strikt
read-only, Bestand gewinnt, Counts-Report als Antwort. Dedup: 409
``import_laeuft`` mit EIGENEM Lock (unabhängig vom Universum-Job).

Wettbewerb-Dashboard (H7/E101): ``GET /planspiel/wettbewerb/ueberblick``
liefert die vier read-only Blöcke (Top-Mover, Analysten-Upside,
Positions-Monitor, Aktivitätszähler) aus :mod:`quantlab.services.planspiel.
wettbewerb` — E072-Statusmuster: Fachfälle (z. B. nicht konfiguriert)
stehen als ``fehler``-Feld in einer 200-Antwort, nie 5xx.
 """

from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import APIRouter

from quantlab import db
from quantlab.agent.tools.daten import ist_isin, loese_isin_zu_kuerzel
from quantlab.errors import ApiError
from quantlab.services.planspiel import (
    PlanspielFehler,
    anmeldeinfos_aus_settings,
    client_zuruecksetzen,
    hole_client,
)
from quantlab.services.planspiel import altbestand as altbestand_service
from quantlab.services.planspiel import backfill as backfill_service
from quantlab.services.planspiel import universum as universum_service
from quantlab.services.planspiel import vwd as vwd_service
from quantlab.services.planspiel import wettbewerb as wettbewerb_service
from quantlab.services.planspiel.altbestand import AltbestandFehler
from quantlab.storage.repos import psb_repo, settings_repo

router = APIRouter(tags=["planspiel"])

#: Geschlossene Status-Filter von ``GET /planspiel/orders`` (E072).
_STATUS_FILTER: frozenset[str] = frozenset({"alle", "offen", "ausgefuehrt"})

#: Order-Status „abgerechnet“ (RECHERCHE P.3, [LIVE] verifiziert); alle
#: anderen/unbekannten Status gelten tolerant als „offen“ (ANNAHME P.7.7).
_STATUS_AUSGEFUEHRT = "EXECUTED"

#: P4-Live-Befund (2026-09-21): ``getOrders`` verlangt ``status`` zwingend
#: (ohne → HTTP 400); „alle“ fragt daher die beiden verifizierten Enum-Werte
#: einzeln ab (EXPIRED ist der zweite [LIVE]-verifizierte Filter-Wert;
#: PLACED ist nur Feldwert der placeOrder-Antwort, kein Abfragefilter).
_STATUS_ABGELAUFEN = "EXPIRED"


def _planspiel_sektion() -> dict[str, Any]:
    """Settings-Sektion ``planspiel`` — leeres Dict, falls der Bestand sie fehlt."""
    try:
        return settings_repo.get_section("planspiel")
    except ValueError:  # alter Bestand ohne Migration der Defaults (P1-Muster)
        return {}


def _loese_depot_id(portfolio_id: int | None) -> int:
    """E070-Lesefallback: Argument → ``standard_depot_id`` → erstes Depot."""
    if portfolio_id is not None:
        return portfolio_id
    standard = int(_planspiel_sektion().get("standard_depot_id") or 0)
    if standard:
        return standard
    try:
        user = hole_client().user_holen()
    except PlanspielFehler as exc:
        raise exc.to_api_error() from exc
    for depot in user.get("userPortfolios") or []:
        if isinstance(depot, dict) and depot.get("id") is not None:
            return int(depot["id"])
    raise ApiError(
        502,
        "vertrag_geaendert",
        "Die Planspiel-Börse liefert keine Depots — bitte die Verbindung in den "
        "Einstellungen testen.",
    )


def _depots_aus_user(user: dict[str, Any]) -> list[dict[str, Any]]:
    """``userPortfolios`` → kompakte Depot-Zeilen (id/name [+ status])."""
    return [
        {
            "id": depot.get("id"),
            "name": depot.get("depotName"),
            "status": depot.get("status"),
        }
        for depot in user.get("userPortfolios") or []
        if isinstance(depot, dict)
    ]


def _position_zeile(item: dict[str, Any]) -> dict[str, Any]:
    """Positions-Item der ``getPortfolioWithItems``-Antwort → flache Zeile (E072)."""
    instrument = item.get("instrument") or {}
    performance = item.get("performanceRelTotal")
    if performance is None:
        performance = item.get("performanceRel")
    return {
        "isin": instrument.get("isin"),
        "wkn": instrument.get("wkn"),
        "name": instrument.get("name"),
        "menge": item.get("quantity"),
        "wert": item.get("currentValue"),
        "performance_rel": performance,
        "bid": instrument.get("bid"),
        "ask": instrument.get("ask"),
        "typ": instrument.get("instrumentType"),
    }


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


# ---------------------------------------------------------------------------
# Status (niemals 5xx — Fehler strukturiert statt Throw, E072)
# ---------------------------------------------------------------------------


@router.get("/planspiel/status")
def get_planspiel_status() -> dict[str, Any]:
    """Konfiguration, Login-Status, Depots, Contest und Team (E072-Struktur)."""
    client = hole_client()
    konfiguriert = anmeldeinfos_aus_settings() is not None
    try:
        user = client.user_holen()
    except PlanspielFehler as exc:
        # Fachfehler (z. B. nicht_konfiguriert/netzwerk) → HTTP 200 + fehler-Feld.
        return {
            "konfiguriert": konfiguriert,
            "eingeloggt": client.eingeloggt,
            "depots": [],
            "contest": None,
            "team": None,
            "fehler": {"code": exc.code, "message": exc.message},
        }
    contest = user.get("contest") or {}
    team = user.get("teamInfo") or {}
    return {
        "konfiguriert": konfiguriert,
        "eingeloggt": client.eingeloggt,
        "depots": _depots_aus_user(user),
        "contest": {
            "name": contest.get("publicName"),
            "phase": contest.get("contestPhase"),
            "start": contest.get("startDate"),
            "ende": contest.get("endDate"),
            "credit": contest.get("credit"),
        }
        if isinstance(contest, dict) and contest
        else None,
        "team": (
            {"name": team.get("name")} if isinstance(team, dict) and team else None
        ),
        "fehler": None,
    }


# ---------------------------------------------------------------------------
# Depot + Orders (Fachfehler → Mapping-Tabelle E069)
# ---------------------------------------------------------------------------


@router.get("/planspiel/depot")
def get_planspiel_depot(portfolio_id: int | None = None) -> dict[str, Any]:
    """Depot-Kennzahlen (Wert/Barbestand/Performance) + Positionen (E072)."""
    try:
        client = hole_client()
        depot_id = _loese_depot_id(portfolio_id)
        kennzahlen = client.portfolio_holen(depot_id)
        mit_positionen = client.positionen_holen(depot_id)
    except PlanspielFehler as exc:
        raise exc.to_api_error() from exc
    positionen = [
        _position_zeile(item)
        for item in mit_positionen.get("items") or []
        if isinstance(item, dict)
    ]
    return {
        "portfolio": {
            "wert": kennzahlen.get("value"),
            "barbestand": kennzahlen.get("credit"),
            "verfuegbar": kennzahlen.get("availableCredit"),
            "instrumentswert": kennzahlen.get("valueOfInstruments"),
            "performance_abs": kennzahlen.get("performanceAbs"),
            "performance_rel": kennzahlen.get("performanceRel"),
            "anzahl_trades": kennzahlen.get("numTrades"),
        },
        "positionen": positionen,
    }


@router.get("/planspiel/orders")
def get_planspiel_orders(
    portfolio_id: int | None = None, status: str = "alle"
) -> dict[str, Any]:
    """Orders eines Depots — offen (platziert) vs. ausgeführt getrennt (E072)."""
    if status not in _STATUS_FILTER:
        raise ApiError(
            400,
            "ungueltiger_status",
            f'Ungültiger Status-Filter „{status}“ — erlaubt sind: '
            f"{', '.join(sorted(_STATUS_FILTER))}.",
        )
    try:
        client = hole_client()
        depot_id = _loese_depot_id(portfolio_id)
        if status == "ausgefuehrt":
            roh = client.orders_holen(depot_id, _STATUS_AUSGEFUEHRT)
        else:
            # „alle“/„offen“: EXECUTED + EXPIRED einzeln abfragen
            # (P4-Live: getOrders verlangt den status-Parameter zwingend).
            roh = client.orders_holen(
                depot_id, _STATUS_AUSGEFUEHRT
            ) + client.orders_holen(depot_id, _STATUS_ABGELAUFEN)
    except PlanspielFehler as exc:
        raise exc.to_api_error() from exc
    offen: list[dict[str, Any]] = []
    ausgefuehrt: list[dict[str, Any]] = []
    for order in roh:
        if not isinstance(order, dict):
            continue
        zeile = _order_zeile(order)
        if zeile["status"] == _STATUS_AUSGEFUEHRT:
            ausgefuehrt.append(zeile)
        else:
            # Unbekannte Status gelten als offen (PLAN §P2a-Risiko).
            offen.append(zeile)
    ergebnis: dict[str, Any] = {"offen": offen, "ausgefuehrt": ausgefuehrt}
    if status == "offen":
        ergebnis["ausgefuehrt"] = []
    elif status == "ausgefuehrt":
        ergebnis["offen"] = []
    return ergebnis


# ---------------------------------------------------------------------------
# Credentials (E071: E-Mail Settings, Passwort Keyring/Fallback — nie Klartext)
# ---------------------------------------------------------------------------


def _credentials_status() -> dict[str, Any]:
    """Maskierter Credentials-Stand — die Passwort-Maske verlässt nie das Haus."""
    status = settings_repo.planspiel_passwort_status()
    return {
        "email": str(_planspiel_sektion().get("email") or ""),
        "passwort": {
            "konfiguriert": status.konfiguriert,
            "maske": status.maske,
            "quelle": status.quelle,
        },
    }


@router.get("/planspiel/credentials")
def get_planspiel_credentials() -> dict[str, Any]:
    """Hinterlegte Zugangsdaten — ausschließlich Maske, nie Klartext (E071)."""
    return _credentials_status()


@router.put("/planspiel/credentials")
def put_planspiel_credentials(payload: dict[str, Any]) -> dict[str, Any]:
    """Speichert E-Mail (Settings) und/oder Passwort (Keyring/Fallback, E071)."""
    email = payload.get("email")
    passwort = payload.get("passwort")
    if email is None and passwort is None:
        raise ApiError(
            400,
            "ungueltige_anmeldedaten",
            "Weder E-Mail noch Passwort übergeben — bitte mindestens eines der "
            "beiden Felder angeben.",
            hint='Body-Format: {"email": "…", "passwort": "…"}',
        )
    if email is not None:
        if not isinstance(email, str) or not email.strip():
            raise ApiError(
                400,
                "ungueltige_anmeldedaten",
                "Die E-Mail fehlt oder ist leer.",
            )
        settings_repo.update_section("planspiel", {"email": email.strip()})
    if passwort is not None:
        if not isinstance(passwort, str) or not passwort.strip():
            raise ApiError(
                400,
                "ungueltige_anmeldedaten",
                "Das Passwort fehlt oder ist leer.",
            )
        settings_repo.planspiel_passwort_save(passwort.strip())
    # Alte Sitzung verwerfen (ANNAHME): nach dem Credentials-Wechsel soll der
    # nächste Zugriff mit den neuen Anmeldedaten neu einloggen (P1-API).
    client_zuruecksetzen()
    return _credentials_status()


@router.delete("/planspiel/credentials")
def delete_planspiel_credentials() -> dict[str, Any]:
    """Räumt NUR das Passwort (Keyring + Fallback-Datei) — E-Mail bleibt (E071)."""
    settings_repo.planspiel_passwort_loeschen()
    client_zuruecksetzen()  # s. PUT — Sitzung an die alten Credentials gebunden
    return _credentials_status()


# ---------------------------------------------------------------------------
# Verbindungstest (Button der Settings-Sektion, E072)
# ---------------------------------------------------------------------------


@router.post("/planspiel/verbindung_testen")
def post_planspiel_verbindung_testen() -> dict[str, Any]:
    """Führt einen echten Login aus → ``{ok, benutzer, depots}`` oder Fachfehler."""
    try:
        user = hole_client().authentifizieren()
    except PlanspielFehler as exc:
        raise exc.to_api_error() from exc
    benutzer = user.get("username") or user.get("email") or ""
    return {
        "ok": True,
        "benutzer": str(benutzer),
        "depots": [
            {"id": depot.get("id"), "name": depot.get("name")}
            for depot in _depots_aus_user(user)
        ],
    }


# ---------------------------------------------------------------------------
# Universum (Welle H1, E093): lokaler Spiegel + Hintergrund-Import
# ---------------------------------------------------------------------------

#: Zustand des Universum-Imports — prozessweit genau EIN Lauf (Dedup, H1).
#: Bewusst Router-Modulzustand (Desktop-App, ein Prozess); ``_import_lock``
#: schützt Read-modify-write zwischen Threadpool-Endpoints und dem asyncio-
#: Task des Laufs (Client ist sync + Pacing → läuft via ``asyncio.to_thread``).
_import_lock = threading.Lock()
_import_zustand: dict[str, Any] = {
    "status": "leer",  # leer | laeuft | fertig | fehler
    "gruppe": None,  # aktuell laufende Gruppe (nur bei „laeuft“)
    "fortschritt": 0,  # abgeschlossene Gruppen
    "gruppen_gesamt": 0,
    "zeilen": 0,  # verarbeitete Tab-Zeilen (Upserts)
    "instrumente": None,  # Gesamtzähler der Tabelle (bei „fertig“ gesetzt)
    "fehler": None,  # {"code", "message"} — deutsch (E069-Codes)
    "gestartet_am": None,
    "beendet_am": None,
}
#: Starke Referenz auf den Lauf-Task (GC-Schutz; asyncio wirft sonst gelegentlich
#: „Task was destroyed but it is pending“).
_import_task: asyncio.Task[None] | None = None


def _import_status() -> dict[str, Any]:
    """Momentaufnahme des Import-Zustands (Kopie — Router-Vertrag bleibt stabil)."""
    with _import_lock:
        return dict(_import_zustand)


def _instrument_zeile(instrument: psb_repo.PsbInstrument) -> dict[str, Any]:
    """Repo-Zeile → deutsch-flache Antwort (E072-Feldsatz, Gruppen als Liste)."""
    return {
        "isin": instrument.isin,
        "wkn": instrument.wkn,
        "name": instrument.name,
        "typ": instrument.typ,
        "branchen": instrument.branchen,
        "gruppen": instrument.gruppen_liste(),
        "id_external": instrument.id_external,
        "waehrung": instrument.waehrung,
        "yahoo_kuerzel": instrument.yahoo_kuerzel,
        "aktualisiert_am": instrument.aktualisiert_am,
    }


@router.get("/planspiel/universum")
def get_planspiel_universum(gruppe: str | None = None) -> dict[str, Any]:
    """Lokales PSB-Universum — optional nach Marktgruppe gefiltert (H1/E093).

    Reiner DB-Lesepfad: E072-Stil heißt hier, dass auch Randfälle (leeres
    Universum, unbekannte Gruppe) **200 mit leerer Liste** liefern — nie 5xx.
    """
    instrumente = psb_repo.liste(gruppe=(gruppe or "").strip() or None)
    return {
        "anzahl": len(instrumente),
        "gruppen": psb_repo.gruppen(),
        "instrumente": [_instrument_zeile(i) for i in instrumente],
    }


@router.get("/planspiel/universum/import")
def get_planspiel_universum_import() -> dict[str, Any]:
    """Status des Universum-Imports (Fortschrittsanzeige der UI, H1)."""
    return _import_status()


async def _universum_import_lauf() -> None:
    """Hintergrundlauf: Pipeline aus :mod:`...planspiel.universum` (E093).

    Endzustand garantiert („fertig“/„fehler“) — der Job bleibt nie hängen;
    PlanspielFehler wird deutsch niedergelegt (E069-Codes), alles andere
    ebenfalls (kein Fachcode-Raterei im Fehlerfall).
    """

    def _fortschritt(gruppe: str, index: int, gesamt: int) -> None:
        with _import_lock:
            _import_zustand.update(
                gruppe=gruppe, fortschritt=index - 1, gruppen_gesamt=gesamt
            )

    try:
        ergebnis = await asyncio.to_thread(
            universum_service.importiere_universum,
            hole_client(),
            kuerzel_aufloeser=loese_isin_zu_kuerzel,
            fortschritt=_fortschritt,
        )
    except PlanspielFehler as exc:
        with _import_lock:
            _import_zustand.update(
                status="fehler",
                gruppe=None,
                fehler={"code": exc.code, "message": exc.message},
                beendet_am=db.utc_now_iso(),
            )
    except Exception as exc:  # noqa: BLE001 — Job-Endzustand ist Pflicht (H1)
        with _import_lock:
            _import_zustand.update(
                status="fehler",
                gruppe=None,
                fehler={"code": "import_fehlgeschlagen", "message": str(exc)},
                beendet_am=db.utc_now_iso(),
            )
    else:
        with _import_lock:
            _import_zustand.update(
                status="fertig",
                gruppe=None,
                zeilen=ergebnis["zeilen"],
                instrumente=ergebnis["instrumente"],
                beendet_am=db.utc_now_iso(),
            )


@router.post("/planspiel/universum/import")
async def post_planspiel_universum_import() -> dict[str, Any]:
    """Startet den Universum-Import als Hintergrundjob (H1/E093).

    Pipeline: ``tabs_lesen`` → je Gruppe ``markt_tab`` → ``psb_repo.upsert``.
    Dedup: läuft bereits ein Import, wird der Request mit 409
    ``import_laeuft`` abgewiesen (Fortschritt via GET-Endpunkt beobachtbar).
    """
    with _import_lock:
        if _import_zustand["status"] == "laeuft":
            raise ApiError(
                409,
                "import_laeuft",
                "Ein Universum-Import läuft bereits — bitte das Ergebnis "
                "abwarten (aktuell Gruppe "
                f"{_import_zustand.get('gruppe') or '…'}).",
                hint="Der Fortschritt lässt sich mit GET "
                "/api/planspiel/universum/import beobachten.",
            )
        _import_zustand.update(
            status="laeuft",
            gruppe=None,
            fortschritt=0,
            gruppen_gesamt=0,
            zeilen=0,
            instrumente=None,
            fehler=None,
            gestartet_am=db.utc_now_iso(),
            beendet_am=None,
        )
    global _import_task
    _import_task = asyncio.create_task(_universum_import_lauf())
    return _import_status()


# ---------------------------------------------------------------------------
# Historien-Backfill (Welle H5, E097): Start/Pause + Fortschritt des resuma-
# blen Byteworx-Bulk-Laufs (5Y/1d, asyncio-Hintergrundjob — kein scheduler).
# Zustand/Mechanik leben in :mod:`quantlab.services.planspiel.backfill`; der
# Router bleibt dünn (Dedup → 409, Fachfehler deutsch, nie 5xx für Fachfälle).
# ---------------------------------------------------------------------------


@router.get("/planspiel/universum/historie")
def get_planspiel_universum_historie() -> dict[str, Any]:
    """Fortschritts-Snapshot des Bulk-Backfills + je-ISIN-Statistik (H5/E097).

    Netzfreier DB-Lesepfad (Buchhaltungstabelle ``psb_backfill_fortschritt``);
    der ehrliche Hinweis zur Laufzeit liegt im Vertrag: ``geschaetzt_rest_s``
    ist eine Untergrenze (nur Pacing) — ein Voll-Lauf dauert mehrere Minuten.
    """
    return backfill_service.snapshot()


@router.post("/planspiel/universum/historie")
async def post_planspiel_universum_historie(payload: dict[str, Any]) -> dict[str, Any]:
    """Steuerung des Bulk-Backfills: ``{aktion: "starten"|"pausieren"}`` (H5).

    ``starten`` startet (bzw. nimmt wieder auf — fertige ISINs werden
    übersprungen) und akzeptiert zusätzlich den 409-Dedup
    ``backfill_laeuft``; ein leeres Universum ist ein deutscher 400-Fachfehler
    mit Handlungshinweis (E072). ``pausieren`` ist idempotent (auch ohne Lauf
    200 mit dem Snapshot).
    """
    aktion = str(payload.get("aktion") or "")
    if aktion == "starten":
        if psb_repo.zaehler() == 0:
            raise ApiError(
                400,
                "universum_leer",
                "Es ist kein Instrument im lokalen Universum — der Historien-"
                "Backfill hat nichts zu laden.",
                hint="Universum importieren: POST /api/planspiel/universum/import "
                "(bzw. Agent-Tool planspiel_universum, aktion=importiere).",
            )
        try:
            return await backfill_service.starte_backfill()
        except backfill_service.BackfillLaeuftBereits as exc:
            raise ApiError(
                409,
                "backfill_laeuft",
                f"{exc} Bitte den laufenden Backfill abwarten oder pausieren.",
                hint="Der Fortschritt lässt sich mit GET "
                "/api/planspiel/universum/historie beobachten.",
            ) from exc
    if aktion == "pausieren":
        return await backfill_service.pausiere_backfill()
    raise ApiError(
        400,
        "ungueltige_aktion",
        f'Unbekannte Aktion „{aktion}“ — erlaubt sind: starten, pausieren.',
        hint='Body-Format: {"aktion": "starten"} | {"aktion": "pausieren"}',
    )


# ---------------------------------------------------------------------------
# Kennzahlen (Welle H3, E095): VWD-Cache + frisch-Refresh (Referenz-Muster
# ``markt.kennzahlen`` — Fehler sind ein strukturiertes fehler-Objekt, nie 5xx)
# ---------------------------------------------------------------------------


def _kennzahlen_fehler(
    code: str, message: str, hint: str | None = None
) -> dict[str, Any]:
    """Fachfehler als Antwort-Objekt (E072-Konvention dieses Endpunkts)."""
    fehler: dict[str, Any] = {"code": code, "message": message}
    if hint:
        fehler["hint"] = hint
    return fehler


@router.get("/planspiel/kennzahlen")
def get_planspiel_kennzahlen(
    isin: str | None = None, sektion: str | None = None, frisch: bool = False
) -> dict[str, Any]:
    """VWD-Daten je ISIN (Kennzahlen, Kursdaten, Performance, Analysten, Profil).

    Pfad 1 (Default): Cache ``psb_kennzahlen`` — netzfrei, mit
    ``aktualisiert``-Map je Sektion. Pfad 2: ``frisch=true`` **oder** Cache-
    Miss → Live-Abruf über die idNotation-Brücke (``psb_instrumente.id_external``
    → ``instrument_mit_kursen`` → ``quoteList[0].idNotation``, Referenz
    ``markt.py``) + Upsert je Sektion. Fehlt die idNotation, kommt ein
    deutscher Fachfehler mit Lösungshinweis („Universum importieren“).
    """
    geprueft = (isin or "").strip().upper()
    if not ist_isin(geprueft):
        raise ApiError(
            400,
            "ungueltige_isin",
            "Die ISIN fehlt oder hat nicht die ISIN-Form (12 Zeichen, "
            'z. B. "DE0007164600").',
        )
    gefilterte_sektion = (sektion or "").strip() or None
    if gefilterte_sektion is not None and gefilterte_sektion not in vwd_service.VWD_SEKTIONEN:
        raise ApiError(
            400,
            "ungueltige_sektion",
            f'Unbekannte VWD-Sektion „{gefilterte_sektion}“ — erlaubt sind: '
            f"{', '.join(vwd_service.VWD_SEKTIONEN)}.",
        )

    erg: dict[str, Any] = {"ok": True, "isin": geprueft}
    gecacht = psb_repo.kennzahlen_lesen(geprueft, gefilterte_sektion)
    aktualisiert = psb_repo.kennzahlen_aktualisiert(geprueft)
    if not frisch and gecacht:
        for name, daten in gecacht.items():
            erg[name] = daten
        erg["aktualisiert"] = {
            name: zeit
            for name, zeit in aktualisiert.items()
            if name in gecacht
        }
        return erg

    instrument = psb_repo.instrument(geprueft)
    if instrument is None or not instrument.id_external:
        erg["ok"] = False
        erg["fehler"] = _kennzahlen_fehler(
            "keine_id_notation",
            f"Für {geprueft} liegt keine idNotation vor — das Instrument ist "
            "nicht im lokalen Universum (oder ohne Börsen-Referenz).",
            hint="Universum importieren: POST /api/planspiel/universum/import "
            "(bzw. Agent-Tool planspiel_universum, aktion=importiere).",
        )
        return erg
    try:
        detail = hole_client().instrument_mit_kursen(geprueft, instrument.id_external)
        quotes = detail.get("quoteList") or []
        id_notation = (
            quotes[0].get("idNotation")
            if quotes and isinstance(quotes[0], dict)
            else None
        )
    except PlanspielFehler as exc:
        erg["ok"] = False
        erg["fehler"] = _kennzahlen_fehler(exc.code, exc.message, exc.hint)
        return erg
    if not id_notation:
        erg["ok"] = False
        erg["fehler"] = _kennzahlen_fehler(
            "keine_id_notation",
            f"Die Planspiel-Börse liefert für {geprueft} keine idNotation — "
            "VWD-Kennzahlen sind ohne sie nicht abrufbar.",
            hint="Universum importieren und es später erneut versuchen.",
        )
        return erg
    try:
        daten = vwd_service.kennzahlen_laden(str(id_notation))
    except PlanspielFehler as exc:
        erg["ok"] = False
        erg["fehler"] = _kennzahlen_fehler(exc.code, exc.message, exc.hint)
        return erg
    for name, inhalt in daten.items():
        if gefilterte_sektion is not None and name != gefilterte_sektion:
            continue
        zeit = psb_repo.kennzahlen_speichern(geprueft, name, inhalt)
        erg[name] = inhalt
        aktualisiert[name] = zeit
    erg["aktualisiert"] = {
        name: zeit
        for name, zeit in aktualisiert.items()
        if name in erg  # nur mitgelieferte Sektionen melden (kompakt)
    }
    return erg


# ---------------------------------------------------------------------------
# Altbestand-Import (Welle H6, E098): planspiel.db → QuantLab — idempotenter
# Einmal-Import. Dedup mit bewusst EIGENEM Lock/Zustand (``_altbestand_lock``,
# NICHT ``_import_lock`` des Universum-Jobs): beide Imports sind unabhängig
# startbar; nur ein Zweitstart DESSELBEN Imports wird mit 409 abgewiesen.
# ---------------------------------------------------------------------------

_altbestand_lock = threading.Lock()
_altbestand_laeuft = False


@router.post("/planspiel/altbestand_import")
async def post_planspiel_altbestand_import(payload: dict[str, Any]) -> dict[str, Any]:
    """Importiert den Altbestand (planspiel.db) — Antwort ist der Counts-Report.

    Die Quelle (``pfad`` im Body) wird strikt read-only geöffnet und auf das
    erwartete Schema geprüft — Verstöße sind deutsche 400-Fachfehler
    (:class:`~quantlab.services.planspiel.altbestand.AltbestandFehler`), nie
    5xx (E072). Die Ausführung läuft im Thread (``asyncio.to_thread``, Muster
    universum/import); der Request wartet das Ergebnis ab und liefert den
    Report (``{instrumente_*, kurse_*, historie_*, kennzahlen_*, fehler,
    dauer_s}``). Dedup: läuft bereits ein Import, antwortet ein Zweitstart
    mit 409 ``import_laeuft`` (eigener Lock — Universum-Import und H5-Backfill
    bleiben daneben startbar).
    """
    global _altbestand_laeuft
    pfad = payload.get("pfad")
    if not isinstance(pfad, str) or not pfad.strip():
        raise ApiError(
            400,
            "ungueltiger_pfad",
            "Der Pfad zur planspiel.db fehlt oder ist leer.",
            hint='Body-Format: {"pfad": "/pfad/zu/planspiel.db"}',
        )
    with _altbestand_lock:
        if _altbestand_laeuft:
            raise ApiError(
                409,
                "import_laeuft",
                "Ein Altbestand-Import läuft bereits — bitte das Ergebnis "
                "abwarten.",
                hint="Der Import ist idempotent; ein erneuter Start nach dem "
                "Abschluss ist gefahrlos.",
            )
        _altbestand_laeuft = True
    try:
        return await asyncio.to_thread(altbestand_service.importiere_altbestand, pfad)
    except AltbestandFehler as exc:
        raise ApiError(400, exc.code, exc.message, hint=exc.hint) from exc
    finally:
        with _altbestand_lock:
            _altbestand_laeuft = False


# ---------------------------------------------------------------------------
# Wettbewerb-Dashboard (Welle H7, E101): vier read-only Blöcke (Top-Mover,
# Analysten-Upside, Positions-Monitor, Aktivitätszähler) — Service
# :mod:`quantlab.services.planspiel.wettbewerb`. E072-Statusmuster: Fachfälle
# (nicht konfiguriert, Netz) stehen als ``fehler``-Feld in der 200-Antwort,
# die netzfreien Blöcke (Top-Mover/Upside) bleiben damit nutzbar; nie 5xx.
# ---------------------------------------------------------------------------


@router.get("/planspiel/wettbewerb/ueberblick")
def get_planspiel_wettbewerb_ueberblick() -> dict[str, Any]:
    """Wettbewerbs-Übersicht: Top-Mover, Upside-Ranking, Positionen, Aktivität.

    Read-only — kein PSB-Schreibpfad. Blöcke 1–2 (Top-Mover, Upside) sind
    netzfrei aus Parquet/SQLite; Blöcke 3–4 konsumieren den Planspiel-Client
    (Standard-Depot, E070-Lesefallback) und sind bei Fachfällen ``null``.
    """
    return wettbewerb_service.ueberblick(hole_client())
