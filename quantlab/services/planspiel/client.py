"""HTTP-Client für die Planspiel-Börse (Welle P, E068) — Whitelist, Pacing, Token nur im Speicher.

Der Client spricht ausschließlich die in RECHERCHE P.2/P.3 verifizierten
JSON-Endpunkte von ``trading.planspiel-boerse.de`` an (HTTP-First, ORCH A-P3):

- **Whitelist-Guard:** Es gibt **keinen** generischen öffentlichen Request-
  Eingang — nur die benannten, lesenden Fachmethoden (plus :meth:`PlanspielClient.
  order_platzieren` für P2b). Jeder Request — auch diese — läuft durch den
  internen :func:`_guard`, der die hart verbotene Menge (``resetPortfolio``,
  ``cancelOrder``, schreibende ``note``-Calls, ``watchlist``, ``ranking``,
  ``logout``) **ohne Senden** mit ``protokoll_verletzt`` abweist. Ein neuer
  Endpunkt erfordert bewusst eine Whitelist-Erweiterung mit Review.
- **Pacing:** Mindestens :data:`PACING_S` Sekunden zwischen zwei HTTP-Requests
  (RECHERCHE P.7.1 „Rate-Limits sind nicht dokumentiert“) — ``threading.Lock``
  + monotone Uhr, Uhr/Schlafer sind für Tests injizierbar.
- **Login-Single-Flight:** Gleichzeitige Aufrufe teilen sich **einen** Login
  (``threading.Lock``, Muster „Single-Flight“ aus quotes/scheduler); der
  Konsum läuft via ``asyncio.to_thread`` (Muster portfolio.py), deshalb
  bewusst Threads, keine asyncio-Locks.
- **Token nur im Speicher:** Das AccessToken (48 h, RECHERCHE P.2) ist
  ausschließlich ein Attribut der Client-Instanz — nie DB, Settings, Datei
  oder Log. Auf HTTP 401 folgt **genau ein** Re-Login, danach Fachfehler
  ``nicht_eingeloggt`` (kein Re-Login-Loop); es gibt keinen Refresh-Endpunkt.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from quantlab.services.planspiel.fehler import PlanspielFehler

#: API-Basis der Planspiel-Börse (RECHERCHE P.3: Base aller JSON-Endpunkte).
BASE_URL = "https://trading.planspiel-boerse.de/stockcontest/services/api"

#: Mindestabstand zwischen zwei HTTP-Requests in Sekunden (E068-Pacing).
PACING_S = 1.0

#: Request-Timeout (großzügig, aber begrenzt — netzwerk-Fehler ist ein Fachfall).
TIMEOUT_S = 15.0

#: Maximalänge des Order-Notiztexts (RECHERCHE P.5: Form-Validator ``maxLength(1e3)``).
NOTIZ_MAX_ZEICHEN = 1000

_PFAD_AUTHENTICATE = "/v-ms7/authenticate"
_PFAD_GET_USER = "/v-ms8/user/getUser"
_PFAD_GET_PORTFOLIO = "/v-ms1/portfolio/getPortfolio"
_PFAD_GET_PORTFOLIO_WITH_ITEMS = "/v-ms1/portfolio/getPortfolioWithItems"
_PFAD_GET_DEPOT_VALUES = "/v-ms1/chart/getDepotValues"
_PFAD_GET_ALLOCATION = "/v-ms1/allocation/getAllocation"
_PFAD_GET_ORDERS = "/v-ms1/order/getOrders"
_PFAD_GET_TRANSACTIONS = "/v-ms1/transaction/getTransactions"
_PFAD_SEARCH_INSTRUMENTS = "/v-ms2/instrument/getSearchInstrumentsPage"
_PFAD_INSTRUMENT_WITH_QUOTES = "/v-ms1/instrument/getInstrumentWithQuotes"
_PFAD_GET_FEE_MODEL = "/v-ms1/order/getFeeModel"
_PFAD_TRADING_HOURS = "/v-ms1/order/getTradingHours"
_PFAD_PUBLIC_HOLIDAYS = "/v-ms1/order/getPublicHolidays"
_PFAD_ALLOWED_SUBTYPES = "/v-ms1/order/allowed/subtypes"
_PFAD_CHECK_CAPITAL = "/v-ms1/order/check/capital"
_PFAD_PLACE_ORDER = "/v-ms3/order/placeOrder"
_PFAD_NOTE_LIST = "/v-ms1/note/list"
# Welle H1 (E093): Tab-Endpunkte des Aktien-Universums (RECHERCHE H.2 —
# ~18 Marktgruppen, ~800 Instrumente; beide lesend, GET).
_PFAD_TAB_LIST_KEY_VALUE = "/v-ms1/instrument/getTabListKeyValue"
_PFAD_INSTRUMENTS_TAB = "/v-ms1/instrument/getInstrumentsTab"

#: Hart verbotene Teilpfade (E068) — Treffer wird **ohne Senden** abgewiesen.
#: ``resetPortfolio`` (Depot-Reset, „GEFAHR“, RECHERCHE P.3) und ``cancelOrder``
#: (kostet 2,50 €) existieren im Client bewusst NICHT als Fachmethoden;
#: Watchlist/Ranking/Logout sind für QuantLab ohne Nutzen bzw. unerwünscht.
_VERBOTENE_TEILPFADE: frozenset[str] = frozenset(
    {"resetPortfolio", "cancelOrder", "watchlist", "ranking", "logout"}
)

#: Schreibende Endpunkte (POST/PUT/DELETE), die verboten sind — das **lesende**
#: ``note/list`` (E066-Verifikation der Agent-Marker-Notiz) bleibt erlaubt.
_VERBOTENE_SCHREIB_TEILPFADE: frozenset[str] = frozenset({"note"})

_SCHREIBMETHODEN: frozenset[str] = frozenset({"POST", "PUT", "DELETE"})

#: Liefert ``(benutzername_oder_email, passwort)`` — oder ``None``, wenn keine
#: Zugangsdaten hinterlegt sind (E065: E-Mail aus Settings, Passwort Keyring).
AnmeldeinfosLader = Callable[[], tuple[str, str] | None]

_ORDERTYPEN = ("BEST", "LIMIT", "STOP")
_ORDERTYPEN_ART = ("BUY", "SELL")


def _guard(pfad: str, methode: str) -> None:
    """Whitelist-Guard (E068): weist verbotene Endpunkte VOR dem Senden ab.

    Läuft als erstes in :meth:`PlanspielClient._anfrage` — jede Fachmethode
    (auch ``order_platzieren`` und ``notizen_lesen``) ist hierdurch abgesichert;
    Pfad-Konkatenationen aus Argumenten können die Verbots-Menge nicht umgehen,
    weil gegen Teilpfad-Fragmente geprüft wird, nicht gegen exakte URLs.
    """
    for teilpfad in _VERBOTENE_TEILPFADE:
        if teilpfad in pfad:
            raise PlanspielFehler(
                "protokoll_verletzt",
                f"Zugriff auf „{teilpfad}“ ist im Planspiel-Client dauerhaft gesperrt "
                "(Depot-Schutz bzw. nicht vorgesehener Endpunkt).",
                hint="Solche Aktionen sind bewusst nicht vorgesehen — bitte die "
                "Website der Planspiel-Börse nutzen.",
            )
    if methode.upper() in _SCHREIBMETHODEN:
        for teilpfad in _VERBOTENE_SCHREIB_TEILPFADE:
            if teilpfad in pfad:
                raise PlanspielFehler(
                    "protokoll_verletzt",
                    f"Schreibender Zugriff auf „{teilpfad}“ ist im Planspiel-Client "
                    "gesperrt (Notizen entstehen ausschließlich als Order-Notiz).",
                )


def _feld(objekt: Any, schluessel: str, kontext: str) -> Any:
    """Liest ein Pflichtfeld aus der Response — fehlt es, ist der Vertrag geändert (E069)."""
    if not isinstance(objekt, dict) or schluessel not in objekt:
        raise PlanspielFehler(
            "vertrag_geaendert",
            f"Unerwartete Antwort der Planspiel-Börse bei „{kontext}“: "
            f"Feld „{schluessel}“ fehlt.",
        )
    return objekt[schluessel]


def _liste(objekt: Any, schluessel: str, kontext: str) -> list[Any]:
    """Pflichtfeld, das eine Liste sein muss (``vertrag_geaendert`` sonst)."""
    wert = _feld(objekt, schluessel, kontext) if schluessel else objekt
    if not isinstance(wert, list):
        raise PlanspielFehler(
            "vertrag_geaendert",
            f"Unerwartete Antwort der Planspiel-Börse bei „{kontext}“: "
            f"Feld „{schluessel or 'Antwort'}“ ist keine Liste.",
        )
    return wert


#: Bekannte Order-Ablehnungsgruende (errorKey der Live-API, P4-Befund
#: 2026-09-21: Wrapper {success, httpStatus, errorKey, errorDetails}) mit
#: deutschen Deutungen — Erweiterung nach weiteren Live-Befunden.
_ABLEHNUNGS_DEUTUNGEN: dict[str, str] = {
    "TRADING_PERIOD_NOT_ACTIVE": (
        "Der Handel ist gerade geschlossen (Handelszeiten 08:00–22:00) — "
        "die Order bitte während der Handelszeiten erneut platzieren."
    ),
    "POSITION_LIMIT": (
        "Das Wertpapier würde die Grenze von 20 % des Depotwerts "
        "überschreiten — Menge reduzieren oder anderes Wertpapier wählen."
    ),
    "ORDER_LIMIT": (
        "Für dieses Wertpapier ist das Tageslimit von 5 Orders erreicht — "
        "morgen wieder versuchen."
    ),
}


def _order_ablehnungsmeldung(antwort: httpx.Response) -> str:
    """Deutsche Ablehnungsmeldung inkl. Server-Grund (errorKey/Fehler-Body).

    Die Live-API wrappernt Order-Ablehnungen als
    ``{success: false, httpStatus, errorKey, errorDetails}`` — ``errorKey``
    wird gedeutet (bekannte Keys), sonst roh genannt; ohne parsbaren Body
    genügt der Status.
    """
    grund = ""
    try:
        body = antwort.json()
        if isinstance(body, dict):
            grund = str(body.get("errorKey") or body.get("validationErrorCode") or "")
    except ValueError:
        grund = ""
    if grund in _ABLEHNUNGS_DEUTUNGEN:
        return (
            f"Die Order wurde von der Planspiel-Börse abgelehnt: "
            f"{_ABLEHNUNGS_DEUTUNGEN[grund]}"
        )
    if grund:
        return (
            f"Die Order wurde von der Planspiel-Börse abgelehnt "
            f"(HTTP {antwort.status_code}, Grund: {grund})."
        )
    return (
        f"Die Order wurde von der Planspiel-Börse abgelehnt "
        f"(HTTP {antwort.status_code})."
    )


def _order_body(
    *,
    isin: str,
    order_type: str,
    menge: int,
    ordertyp: str,
    id_exchange: str | int,
    portfolio_id: int,
    valid_until: str,
    notiz: str,
    limit: float | None,
    stop: float | None,
) -> dict[str, Any]:
    """Baut den placeOrder-Body exakt nach RECHERCHE P.5 (Bundle-Ableitung).

    ``limit`` nur bei ``orderSubType="LIMIT"``, ``stop`` nur bei ``"STOP"``
    (je auf 2 Nachkommastellen gerundet), ``stopOrLimit`` bleibt als Artifakt
    ``null`` im Body. ``noteText`` wird hart auf :data:`NOTIZ_MAX_ZEICHEN`
    gekappt — der E074-Marker-Prefix bleibt Sache des Tools (P2b), hier ist
    die Kappung nur letzte Verteidigungslinie.
    """
    if ordertyp not in _ORDERTYPEN:
        raise PlanspielFehler(
            "order_fehler",
            f"Unbekannter Ordertyp „{ordertyp}“ — erlaubt sind: "
            f"{', '.join(_ORDERTYPEN)}.",
        )
    if order_type not in _ORDERTYPEN_ART:
        raise PlanspielFehler(
            "order_fehler",
            f"Unbekannte Order-Art „{order_type}“ — erlaubt sind: "
            f"{', '.join(_ORDERTYPEN_ART)}.",
        )
    body: dict[str, Any] = {
        "isin": isin,
        "orderSubType": ordertyp,
        "orderType": order_type,
        "portfolioId": int(portfolio_id),
        "quantity": int(menge),
        "validUntil": valid_until,
        "idExchange": str(id_exchange),
        "noteText": (notiz or "")[:NOTIZ_MAX_ZEICHEN],
        "stopOrLimit": None,
    }
    if ordertyp == "LIMIT":
        if limit is None:
            raise PlanspielFehler(
                "order_fehler",
                "Ordertyp „LIMIT“ benötigt ein Limit — bitte Limit angeben.",
            )
        body["limit"] = round(float(limit), 2)
    if ordertyp == "STOP":
        if stop is None:
            raise PlanspielFehler(
                "order_fehler",
                "Ordertyp „STOP“ benötigt eine Stop-Schwelle — bitte Stop angeben.",
            )
        body["stop"] = round(float(stop), 2)
    return body


class PlanspielClient:
    """Sync HTTP-Client für die Planspiel-Börse (E068).

    Instanzen halten den Zustand (Cookie-Jar für LB-Stickiness, AccessToken)
    **ausschließlich im Speicher**. Für Tests sind ``transport`` (httpx.
    MockTransport, E076), ``pacing_s``, Uhr und Schlafer injizierbar;
    ``anmeldeinfos`` liefert die Zugangsdaten zur Laufzeit (E065 — der Client
    speichert sie selbst nie ab und protokolliert sie nie).
    """

    def __init__(
        self,
        *,
        basis_url: str = BASE_URL,
        transport: httpx.BaseTransport | None = None,
        pacing_s: float = PACING_S,
        timeout_s: float = TIMEOUT_S,
        anmeldeinfos: AnmeldeinfosLader | None = None,
        uhr: Callable[[], float] = time.monotonic,
        schlafer: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = httpx.Client(
            base_url=basis_url, transport=transport, timeout=timeout_s
        )
        self._pacing_s = pacing_s
        self._uhr = uhr
        self._schlafer = schlafer
        self._pacing_lock = threading.Lock()
        self._letzter_request: float | None = None
        self._login_lock = threading.Lock()
        self._token: str | None = None
        self._anmeldeinfos = anmeldeinfos

    # -- Sitzung (Token nur im Speicher) ----------------------------------

    @property
    def eingeloggt(self) -> bool:
        """``True``, solange ein AccessToken im Speicher gehalten wird."""
        return self._token is not None

    def token_verwerfen(self) -> None:
        """Wirft die Sitzung lokal weg (KEIN ``logout`` — der ist verboten, E068)."""
        with self._login_lock:
            self._token = None

    def schliessen(self) -> None:
        """Schließt die zugrundeliegende HTTP-Verbindung (Sitzung geht verloren)."""
        self._http.close()

    def authentifizieren(self) -> dict[str, Any]:
        """Führt einen **neuen** Login aus (alte Sitzung wird verworfen, E068).

        Liefert ``result.user`` der authenticate-Antwort (RECHERCHE P.2:
        ``userPortfolios``, ``teamInfo``, ``contest``). Fehler:
        ``nicht_konfiguriert``/``login_fehlgeschlagen``/``rate_limit``/
        ``vertrag_geaendert``/``netzwerk``.
        """
        with self._login_lock:
            self._token = None
            return self._einloggen_unter_lock()

    def _einloggen_unter_lock(self) -> dict[str, Any]:
        """Login — Login-Lock MUSS gehalten werden (Single-Flight, E068)."""
        benutzername, passwort = self._anmeldeinfos_beschaffen()
        # E068-Konformität (QA-P1): AUCH der Login läuft formal durch den Guard.
        # Der fixe Authenticate-Pfad ist erlaubt (trifft keine verbotene Menge) —
        # reiner Konformitätsgewinn, Pacing/Single-Flight bleiben unverändert.
        _guard(_PFAD_AUTHENTICATE, "POST")
        antwort = self._senden(
            "POST",
            _PFAD_AUTHENTICATE,
            json_body={"userName": benutzername, "password": passwort},
        )
        if antwort.status_code in (429, 503):
            raise PlanspielFehler(
                "rate_limit",
                "Die Planspiel-Börse bittet gerade um Entlastung — bitte kurz "
                "warten und es erneut versuchen.",
            )
        if antwort.status_code != 200:
            raise PlanspielFehler(
                "login_fehlgeschlagen",
                "Anmeldung bei der Planspiel-Börse fehlgeschlagen — bitte E-Mail "
                "und Passwort in den Einstellungen prüfen.",
                hint="Einstellungen → Planspiel",
            )
        daten = self._json(antwort, "authenticate")
        if not isinstance(daten, dict) or daten.get("success") is not True:
            raise PlanspielFehler(
                "login_fehlgeschlagen",
                "Anmeldung bei der Planspiel-Börse wurde abgelehnt — E-Mail oder "
                "Passwort ist nicht gültig.",
                hint="Einstellungen → Planspiel",
            )
        ergebnis = _feld(daten, "result", "authenticate")
        token_objekt = _feld(ergebnis, "token", "authenticate")
        access_token = _feld(token_objekt, "accessToken", "authenticate")
        if not isinstance(access_token, str) or not access_token:
            raise PlanspielFehler(
                "vertrag_geaendert",
                "Unerwartete Antwort der Planspiel-Börse bei „authenticate“: "
                "Feld „accessToken“ ist leer.",
            )
        benutzer = _feld(ergebnis, "user", "authenticate")
        self._token = access_token
        return benutzer  # type: ignore[no-any-return]

    def _anmeldeinfos_beschaffen(self) -> tuple[str, str]:
        """Zugangsdaten über den injizierten Lader — ohne → ``nicht_konfiguriert``."""
        geladen = self._anmeldeinfos() if self._anmeldeinfos is not None else None
        if geladen is None:
            raise PlanspielFehler(
                "nicht_konfiguriert",
                "Es sind noch keine Zugangsdaten für die Planspiel-Börse "
                "hinterlegt. Bitte E-Mail und Passwort in den Einstellungen "
                "speichern.",
                hint="Einstellungen → Planspiel",
            )
        benutzername, passwort = geladen
        if not benutzername or not passwort:
            raise PlanspielFehler(
                "nicht_konfiguriert",
                "Die Zugangsdaten für die Planspiel-Börse sind unvollständig — "
                "bitte E-Mail und Passwort in den Einstellungen prüfen.",
                hint="Einstellungen → Planspiel",
            )
        return benutzername, passwort

    def _token_beschaffen(self) -> str:
        """Token aus dem Speicher; bei Bedarf EIN Login (Single-Flight, E068)."""
        with self._login_lock:
            if self._token is None:
                self._einloggen_unter_lock()
            assert self._token is not None  # Login setzt das Token zwingend
            return self._token

    def _neu_anmelden_nach(self, gescheitertes_token: str) -> str:
        """Genau EIN Re-Login nach 401 (E068) — Single-Flight über den Login-Lock.

        Läuft zwischenzeitlich bereits ein frischer Login eines anderen Threads,
        wird dessen Token übernommen statt ein zweites Mal anzumelden.
        """
        with self._login_lock:
            if self._token == gescheitertes_token:
                self._token = None
            if self._token is None:
                self._einloggen_unter_lock()
            assert self._token is not None
            return self._token

    # -- Transport (Guard → Pacing → Senden) ------------------------------

    def _anfrage(
        self,
        methode: str,
        pfad: str,
        *,
        kontext: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        order_endpunkt: bool = False,
    ) -> Any:
        """Interner **einziger** Request-Weg (E068): Guard → Token → Senden.

        401 → genau EIN Re-Login + ein Wiederholungsversuch; danach
        ``nicht_eingeloggt`` (kein Loop). 429/503 → ``rate_limit``; sonstige
        Nicht-2xx → ``vertrag_geaendert`` (bzw. ``order_fehler`` auf dem
        Order-Endpunkt). Liefert die geparste JSON-Antwort.
        """
        _guard(pfad, methode)
        token = self._token_beschaffen()
        antwort = self._senden(methode, pfad, token=token, params=params, json_body=json_body)
        if antwort.status_code == 401:
            token = self._neu_anmelden_nach(token)
            antwort = self._senden(
                methode, pfad, token=token, params=params, json_body=json_body
            )
            if antwort.status_code == 401:
                with self._login_lock:
                    if self._token == token:
                        self._token = None
                raise PlanspielFehler(
                    "nicht_eingeloggt",
                    "Die Anmeldung bei der Planspiel-Börse ist trotz erneutem "
                    "Login ungültig — bitte die Verbindung in den Einstellungen "
                    "testen.",
                    hint="Einstellungen → Planspiel",
                )
        if antwort.status_code in (429, 503):
            raise PlanspielFehler(
                "rate_limit",
                "Die Planspiel-Börse bittet gerade um Entlastung — bitte kurz "
                "warten und es erneut versuchen.",
            )
        if not 200 <= antwort.status_code < 300:
            if order_endpunkt:
                raise PlanspielFehler(
                    "order_fehler",
                    _order_ablehnungsmeldung(antwort),
                )
            raise PlanspielFehler(
                "vertrag_geaendert",
                f"Unerwartete HTTP-Antwort der Planspiel-Börse bei „{kontext}“ "
                f"(Status {antwort.status_code}).",
            )
        return self._json(antwort, kontext)

    def _senden(
        self,
        methode: str,
        pfad: str,
        *,
        token: str | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Pacing (E068) + Auth-Header + Senden; Netzstörungen → ``netzwerk``."""
        self._pacing_einhalten()
        header = {"Accept": "application/json", "Accept-Language": "de"}
        if token:
            header["Authorization"] = f"Bearer {token}"
        try:
            return self._http.request(
                methode, pfad, params=params, json=json_body, headers=header
            )
        except httpx.HTTPError as exc:
            raise PlanspielFehler(
                "netzwerk",
                "Die Planspiel-Börse ist gerade nicht erreichbar "
                f"(Netzwerkfehler: {type(exc).__name__}). Bitte Verbindung prüfen.",
            ) from exc

    def _pacing_einhalten(self) -> None:
        """Blockt, bis seit dem letzten Request min. ``pacing_s`` vergangen ist (E068)."""
        with self._pacing_lock:
            if self._letzter_request is not None:
                verbleibend = self._pacing_s - (self._uhr() - self._letzter_request)
                if verbleibend > 0:
                    self._schlafer(verbleibend)
            self._letzter_request = self._uhr()

    def _json(self, antwort: httpx.Response, kontext: str) -> Any:
        """Response-Body parsen — kein JSON → ``vertrag_geaendert`` (E069)."""
        try:
            return antwort.json()
        except ValueError as exc:
            raise PlanspielFehler(
                "vertrag_geaendert",
                f"Unerwartete Antwort der Planspiel-Börse bei „{kontext}“: "
                "kein gültiges JSON.",
            ) from exc

    # -- Benannte Fachmethoden (Whitelist, RECHERCHE P.2/P.3) --------------

    def user_holen(self) -> dict[str, Any]:
        """Aktueller User + Contest + Depots + Team (``getUser``).

        Pflichtfeld ``userPortfolios`` (Liste der Depots) wird gegen
        Vertragsänderungen geprüft (E069).
        """
        daten = self._anfrage("GET", _PFAD_GET_USER, kontext="getUser")
        _liste(daten, "userPortfolios", "getUser")
        return daten  # type: ignore[no-any-return]

    def portfolio_holen(self, portfolio_id: int) -> dict[str, Any]:
        """Portfoliowert + Barbestand (``getPortfolio``, Dashboard-Kennzahlen)."""
        daten = self._anfrage(
            "GET",
            _PFAD_GET_PORTFOLIO,
            kontext="getPortfolio",
            params={"portfolioId": portfolio_id},
        )
        _feld(daten, "value", "getPortfolio")
        _feld(daten, "credit", "getPortfolio")
        return daten  # type: ignore[no-any-return]

    def positionen_holen(self, portfolio_id: int) -> dict[str, Any]:
        """Portfolio inkl. Positionen (``getPortfolioWithItems``, ``withItems=true``).

        Leeres Depot liefert ``items: null`` statt ``[]`` (P4-Live-Befund
        2026-09-21, Depot 728439) — ``None`` wird hier vertragsgerecht als
        leere Positionenliste interpretiert; ein anderer Nicht-Listen-Typ
        bleibt ein ``vertrag_geaendert``-Fachfehler.
        """
        daten = self._anfrage(
            "GET",
            _PFAD_GET_PORTFOLIO_WITH_ITEMS,
            kontext="getPortfolioWithItems",
            params={"portfolioId": portfolio_id, "withItems": "true"},
        )
        _feld(daten, "portfolio", "getPortfolioWithItems")
        if daten.get("items") is None:
            daten["items"] = []
        else:
            _liste(daten, "items", "getPortfolioWithItems")
        return daten  # type: ignore[no-any-return]

    def depotwerte_holen(self, portfolio_id: int) -> list[Any]:
        """Depotwert-Historie (``getDepotValues``) — Liste der Zeitreihe."""
        return _liste(
            self._anfrage(
                "GET",
                _PFAD_GET_DEPOT_VALUES,
                kontext="getDepotValues",
                params={"portfolioId": portfolio_id},
            ),
            "",
            "getDepotValues",
        )

    def allokation_holen(self, portfolio_id: int) -> dict[str, Any]:
        """Branchen-Allokation inkl. Cash-Anteil (``getAllocation``)."""
        daten = self._anfrage(
            "GET",
            _PFAD_GET_ALLOCATION,
            kontext="getAllocation",
            params={"portfolioId": portfolio_id},
        )
        _liste(daten, "branches", "getAllocation")
        return daten  # type: ignore[no-any-return]

    def orders_holen(self, portfolio_id: int, status: str | None = None) -> list[Any]:
        """Order-/Trade-Liste (``getOrders``) — ``status`` z. B. ``EXECUTED``.

        ANNAHME (RECHERCHE P.7.7): Das Status-Enum ist nur teilweise
        verifiziert — unbekannte Status-Werte werden unverändert durchgereicht.
        """
        params: dict[str, Any] = {"portfolioId": portfolio_id}
        if status:
            params["status"] = status
        return _liste(
            self._anfrage(
                "GET", _PFAD_GET_ORDERS, kontext="getOrders", params=params
            ),
            "",
            "getOrders",
        )

    def transaktionen_holen(self, portfolio_id: int, isin: str | None = None) -> list[Any]:
        """Abgerechnete Transaktionen (``getTransactions``)."""
        params: dict[str, Any] = {"portfolioId": portfolio_id}
        if isin:
            params["isin"] = isin
        return _liste(
            self._anfrage(
                "GET",
                _PFAD_GET_TRANSACTIONS,
                kontext="getTransactions",
                params=params,
            ),
            "",
            "getTransactions",
        )

    def instrument_suchen(self, name: str, seite: int = 0, groesse: int = 5) -> list[Any]:
        """Instrumentensuche (``getSearchInstrumentsPage``).

        Die API antwortet mit einem Paginierungs-Objekt
        ``{totalPages, totalElements, page, size, content: [...]}``
        (P4-Live-Befund 2026-09-21) — nur ``content`` ist die Trefferliste;
        fehlt es oder ist ``null`` → leere Liste, ein anderer Typ bleibt
        ``vertrag_geaendert``.
        """
        daten = self._anfrage(
            "GET",
            _PFAD_SEARCH_INSTRUMENTS,
            kontext="getSearchInstrumentsPage",
            params={
                "instrumentName": name,
                "page": seite,
                "size": groesse,
            },
        )
        if isinstance(daten, list):
            return daten
        if not isinstance(daten, dict):
            raise PlanspielFehler(
                "vertrag_geaendert",
                "Unerwartete Antwort der Planspiel-Börse bei "
                "„getSearchInstrumentsPage“: Antwort ist kein Objekt.",
            )
        inhalt = daten.get("content")
        if inhalt is None:
            return []
        return _liste(inhalt, "", "getSearchInstrumentsPage")

    def instrument_mit_kursen(self, isin: str, id_external: str | int) -> dict[str, Any]:
        """Instrument-Details + Kurse (``getInstrumentWithQuotes``)."""
        daten = self._anfrage(
            "GET",
            _PFAD_INSTRUMENT_WITH_QUOTES,
            kontext="getInstrumentWithQuotes",
            params={"isin": isin, "idExternal": id_external},
        )
        _liste(daten, "quoteList", "getInstrumentWithQuotes")
        return daten  # type: ignore[no-any-return]

    def gebuehren_modell(self, isin: str) -> dict[str, Any]:
        """Gebührenmodell je Instrument (``getFeeModel``)."""
        daten = self._anfrage(
            "GET",
            _PFAD_GET_FEE_MODEL,
            kontext="getFeeModel",
            params={"isin": isin},
        )
        _feld(daten, "minimum", "getFeeModel")
        _feld(daten, "percentage", "getFeeModel")
        return daten  # type: ignore[no-any-return]

    def handelszeiten_holen(self) -> list[Any]:
        """Handelsfenster je Instrumenttyp/Börse (``getTradingHours``)."""
        return _liste(
            self._anfrage(
                "GET", _PFAD_TRADING_HOURS, kontext="getTradingHours"
            ),
            "",
            "getTradingHours",
        )

    def feiertage_holen(self) -> dict[str, Any]:
        """Feiertage + Wochenend-Flag (``getPublicHolidays``)."""
        daten = self._anfrage(
            "GET", _PFAD_PUBLIC_HOLIDAYS, kontext="getPublicHolidays"
        )
        _liste(daten, "publicHolidays", "getPublicHolidays")
        return daten  # type: ignore[no-any-return]

    def ordertypen_holen(self) -> list[Any]:
        """Erlaubte Ordertypen (``allowed/subtypes``)."""
        return _liste(
            self._anfrage(
                "GET", _PFAD_ALLOWED_SUBTYPES, kontext="allowed/subtypes"
            ),
            "",
            "allowed/subtypes",
        )

    def kapital_check(self, portfolio_id: int) -> dict[str, Any]:
        """Kapital-Check vor einer Order (``order/check/capital``)."""
        daten = self._anfrage(
            "GET",
            _PFAD_CHECK_CAPITAL,
            kontext="order/check/capital",
            params={"portfolioId": portfolio_id},
        )
        _feld(daten, "amount", "order/check/capital")
        return daten  # type: ignore[no-any-return]

    def order_platzieren(
        self,
        *,
        isin: str,
        order_type: str,
        menge: int,
        ordertyp: str,
        id_exchange: str | int,
        portfolio_id: int,
        valid_until: str,
        notiz: str = "",
        limit: float | None = None,
        stop: float | None = None,
    ) -> dict[str, Any]:
        """Order anlegen (``placeOrder``, RECHERCHE P.5 — erste echte Sendung: §P4).

        Body-Aufbau siehe :func:`_order_body`. Setzt die Börse
        ``validationErrorCode`` in der Antwort, wird ``order_abgelehnt``
        geworfen (E069) — Meldung nennt den Code.
        """
        body = _order_body(
            isin=isin,
            order_type=order_type,
            menge=menge,
            ordertyp=ordertyp,
            id_exchange=id_exchange,
            portfolio_id=portfolio_id,
            valid_until=valid_until,
            notiz=notiz,
            limit=limit,
            stop=stop,
        )
        daten = self._anfrage(
            "POST",
            _PFAD_PLACE_ORDER,
            kontext="placeOrder",
            json_body=body,
            order_endpunkt=True,
        )
        if not isinstance(daten, dict):
            raise PlanspielFehler(
                "vertrag_geaendert",
                "Unerwartete Antwort der Planspiel-Börse bei „placeOrder“: "
                "kein Order-Objekt.",
            )
        fehler_code = daten.get("validationErrorCode")
        if fehler_code:
            raise PlanspielFehler(
                "order_abgelehnt",
                f"Die Börse hat die Order abgelehnt (Grund: {fehler_code}). Bitte "
                "Spielregeln beachten: max. 20 % des Depotwerts je Wertpapier, "
                "max. 5 Orders je Tag und Wertpapier, Mindestkurs 1,00 €.",
            )
        return daten

    def notizen_lesen(self) -> list[Any]:
        """Trade-Notizen (``note/list``, GET — lesend; Verifikation E066-Marker)."""
        return _liste(
            self._anfrage("GET", _PFAD_NOTE_LIST, kontext="note/list"),
            "",
            "note/list",
        )

    def tabs_lesen(self) -> list[dict[str, Any]]:
        """Marktgruppen des Aktien-Universums (``getTabListKeyValue``, H1/E093).

        Die API liefert ``{key, value}``-Zeilen (RECHERCHE H.2: ~18 Gruppen,
        z. B. DAX, MDAX, ETC); Einträge ohne ``key`` werden tolerant
        gefiltert (Muster Referenz-Client ``tabs_lesen``). Keine Liste →
        ``vertrag_geaendert`` (E069, via :func:`_liste`).
        """
        zeilen = _liste(
            self._anfrage(
                "GET", _PFAD_TAB_LIST_KEY_VALUE, kontext="getTabListKeyValue"
            ),
            "",
            "getTabListKeyValue",
        )
        return [z for z in zeilen if isinstance(z, dict) and "key" in z]

    def markt_tab(self, gruppe: str) -> list[dict[str, Any]]:
        """Instrumente einer Marktgruppe (``getInstrumentsTab``, H1/E093).

        ``gruppe`` ist der ``key`` aus :meth:`tabs_lesen`; die Antwort-Zeilen
        tragen ``isin``/``wkn``/``name``/``instrumentType``/``branch``/
        ``idExternal``/``isoCurrency`` plus Kursfelder. Zeilen ohne ``isin``
        werden tolerant gefiltert (Muster Referenz-Client ``markt_tab``);
        keine Liste → ``vertrag_geaendert``.
        """
        zeilen = _liste(
            self._anfrage(
                "GET",
                _PFAD_INSTRUMENTS_TAB,
                kontext="getInstrumentsTab",
                params={"tabName": gruppe},
            ),
            "",
            "getInstrumentsTab",
        )
        return [z for z in zeilen if isinstance(z, dict) and z.get("isin")]
