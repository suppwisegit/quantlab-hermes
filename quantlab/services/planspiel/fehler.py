"""Fachfehler-Typologie des Planspiel-Clients (Welle P, E069).

Eine geschlossene Liste deutscher Fachcodes — damit sehen Agent/UI nie ein
„500 intern", wo ein Fachfehler vorliegt. Die Tabelle :data:`STATUS_NACH_CODE`
ist die **feste** HTTP-Mapping-Vorgabe für Router (Haus-Fehlerformat
``{detail: {code, message, hint?}}``); der Code ist stabil und
maschinenlesbar, ``message`` immer deutsch (E001).
"""

from __future__ import annotations

from quantlab.errors import ApiError

#: Feste HTTP-Mapping-Tabelle (E069): Fachcode → HTTP-Status für Router.
STATUS_NACH_CODE: dict[str, int] = {
    "nicht_konfiguriert": 409,  # Zugangsdaten fehlen (E-Mail/Passwort nicht hinterlegt)
    "login_fehlgeschlagen": 401,  # 401/„success:false“ auf authenticate
    "nicht_eingeloggt": 401,  # 401 auch nach dem einen Re-Login (E068)
    "vertrag_geaendert": 502,  # Response-Shape weicht RECHERCHE P.2/P.3 ab
    "netzwerk": 503,  # Timeout/ConnectError — Quelle nicht erreichbar
    "rate_limit": 429,  # HTTP 429/503 der Börse — Entlastung erzwingen
    "order_abgelehnt": 422,  # validationErrorCode in der placeOrder-Antwort
    "order_fehler": 422,  # sonstiger 4xx auf placeOrder
    "protokoll_verletzt": 500,  # E068-Guard: verbotener Endpunkt angefragt
}


class PlanspielFehler(Exception):
    """Fachlicher Fehler der Planspiel-Domäne (E069).

    ``code`` ist einer der geschlossenen Codes aus :data:`STATUS_NACH_CODE`,
    ``message`` deutsch und nutzerlesbar, ``hint`` optional ein Handungshinweis
    (z. B. „Einstellungen → Planspiel“). ``status_code`` liefert den Router-
    Status gemäß fester Mapping-Tabelle — analog :class:`~quantlab.providers.base.ProviderError`.
    """

    def __init__(self, code: str, message: str, hint: str | None = None) -> None:
        if code not in STATUS_NACH_CODE:
            raise ValueError(f"Unbekannter Planspiel-Fehlercode: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.status_code = STATUS_NACH_CODE[code]

    def to_api_error(self) -> ApiError:
        """Überführt den Fehler in den ApiError-Vertrag (C.1) — Felder unverändert."""
        return ApiError(self.status_code, self.code, self.message, hint=self.hint)
