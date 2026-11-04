"""Planspiel-Service (Welle P, E068) — öffentliche API des Börsen-Clients.

- :class:`PlanspielClient` — sync httpx-Client (Whitelist-Guard, Pacing ≥ 1 s,
  Login-Single-Flight, AccessToken ausschließlich im Speicher).
- :class:`PlanspielFehler` — deutsche Fachfehler-Typologie (E069) mit fester
  HTTP-Mapping-Tabelle :data:`STATUS_NACH_CODE` für Router.
- :func:`hole_client` — Prozess-Singleton; der Token lebt **ausschließlich**
  in dieser Instanz im Speicher (nie DB/Settings/Datei/Log).
- :func:`anmeldeinfos_aus_settings` — Credentials-Lader nach E065/E071:
  E-Mail aus der Settings-Sektion ``planspiel``, Passwort über Keyring
  (Fallback-Datei) — wird nur zur Laufzeit gelesen, nie abgelegt.
"""

from __future__ import annotations

import threading

from quantlab.services.planspiel.client import (
    BASE_URL,
    NOTIZ_MAX_ZEICHEN,
    PACING_S,
    AnmeldeinfosLader,
    PlanspielClient,
)
from quantlab.services.planspiel.fehler import STATUS_NACH_CODE, PlanspielFehler

__all__ = [
    "BASE_URL",
    "NOTIZ_MAX_ZEICHEN",
    "PACING_S",
    "AnmeldeinfosLader",
    "PlanspielClient",
    "PlanspielFehler",
    "STATUS_NACH_CODE",
    "anmeldeinfos_aus_settings",
    "client_zuruecksetzen",
    "hole_client",
]

_singleton_lock = threading.Lock()
_singleton: PlanspielClient | None = None


def anmeldeinfos_aus_settings() -> tuple[str, str] | None:
    """Liest die hinterlegten Zugangsdaten (E065) — ``None``, wenn unvollständig.

    E-Mail: Settings-Sektion ``planspiel`` (kein Secret). Passwort:
    :func:`settings_repo.planspiel_passwort_lesen` (Keyring vor Fallback-Datei).
    """
    from quantlab.storage.repos import settings_repo  # lazy: DB erst bei Bedarf

    try:
        email = str(settings_repo.get_section("planspiel").get("email") or "")
    except ValueError:  # Sektion fehlt (alter Bestand ohne Migration der Defaults)
        return None
    passwort = settings_repo.planspiel_passwort_lesen()
    if not email or not passwort:
        return None
    return email, passwort


def hole_client() -> PlanspielClient:
    """Prozessweiter Client-Singleton (E068) — Token lebt nur hier im Speicher."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = PlanspielClient(anmeldeinfos=anmeldeinfos_aus_settings)
        return _singleton


def client_zuruecksetzen() -> None:
    """Setzt das Singleton zurück (für Tests) — verwirft Sitzung + Token."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.schliessen()
        _singleton = None
