"""Universum-Import (Welle H1, E093): Planspiel-Marktgruppen → ``psb_instrumente``.

Pipeline (RECHERCHE H.2): ``tabs_lesen`` (~18 Marktgruppen) → je Gruppe
``markt_tab`` → :func:`psb_repo.upsert` (Gruppen-Union bei Mehrfach-
Zugehörigkeit). Der Client Pacingt selbst (≥ 1 s je Request, E068) — ein
vollständiger Lauf dauert daher real ~20 s; Router (Hintergrundjob) und
Agent-Tool führen die Funktion deshalb ausschließlich im Thread aus
(``asyncio.to_thread``). Die Funktion ist synchron und client-injiziert —
netzfreie Tests fahren sie mit Fakes/MockTransport.

Konsumiert ausschließlich :mod:`quantlab.services.planspiel` + ``psb_repo``
(keine Router-Imports, kein eigenes httpx — Schichtungsvertrag der Tools).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quantlab.storage.repos import psb_repo

#: Fortschritts-Hook: ``(gruppe, index_1basiert, gesamt)`` — je begonnener
#: Gruppe; ein wirfender Hook bricht den Import ab — der Router-Job
#: garantiert dennoch den Endzustand (``import_fehlgeschlagen``).
Fortschritt = Callable[[str, int, int], None]

#: Best-effort-Auflöser ISIN → Yahoo-Kürzel (E093-Brücke); ``None`` = keine
#: Auflösung (das Repo behält dann einen gepflegten Wert per COALESCE).
KuerzelAufloeser = Callable[[str], str | None]


def _text(wert: Any) -> str | None:
    """API-Wert → getrimmter Text oder ``None`` (leere Strings sind „fehlt“)."""
    if wert is None:
        return None
    text = str(wert).strip()
    return text or None


def importiere_universum(
    client: Any,
    *,
    kuerzel_aufloeser: KuerzelAufloeser | None = None,
    fortschritt: Fortschritt | None = None,
) -> dict[str, int]:
    """Fährt die vollständige Import-Pipeline aus (synchron, E093).

    Liefert ``{"gruppen": <Anzahl Gruppen>, "zeilen": <verarbeitete Tab-
    Zeilen>, "instrumente": <Gesamtzähler der Tabelle>}`` — Mehrfach-
    Zugehörigkeit zählt je Zeile (Union liegt im Repo). Ein
    :class:`~quantlab.services.planspiel.PlanspielFehler` aus dem Client
    bricht ab; bereits geschriebene Zeilen bleiben (Upserts sind inkrementell).
    """
    tabs = client.tabs_lesen()
    gruppen = [
        str(tab["key"]) for tab in tabs if isinstance(tab, dict) and tab.get("key")
    ]
    gruppen = list(dict.fromkeys(gruppen))  # Dedup, Reihenfolge der API bleibt
    zeilen = 0
    for index, gruppe in enumerate(gruppen, start=1):
        if fortschritt is not None:
            fortschritt(gruppe, index, len(gruppen))
        for eintrag in client.markt_tab(gruppe):
            isin = _text(eintrag.get("isin"))
            if isin is None:
                continue  # markt_tab filtert bereits; Doppelsicherung
            psb_repo.upsert(
                isin=isin,
                wkn=_text(eintrag.get("wkn")),
                name=_text(eintrag.get("name")),
                typ=_text(eintrag.get("instrumentType")),
                branchen=_text(eintrag.get("branch")),
                gruppe=gruppe,
                id_external=_text(eintrag.get("idExternal")),
                waehrung=_text(eintrag.get("isoCurrency")),
                yahoo_kuerzel=(
                    kuerzel_aufloeser(isin)
                    if kuerzel_aufloeser is not None
                    else None
                ),
            )
            zeilen += 1
    return {
        "gruppen": len(gruppen),
        "zeilen": zeilen,
        "instrumente": psb_repo.zaehler(),
    }
