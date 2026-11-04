"""Wettbewerb-Dashboard-Service (Welle H, E101) — read-only Übersichtsblöcke
für ``GET /api/planspiel/wettbewerb/ueberblick``.

Vier Blöcke (RECHERCHE H.6/H.3; Regeln aus :mod:`quantlab.services.planspiel.
regeln`):

1. **Top-Mover** — letzte 2 Tages-Schlüsse je ISIN aus dem Parquet-Store
   (``1d``; Quellen-Mix ``psb_alt``/``byteworx``/``yahoo`` ist bewusst NORMAL,
   bar_meta-``source`` wird je Zeile mitgeliefert), Δ% je ISIN, Top 5 auf/ab.
2. **Analysten-Upside-Ranking** — ``psb_kennzahlen``
   (``getAnalystenempfehlungen.kursziel``) gegen den letzten 1d-Kurs
   (Upside als Bruchzahl, Konvention H3), Top 5.
3. **Positions-Monitor** — aktuelle Depot-Positionen über die bestehenden
   Client-Zugriffe (``getPortfolio``/``getPortfolioWithItems``, E070-Fallback)
   mit Gewicht % und 20 %-Warnung (:data:`regeln.MAX_DEPOTGEWICHT`).
4. **Aktivitätszähler** — abgerechnete Käufe im heutigen Saisonmonat
   (:func:`regeln.abgerechnete_kaeufe`) gegen :data:`regeln.MIN_KAEUFE`.

Blöcke 1–2 sind netzfrei (Parquet/SQLite); 3–4 konsumieren den Planspiel-
Client. Scheitert der Client (nicht konfiguriert/Netz), bleiben 1–2 lieferbar
und 3–4 sind ``None`` mit strukturiertem ``fehler``-Feld (E072: nie 5xx).
Alles rein lesend — kein PSB-Schreibpfad.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from quantlab import db
from quantlab.services.planspiel import regeln
from quantlab.services.planspiel.fehler import PlanspielFehler
from quantlab.storage import parquet_store
from quantlab.storage.repos import psb_repo

#: Top-N je Mover-Liste (auf/ab) und je Upside-Ranking.
_TOP_N = 5

#: Sektion der Analysten-Snippets im VWD-Cache (Migration 8).
_SEKTION_ANALYSTEN = "getAnalystenempfehlungen"


def _jetzt() -> datetime:
    """Injizierbare Uhr (Tests) — aktuelle UTC-Zeit."""
    return datetime.now(UTC)


def _depot_id_ermitteln(client: Any) -> int:
    """E070-Lesefallback: ``standard_depot_id`` → erstes Depot aus ``getUser``.

    Spiegelt das Muster der Planspiel-Tools/Routers (ohne ``portfolio_id``-
    Argument — das Dashboard liest das Standard-Depot).
    """
    try:
        standard = int(
            db_settings_planspiel().get("standard_depot_id") or 0
        )
    except Exception:  # noqa: BLE001 — Settings-Störung: Fallback-Kette fortsetzen
        standard = 0
    if standard:
        return standard
    for depot in client.user_holen().get("userPortfolios") or []:
        if isinstance(depot, dict) and depot.get("id") is not None:
            return int(depot["id"])
    raise PlanspielFehler(
        "vertrag_geaendert",
        "Die Planspiel-Börse liefert keine Depots — bitte die Verbindung in den "
        "Einstellungen testen.",
    )


def db_settings_planspiel() -> dict[str, Any]:
    """Settings-Sektion ``planspiel`` — leeres Dict, falls der Bestand sie fehlt."""
    from quantlab.storage.repos import settings_repo

    try:
        return settings_repo.get_section("planspiel")
    except ValueError:
        return {}


def _name_zu_isin(isin: str) -> str | None:
    """Instrumenten-Name aus dem lokalen Universum (``None`` = unbekannt)."""
    instrument = psb_repo.instrument(isin)
    return instrument.name if instrument is not None else None


def _letzter_kurs(isin: str) -> tuple[float, int] | None:
    """Letzter Schlusskurs + Balkenzahl des 1d-Bestands — ``None`` ohne Daten."""
    try:
        balken = parquet_store.read_bars(isin, "1d", limit=1)
    except Exception:  # noqa: BLE001 — defekter Bestand → ISIN überspringen
        return None
    if balken.empty:
        return None
    kurs = float(balken["close"].iloc[-1])
    if kurs <= 0:
        return None
    return kurs, len(balken)


def top_mover(*, jetzt: datetime | None = None) -> dict[str, Any]:
    """Top 5 Auf/Ab über die letzten zwei Tages-Schlüsse je ISIN (Parquet 1d).

    Quelle-Mix (``psb_alt``/``byteworx``/``yahoo``) ist NORMAL — je Zeile wird
    die ``source`` aus ``bar_meta`` mitgeliefert. Nur ISIN-förmige Symbole
    (Wettbewerbs-Universum); nur ISINs mit mindestens zwei Balken.
    """
    del jetzt  # Δ% ist rein reihenbasiert — Uhr nur für Vertrags-Symmetrie
    zeilen: list[dict[str, Any]] = []
    for meta in parquet_store.list_bar_meta():
        if meta.interval != "1d" or not regeln.ist_isin_form(meta.symbol):
            continue
        if meta.row_count is not None and meta.row_count < 2:
            continue
        try:
            balken = parquet_store.read_bars(meta.symbol, "1d", limit=2)
        except Exception:  # noqa: BLE001 — defekte Datei überspringen
            continue
        if len(balken) < 2:
            continue
        alt = float(balken["close"].iloc[0])
        neu = float(balken["close"].iloc[-1])
        if alt <= 0:
            continue
        zeilen.append(
            {
                "isin": meta.symbol,
                "name": _name_zu_isin(meta.symbol),
                "delta": (neu - alt) / alt,
                "kurs": neu,
                "quelle": meta.source,
            }
        )
    auf = sorted(zeilen, key=lambda z: z["delta"], reverse=True)[:_TOP_N]
    ab = sorted(zeilen, key=lambda z: z["delta"])[:_TOP_N]
    return {
        "auf": auf,
        "ab": ab,
        "basis": len(zeilen),
        "hinweis": (
            "Δ % aus den letzten zwei Tages-Schlüssen des lokalen 1d-Bestands "
            "(Quellen-Mix); ohne Historie fehlt eine ISIN — Universum/Historie "
            "importieren (Planspiel-Ansicht)."
        ),
    }


def upside_ranking() -> dict[str, Any]:
    """Top 5 Analysten-Upside: Kursziel (VWD-Cache) vs. letzter 1d-Kurs.

    Upside als Bruchzahl (Konvention H3); ISINs ohne Kursziel **oder** ohne
    lokalen 1d-Kurs fließen bewusst nicht ein (keine Erfindung von Werten).
    """
    try:
        conn = db.connect()
        try:
            rows = conn.execute(
                "SELECT isin, daten FROM psb_kennzahlen WHERE sektion = ?",
                (_SEKTION_ANALYSTEN,),
            ).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — fehlende Tabelle (alter Bestand) → leer
        rows = []
    zeilen: list[dict[str, Any]] = []
    for row in rows:
        isin = str(row["isin"])
        try:
            analysten = json.loads(row["daten"])
        except (json.JSONDecodeError, TypeError):
            continue
        kursziel = analysten.get("kursziel") if isinstance(analysten, dict) else None
        if not isinstance(kursziel, int | float) or isinstance(kursziel, bool):
            continue
        kurs_info = _letzter_kurs(isin)
        if kurs_info is None:
            continue
        kurs = kurs_info[0]
        zeilen.append(
            {
                "isin": isin,
                "name": _name_zu_isin(isin),
                "kursziel": float(kursziel),
                "kurs": kurs,
                "upside": (float(kursziel) - kurs) / kurs,
            }
        )
    zeilen.sort(key=lambda z: z["upside"], reverse=True)
    return {
        "liste": zeilen[:_TOP_N],
        "basis": len(zeilen),
        "hinweis": (
            "Upside = (Analysten-Kursziel − letzter Kurs) / letzter Kurs; "
            "Kursziele aus dem VWD-Cache (planspiel_kennzahlen), Kurse aus dem "
            "1d-Bestand. Ohne Historie fehlt eine ISIN — Historie importieren."
        ),
    }


def positions_monitor(client: Any) -> dict[str, Any]:
    """Aktuelle Depot-Positionen mit Gewicht % und 20 %-Warnung (read-only)."""
    depot_id = _depot_id_ermitteln(client)
    kennzahlen = client.portfolio_holen(depot_id)
    mit_positionen = client.positionen_holen(depot_id)
    roh_wert = kennzahlen.get("value")
    depotwert = (
        float(roh_wert)
        if isinstance(roh_wert, int | float) and not isinstance(roh_wert, bool)
        else None
    )
    positionen: list[dict[str, Any]] = []
    for item in mit_positionen.get("items") or []:
        if not isinstance(item, dict):
            continue
        instrument = item.get("instrument") or {}
        isin = instrument.get("isin")
        wert = item.get("currentValue")
        if not isinstance(isin, str) or not isinstance(wert, int | float) or isinstance(wert, bool):
            continue
        gewicht = (float(wert) / depotwert) if depotwert and depotwert > 0 else None
        positionen.append(
            {
                "isin": isin,
                "name": instrument.get("name"),
                "menge": item.get("quantity"),
                "wert": float(wert),
                "gewicht": gewicht,
                "warnung": gewicht is not None and gewicht > regeln.MAX_DEPOTGEWICHT + 1e-9,
            }
        )
    return {
        "portfolio_id": depot_id,
        "depotwert": depotwert,
        "positionen": positionen,
        "hinweis": (
            "Gewicht = Positionswert / Depotwert; Warnung ab mehr als "
            f"{int(round(regeln.MAX_DEPOTGEWICHT * 100))} % je Wertpapier "
            "(Spielregel inkl. offener Käufe)."
        ),
    }


def aktivitaet(client: Any, *, jetzt: datetime | None = None) -> dict[str, Any]:
    """Abgerechnete Käufe des heutigen Saisonmonats vs. Mindestaktivität."""
    moment = jetzt or _jetzt()
    depot_id = _depot_id_ermitteln(client)
    orders = client.orders_holen(depot_id, regeln.STATUS_AUSGEFUEHRT)
    if not isinstance(orders, list):
        return {"kaeufe": None, "grenze": regeln.MIN_KAEUFE, "hinweis": (
            "Orders ließen sich nicht ermitteln (unerwartetes Format) — "
            "Aktivitätszähler entfällt."
        )}
    kaeufe = regeln.abgerechnete_kaeufe(orders, moment)
    lokal = moment.astimezone(ZoneInfo(regeln.HANDELSZEIT_ZONE))
    return {
        "portfolio_id": depot_id,
        "saisonmonat": f"{lokal.year:04d}-{lokal.month:02d}",
        "kaeufe": kaeufe,
        "grenze": regeln.MIN_KAEUFE,
        "erfuellt": kaeufe >= regeln.MIN_KAEUFE,
        "hinweis": (
            "Abgerechnete Käufe (getOrders EXECUTED) im laufenden Saisonmonat; "
            f"die Börse verlangt mindestens {regeln.MIN_KAEUFE} "
            "(sonst Ausschluss wegen Passivität)."
        ),
    }


def ueberblick(client: Any, *, jetzt: datetime | None = None) -> dict[str, Any]:
    """Gesamtes Dashboard (Blöcke 1–4) — E072-Stil: Fachfälle als Fehler-Feld."""
    erg: dict[str, Any] = {
        "top_mover": None,
        "upside": None,
        "positionen": None,
        "aktivitaet": None,
        "fehler": None,
    }
    # Blöcke 1–2: netzfrei — sie kippen nur bei Speicher-/DB-Störungen, die
    # hier bewusst als Fachfall (statt 5xx) niedergelegt werden.
    try:
        erg["top_mover"] = top_mover(jetzt=jetzt)
        erg["upside"] = upside_ranking()
    except Exception as exc:  # noqa: BLE001 — Dashboard bleibt antwortfähig
        erg["fehler"] = {
            "code": "dashboard_daten",
            "message": f"Lokale Kurs-/Kennzahlen-Daten konnten nicht gelesen werden: {exc}",
        }
    # Blöcke 3–4: Planspiel-Client — PlanspielFehler (E069) wird strukturiert,
    # alles Übrige generisch deutsch niedergelegt; nie ein Throw (E072).
    try:
        erg["positionen"] = positions_monitor(client)
        erg["aktivitaet"] = aktivitaet(client, jetzt=jetzt)
    except PlanspielFehler as exc:
        erg["fehler"] = erg["fehler"] or {"code": exc.code, "message": exc.message}
    except Exception as exc:  # noqa: BLE001 — Client-Störung: Blöcke null + Fehler
        erg["fehler"] = erg["fehler"] or {
            "code": "vertrag_geaendert",
            "message": "Depot-Daten konnten nicht geladen werden — bitte die "
            "Verbindung in den Einstellungen testen.",
            "detail": str(exc),
        }
    return erg
