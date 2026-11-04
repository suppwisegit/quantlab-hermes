"""Alert-Evaluator (T33, §T33/§C.5): Bedingungen, Cooldown/Re-Arm, Auslösen.

Fachlogik zwischen Alerts-Repo (T18), Quotes-Service (T22) und Scheduler (T21):

- **Bedingungen (§D.2-Arten, durch den CHECK-Constraint eingefroren):**
  ``price_above``/``price_below`` (Greater/Less than), ``price_crossing_up``/
  ``price_crossing_down`` (Crossing, TradingView-Block 7) und ``percent_move``
  (± % über ``bars`` Tagesbalken gegen den lokalen Parquet-Bestand).
  ANNAHME (T33): Indikator-Bedingungen (z. B. „RSI < 30“) sind im Schema §D.2
  nicht vorgesehen (CHECK-Constraint, E028 verbietet nachträgliche ALTER in
  Feature-Tasks) — sie bleiben v3-Kandidat; „Preis vs. SMA“ lässt sich heute
  über ``price_above``/``price_below`` mit handberechneter Schwelle abbilden.
- **Kursquelle:** T22-Quotes-Service (TTL-Cache/Single-Flight dort; der
  Poll-Ansatz folgt dem quotes-Job — einen Tick-Hook im Quotes-Service gibt
  es bewusst nicht, ANNAHME T33). Liefert ein Symbol keinen Live-Quote
  (Fehler/delistet), fällt der Evaluator auf den letzten gespeicherten
  Tageschluss (Parquet 1d) zurück — Symbole ohne beide Quellen werden
  **übersprungen** (kein Fehler, DoD e).
- **Zustandsmaschine/Cooldown:** Ein Auslöser setzt den Alert atomar auf
  ``triggered`` (``UPDATE … WHERE state='active'`` — PLAN §T33-Risiko
  „Doppel-Trigger durch Poll+Scheduler“) und schreibt ein ``alert_events``-Ereignis.
  Ein Alert feuert frühestens wieder, wenn ``cooldown_min`` seit dem letzten
  Ereignis vergangen ist (Re-Arm: ``triggered → active``). Der Cooldown wird
  aus der **Historie** abgeleitet (letztes Ereignis) — damit überlebt er
  Neustarts, ohne zusätzliche Tabellen-/Memory-Zustände.
- **Crossing-Gedächtnis:** zuletzt gesehener Preis je Alert im Prozess (nach
  einem Neustart gültigt der erste Tick als Referenz, ohne auszulösen —
  „kein Crossing aus dem Nichts“). Pausierte Alerts werden nicht evaluiert
  (DoD d) und verlieren ihr Preis-Gedächtnis (beim Reaktivieren neu gültigend).

Auslöser emittieren ``alert_triggered`` (§C.5:
``{alertId, symbol, kind, value, message, triggeredAt}``) direkt über
:func:`quantlab.events.publizieren` — Muster ``SCHEDULER_TICK`` im Scheduler.
Der OS-Notify-Weg (IPC ``app:notify``) ist Renderer-Sache: das Frontend
empfängt das SSE-Event und ruft ``window.quantlab.notifyAlert(...)`` (T33).

Der Scheduler-Job ``alerts`` (reservierter Name, T21) ruft
:func:`tick_alerts` unter dem Job-Lock; ``run-now`` erzwingt frische Quotes
(TTL-Bypass — „jetzt prüfen“ für Tests/Agent).
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from quantlab import db, events
from quantlab.providers.yfinance_provider import us_zu_datetime
from quantlab.services import quotes as quotes_service
from quantlab.storage import parquet_store
from quantlab.storage.repos import alerts_repo

logger = logging.getLogger(__name__)

#: SSE-Event-Name (§C.5) — bewusst direktes ``publizieren`` (Muster Scheduler).
ALERT_TRIGGERED = "alert_triggered"

#: Parquet-Intervall des Fallbacks/der %-Basis (Daily-Close, PLAN §T33).
INTERVAL_TAGESBARS = "1d"

#: Preis-Arten mit Schwelle ``price``; Crossing-Arten teilen dieselbe Parametrik.
_PREIS_ARTEN: frozenset[str] = frozenset(
    {"price_above", "price_below", "price_crossing_up", "price_crossing_down"}
)


# ---------------------------------------------------------------------------
# Params-Validierung (Router → 422 deutsch; Evaluator vertraut dem Bestand)
# ---------------------------------------------------------------------------


def validiere_params(kind: str, params: dict[str, Any]) -> dict[str, float]:
    """Prüft Art + Params und liefert die normalisierte Parametersammlung.

    ``price_above``/``price_below``/``price_crossing_*`` verlangen
    ``{"price": > 0}``; ``percent_move`` verlangen ``{"pct": > 0, "bars": ≥ 1}``
    (DoD API: „price > 0, bars ≥ 1“). Unbekannte Art/Schlüssel/Werte →
    :class:`ValueError` (Router übersetzt in 422, deutsch).
    """
    if kind not in alerts_repo.ALERT_KINDS:
        raise ValueError(
            f"Unbekannte Alert-Art „{kind}“. Gültig: "
            f"{', '.join(sorted(alerts_repo.ALERT_KINDS))}."
        )
    if not isinstance(params, dict):
        raise ValueError("Alert-Parameter müssen ein Objekt (dict) sein.")

    def _positive_zahl(schluessel: str) -> float:
        wert = params.get(schluessel)
        if isinstance(wert, bool) or not isinstance(wert, (int, float)) or not math.isfinite(
            float(wert)
        ):
            raise ValueError(
                f"„{schluessel}“ muss eine Zahl größer 0 sein (erhalten: {wert!r})."
            )
        if float(wert) <= 0:
            raise ValueError(
                f"„{schluessel}“ muss eine Zahl größer 0 sein (erhalten: {wert!r})."
            )
        return float(wert)

    if kind == "percent_move":
        unbekannt = set(params) - {"pct", "bars"}
        if unbekannt:
            raise ValueError(
                f"Unbekannte Parameter {sorted(unbekannt)} für „percent_move“. "
                "Erlaubt: pct, bars."
            )
        pct = _positive_zahl("pct")
        bars = params.get("bars")
        if (
            isinstance(bars, bool)
            or not isinstance(bars, int)
            or isinstance(bars, float)
            or int(bars) < 1
        ):
            raise ValueError(
                f"„bars“ muss eine ganze Zahl ≥ 1 sein (erhalten: {bars!r})."
            )
        return {"pct": pct, "bars": float(bars)}

    if kind in _PREIS_ARTEN:
        unbekannt = set(params) - {"price"}
        if unbekannt:
            raise ValueError(
                f"Unbekannte Parameter {sorted(unbekannt)} für „{kind}“. Erlaubt: price."
            )
        return {"price": _positive_zahl("price")}
    raise ValueError(f"Unbekannte Alert-Art „{kind}“.")  # pragma: no cover — oben abgefangen


# ---------------------------------------------------------------------------
# Kursquelle: Quotes (T22) mit Parquet-Fallback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Kurs:
    """Ein auswertbarer Kurs: Wert + Herkunftszeit (ISO, ``None`` bei Fallback-Leer)."""

    preis: float
    quote_ts: str | None


def _us_zu_iso(ts_us: int) -> str:
    """int64 µs UTC → ISO-8601 (``alert_events.quote_ts``-Konvention)."""
    return us_zu_datetime(int(ts_us)).isoformat()


def _fallback_kurs(symbol: str) -> Kurs | None:
    """Letzter gespeicherter Tageschluss (Parquet 1d) — ``None`` ohne Bestand."""
    balken = parquet_store.read_bars(symbol, INTERVAL_TAGESBARS, limit=1)
    if balken.empty:
        return None
    zeile = balken.iloc[0]
    return Kurs(float(zeile["close"]), _us_zu_iso(int(zeile["ts"])))


def hole_kurse(symbole: list[str], *, erzwingen: bool = False) -> dict[str, Kurs]:
    """Kurse je Symbol über den Quotes-Service (TTL/Single-Flight dort).

    Fehler- oder leere Live-Einträge fallen auf den letzten Tageschluss zurück
    (Parquet-Fallback); ohne beide Quellen fehlt das Symbol im Ergebnis —
    der Aufrufer überspringt es (DoD e, „kein Fehler“).
    """
    kurse: dict[str, Kurs] = {}
    if not symbole:
        return kurse
    eintraege = quotes_service.get_service().liefere(symbole, erzwingen=erzwingen)
    for eintrag in eintraege:
        symbol = str(eintrag.get("symbol", "")).strip()
        if not symbol:
            continue
        preis = eintrag.get("price")
        if "error" not in eintrag and isinstance(preis, (int, float)) and preis > 0:
            quote_ts = eintrag.get("quote_ts")
            kurse[symbol] = Kurs(
                float(preis),
                _us_zu_iso(quote_ts) if isinstance(quote_ts, int) else None,
            )
            continue
        fallback = _fallback_kurs(symbol)
        if fallback is not None:
            kurse[symbol] = fallback
    return kurse


# ---------------------------------------------------------------------------
# Laufzeitzustand: Crossing-Gedächtnis + Cooldown aus der Historie
# ---------------------------------------------------------------------------

#: Zuletzt gesehener Preis je Alert (Crossing-Referenz; Prozess-Laufzeit).
_LETZTER_PREIS: dict[int, float] = {}


def _jetzt() -> datetime:
    """Injektionspunkt für Tests (gemockte Clock, DoD a)."""
    return datetime.now(UTC)


def _letztes_ereignis(alert_id: int) -> str | None:
    """``triggered_at`` des jüngsten Ereignisses (Historie = Cooldown-Quelle)."""
    ereignisse = alerts_repo.list_events(alert_id, limit=1)
    return ereignisse[0].triggered_at if ereignisse else None


def _in_cooldown(letztes_ereignis: str | None, cooldown_min: int) -> bool:
    """``True``, solange ``cooldown_min`` seit dem letzten Ereignis laufen."""
    if letztes_ereignis is None:
        return False
    try:
        zuletzt = datetime.fromisoformat(letztes_ereignis)
    except (TypeError, ValueError):
        return False  # defekter Historieneintrag → nicht ewig sperren
    if zuletzt.tzinfo is None:
        zuletzt = zuletzt.replace(tzinfo=UTC)
    return (_jetzt() - zuletzt) < timedelta(minutes=int(cooldown_min))


def _setze_triggered_atomar(alert_id: int) -> bool:
    """``active → triggered`` in einer Transaktion (PLAN §T33-Risiko-Guard).

    ``True`` = dieser Aufruf hat den Trigger „gewonnen“; ``False`` = ein
    paralleler Evaluator (Poll-Tick × run-now × evaluate-Endpunkt) war
    schneller — der Kandidat feuert bewusst nicht doppelt. Direktes SQL nach
    Repo-Muster (``BEGIN IMMEDIATE``), weil ``alerts_repo`` diese atomare
    Transition (bewusst) nicht anbietet — die Übergabe T18→T33 nennt den
    ``UPDATE … WHERE state``-Guard als Service-Aufgabe.
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE alerts SET state = 'triggered', updated_at = ? "
            "WHERE id = ? AND state = 'active'",
            (db.utc_now_iso(), int(alert_id)),
        )
        gewonnen = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return gewonnen


# ---------------------------------------------------------------------------
# Bedingungen (rein — Goldwert-testbar ohne Clock/Netz)
# ---------------------------------------------------------------------------


def _format(wert: float) -> str:
    """Kompakte Zahlenformatierung für deutsche Meldungen (``150`` / ``151.25``)."""
    return f"{wert:g}"


def baue_meldung(alert: alerts_repo.Alert, wert: float, schwelle: float) -> str:
    """Deutsche Auslöse-Meldung je Art (Historie + Notification-Text)."""
    symbol = alert.symbol
    if alert.kind == "price_above":
        return f"„{symbol}“ über {schwelle:g} — aktuell {_format(wert)}"
    if alert.kind == "price_below":
        return f"„{symbol}“ unter {schwelle:g} — aktuell {_format(wert)}"
    if alert.kind == "price_crossing_up":
        return f"„{symbol}“ kreuzt {schwelle:g} nach oben — aktuell {_format(wert)}"
    if alert.kind == "price_crossing_down":
        return f"„{symbol}“ kreuzt {schwelle:g} nach unten — aktuell {_format(wert)}"
    bars = int(alert.params.get("bars", 0))
    richtung = "gestiegen" if wert >= 0 else "gesunken"
    return (
        f"„{symbol}“ hat sich über {bars} Balken um {_format(abs(wert))} % "
        f"{richtung} (Schwelle {_format(schwelle)} %)"
    )


def _basis_prozent_move(symbol: str, bars: int) -> float | None:
    """Referenzschluss für ``percent_move``: Schluss von ``bars`` Balken zurück.

    Benötigt mindestens ``bars + 1`` gespeicherte Tagesbalken; weniger Bestand
    → ``None`` (Symbol wird übersprungen, kein Fehler).
    """
    balken = parquet_store.read_bars(symbol, INTERVAL_TAGESBARS, limit=int(bars) + 1)
    if len(balken) < int(bars) + 1:
        return None
    return float(balken.iloc[0]["close"])


def pruefe_bedingung(
    alert: alerts_repo.Alert, kurs: Kurs
) -> tuple[bool, float, float] | None:
    """Wertet die Bedingung aus — ``(ausgeloest, wert, schwelle)`` oder ``None``.

    ``None`` heißt „nicht bewertbar“ (z. B. Crossing ohne vorherigen Preis,
    ``percent_move`` ohne genug Bestand) — der Aufrufer überspringt dann.
    ``wert`` ist der auslösende Kurs bzw. (bei ``percent_move``) die Bewegung
    in Prozent (``alert_events.value``: „Kurs/Metrik beim Auslösen“).
    """
    params = validiere_params(alert.kind, alert.params)  # Bestand defensiv nachgeprüft
    preis = kurs.preis
    if alert.kind == "price_above":
        schwelle = params["price"]
        return (preis > schwelle, preis, schwelle)
    if alert.kind == "price_below":
        schwelle = params["price"]
        return (preis < schwelle, preis, schwelle)
    if alert.kind in ("price_crossing_up", "price_crossing_down"):
        schwelle = params["price"]
        vorher = _LETZTER_PREIS.get(alert.id)
        if vorher is None:
            return None  # erster Tick nach (Re)Start: nur Referenz gültigen
        if alert.kind == "price_crossing_up":
            kreuzt = vorher < schwelle <= preis
        else:
            kreuzt = vorher > schwelle >= preis
        return (kreuzt, preis, schwelle)
    # percent_move: Live-Kurs gegen Tageschluss von `bars` Balken zurück.
    bars = int(params["bars"])
    basis = _basis_prozent_move(alert.symbol, bars)
    if basis is None or basis <= 0:
        return None
    bewegung = (preis - basis) / basis * 100.0
    return (abs(bewegung) >= params["pct"], bewegung, params["pct"])


# ---------------------------------------------------------------------------
# Auslösen + Einzelauswertung (geteilt von Tick und evaluate-Endpunkt)
# ---------------------------------------------------------------------------


def loese_aus(
    alert: alerts_repo.Alert, kurs: Kurs, wert: float, schwelle: float, meldung: str
) -> alerts_repo.AlertEvent:
    """Triggert: atomarer Zustandswechsel, Historieneintrag, SSE ``alert_triggered``.

    Der atomare ``UPDATE … WHERE state='active'`` entscheidet, wer bei einer
    Konkurrenz (Poll-Tick × run-now × evaluate) den Trigger bekommt —
    ``RuntimeError`` bei Verlust des Rennens (Aufrufer fängt ab).
    """
    if not _setze_triggered_atomar(alert.id):
        raise RuntimeError(f"Alert {alert.id} wurde parallel bereits ausgelöst.")
    ereignis = alerts_repo.add_event(
        alert.id,
        value=float(wert),
        message=meldung,
        quote_ts=kurs.quote_ts,
        triggered_at=_jetzt().isoformat(),
    )
    events.publizieren(
        ALERT_TRIGGERED,
        {
            "alertId": alert.id,
            "symbol": alert.symbol,
            "kind": alert.kind,
            "value": float(wert),
            "message": meldung,
            "triggeredAt": ereignis.triggered_at,
        },
    )
    logger.info("Alert %s ausgelöst: %s", alert.id, meldung)
    return ereignis


def evaluiere_alert(alert: alerts_repo.Alert, kurs: Kurs) -> dict[str, Any]:
    """Ein Alert gegen einen Kurs — Zustandsmaschine inkl. Re-Arm + Cooldown.

    Reihenfolge (Plan §T33 „einmal triggern, re-arm nach cooldown_min“):
    1. ``triggered`` + Cooldown abgelaufen → Re-Arm auf ``active``.
    2. Cooldown läuft → Preis-Gedächtnis pflegen, ohne zu feuern.
    3. Bedingung prüfen; Auslöser gewinnt atomar (siehe :func:`loese_aus`).
    Rückgabe: ``{"ausgeloest": bool, "wert": float|None, "meldung": str|None,
    "bewertet": bool}`` — ``bewertet=False`` heißt „Bedingung nicht prüfbar“
    (Crossing ohne Referenz, ``percent_move`` ohne Bestand; der Tick meldet
    das Symbol dann als übersprungen, DoD e).
    """
    letztes = _letztes_ereignis(alert.id)
    if alert.state == "triggered" and not _in_cooldown(letztes, alert.cooldown_min):
        alert = alerts_repo.update_alert(alert.id, state="active") or alert  # Re-Arm (§T33)
    if _in_cooldown(letztes, alert.cooldown_min):
        _LETZTER_PREIS[alert.id] = kurs.preis  # Gedächtnis pflegen, ohne zu feuern
        return {"ausgeloest": False, "wert": None, "meldung": None, "bewertet": True}
    bewertet = pruefe_bedingung(alert, kurs)
    # Gedächtnis NACH dem Lesen der Referenz aktualisieren (sonst vergleicht
    # der Crossing-Vergleich den Kurs mit sich selbst und feuert nie).
    _LETZTER_PREIS[alert.id] = kurs.preis
    if bewertet is None:
        return {"ausgeloest": False, "wert": None, "meldung": None, "bewertet": False}
    erfuellt, wert, schwelle = bewertet
    if not erfuellt:
        return {"ausgeloest": False, "wert": float(wert), "meldung": None, "bewertet": True}
    meldung = baue_meldung(alert, wert, schwelle)
    try:
        loese_aus(alert, kurs, wert, schwelle, meldung)
    except RuntimeError:
        return {"ausgeloest": False, "wert": float(wert), "meldung": None, "bewertet": True}
    return {"ausgeloest": True, "wert": float(wert), "meldung": meldung, "bewertet": True}


# ---------------------------------------------------------------------------
# Scheduler-Tick (Job „alerts“, T21-Gerüst) + evaluate-now
# ---------------------------------------------------------------------------


async def tick_alerts(*, erzwingen: bool = False) -> dict[str, Any]:
    """Ein Alerts-Tick: aktive/„frisch ausgelöste“ Alerts je Kurs prüfen.

    Läuft unter dem Scheduler-Job-Lock (Single-Flight je Job-Art). Ohne
    relevante Alerts Netto-No-Op (kein Quotes-Abruf). Pausierte Alerts
    werden nicht evaluiert (DoD d); Symbole ohne Kursdaten übersprungen
    (DoD e). ``erzwingen=True`` (``run-now``) umgeht den Quotes-TTL-Cache.
    """
    kandidaten = [
        alert
        for alert in await asyncio.to_thread(alerts_repo.list_alerts)
        if alert.state in ("active", "triggered")
    ]
    if not kandidaten:
        return {"ok": True, "aktive": 0, "ausgeloest": [], "uebersprungen": []}

    symbole = sorted({alert.symbol for alert in kandidaten})
    kurse = await asyncio.to_thread(hole_kurse, symbole, erzwingen=erzwingen)

    ausgeloest: list[int] = []
    uebersprungen: list[str] = []
    for alert in kandidaten:
        kurs = kurse.get(alert.symbol)
        if kurs is None:
            uebersprungen.append(alert.symbol)
            continue
        try:
            ergebnis = await asyncio.to_thread(evaluiere_alert, alert, kurs)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — Isolation: ein Alert kippt nie den Tick
            logger.exception("Alert %s konnte nicht ausgewertet werden", alert.id)
            uebersprungen.append(alert.symbol)
            continue
        if ergebnis["ausgeloest"]:
            ausgeloest.append(alert.id)
        if not ergebnis.get("bewertet", True):
            uebersprungen.append(alert.symbol)  # Bedingung nicht prüfbar (z. B. Bestand)

    # Crossing-Gedächtnis verloren gegangener Alerts entsorgen (gelöscht/pausiert).
    lebende = {alert.id for alert in kandidaten}
    for alert_id in list(_LETZTER_PREIS):
        if alert_id not in lebende:
            _LETZTER_PREIS.pop(alert_id, None)

    return {
        "ok": True,
        "aktive": len(kandidaten),
        "ausgeloest": ausgeloest,
        "uebersprungen": sorted(set(uebersprungen)),
    }


async def evaluate_now(alert_id: int) -> dict[str, Any]:
    """„Jetzt prüfen“ für UI/Agent (Test-Trigger): genau dieser Alert, frischer Kurs.

    Wendet dieselbe Zustandsmaschine an (ein manuelles Auslösen startet den
    Cooldown wie ein automatischer — kein Doppel-Trigger). Pausierte Alerts
    werden nicht ausgewertet (DoD d) — Rückgabe mit ``hinweis`` statt Fehler.
    """
    alert = await asyncio.to_thread(alerts_repo.get_alert, int(alert_id))
    if alert is None:
        raise LookupError(f"Alert {int(alert_id)} existiert nicht.")
    if alert.state == "paused":
        return {"alert_id": alert.id, "ausgeloest": False, "hinweis": "pausiert"}
    kurse = await asyncio.to_thread(
        hole_kurse, [alert.symbol], erzwingen=True
    )
    kurs = kurse.get(alert.symbol)
    if kurs is None:
        return {
            "alert_id": alert.id,
            "ausgeloest": False,
            "hinweis": "keine Kursdaten",
        }
    ergebnis = await asyncio.to_thread(evaluiere_alert, alert, kurs)
    return {
        "alert_id": alert.id,
        "ausgeloest": bool(ergebnis["ausgeloest"]),
        "wert": ergebnis["wert"],
        "meldung": ergebnis["meldung"],
    }


__all__ = [
    "ALERT_TRIGGERED",
    "INTERVAL_TAGESBARS",
    "Kurs",
    "baue_meldung",
    "evaluate_now",
    "evaluiere_alert",
    "hole_kurse",
    "loese_aus",
    "pruefe_bedingung",
    "tick_alerts",
    "validiere_params",
]
