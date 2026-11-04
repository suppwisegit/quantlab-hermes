"""Daten-Vorbereitungsphase (B-Fix-4, E051–E056): fehlende Marktdaten automatisch laden.

Zwei-Phasen-Muster vor jedem Backtest („Wenn Daten nicht da sind, sollen sie
noch heruntergeladen werden"): Phase 1 prüft je Symbol die Abdeckung gegen das
Wunschfenster, lädt Fehlendes über die bestehende Download-Schleuse
(:mod:`quantlab.services.downloads` — Pacing/Circuit-Breaker/Dedup/SSE werden
wiederverwendet, kein zweiter Fetch-Pfad) und Phase 2 (der eigentliche Lauf)
startet erst, wenn die Daten stehen oder fachlich abgebrochen wird.

Ablauf von :func:`bereite_daten_vor`:

1. **Cover-Check** (:func:`fenster_gedeckt`) je Symbol über
   ``parquet_store.get_bar_meta`` vs. Wunschfenster — ohne Fenster genügt
   irgendein Bestand (E054), mit Fenster müssen Bestandsanfang bzw. -ende das
   Fenster abdecken (Toleranz s. :data:`_FENSTER_TOLERANZ_US`).
2. Fehlende Symbole gehen in **einen** ``downloads.submit_job(...)`` mit
   explizitem ``start``/``end`` als ISO-String, sobald ein Wunschfenster
   vorliegt (µs→ISO im Service; ``create_job`` nimmt nur Strings). Expliziter
   Start erzwingt den Voll-Fenster-Fetch statt des 5-Balken-Inkrements (E054 —
   fehlt der Anfang, wäre der Backtest sonst trotzdem falsch leer; der Upsert
   ist idempotent). Es gibt bewusst **keinen** Cross-Job-Dedup: zwei
   parallele Preps mit denselben fehlenden Symbolen stoßen zwei Jobs an —
   die FIFO-Schleuse serialisiert sie, der Upsert ist idempotent; der
   Zweit-Job kostet nur Pacing-Zeit (REV-B1, dokumentierte Grenze).
3. Warten auf den Terminal-Status mit Timeout (:data:`VORBEREIT_TIMEOUT_S`,
   konfigurierbar). Während des Wartens werden die ``download_progress``-
   Events des Jobs je Symbol weitergeleitet (SSE-Vertrag unten bzw.
   ``fortschritt``-Callback für den Job-Pfad). Bei Timeout:
   ``downloads.cancel_job_anfordern`` + deutscher Fachfehler (E055).
4. **Frischer Cover-Check**: nur Symbole, die danach das Wunschfenster
   abdecken, gelten als geladen; der Rest landet mit Grund im Bericht
   ``{"geprüft": n, "geladen": [sym, …], "fehlgeschlagen": {sym: meldung}}``.
   Bleibt irgendein Symbol leer, bricht :func:`bereite_daten_vor` mit 422 und
   deutscher Sammelmeldung (je Symbol + Grund) ab — konservativ, keine stille
   Verkürzung der Historie (E055). Fachfehler aus ``create_job`` (z. B.
   Intraday-60-Tage-Grenze, ``intraday_limit``) werden unverändert als
   :class:`~quantlab.providers.base.ProviderError` durchgereicht.

SSE-Vertrag (E056): neues Event ``backtest_prepare_progress`` mit Payload
``{token, done, total, symbol}`` für den Sync-Pfad (``POST /api/backtest/runs``).
``token`` ist das vom Frontend je Submit erzeugte ``prepare_token``; wurde kein
Token mitgegeben, wird mit leerem Token (``""``) emittiert — das Frontend
filtert auf sein eigenes Token und ignoriert Fremde/Leere. ``token=None``
(Optimierungs-Jobs) emittiert bewusst **keine** Prepare-Events: Jobs melden
ihre Vorbereitungs-Phase über das bestehende ``backtest_job_progress`` mit
``phase: "preparing_data"`` (Callback ``fortschritt``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from quantlab import events
from quantlab.errors import ApiError
from quantlab.services import downloads
from quantlab.storage import parquet_store
from quantlab.storage.repos import downloads_repo

#: Standard-Timeout für die Vorbereitungsphase (E055) in Sekunden.
VORBEREIT_TIMEOUT_S = 300.0

#: SSE-Event-Name des Sync-Pfads (E056) — bewusst hier (statt events.py)
#: definiert, damit der B1-Vertrag an einem Ort steht (Muster backtest_jobs).
BACKTEST_PREPARE_PROGRESS = "backtest_prepare_progress"

#: Toleranz der Fenster-Grenzen gegen Handelskalender-Lücken (Wochenende,
#: Feiertage): Ein Bestand, dessen erster/letzter Balken innerhalb der Toleranz
#: am Wunschfenster liegt, gilt als gedeckt. ANNAHME: Yahoo liefert den ersten
#: Handelstag **nach** einem Feiertags-Start und keinen Balken am Wochenende —
#: die strenge E051-Grenze (``first_ts ≤ start`` bzw. ``last_ts ≥ end−1µs``)
#: würde solche legitimen Fenster sonst dauerhaft als Lücke melden (Provider
#: liefert dieselben Grenzen erneut → permanenter 422). Echte Lücken sind
#: deutlich größer als 4 Kalendertage (längste reguläre Börsenschließung).
_FENSTER_TOLERANZ_US = 4 * parquet_store.US_PER_DAY

#: Abfrage-Intervall des Warte-Loops (Sekunden).
_POLL_INTERVALL_S = 0.05

_TERMINAL_STATUS: frozenset[str] = frozenset(
    {
        downloads_repo.STATUS_DONE,
        downloads_repo.STATUS_ERROR,
        downloads_repo.STATUS_CANCELLED,
    }
)

#: Deutscher Sammelmeldungs-Code des Sync-Pfads (422, E055).
CODE_VORBEREITUNG_UNVOLLSTAENDIG = "daten_vorbereitung_fehlgeschlagen"

#: Deutscher Fachfehler bei Timeout (E055).
CODE_VORBEREIT_TIMEOUT = "datenladen_timeout"

#: Rückgabe-Form des ``fortschritt``-Callbacks: (done, total, symbol).
FortschrittCallback = Callable[[int, int, str], None]

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Cover-Check (E051/E054)
# ---------------------------------------------------------------------------


def fenster_gedeckt(
    symbol: str,
    interval: str,
    start_us: int | None,
    end_us: int | None,
) -> bool:
    """Deckt der gespeicherte Bestand das Wunschfenster ab?

    ``None``-Meta/leerer Bestand → nie gedeckt. Ohne Fenster genügt irgendein
    Bestand (E054 — dann darf der Download inkrementell laufen). Mit Fenster
    müssen ``first_ts`` (inklusiv, binnen Toleranz) und ``last_ts`` (binnen
    Toleranz an ``end`` heran) das Fenster abdecken — Details
    :data:`_FENSTER_TOLERANZ_US`.
    """
    meta = parquet_store.get_bar_meta(symbol, interval)
    if meta is None or meta.row_count <= 0 or meta.first_ts is None:
        return False
    if start_us is None and end_us is None:
        return True
    if start_us is not None and meta.first_ts > start_us + _FENSTER_TOLERANZ_US:
        return False
    if end_us is not None and (
        meta.last_ts is None or meta.last_ts < end_us - _FENSTER_TOLERANZ_US
    ):
        return False
    return True


def _us_zu_iso(ts_us: int) -> str:
    """int64 µs UTC → ISO-8601-String (Vertrag ``downloads.create_job``: Strings)."""
    return (_EPOCH_UTC + timedelta(microseconds=ts_us)).isoformat()


# ---------------------------------------------------------------------------
# Fortschritt (E056)
# ---------------------------------------------------------------------------


def _publiziere_prepare(
    token: str | None,
    done: int,
    total: int,
    symbol: str,
    fortschritt: FortschrittCallback | None,
) -> None:
    """Ein Prepare-Fortschritt → SSE (Sync-Pfad) und/oder Callback (Job-Pfad)."""
    if token is not None:
        events.publizieren(
            BACKTEST_PREPARE_PROGRESS,
            {"token": token, "done": done, "total": total, "symbol": symbol},
        )
    if fortschritt is not None:
        fortschritt(done, total, symbol)


def _drainiere_download_fortschritt(
    queue: asyncio.Queue[dict[str, Any]],
    job_id: str,
    token: str | None,
    fortschritt: FortschrittCallback | None,
) -> None:
    """Leitet wartende ``download_progress``-Events des Jobs als Prepare-Fortschritt weiter."""
    while True:
        try:
            nachricht = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        if nachricht.get("name") != events.DOWNLOAD_PROGRESS:
            continue
        daten = nachricht.get("daten") or {}
        if daten.get("jobId") != job_id:
            continue
        _publiziere_prepare(
            token,
            int(daten.get("done") or 0),
            int(daten.get("total") or 0),
            str(daten.get("symbol") or ""),
            fortschritt,
        )


async def _warte_mit_fortschritt(
    job_id: str,
    timeout_s: float,
    token: str | None,
    fortschritt: FortschrittCallback | None,
) -> downloads_repo.Download | None:
    """Wartet auf den Terminal-Status des Download-Jobs (``None`` bei Timeout).

    Nebeneffekt: ``download_progress``-Events des Jobs werden eingesammelt und
    als Prepare-Fortschritt weitergeleitet (E056) — im Loop-Thread, daher darf
    :func:`events.abonnieren` hier direkt laufen.
    """
    queue: asyncio.Queue[dict[str, Any]] = events.abonnieren()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    try:
        while loop.time() < deadline:
            job = await asyncio.to_thread(downloads_repo.get_download, job_id)
            if job is not None and job.status in _TERMINAL_STATUS:
                # Restliche Events einsammeln, damit der letzte Fortschritt
                # ankommt (der Status erscheint in der DB kurz VOR dem letzten
                # Bus-Event — Muster-Befund aus tests/test_download_service.py).
                await asyncio.sleep(_POLL_INTERVALL_S)
                _drainiere_download_fortschritt(queue, job_id, token, fortschritt)
                return job
            _drainiere_download_fortschritt(queue, job_id, token, fortschritt)
            await asyncio.sleep(_POLL_INTERVALL_S)
        return None
    finally:
        events.abmelden(queue)


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


async def bereite_daten_vor(
    symbole: list[str],
    interval: str,
    *,
    start_us: int | None = None,
    end_us: int | None = None,
    token: str | None = "",
    timeout_s: float = VORBEREIT_TIMEOUT_S,
    fortschritt: FortschrittCallback | None = None,
) -> dict[str, Any]:
    """Daten-Vorbereitungsphase (E051): fehlende Marktdaten laden → Bericht.

    Rückgabe: ``{"geprüft": n, "geladen": [sym, …], "fehlgeschlagen":
    {sym: meldung}}``. Wirft:

    - :class:`~quantlab.providers.base.ProviderError` — Fachfehler aus
      ``downloads.create_job`` (u. a. 422 ``intraday_limit`` bei
      Intraday-Wunschfenstern jenseits 60 Tage), unverändert durchgereicht;
    - :class:`~quantlab.errors.ApiError` 504 ``datenladen_timeout`` — der
      Download-Job wurde nach ``timeout_s`` abgebrochen (E055);
    - :class:`~quantlab.errors.ApiError` 422
      ``daten_vorbereitung_fehlgeschlagen`` — Sammelmeldung mit **jedem**
      Symbol, das nach Phase 1 leer geblieben ist, samt Grund (E055).

    ``token``: ``""`` → Prepare-SSE mit leerem Token (Sync-Pfad ohne
    ``prepare_token``), ``None`` → keine Prepare-SSE (Job-Pfad), sonst Wert
    des Frontend-Tokens (E056-Vertrag, Modul-Docstring).
    """
    geprueft = list(dict.fromkeys(symbole))  # Dedup, Reihenfolge erhalten
    bericht: dict[str, Any] = {
        "geprüft": len(geprueft),
        "geladen": [],
        "fehlgeschlagen": {},
    }
    fehlende = [
        sym
        for sym in geprueft
        if not fenster_gedeckt(sym, interval, start_us, end_us)
    ]
    if not fehlende:
        return bericht

    # E054: mit Wunschfenster immer explizites start/end (Voll-Fenster-Fetch,
    # nicht Inkrement); ohne Fenster (Gesamthistorie) darf inkrementell laufen.
    job = await downloads.submit_job(
        fehlende,
        interval,
        start=_us_zu_iso(start_us) if start_us is not None else None,
        end=_us_zu_iso(end_us) if end_us is not None else None,
    )

    _publiziere_prepare(token, 0, len(fehlende), "", fortschritt)
    fertig = await _warte_mit_fortschritt(
        job.job_id, timeout_s, token, fortschritt
    )
    if fertig is None:
        await asyncio.to_thread(downloads.cancel_job_anfordern, job.job_id)
        downloads.abbruch_signalieren(job.job_id)
        raise ApiError(
            504,
            CODE_VORBEREIT_TIMEOUT,
            "Datenladen nicht rechtzeitig abgeschlossen — der Download-Job "
            "wurde abgebrochen.",
            hint=(
                "Bitte erneut versuchen, den Zeitraum verkleinern oder die "
                "Daten vorab im Dashboard → Download laden."
            ),
        )

    gruende = _fehlergruende(fertig)
    for sym in fehlende:
        if fenster_gedeckt(sym, interval, start_us, end_us):
            bericht["geladen"].append(sym)
        else:
            bericht["fehlgeschlagen"][sym] = gruende.get(sym) or (
                "Der Download hat keine Daten für das Wunschfenster geliefert."
            )

    if bericht["fehlgeschlagen"]:
        details = "; ".join(
            f"{sym}: {grund}" for sym, grund in bericht["fehlgeschlagen"].items()
        )
        raise ApiError(
            422,
            CODE_VORBEREITUNG_UNVOLLSTAENDIG,
            f"Die Daten-Vorbereitung ist unvollständig — {details}.",
            hint=(
                "Symbol/Zeitraum prüfen (Intraday reicht maximal 60 Tage "
                "zurück; Tagesdaten empfehlen sich für lange Historien) oder "
                "die genannten Symbole im Dashboard → Download manuell laden."
            ),
        )
    return bericht


def _fehlergruende(job: downloads_repo.Download | None) -> dict[str, str]:
    """Job-Fehlerzusammenfassung (``sym: meldung; …``, downloads._zusammenfassung)
    → je-Symbol-Grund. Heuristik: Trennung am ersten ``": "`` je ``"; "``-Teil —
    Symbol-Kürzel enthalten kein ``": "``, Meldungen können, der Rest bleibt
    Teil des Grundes."""
    if job is None or not job.error:
        return {}
    gruende: dict[str, str] = {}
    for teil in str(job.error).split("; "):
        if ": " in teil:
            sym, meldung = teil.split(": ", 1)
            gruende[sym.strip()] = meldung.strip()
    return gruende
