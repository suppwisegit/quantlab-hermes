"""Fetch-Härtung für Yahoo (T19, E035): Pacing + Circuit-Breaker gegen 429.

yfinance reicht :class:`yfinance.exceptions.YFRateLimitError` bewusst ohne
eigenes Retry weiter (RECHERCHE v2 Block 1.8, ``history.py:297``). Diese
Schicht setzt genau dort an:

- **Pacing:** prozessweiter Mindestabstand (:data:`DEFAULT_PACING_S` = 0,75 s)
  **vor jedem** Yahoo-Call — über alle Threads hinweg (Slot-Reservierung unter
  Lock, geschlafen wird außerhalb). Der Abstand ist konfigurierbar bzw. für
  Tests injizierbar (:class:`YahooRateLimiter` mit eigener ``clock``/``sleep``).
- **Circuit-Breaker:** das Auftreten von ``YFRateLimitError`` öffnet den
  Circuit **sofort** — der nächste Yahoo-Call wird gar nicht mehr ausgeführt,
  bis der Cool-down abgelaufen ist. Cool-down 60 s, exponentiell verdoppelnd
  (60 → 120 → 240 → …) bis max. 900 s; ein **erfolgreicher** Call schließt
  den Circuit und setzt die Exponential-Stufe zurück. Ein nur abgelaufener
  Cool-down (kein Erfolg) lässt die Stufe stehen — erneute Drosselung
  eskaliert weiter.
- **Sichtbarkeit:** :meth:`YahooRateLimiter.status` speist
  ``GET /api/data/provider-status`` (``state: ok|cooldown`` + ``cooldown_until``).

Seit E087 (Welle G) gibt es **getrennte Circuits** je Verbrauchergruppe
(damit öffnet ein 429 in einem Bein nie mehr die anderen — Anlass: reale
Nutzung 2026-09-21, ein Yahoo-429 legte beide Download-Beine + Suche lahm):

- :data:`RATE_LIMITER_DIRECT` — YahooDirect-Bein (v8/chart-Downloads),
- :data:`RATE_LIMITER_YF` — yfinance-Bein,
- :data:`RATE_LIMITER_LEICHT` — „leichte“ Abrufe (Quotes-Poll, Symbolsuche,
  Fundamentals-Refresh); ein 429 hier eskaliert nie einen Download-Circuit,
- :data:`RATE_LIMITER` — Legacy-Default, bleibt **Konstruktor-Default aller
  Provider** (Fixture-Kompatibilität: Tests patchen dieses Modulattribut).
  Produktiv verdrahtet nur noch ``downloads.get_provider``/``quotes``/
  ``fundamentals``/Suche die neuen Singletons (E087).

Die Zustände sind rein prozesslokal (ein Sidecar-Prozess, E009); Yahoo- und
Stooq-Status werden im Router (:mod:`quantlab.api.routers.data`) zusammen-
geführt.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from quantlab.providers.base import ProviderError

#: Prozessweiter Mindestabstand zwischen zwei Yahoo-Calls (E035: Default 0,75 s).
DEFAULT_PACING_S = 0.75

#: Erster Cool-down nach einer Drosselung (E035: 60 s).
COOLDOWN_START_S = 60.0

#: Obergrenze des exponentiellen Cool-downs (E035: max. 900 s).
COOLDOWN_MAX_S = 900.0


class YahooGedrosseltError(ProviderError):
    """Yahoo ist im Cool-down — weitere Yahoo-Abrufe werden blockiert (HTTP 429).

    Hebt sich von „Quelle nicht erreichbar“ (502) ab: Der Aufrufer weiß, dass
    ein **früherer, zu schneller** Abruf schuld war. Die Meldung verspricht
    bewusst **keinen** Fallback (E087): Der einzelne Provider kennt die
    Bein-Kette nicht — die ehrliche, dynamische Quellen-Erwähnung baut der
    :class:`~quantlab.providers.fallback_provider.FallbackProvider`.

    Strukturierte Restzeit: ``rest_s`` (float, Sekunden bis Circuit-Ende) —
    Lesepfad für die Auto-Retry-Planung (E089/T-G3):
    ``rate_limit.RATE_LIMITER_DIRECT.gesperrt_fehler().rest_s``.
    """

    def __init__(self, rest_s: float) -> None:
        self.rest_s = rest_s
        super().__init__(
            f"Yahoo drosselt die Anfragen (Rate-Limit). Weitere Yahoo-Abrufe "
            f"werden für {rest_s:.0f} s ausgesetzt.",
            429,
            "yahoo_gedrosselt",
            hint="Etwas seltener anfragen — der Cool-down verdoppelt sich "
            "bei jeder weiteren Drosselung (max. 900 s).",
        )


class YahooRateLimiter:
    """Pacing + Circuit-Breaker für alle Yahoo-Calls eines Prozesses.

    ``clock``/``sleep`` sind injizierbar — Tests steuern die Zeit deterministisch
    (gemockte Clock, Schlafzeiten werden nur protokolliert). Der Limiter ist
    thread-safe: Slots werden unter Lock reserviert, geschlafen wird außerhalb
    (kein Lock-Blocking durch den Schlaf selbst).
    """

    def __init__(
        self,
        *,
        pacing_s: float = DEFAULT_PACING_S,
        cooldown_start_s: float = COOLDOWN_START_S,
        cooldown_max_s: float = COOLDOWN_MAX_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.pacing_s = max(0.0, float(pacing_s))  # public: zur Laufzeit stellbar
        self._cooldown_start_s = float(cooldown_start_s)
        self._cooldown_max_s = float(cooldown_max_s)
        self._clock = clock
        self._sleep = sleep if sleep is not None else time.sleep
        self._lock = threading.Lock()
        self._letzter_aufruf: float | None = None  # clock()-Wert des letzten Slots
        self._cooldown_bis: float | None = None  # clock()-Wert; None = zu
        self._stufe = 0  # aufeinanderfolgende Drosselungen (60 s · 2^stufe)

    # -- Guard vor jedem Yahoo-Call -------------------------------------

    def vor_yahoo_aufruf(self) -> float:
        """Pacing + Circuit-Guard — **vor** jeden Yahoo-Call zu rufen.

        Offener Circuit → :class:`YahooGedrosseltError` **sofort**, ohne dass
        der Yahoo-Call stattfindet (E035 „kein weiterer Yahoo-Call“). Sonst
        wird der nötige Pacing-Abstand geschlafen und der eigene Slot
        reserviert. Rückgabe: tatsächlich geschlafene Sekunden (Tests).

        Bekannte Grenze (M1-Doku): Der Pacing-Slot wird VOR der Semaphore(4)
        des yfinance-Providers reserviert (s.
        ``YfinanceProvider._roh_download``). Steht ein Thread an der
        Semaphore an, verstreicht seine Slot-Zeit wartend — reale Yahoo-Calls
        können damit NÄHER als ``pacing_s`` aufeinander folgen. Downloads
        laufen FIFO-serialisiert → praktisch irrelevant.
        """
        with self._lock:
            moment = self._clock()
            if self._cooldown_bis is not None:
                if moment < self._cooldown_bis:
                    rest = self._cooldown_bis - moment
                    raise YahooGedrosseltError(round(rest, 3))
                self._cooldown_bis = None  # abgelaufen → Yahoo wieder erlaubt
            warte = 0.0
            if self._letzter_aufruf is not None:
                warte = self._letzter_aufruf + self.pacing_s - moment
                if warte < 0.0:
                    warte = 0.0
            self._letzter_aufruf = moment + warte
        if warte > 0.0:
            self._sleep(warte)
        return warte

    # -- Zustandsübergänge ----------------------------------------------

    def verzeichne_erfolg(self) -> None:
        """Erfolgreicher Yahoo-Call: schließt den Circuit, resettet die Stufe."""
        with self._lock:
            self._cooldown_bis = None
            self._stufe = 0

    def verzeichne_rate_limit(self) -> float:
        """Drosselung erkannt: öffnet den Circuit; liefert die Cool-down-Dauer.

        60 s · 2^stufe, gedeckelt auf :data:`COOLDOWN_MAX_S` (E035).
        """
        with self._lock:
            dauer = min(
                self._cooldown_start_s * (2**self._stufe), self._cooldown_max_s
            )
            self._stufe += 1
            self._cooldown_bis = self._clock() + dauer
            return dauer

    def zustand_zuruecksetzen(self) -> None:
        """Pacing-Slot, Cool-down und Stufe zurücksetzen (Test-Hygiene).

        Die drei Prozess-Singletons (:data:`RATE_LIMITER_DIRECT`,
        :data:`RATE_LIMITER_YF`, :data:`RATE_LIMITER_LEICHT`) sind
        prozessglobal — Suite-Tests, die ihren Zustand direkt assertern
        (provider-status, ``_stufe``/``_cooldown_bis``), setzen ihn vor jedem
        Test auf „frisch“, damit die Suite reihenfolge- und lastunabhängig
        ist (REV-G MAJOR-2; Präzedenz ``leeway_provider.
        zustand_zuruecksetzen``, Muster websuche). Konfiguration
        (``pacing_s``, Cool-down-Grenzen, ``clock``/``sleep``) bleibt
        unberührt — es wird nur der **Zustand** gelöscht.
        """
        with self._lock:
            self._letzter_aufruf = None
            self._cooldown_bis = None
            self._stufe = 0

    # -- Abfragen --------------------------------------------------------

    def ist_gesperrt(self) -> bool:
        """Ist der Circuit gerade offen (Cool-down noch nicht abgelaufen)?"""
        with self._lock:
            if self._cooldown_bis is None:
                return False
            return self._clock() < self._cooldown_bis

    def gesperrt_fehler(self) -> YahooGedrosseltError:
        """Fehler-Objekt zum aktuellen Cool-down (für Aufrufer ohne eigenen Abgleich).

        Die Meldung trägt die Restzeit strukturiert im Attribut ``rest_s``
        (Sekunden, ``0.0`` wenn der Circuit gerade zu ist) — Zugriffspfad für
        die Retry-Planung (E089/T-G3):
        ``rate_limit.RATE_LIMITER_DIRECT.gesperrt_fehler().rest_s``.
        """
        with self._lock:
            rest = 0.0
            if self._cooldown_bis is not None:
                rest = max(0.0, self._cooldown_bis - self._clock())
        return YahooGedrosseltError(round(rest, 3))

    def status(self) -> dict[str, Any]:
        """Zustand für ``GET /api/data/provider-status`` (§C.4/E035).

        ``{"state": "ok"}`` bzw. ``{"state": "cooldown", "cooldown_until":
        <ISO-8601 UTC>, "cooldown_rest_s": <float>}`` — ``cooldown_until`` nur
        im Cool-down (Vertrag: optionales Feld). Der ISO-Wert kombiniert die
        echte Wanduhr mit der (ggf. gefakten) Restdauer — Anzeigezweck.
        """
        with self._lock:
            offen = (
                self._cooldown_bis is not None
                and self._clock() < self._cooldown_bis
            )
            rest = 0.0
            if self._cooldown_bis is not None:
                rest = max(0.0, self._cooldown_bis - self._clock())
        if not offen:
            return {"state": "ok"}
        bis = datetime.now(UTC) + timedelta(seconds=rest)
        return {
            "state": "cooldown",
            "cooldown_until": bis.isoformat(),
            "cooldown_rest_s": round(rest, 1),
        }


#: Prozessweiter Default-Limiter (Sidecar hat genau einen, E009). Tests
#: ersetzen ihn per ``monkeypatch.setattr(rate_limit, "RATE_LIMITER", …)``
#: durch eine Instanz mit eigener Clock — niemals den Zustand des Modul-
#: Defaults anfassen (-suite-übergreifende Schmutzeffekte).
RATE_LIMITER = YahooRateLimiter()

#: Circuit des YahooDirect-Beins (E087) — Downloads über v8/chart. Produktiv
#: verdrahtet in ``downloads.get_provider``; ein 429 hier sperrt nie das
#: yfinance-Bein oder die „leichten“ Abrufe.
RATE_LIMITER_DIRECT = YahooRateLimiter()

#: Circuit des yfinance-Beins (E087) — zweites Download-Bein, eigenständig
#: gegenüber :data:`RATE_LIMITER_DIRECT`.
RATE_LIMITER_YF = YahooRateLimiter()

#: Circuit der „leichten“ Yahoo-Abrufe (E087): Quotes-Poll, Symbolsuche,
#: Fundamentals-Refresh. Ein 429 hier eskaliert nie einen Download-Circuit
#: (Anlass-Fall: der Poll-Pull zählte den Download-Cooldon 60 → 900 s hoch).
RATE_LIMITER_LEICHT = YahooRateLimiter()


def status_aggregiert(limiter: YahooRateLimiter | None = None) -> dict[str, Any]:
    """Status eines Circuits (Router-Helfer, §C.4).

    Ohne Argument der Legacy-Default :data:`RATE_LIMITER`; der Router liest
    die getrennten Circuits explizit (``RATE_LIMITER_DIRECT`` etc., E087).
    """
    return (limiter if limiter is not None else RATE_LIMITER).status()


def ist_rate_limit_fehler(exc: BaseException) -> bool:
    """Ist ``exc`` eine yfinance-``YFRateLimitError``?

    yfinance wird lazy importiert (Netzfreiheit der Suite, Muster
    :mod:`quantlab.providers.yfinance_provider`) — der Import hier ist billig
    und ohne Nebenwirkungen, sobald yfinance ohnehin geladen ist.
    """
    try:
        from yfinance.exceptions import YFRateLimitError
    except Exception:  # pragma: no cover — yfinance ist Haupt-Dependency
        return False
    return isinstance(exc, YFRateLimitError)


__all__ = [
    "COOLDOWN_MAX_S",
    "COOLDOWN_START_S",
    "DEFAULT_PACING_S",
    "RATE_LIMITER",
    "RATE_LIMITER_DIRECT",
    "RATE_LIMITER_LEICHT",
    "RATE_LIMITER_YF",
    "YahooGedrosseltError",
    "YahooRateLimiter",
    "ist_rate_limit_fehler",
    "status_aggregiert",
]
