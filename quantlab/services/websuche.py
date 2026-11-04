"""Websuche-Service (Welle F, T-F6/E083): ddgs-Suche + trafilatura-Leseabruf.

Grenzen des SSRF-Guards (REV-F2-Doku): IPv4-mapped IPv6 (``::ffff:0:0/96``)
wird blockiert (Verhalten ab Python 3.13, gh-113171 — per Test gepinnt).
Nicht abgedeckt: NAT64-Präfix ``64:ff9b::/96`` (läuft durch die Blocklist,
da ``ipaddress`` es als global einstuft) und TOCTOU-DNS-Rebinding (der Guard
löst auf, httpx löst erneut auf — für einen Desktop-Leseabruf akzeptiertes
Risiko, bewusst dokumentiert statt gecacht).

Zwei asynchrone Einstiege (die schwere Arbeit läuft via ``asyncio.to_thread``,
damit der Event-Loop nicht blockiert — Muster planspiel/portfolio):

- :func:`suche` — DuckDuckGo-Websuche über ``ddgs`` (keyless, RECHERCHE F.5),
  normalisiert auf ``{titel, url, snippet, datum?}``, mit 24-h-SQLite-Cache
  (:mod:`quantlab.storage.repos.websuche_cache_repo`), prozessweitem Pacing
  (min. 1,5 s zwischen externen Such-Calls) und Backoff-Merkung nach
  ``RatelimitException``/``TimeoutException`` (60 s Sperrfenster in-memory —
  der nächste sofortige Call wird mit deutschem Hinweis abgewiesen).
  Welle I (E104/I3, additiv): ``kategorie="news"`` schaltet auf
  ``DDGS().news(...)`` um (RECHERCHE §Welle I — Felder ``date/title/body/url/
  source``); Treffer erhalten zusätzlich ``quelle`` (aus ``source``) und der
  Cache-Schlüssel bekommt das Präfix ``news|`` (sonst web/news-Kollision bei
  gleicher Query). Der Alt-Pfad (``kategorie="web"``, Default) bleibt
  byte-identisch — gleicher Schlüssel wie vor Welle I, keine ``quelle``.
- :func:`lese` — holt eine Seite per httpx (Browser-UA, Timeout 15 s) und
  extrahiert den Artikeltext mit ``trafilatura.extract`` (RECHERCHE F.6:
  Positional-Argument, ``include_comments=False``; leeres Ergebnis → ein
  Retry mit ``favor_recall=True``).

Sicherheit (web_lesen): SSRF-Guard **vor jedem Request** — der Hostname wird
aufgelöst (``socket.getaddrinfo``) und **alle** resultierenden IPs gegen die
Blocklist geprüft (loopback, privat, link-local 169.254/16, ULA fc00::/7,
``::1``, ``0.0.0.0``, reservierte/multicast Bereiche). Redirects werden
**manuell** gefolgt (max. 5 Hops) und **jeder Ziel-URL läuft erneut durch den
Guard** (ANNAHME: „Re-Check je Hop“ statt blindem Follow). ``http://``-URLs
werden vorab auf ``https`` angehoben (Downgrade-Schutz, getestet).

Die Bibliotheken ``ddgs``/``trafilatura`` werden **lazy** importiert (Muster
yfinance lazy in ``providers/yfinance_provider._roh_download``) — der Import
des Moduls und die Testsuite bleiben netzfrei; Tests monkeypatchen
``ddgs.DDGS`` bzw. :func:`_neuer_client` (httpx MockTransport).
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from quantlab.storage.repos import websuche_cache_repo

#: Mindestabstand zwischen zwei externen **Such**-Calls (T-F6: eigener Pacing-
#: Abstand, min. 1,5 s — ddgs/Backends drosseln aggressive Serien, F.1).
_SUCHE_PACING_S = 1.5

#: Backoff-Sperrfenster nach Ratelimit-/Timeout-Signal (F.1: „Cache + Backoff
#: bei 403 + eigener Pacing“) — in-memory, prozesslokal (ein Sidecar, E009).
_SPERRFENSTER_S = 60.0

#: Explizites ddgs-Timeout (F.5: Default 5 s ist knapp, EMPFEHLUNG explizit).
_DDGS_TIMEOUT_S = 10

#: Suchregion (ANNAHME: deutschsprachiger Nutzerkreis der Desktop-App →
#: „de-de“; ddgs-Vertrag laut RECHERCHE F.5: ``region``-Parameter von text()).
_REGION = "de-de"

#: Request-Timeout des Seiten-Abrufs (T-F6-Spezifikation: 15 s).
_LESE_TIMEOUT_S = 15.0

#: Größenlimit des HTML-Rohkörpers (T-F6: „z. B. 2 MB HTML“).
_MAX_HTML_BYTES = 2_000_000

#: Kappung des extrahierten Textes im Service (T-F6: „20k Zeichen Output-
#: Kappung“). Das Tool kapt zusätzlich enger (≤ ~6000), damit das Dispatch-
#: Kapping (8000, registry.TOOL_RESULT_MAX_CHARS) nie sinnlos frisst.
_MAX_TEXT_ZEICHEN = 20_000

#: Maximale Redirect-Hops (manuelle Schleife — jeder Hop erneut durch den Guard).
_MAX_REDIRECTS = 5

#: Redirect-Statuscodes, denen manuell gefolgt wird.
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})

#: Browser-UA für den Seiten-Abruf (v8-F-Erfahrung: viele Seiten antworten
#: krummen Bot-Filtern auf fremde UAs; ein aktueller Chrome-UA ist Hauskonvention).
_LESE_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

_LESE_HEADER = {"User-Agent": _LESE_UA, "Accept": "text/html,application/xhtml+xml"}

#: Cache-TTL (T-F6/E083: 24 h, Muster symbol_search_cache/T19).
_CACHE_TTL_S = 24 * 3600

#: Cache-TTL der News-Kategorie (Welle I/I3: News sind kurzlebig — 45 min).
_CACHE_TTL_NEWS_S = 45 * 60

#: Schlüssel-Präfix der News-Kategorie (ALTER-frei — Präfix im Schlüsselstring
#: statt Schema-Änderung; verhindert die web/news-Kollision bei gleicher Query).
_NEWS_SCHLUESSEL_PRAEFIX = "news|"


class WebsucheFehler(Exception):
    """Fachlicher Websuche-Fehler mit deutscher Meldung (Tools → ``ok:false``)."""


# ---------------------------------------------------------------------------
# Zeit/Modul-Zustand (Uhr + Schlaf injizierbar — Muster YahooRateLimiter)
# ---------------------------------------------------------------------------


def _uhr() -> float:
    """Monotone Uhr — Tests monkeypatchen diese Funktion (netzfreie Zeitsteuerung)."""
    return time.monotonic()


def _schlaf(sekunden: float) -> None:
    """Schläft die Pacing-Restzeit — Tests protokollieren statt schlafen."""
    time.sleep(sekunden)


_ZUSTAND_LOCK = threading.Lock()
_GEDROSSELT_BIS: float | None = None  # _uhr()-Wert; None = kein Sperrfenster
_LETZTER_SUCHE_SLOT: float | None = None  # _uhr()-Wert des letzten Pacing-Slots


def _pruefe_sperrfenster() -> None:
    """Offenes Backoff-Fenster → :class:`WebsucheFehler` **ohne** externen Call."""
    with _ZUSTAND_LOCK:
        if _GEDROSSELT_BIS is None:
            return
        rest = _GEDROSSELT_BIS - _uhr()
        if rest <= 0.0:
            return  # abgelaufen → Websuche wieder erlaubt (Fenster bleibt stehen)
    raise WebsucheFehler(
        f"Die Websuche ist nach einer Drosselung (Rate-Limit/Zeitüberschreitung) "
        f"noch für {max(rest, 1.0):.0f} s ausgesetzt — bitte kurz warten und es "
        "dann erneut versuchen."
    )


def _verzeichne_drossel() -> float:
    """Öffnet das Backoff-Fenster; liefert die Sperrdauer (Tests/Log)."""
    global _GEDROSSELT_BIS
    with _ZUSTAND_LOCK:
        _GEDROSSELT_BIS = _uhr() + _SPERRFENSTER_S
    return _SPERRFENSTER_S


def _wahre_suche_abstand() -> float:
    """Prozessweiter Mindestabstand zwischen zwei Such-Calls (Slot unter Lock).

    Muster ``YahooRateLimiter.vor_yahoo_aufruf`` (providers/rate_limit.py:96):
    Slot-Reservierung unter Lock, geschlafen wird **außerhalb** — der Schlaf
    blockiert keine parallelen Aufrufe der Reservierung selbst.
    """
    global _LETZTER_SUCHE_SLOT
    with _ZUSTAND_LOCK:
        moment = _uhr()
        warte = 0.0
        if _LETZTER_SUCHE_SLOT is not None:
            warte = _LETZTER_SUCHE_SLOT + _SUCHE_PACING_S - moment
            if warte < 0.0:
                warte = 0.0
        _LETZTER_SUCHE_SLOT = moment + warte
    if warte > 0.0:
        _schlaf(warte)
    return warte


def _zustand_zuruecksetzen() -> None:
    """Setzt Sperrfenster + Pacing-Slot zurück (Test-Hygiene, Muster
    ``stooq_provider._zustand_ok`` in tests/conftest.py)."""
    global _GEDROSSELT_BIS, _LETZTER_SUCHE_SLOT
    with _ZUSTAND_LOCK:
        _GEDROSSELT_BIS = None
        _LETZTER_SUCHE_SLOT = None


# ---------------------------------------------------------------------------
# suche
# ---------------------------------------------------------------------------


def _cache_frisch(fetched_at: str, ttl_s: float = _CACHE_TTL_S) -> bool:
    """Ist der Cache-Eintrag jünger als ``ttl_s`` (Web: 24 h, News: 45 min)?

    TTL-Bewertung beim Aufrufer (Muster ``_suche_cache_frisch``,
    api/routers/data.py:233) — defekte Zeitstempel gelten als abgelaufen.
    """
    try:
        moment = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return datetime.now(UTC) - moment < timedelta(seconds=ttl_s)


def _normalisiere(
    roh: list[Any], kategorie: str = "web"
) -> list[dict[str, str]]:
    """ddgs-Rohzeilen → Vertrag ``{titel, url, snippet, datum?}`` (F.5-Keys
    ``title``/``href``/``body``; ``datum`` nur, wenn die Quelle ein non-empty
    ``date`` liefert). Einträge ohne URL entfallen (Muster
    ``_suchergebnis_aufbereiten``).

    News-Kategorie (RECHERCHE §Welle I): ddgs liefert die URL unter ``url``
    (nicht ``href``) und zusätzlich ``source`` → wird als ``quelle``
    durchgereicht. Der Web-Pfad bleibt byte-identisch (kein ``quelle``, auch
    wenn ein Roh-Eintrag ``source`` mitbrächte).
    """
    ergebnisse: list[dict[str, str]] = []
    for eintrag in roh:
        if not isinstance(eintrag, dict):
            continue
        if kategorie == "news":
            href = eintrag.get("url") or eintrag.get("href")
        else:
            href = eintrag.get("href")
        if not href:
            continue
        zeile = {
            "titel": str(eintrag.get("title") or ""),
            "url": str(href),
            "snippet": str(eintrag.get("body") or ""),
        }
        datum = eintrag.get("date")
        if datum:
            zeile["datum"] = str(datum)
        if kategorie == "news":
            quelle = eintrag.get("source")
            if quelle:
                zeile["quelle"] = str(quelle)
        ergebnisse.append(zeile)
    return ergebnisse


def _suche_sync(
    query: str, max_results: int, timelimit: str | None, kategorie: str = "web"
) -> list[dict[str, str]]:
    """Sync-Teil von :func:`suche` (läuft im Thread via ``asyncio.to_thread``).

    Cache vor Netz: ein frischer (Web 24 h / News 45 min) Eintrag, der die
    angefragte Trefferzahl **decken** kann (``len >= max_results`` — Muster
    ``_symbolsuche`` M1), bedient ohne ddgs-Call; ein größerer ``max_results``
    refetcht und überschreibt. Der Cache-Schlüssel schließt ``timelimit`` ein
    (ANNAHME: der Zeitfenster-Filter ändert die Ergebnissemantik — „d/w/m/y“
    gehört zur Identität der Suche, anders als das reine ``limit`` bei T19)
    und — für die News-Kategorie — das Präfix ``news|`` (Welle I/I3: gleiche
    Query in web vs. news sind verschiedene Suchen, ALTER-frei im selben
    ``websuche_cache``-Schema).

    Pacing/Backoff (1,5 s / 60 s) gelten je Kategorie weiter; der
    Sperrfenster-Zustand (``_GEDROSSELT_BIS``) ist bewusst GEMEINSAM — eine
    Drosselung des ddgs-Backends betrifft beide Kategorien gleichermaßen
    (dasselbe Backend, derselbe Prozess; dokumentiert, I3-DoD).
    """
    schluessel = f"{query.casefold()}|{timelimit or ''}"
    if kategorie == "news":
        schluessel = f"{_NEWS_SCHLUESSEL_PRAEFIX}{schluessel}"
    ttl_s = _CACHE_TTL_NEWS_S if kategorie == "news" else _CACHE_TTL_S
    gecacht = websuche_cache_repo.get(schluessel)
    if (
        gecacht is not None
        and _cache_frisch(gecacht.fetched_at, ttl_s)
        and len(gecacht.results) >= max_results
    ):
        return list(gecacht.results)[:max_results]

    _pruefe_sperrfenster()
    _wahre_suche_abstand()
    try:
        # lazy (Muster yfinance lazy :417): Import/Testsuite bleibt netzfrei;
        # Tests monkeypatchen ddgs.DDGS (RECHERCHE F.5: Konstruktor- und
        # text()-Vertrag — proxy/timeout/verify gehören zum Konstruktor;
        # news()-Vertrag laut RECHERCHE §Welle I: gleiche Kern-Argumente).
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        client = DDGS(proxy=None, timeout=_DDGS_TIMEOUT_S, verify=True)
        if kategorie == "news":
            roh = client.news(
                query=query,
                region=_REGION,
                safesearch="moderate",
                timelimit=timelimit,
                max_results=max_results,
            )
        else:
            roh = client.text(
                query=query,
                region=_REGION,
                safesearch="moderate",
                timelimit=timelimit,
                max_results=max_results,
            )
    except RatelimitException as exc:
        dauer = _verzeichne_drossel()
        raise WebsucheFehler(
            "Die Websuche wird gerade gedrosselt (HTTP 403/Rate-Limit des "
            f"Suchbackends). Sie ist für {dauer:.0f} s ausgesetzt — danach "
            "automatisch wieder verfügbar; frühere Ergebnisse kommen aus dem "
            "Cache."
        ) from exc
    except TimeoutException as exc:
        _verzeichne_drossel()  # analog zum Rate-Limit: erst einmal rasten (T-F6-DoD)
        raise WebsucheFehler(
            "Die Websuche ist fehlgeschlagen (Zeitüberschreitung im "
            f"Suchbackend). Zur Schonung ist sie für {_SPERRFENSTER_S:.0f} s "
            "ausgesetzt — bitte danach erneut versuchen."
        ) from exc
    except DDGSException as exc:
        raise WebsucheFehler(f"Die Websuche ist fehlgeschlagen: {exc}") from exc

    # Voller normalisierter Satz in den Cache (Deckungsregel: ein späterer
    # größerer max_results refetcht, ein kleinerer wird aus dem Cache bedient),
    # die Rückgabe selbst kappet auf max_results — Vertrag „max_results-Respekt“.
    ergebnisse = _normalisiere(list(roh), kategorie)
    websuche_cache_repo.save(schluessel, ergebnisse)
    return ergebnisse[:max_results]


async def suche(
    query: str,
    max_results: int = 8,
    timelimit: str | None = None,
    kategorie: str = "web",
) -> list[dict[str, str]]:
    """Websuche über ddgs — normalisiert, gecacht (Web 24 h/News 45 min),
    mit Pacing/Backoff.

    ``timelimit`` (ddgs-Vertrag F.5): ``"d"``/``"w"``/``"m"``/``"y"`` oder
    ``None`` (unbegrenzt); ``max_results`` wird 1:1 an ddgs durchgereicht und
    zusätzlich beim Cache-Slice respektiert. ``kategorie`` (Welle I/I3,
    additiv): ``"web"`` (Default, byte-identischer Alt-Pfad) oder ``"news"``
    (``DDGS().news``, Treffer inkl. ``quelle``).
    """
    return await asyncio.to_thread(_suche_sync, query, max_results, timelimit, kategorie)


# ---------------------------------------------------------------------------
# lese (SSRF-Guard + httpx + trafilatura)
# ---------------------------------------------------------------------------


def _ist_blockiert(ip_str: str) -> bool:
    """IP gegen die SSRF-Blocklist (loopback, privat, link-local, ULA, ::1,
    0.0.0.0, reserviert, multicast). Unbekannte Formate sicherheitshalber blockiert."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    # fc00::/7 (ULA) und 10/8, 172.16/12, 192.168/16 stecken in is_private;
    # 169.254/16 (AWS-Metadaten!) in is_link_local; ::1/127/8 in is_loopback.
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


def _aufloesen(host: str) -> list[str]:
    """Hostname → alle resultierenden IPs (``socket.getaddrinfo``).

    Modulfunktion bewusst als Injektionspunkt: Tests mit MockTransport
    monkeypatchen sie, statt echte DNS-Queries zu stellen (E013-Netzfreiheit).
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise WebsucheFehler(
            f"Der Hostname „{host}“ konnte nicht aufgelöst werden — Seite "
            "existiert nicht (mehr) oder es besteht kein Netz."
        ) from exc
    return [str(info[4][0]) for info in infos]


def _pruefe_url(url: str) -> str:
    """SSRF-Guard (T-F6): Schema-Whitelist + Host-Auflösung + Blocklist.

    Läuft **vor jedem Request** und **je Redirect-Hop** erneut; liefert die
    (geprüfte) URL zurück. Fehler sind deutsche :class:`WebsucheFehler`.
    """
    try:
        geparst = httpx.URL(url)
    except Exception as exc:  # noqa: BLE001 — httpx wirft heterogene URL-Fehler
        raise WebsucheFehler(f"Ungültige URL: „{url}“.") from exc
    if geparst.scheme not in ("http", "https"):
        raise WebsucheFehler(
            f"Nicht unterstütztes URL-Schema „{geparst.scheme}“ — nur "
            "http(s)-Seiten können gelesen werden."
        )
    host = geparst.host
    if not host:
        raise WebsucheFehler(f"URL ohne Host: „{url}“.")
    ips = _aufloesen(host)
    blockiert = [ip for ip in ips if _ist_blockiert(ip)]
    if blockiert:
        raise WebsucheFehler(
            f"Zugriff verweigert: „{host}“ löst zu geschützter Adresse "
            f"({blockiert[0]}) auf — localhost und private/netzinterne Ziele "
            "sind aus Sicherheitsgründen blockiert (SSRF-Schutz)."
        )
    return str(geparst)


def _neuer_client() -> httpx.Client:
    """httpx-Client des Seiten-Abrufs (Injektionspunkt für Tests: MockTransport).

    ANNAHME: ``follow_redirects=False`` — Redirects werden manuell gefolgt
    (max. 5 Hops, **jeder** Ziel-URL erneut durch den SSRF-Guard, „Re-Check
    je Hop“), niemals blind von httpx.
    """
    return httpx.Client(
        headers=_LESE_HEADER,
        timeout=_LESE_TIMEOUT_S,
        follow_redirects=False,
    )


def _extrahiere(html: bytes, url: str) -> str:
    """HTML → Plain-Text via trafilatura (RECHERCHE F.6).

    Erstes Argument **positional** (exakter Parametername der Bibliothek ist
    dort nicht stabil), ``include_comments=False`` explizit (Default wäre
    AN), ``include_tables=False`` (Artikeltext, keine Tabellen-Noise) —
    leeres Ergebnis → genau **ein** Retry mit ``favor_recall=True``.
    """
    from trafilatura import extract  # lazy (Muster yfinance lazy :417)

    text = extract(
        html,
        output_format="txt",
        include_comments=False,
        include_tables=False,
        url=url,
        with_metadata=True,
    )
    if not text or not text.strip():
        text = extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=False,
            url=url,
            with_metadata=True,
            favor_recall=True,
        )
    return (text or "").strip()


def _titel_und_text(mit_metadaten: str) -> tuple[str | None, str]:
    """trafilatura-TXT mit ``with_metadata=True`` → ``(titel, body)``.

    Das Format beginnt optional mit einem YAML-artigen Kopf
    (``---\\ntitle: …\\n---``) — der Titel wird daraus gelöst, der Rest ist
    der Artikeltext. Ohne Kopf ist der Titel ``None`` (Fallback: Host).
    """
    if not mit_metadaten.startswith("---"):
        return None, mit_metadaten
    ende = mit_metadaten.find("\n---", 3)
    if ende < 0:
        return None, mit_metadaten
    kopf = mit_metadaten[3:ende]
    titel: str | None = None
    for zeile in kopf.splitlines():
        if zeile.lower().startswith("title:"):
            wert = zeile.split(":", 1)[1].strip()
            titel = wert or None
            break
    return titel, mit_metadaten[ende + 4 :].lstrip("\n")


def _lese_sync(url: str) -> dict[str, Any]:
    """Sync-Teil von :func:`lese` (läuft im Thread via ``asyncio.to_thread``)."""
    aktuelle_url = _pruefe_url(url)  # Guard vor dem ERSTEN Request
    if aktuelle_url.startswith("http://"):
        # http→https-Anhebung (Downgrade-Schutz): Klartext-Abrufe werden
        # vermieden; reine http-Hosts scheitern natürlich am TLS-Handshake.
        geparst = httpx.URL(aktuelle_url)
        port = None if geparst.port == 80 else geparst.port
        aktuelle_url = str(geparst.copy_with(scheme="https", port=port))

    client = _neuer_client()
    try:
        teile: list[bytes] = []
        for _hop in range(_MAX_REDIRECTS + 1):
            _pruefe_url(aktuelle_url)  # Re-Check je Hop (auch der erste)
            with client.stream("GET", aktuelle_url) as antwort:
                if antwort.status_code in _REDIRECT_CODES:
                    ziel = antwort.headers.get("location")
                    if not ziel:
                        raise WebsucheFehler(
                            f"Die Seite „{aktuelle_url}“ antwortet mit einer "
                            "Umleitung ohne Ziel."
                        )
                    aktuelle_url = str(httpx.URL(aktuelle_url).join(ziel))
                    if aktuelle_url.startswith("http://"):
                        # Downgrade-Schutz auch je Hop (REV-F2 MINOR-4): ein
                        # Redirect auf Klartext-http wird ebenfalls auf https angehoben.
                        gehopst = httpx.URL(aktuelle_url)
                        hport = None if gehopst.port == 80 else gehopst.port
                        aktuelle_url = str(gehopst.copy_with(scheme="https", port=hport))
                    continue  # äußerer Loop — Guard läuft für das Ziel erneut
                if antwort.status_code >= 400:
                    raise WebsucheFehler(
                        f"Die Seite „{aktuelle_url}“ antwortete mit HTTP "
                        f"{antwort.status_code} — nicht lesbar."
                    )
                content_type = (
                    antwort.headers.get("content-type", "").split(";")[0].strip().lower()
                )
                if content_type != "text/html":
                    raise WebsucheFehler(
                        f"Die Seite „{aktuelle_url}“ ist keine HTML-Seite "
                        f"(Content-Type „{content_type or 'unbekannt'}“) — "
                        "web_lesen liest ausschließlich text/html."
                    )
                gelesen = 0
                teile = []
                for chunk in antwort.iter_bytes():
                    gelesen += len(chunk)
                    if gelesen > _MAX_HTML_BYTES:
                        raise WebsucheFehler(
                            f"Die Seite „{aktuelle_url}“ ist größer als "
                            f"{_MAX_HTML_BYTES // 1_000_000} MB und wird bewusst "
                            "nicht vollständig geladen (Größenlimit)."
                        )
                    teile.append(chunk)
                break  # Erfolg — aus der Hop-Schleife
        else:
            raise WebsucheFehler(
                f"Zu viele Umleitungen (mehr als {_MAX_REDIRECTS}) für „{url}“."
            )
    except httpx.TimeoutException as exc:
        raise WebsucheFehler(
            f"Zeitüberschreitung beim Laden von „{aktuelle_url}“ "
            f"(> {_LESE_TIMEOUT_S:.0f} s) — bitte später erneut versuchen."
        ) from exc
    except httpx.HTTPError as exc:
        raise WebsucheFehler(
            f"Die Seite „{aktuelle_url}“ konnte nicht geladen werden: {exc}"
        ) from exc
    finally:
        client.close()

    mit_metadaten = _extrahiere(b"".join(teile), aktuelle_url)
    titel, body = _titel_und_text(mit_metadaten)
    zeichen_gesamt = len(body)
    gekuerzt = zeichen_gesamt > _MAX_TEXT_ZEICHEN
    if gekuerzt:
        body = (
            body[:_MAX_TEXT_ZEICHEN]
            + f"… [Text gekürzt: {zeichen_gesamt} Zeichen gesamt, Anzeige auf "
            f"{_MAX_TEXT_ZEICHEN} begrenzt.]"
        )
    host = httpx.URL(aktuelle_url).host or aktuelle_url
    return {
        "url": aktuelle_url,
        "titel": titel or host,
        "text": body,
        "zeichen": zeichen_gesamt,
        "gekuerzt": gekuerzt,
    }


async def lese(url: str) -> dict[str, Any]:
    """Holt eine Seite (SSRF-geprüft, max. 2 MB, HTML-only) und extrahiert
    den Artikeltext (trafilatura) — Ergebnis ``{url, titel, text, zeichen,
    gekuerzt}`` mit auf ≤ 20 000 Zeichen gekapptem Text."""
    return await asyncio.to_thread(_lese_sync, url)
