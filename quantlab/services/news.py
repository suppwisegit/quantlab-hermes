"""Börsennews-Service (Welle I, E104/I3): keyless RSS-Feeds, stdlib-only.

Zwei asynchrone Einstiege (die schwere Arbeit läuft via ``asyncio.to_thread``,
Muster :mod:`quantlab.services.websuche`):

- :func:`symbol_news` — Yahoo-Finance-RSS je Symbol (RECHERCHE I.B, live
  verifiziert): ``https://feeds.finance.yahoo.com/rss/2.0/headline?s={SYMBOL}``
  mit ``region=US&lang=en-US`` bzw. für ``.DE``-Suffixe
  ``region=DE&lang=de-DE``. Das Symbol kommt als **gültiges Yahoo-Kürzel**
  herein — die ISIN-Auflösung (Brücke ``agent/tools/daten.py::
  loese_isin_zu_kuerzel``) bewusst im TOOL-Layer (Import-Richtung: ein Service
  darf :mod:`quantlab.agent` nicht importieren); hier bleibt nur die
  Symbol-Validierung über :func:`providers.symbols.validate_symbol`
  (Muster ``services/fundamentals.py``).
- :func:`markt_news` — FinanzNachrichten.de-RSS je Kategorie
  (``aktien``/``markt``/``adhoc``; URLs **ohne** ``.rss``-Suffix — mit Suffix
  liefert der Host HTML, RECHERCHE I.B); ``pubDate`` dort ISO-8601.

Parsing bewusst mit der **Standardbibliothek** (``xml.etree.ElementTree``;
Datum per ``email.utils.parsedate_to_datetime`` bzw. ``datetime.
fromisoformat``) — KEINE neue Abhängigkeit (kein feedparser, PLAN §Welle I
„Ausdrücklich NICHT“). Items werden einzeln robust geparst: ein defektes/
unvollständiges Item wird übersprungen (ANNAHME zum I3-DoD „Teilerfolge
aggregieren“: je Kategorie liegt genau **ein** Feed zugrunde — Teilerfolg
bedeutet dort auf Item-Ebene; erst ein Feed ohne **einen** lesbaren Eintrag
ist ein deutscher Fachfehler).

Cache: :mod:`quantlab.storage.repos.websuche_cache_repo` wird ALTER-frei
wiederverwendet (Schlüssel → ``results_json`` + ``fetched_at``; TTL-Bewertung
beim Aufrufer — exakt die Repo-Konvention). Schlüssel mit Kategorie-Präfix
(„``news:symbol:SAP.DE``“, „``news:markt:aktien``“), TTL **45 min** (News sind
kurzlebig — Deckungsregel wie websuche: größerer ``max_items`` refetcht).

Pacing: min. 1 s zwischen Feed-Abrufen (Muster ``websuche.
_wahre_suche_abstand``) — bewusst **eigener, unabhängiger** Zustand neben dem
Websuche-Pacing (andere Hosts, andere Drossel-Politik; dokumentiert, I3-DoD).
Timeout 10 s je Feed. Fehler deutsch (:class:`NewsFehler` → Tool
``ok:false``). Kein SSRF-Guard nötig: Die URLs sind feste Konstanten aus
validiertem Symbol + allowlisted Hosts — keine nutzer-/LLM-gelieferten URLs.
"""

from __future__ import annotations

import asyncio
import html
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import httpx

from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.storage.repos import websuche_cache_repo

#: Request-Timeout je Feed-Abruf (I3-DoD: 10 s).
_FEED_TIMEOUT_S = 10.0

#: Mindestabstand zwischen zwei Feed-Abrufen (I3-DoD: ≥ 1 s; eigener Zustand).
_NEWS_PACING_S = 1.0

#: Cache-TTL (I3-DoD: News sind kurzlebig — 45 min).
_CACHE_TTL_S = 45 * 60

#: Kappung der Kurzbeschreibung je Nachricht („kurz“) — kompakte Tool-Liste.
_KURZ_MAX = 300

#: Yahoo-RSS-Basis (RECHERCHE I.B; Symbol/Region/Lang werden interpoliert).
_YAHOO_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline"

#: FinanzNachrichten-Feeds je Kategorie (RECHERCHE I.B: OHNE .rss-Suffix!).
_MARKT_FEEDS: dict[str, str] = {
    "aktien": "https://www.finanznachrichten.de/rss-aktien-nachrichten",
    "markt": "https://www.finanznachrichten.de/rss-marktberichte",
    "adhoc": "https://www.finanznachrichten.de/rss-aktien-adhoc",
}

#: Feed-UA (Hauskonvention aktueller Chrome-UA, Muster websuche._LESE_UA —
#: FinanzNachrichten antwortet krummen Bot-UAs teils nicht; eigene Konstante,
#: kein privater Import aus websuche).
_FEED_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

_FEED_HEADER = {
    "User-Agent": _FEED_UA,
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

_HTML_TAG_MUSTER = re.compile(r"<[^>]+>")


class NewsFehler(Exception):
    """Fachlicher News-Fehler mit deutscher Meldung (Tools → ``ok:false``)."""


# ---------------------------------------------------------------------------
# Zeit/Modul-Zustand (eigener Pacing-Slot — Muster websuche, unabhängig)
# ---------------------------------------------------------------------------


def _uhr() -> float:
    """Monotone Uhr — Tests monkeypatchen diese Funktion (netzfreie Zeitsteuerung)."""
    return time.monotonic()


def _schlaf(sekunden: float) -> None:
    """Schläft die Pacing-Restzeit — Tests protokollieren statt schlafen."""
    time.sleep(sekunden)


_ZUSTAND_LOCK = threading.Lock()
_LETZTER_FEED_SLOT: float | None = None  # _uhr()-Wert des letzten Pacing-Slots


def _wahre_feed_abstand() -> float:
    """Prozessweiter Mindestabstand zwischen zwei Feed-Abrufen (Slot unter
    Lock, geschlafen wird außerhalb — Muster ``websuche._wahre_suche_abstand``/
    ``YahooRateLimiter.vor_yahoo_aufruf``)."""
    global _LETZTER_FEED_SLOT
    with _ZUSTAND_LOCK:
        moment = _uhr()
        warte = 0.0
        if _LETZTER_FEED_SLOT is not None:
            warte = _LETZTER_FEED_SLOT + _NEWS_PACING_S - moment
            if warte < 0.0:
                warte = 0.0
        _LETZTER_FEED_SLOT = moment + warte
    if warte > 0.0:
        _schlaf(warte)
    return warte


def _zustand_zuruecksetzen() -> None:
    """Setzt den News-Pacing-Slot zurück (Test-Hygiene, Muster websuche)."""
    global _LETZTER_FEED_SLOT
    with _ZUSTAND_LOCK:
        _LETZTER_FEED_SLOT = None


# ---------------------------------------------------------------------------
# Feed-Abruf + RSS-Parsing (stdlib)
# ---------------------------------------------------------------------------


def _neuer_client() -> httpx.Client:
    """httpx-Client der Feed-Abrufe (Injektionspunkt für Tests: MockTransport).

    ANNAHME: ``follow_redirects=True`` — feste allowliste Hosts ohne
    nutzerkontrollierte URLs (anders als ``websuche.lese`` braucht es hier
    keinen je-Hop-Guard); Yahoo/FinanzNachrichten leiten gelegentlich um.
    """
    return httpx.Client(headers=_FEED_HEADER, timeout=_FEED_TIMEOUT_S, follow_redirects=True)


def _text_aus_html(roh: str) -> str:
    """RSS-``description`` (enthält oft HTML) → kompakter Klartext: Tags raus,
    Entities aufgelöst, Whitespace kollabiert (stdlib ``html``)."""
    text = html.unescape(_HTML_TAG_MUSTER.sub(" ", roh))
    return " ".join(text.split())


def _parse_rfc822(roh: str) -> datetime | None:
    """RFC-822-Datum (Yahoo-RSS) → :class:`datetime`, ``None`` bei Defekt."""
    try:
        return parsedate_to_datetime(roh)
    except (TypeError, ValueError):
        return None


def _parse_iso(roh: str) -> datetime | None:
    """ISO-8601-Datum (FinanzNachrichten) → :class:`datetime`, ``None`` bei
    Defekt; ``Z``-Suffix defensiv auf ``+00:00`` angehoben."""
    text = roh.strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    if text.endswith("Z"):
        try:
            return datetime.fromisoformat(text[:-1] + "+00:00")
        except ValueError:
            return None
    return None


def _normiere_datum(roh: str) -> str:
    """``pubDate`` (RFC-822 **oder** ISO-8601) → kanonischer ISO-Text; ein
    nicht parsbares Datum wird roh durchgereicht (lieber roh als fehlend)."""
    moment = _parse_rfc822(roh) or _parse_iso(roh)
    if moment is None:
        return roh
    return moment.isoformat()


def _parse_items(xml_bytes: bytes) -> list[dict[str, str]]:
    """RSS-2.0-XML → Nachrichtenliste ``{titel, url, datum?, kurz?}``.

    Nur ``channel/item``-Kinder: ``title``/``link`` Pflicht (ohne sie wird das
    Item übersprungen — Teilerfolg statt Totalausfall), ``pubDate`` →
    ``datum`` (normiert), ``description`` → ``kurz`` (HTML-frei, gekappt).
    Defektes XML insgesamt → :class:`NewsFehler`.
    """
    try:
        wurzel = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise NewsFehler(
            "Der News-Feed liefert kein gültiges XML (RSS 2.0) — die Quelle "
            "ist aktuell nicht lesbar."
        ) from exc
    items: list[dict[str, str]] = []
    for item in wurzel.iter("item"):
        titel = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        if not titel or not url:
            continue  # unvollständiges Item → überspringen (Teilerfolg)
        zeile = {"titel": titel, "url": url}
        pub_roh = (item.findtext("pubDate") or "").strip()
        if pub_roh:
            zeile["datum"] = _normiere_datum(pub_roh)
        beschreibung = (item.findtext("description") or "").strip()
        if beschreibung:
            zeile["kurz"] = _text_aus_html(beschreibung)[:_KURZ_MAX]
        items.append(zeile)
    return items


def _cache_frisch(fetched_at: str) -> bool:
    """Ist der Cache-Eintrag jünger als :data:`_CACHE_TTL_S` (45 min)?

    Spiegel von ``websuche._cache_frisch`` mit News-TTL (bewusst lokal —
    die 45-min-Politik ist news-spezifisch, das Repo bleibt zeitfrei).
    """
    try:
        moment = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return datetime.now(UTC) - moment < timedelta(seconds=_CACHE_TTL_S)


def _hole_feed(url: str, schluessel: str, max_items: int) -> list[dict[str, str]]:
    """Ein Feed holen (Cache vor Netz, Pacing, Timeout 10 s) und parsen.

    Cache-Deckung (Muster websuche): frischer Eintrag (45 min), der
    ``max_items`` decken kann, bedient ohne HTTP; sonst Fetch + Parsen +
    Upsert in den Cache. Deutsche :class:`NewsFehler` bei 4xx/5xx, Timeout,
    Transportdefekt, kaputtem XML oder null lesbaren Items.
    """
    gecacht = websuche_cache_repo.get(schluessel)
    if (
        gecacht is not None
        and _cache_frisch(gecacht.fetched_at)
        and len(gecacht.results) >= max_items
    ):
        return list(gecacht.results)[:max_items]

    _wahre_feed_abstand()
    client = _neuer_client()
    try:
        antwort = client.get(url)
        if antwort.status_code >= 400:
            raise NewsFehler(
                f"Der News-Feed „{url}“ ist nicht erreichbar (HTTP "
                f"{antwort.status_code}) — bitte später erneut versuchen."
            )
        xml_bytes = antwort.content
    except httpx.TimeoutException as exc:
        raise NewsFehler(
            f"Zeitüberschreitung beim Laden des News-Feeds „{url}“ "
            f"(> {_FEED_TIMEOUT_S:.0f} s) — bitte später erneut versuchen."
        ) from exc
    except httpx.HTTPError as exc:
        raise NewsFehler(
            f"Der News-Feed „{url}“ konnte nicht geladen werden: {exc}"
        ) from exc
    finally:
        client.close()

    items = _parse_items(xml_bytes)
    if not items:
        raise NewsFehler(
            f"Der News-Feed „{url}“ enthält aktuell keine Nachrichten."
        )
    websuche_cache_repo.save(schluessel, items)
    return items[:max_items]


# ---------------------------------------------------------------------------
# symbol_news / markt_news
# ---------------------------------------------------------------------------


def _symbol_news_sync(symbol: str, max_items: int) -> list[dict[str, str]]:
    """Sync-Teil von :func:`symbol_news` (Thread via ``asyncio.to_thread``)."""
    try:
        kuerzel = validate_symbol(symbol)
    except ProviderError as exc:
        # validate_symbol meldet bereits deutsch („ungültige Zeichen“ …).
        raise NewsFehler(str(exc)) from exc
    if kuerzel.upper().endswith(".DE"):
        region, sprache = "DE", "de-DE"
    else:
        region, sprache = "US", "en-US"
    url = f"{_YAHOO_FEED}?s={kuerzel}&region={region}&lang={sprache}"
    schluessel = f"news:symbol:{kuerzel.upper()}"
    return _hole_feed(url, schluessel, max_items)


async def symbol_news(symbol: str, max_items: int = 12) -> list[dict[str, str]]:
    """Aktuelle Nachrichten zu einem Symbol (Yahoo-Finance-RSS, 45-min-Cache).

    ``symbol`` ist ein **gültiges Yahoo-Kürzel** (z. B. ``SAP.DE``, ``AAPL``)
    — ISINs löst der Tool-Layer vorher auf (Import-Richtung Service → agent
    verboten, E104/I3). ``.DE``-Kürzel laufen mit ``region=DE&lang=de-DE``,
    alle anderen mit ``region=US&lang=en-US`` (RECHERCHE I.B).
    """
    return await asyncio.to_thread(_symbol_news_sync, symbol, max_items)


def _markt_news_sync(kategorie: str, max_items: int) -> list[dict[str, str]]:
    """Sync-Teil von :func:`markt_news` (Thread via ``asyncio.to_thread``)."""
    url = _MARKT_FEEDS.get(kategorie)
    if url is None:
        raise NewsFehler(
            f"Unbekannte Markt-Kategorie „{kategorie}“ — erlaubt sind: "
            f"{', '.join(sorted(_MARKT_FEEDS))}."
        )
    schluessel = f"news:markt:{kategorie}"
    return _hole_feed(url, schluessel, max_items)


async def markt_news(
    kategorie: str = "aktien", max_items: int = 15
) -> list[dict[str, str]]:
    """Aktuelle Markt-News je Kategorie (FinanzNachrichten-RSS, 45-min-Cache).

    ``kategorie``: ``"aktien"`` (Aktien-Nachrichten), ``"markt"``
    (Marktberichte) oder ``"adhoc"`` (Ad-hoc-Meldungen); URLs ohne
    ``.rss``-Suffix (RECHERCHE I.B — mit Suffix liefert der Host HTML).
    """
    return await asyncio.to_thread(_markt_news_sync, kategorie, max_items)
