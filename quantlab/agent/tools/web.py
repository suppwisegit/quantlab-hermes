"""Websuche-Tools (Welle F, T-F6/E083; Welle I/E104-I3): ``web_suche``,
``web_lesen`` und ``news_suche``.

Konsumiert ausschließlich :mod:`quantlab.services.websuche` (ddgs-Suche mit
Cache/Pacing/Backoff + httpx/trafilatura-Leseabruf mit SSRF-Guard) und
:mod:`quantlab.services.news` (Yahoo-/FinanzNachrichten-RSS) — keine
Bibliotheks-Imports hier, kein eigenes httpx. Die Service-Aufrufe sind bereits
``async`` (Thread-Lagerung im Service, Muster planspiel); Fachfehler
(:class:`~quantlab.services.websuche.WebsucheFehler`,
:class:`~quantlab.services.news.NewsFehler`, deutsche Meldungen) werden 1:1
auf :class:`ToolFehler` gemappt — ``ok:false``-Tool-Ergebnis, der Run läuft
weiter (§E).

Settings-Gate: Alle drei Handler prüfen die Sektion ``websuche`` (Default
``enabled: true``) je Aufruf — deaktiviert → deutscher Fachfehler
(ANNAHME I3: ``news_suche`` gehört als Recherche-Tool zur selben
Websuche-Familie und teilt deren Schalter; eine eigene Sektion wäre ein
Settings-Schema-Eingriff, der nicht im Task-Scope liegt).

Kappung (T-F6): Ergebnis-Strings werden so gekappt, dass das Dispatch-Kapping
(8000 Zeichen, ``registry.TOOL_RESULT_MAX_CHARS``) nicht sinnlos frisst —
``web_suche`` hält das JSON unter ~6500 Zeichen (Feld-Kappung + Treffer-
Budget), ``web_lesen`` zielt auf ≤ 6000 Zeichen Text, ``news_suche`` auf
< 7000 Zeichen insgesamt (I3-DoD).

ISIN-Brücke (E104/I3): ``news_suche`` im Modus ``symbol`` akzeptiert ISIN
**oder** Kürzel — die Auflösung läuft hier im Tool-Layer über die bestehende
Brücke :func:`daten.loese_isin_zu_kuerzel` (Import-Richtung: der Service darf
``quantlab.agent`` nicht importieren, also bekommt er nur das Kürzel; Muster
``planspiel.py``/``daten.py``).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, pruefe_args, schema_von
from quantlab.agent.tools.daten import ist_isin, loese_isin_zu_kuerzel
from quantlab.services import news as news_service
from quantlab.services import websuche
from quantlab.storage.repos import settings_repo

#: Feld-Kappung je Suchtreffer (Zeichen) — URLs können entartet lang sein.
_TITEL_MAX = 200
_URL_MAX = 500
_SNIPPET_MAX = 300

#: Feld-Kappung je Nachricht in ``news_suche`` (Zeichen).
_QUELLE_MAX = 120
_DATUM_MAX = 40
_NEWS_KURZ_MAX = 300

#: Zeichen-Budget der Trefferliste (JSON-Länge) — hält das Tool-Result unter
#: dem Dispatch-Kapping (8000, registry.py:47), ohne dass dort gekappt wird.
_SUCHE_ERGEBNIS_MAX_ZEICHEN = 6500

#: Zeichen-Budget der Nachrichtenliste in ``news_suche`` (I3-DoD: GESAMT
#: < 7000 Z inkl. Meta-Felder).
_NEWS_ERGEBNIS_MAX_ZEICHEN = 6200

#: Ziel-Kappung des Seiten-Texts in ``web_lesen`` (T-F6: „web_lesen Ziel
#: ≤ ~6000 Z“) — der Service kapt grob bei 20 000 (websuche._MAX_TEXT_ZEICHEN).
_LESE_TEXT_MAX_ZEICHEN = 6000

#: ddgs-``timelimit``-Werte (RECHERCHE F.5) je deutschem Args-Wert.
_ZEITRAUM_NACH_TIMELIMIT: dict[str, str] = {
    "tag": "d",
    "woche": "w",
    "monat": "m",
    "jahr": "y",
}


def _websuche_sektion() -> dict[str, Any]:
    """Settings-Sektion ``websuche`` — leeres Dict, falls der Bestand sie fehlt."""
    try:
        return settings_repo.get_section("websuche")
    except ValueError:  # alter Bestand ohne Migration der Defaults (P1-Muster)
        return {}


def _sichere_aktiviert() -> None:
    """Settings-Gate (T-F6): deaktiviert → deutscher Fachfehler, kein Netz-Call."""
    if not bool(_websuche_sektion().get("enabled", True)):
        raise ToolFehler(
            "Websuche ist in den Einstellungen deaktiviert — sie kann in den "
            "Einstellungen (Sektion „Websuche“) wieder eingeschaltet werden."
        )


# ---------------------------------------------------------------------------
# web_suche
# ---------------------------------------------------------------------------


class WebSucheArgs(BaseModel):
    """Argumente von ``web_suche``."""

    query: str = Field(
        min_length=1,
        max_length=200,
        description='Suchbegriff bzw. Frage, z. B. "DAX Quartalszahlen" oder '
        '"SAP Q2 Ergebnisse".',
    )
    max_results: int = Field(
        default=8,
        ge=1,
        le=15,
        description="Maximale Trefferzahl (1–15), Default 8.",
    )
    zeitraum: Literal["tag", "woche", "monat", "jahr"] | None = Field(
        default=None,
        description='Optionaler Zeitfilter: "tag", "woche", "monat" oder "jahr" '
        "(leer = unbegrenzt).",
    )
    thema: Literal["web", "news"] = Field(
        default="web",
        description='Suchkategorie: "web" (allgemeine Websuche, Default) oder '
        '"news" (Nachrichtensuche — Treffer mit datum und quelle, für '
        "Börsennews bevorzugt).",
    )


def _kappe_treffer(treffer: list[dict[str, str]]) -> list[dict[str, str]]:
    """Feld-Kappung je Treffer + Gesamtbudget (JSON-Zeichen) — siehe Modul-Kopf.

    Der erste Treffer kommt immer durch (eine Zeile bleibt unter jedem Budget),
    danach bricht die Aufnahme ab, sobald das Budget erschöpft wäre — so bleibt
    das serialisierte Ergebnis deterministisch unter dem Dispatch-Kapping.
    News-Treffer können ``quelle`` tragen (Welle I/I3); der Web-Pfad bleibt
    ohne ``quelle`` (byte-identischer Alt-Vertrag).
    """
    budget = _SUCHE_ERGEBNIS_MAX_ZEICHEN
    gehalten: list[dict[str, str]] = []
    for zeile in treffer:
        gekappt = {
            "titel": zeile.get("titel", "")[:_TITEL_MAX],
            "url": zeile.get("url", "")[:_URL_MAX],
            "snippet": zeile.get("snippet", "")[:_SNIPPET_MAX],
            **(
                {"datum": zeile["datum"][:_TITEL_MAX]} if zeile.get("datum") else {}
            ),
            **(
                {"quelle": zeile["quelle"][:_QUELLE_MAX]}
                if zeile.get("quelle")
                else {}
            ),
        }
        laenge = len(json.dumps(gekappt, ensure_ascii=False))
        if gehalten and laenge > budget:
            break
        gehalten.append(gekappt)
        budget -= laenge
    return gehalten


async def web_suche(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Durchsucht das Web (DuckDuckGo via ddgs, 24-h-Cache, Pacing/Backoff);
    ``thema="news"`` schaltet auf die Nachrichtensuche (45-min-Cache)."""
    args = pruefe_args(WebSucheArgs, roh_args, "web_suche")
    _sichere_aktiviert()
    timelimit = (
        _ZEITRAUM_NACH_TIMELIMIT[args.zeitraum] if args.zeitraum is not None else None
    )
    try:
        treffer = await websuche.suche(
            args.query,
            max_results=args.max_results,
            timelimit=timelimit,
            kategorie=args.thema,  # "web"/"news" 1:1 als Kategorie
        )
    except websuche.WebsucheFehler as exc:
        raise ToolFehler(str(exc)) from exc
    gehalten = _kappe_treffer(treffer)
    antwort: dict[str, Any] = {
        "query": args.query,
        "anzahl": len(gehalten),
        "ergebnisse": gehalten,
        "hinweis": "Seiteninhalt einer Quelle liest web_lesen mit der "
        "passenden URL; identische Suchen kommen aus dem Cache "
        "(web 24 h, news 45 min).",
    }
    if args.thema == "news":
        # Additiv nur im news-Fall — Alt-Antwort (thema web) bleibt byte-identisch.
        antwort["thema"] = "news"
    if len(gehalten) < len(treffer):
        antwort["gekuerzt"] = True
        antwort["hinweis"] = (
            f"{len(treffer) - len(gehalten)} weitere Treffer wurden zur "
            "Kontext-Schonung gekürzt. " + antwort["hinweis"]
        )
    return antwort


# ---------------------------------------------------------------------------
# web_lesen
# ---------------------------------------------------------------------------


class WebLesenArgs(BaseModel):
    """Argumente von ``web_lesen``."""

    url: str = Field(
        min_length=1,
        max_length=2048,
        description="Vollständige http(s)-URL der Seite, z. B. aus web_suche "
        '(Feld "url").',
    )


async def web_lesen(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Liest eine Seite (SSRF-geprüft, HTML-only) und extrahiert den Text."""
    args = pruefe_args(WebLesenArgs, roh_args, "web_lesen")
    _sichere_aktiviert()
    try:
        daten = await websuche.lese(args.url)
    except websuche.WebsucheFehler as exc:
        raise ToolFehler(str(exc)) from exc
    text = str(daten.get("text") or "")
    if not text:
        raise ToolFehler(
            f"Auf der Seite „{args.url}“ konnte kein Artikeltext erkannt "
            "werden (möglicherweise eine JS-App oder ein leeres Layout) — "
            "bitte eine andere Quelle wählen."
        )
    zeichen_gesamt = int(daten.get("zeichen") or len(text))
    gekuerzt = bool(daten.get("gekuerzt"))
    if len(text) > _LESE_TEXT_MAX_ZEICHEN:
        text = (
            text[:_LESE_TEXT_MAX_ZEICHEN]
            + f"… [Text gekürzt: {zeichen_gesamt} Zeichen gesamt, Anzeige auf "
            f"{_LESE_TEXT_MAX_ZEICHEN} begrenzt.]"
        )
        gekuerzt = True
    return {
        "titel": str(daten.get("titel") or args.url),
        "quelle": str(daten.get("url") or args.url),
        "text": text,
        "zeichen_gesamt": zeichen_gesamt,
        "gekuerzt": gekuerzt,
    }


# ---------------------------------------------------------------------------
# news_suche (Welle I, E104/I3): Symbol-/Markt-RSS + News-Websuche
# ---------------------------------------------------------------------------


class NewsSucheArgs(BaseModel):
    """Argumente von ``news_suche``."""

    modus: Literal["symbol", "markt", "suche"] = Field(
        description='Wahl des News-Zugangs: "symbol" (Nachrichten zu einem '
        'Wertpapier), "markt" (Markt-/Ad-hoc-News je Kategorie) oder "suche" '
        "(freie News-Schlagwortsuche).",
    )
    symbol: str | None = Field(
        default=None,
        max_length=24,
        description='Nur modus "symbol": Yahoo-Kürzel (z. B. "SAP.DE", "AAPL") '
        'oder ISIN (z. B. "DE0007164600" → wird zu SAP.DE aufgelöst).',
    )
    markt_kategorie: Literal["aktien", "markt", "adhoc"] = Field(
        default="aktien",
        description='Nur modus "markt": "aktien" (Aktien-News), "markt" '
        '(Marktberichte) oder "adhoc" (Ad-hoc-Meldungen), Default "aktien".',
    )
    abfrage: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description='Nur modus "suche": Schlagwort bzw. Frage, z. B. '
        '"EZB Zinsentscheidung".',
    )
    zeitraum: Literal["tag", "woche", "monat", "jahr"] | None = Field(
        default=None,
        description='Nur modus "suche" — optionaler Zeitfilter: "tag", '
        '"woche", "monat" oder "jahr" (leer = unbegrenzt).',
    )
    max_items: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximale Nachrichtenzahl (1–20), Default 10.",
    )


def _kappe_news(nachrichten: list[dict[str, str]]) -> list[dict[str, str]]:
    """Feld-Kappung je Nachricht + Gesamtbudget (JSON-Zeichen) — Muster
    ``_kappe_treffer``. News-Websuche-Treffer liefern ihre Kurzfassung unter
    ``snippet`` (websuche-Vertrag), Symbol-/Markt-Feeds unter ``kurz`` —
    beides geht als ``kurz`` in das Ergebnis (einheitlicher Nachrichten-
    Vertrag: titel/datum/quelle?/url/kurz?)."""
    budget = _NEWS_ERGEBNIS_MAX_ZEICHEN
    gehalten: list[dict[str, str]] = []
    for zeile in nachrichten:
        kurz = zeile.get("kurz") or zeile.get("snippet") or ""
        gekappt = {
            "titel": zeile.get("titel", "")[:_TITEL_MAX],
            "url": zeile.get("url", "")[:_URL_MAX],
            **(
                {"datum": zeile["datum"][:_DATUM_MAX]} if zeile.get("datum") else {}
            ),
            **(
                {"quelle": zeile["quelle"][:_QUELLE_MAX]}
                if zeile.get("quelle")
                else {}
            ),
            **({"kurz": kurz[:_NEWS_KURZ_MAX]} if kurz else {}),
        }
        laenge = len(json.dumps(gekappt, ensure_ascii=False))
        if gehalten and laenge > budget:
            break
        gehalten.append(gekappt)
        budget -= laenge
    return gehalten


async def news_suche(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Aktuelle Börsen-News je Modus (Symbol-RSS, Markt-RSS oder News-Suche)."""
    args = pruefe_args(NewsSucheArgs, roh_args, "news_suche")
    _sichere_aktiviert()  # ANNAHME: news_suche teilt den Websuche-Schalter

    if args.modus == "symbol":
        roh_symbol = (args.symbol or "").strip()
        if not roh_symbol:
            raise ToolFehler(
                'modus "symbol" benötigt ein Symbol — Yahoo-Kürzel '
                '(z. B. "SAP.DE") oder ISIN (z. B. "DE0007164600").'
            )
        isin: str | None = None
        if ist_isin(roh_symbol):
            isin = roh_symbol.upper()
            kuerzel = loese_isin_zu_kuerzel(isin)
            if kuerzel is None:
                raise ToolFehler(
                    f"ISIN unbekannt — Symbol in der Suche suchen oder Registry "
                    f"pflegen (search_symbols mit der ISIN aufrufen oder das "
                    f"Kürzel in der Symbol-Registry hinterlegen): {isin}"
                )
        else:
            kuerzel = roh_symbol
        try:
            nachrichten = await news_service.symbol_news(
                kuerzel, max_items=args.max_items
            )
        except news_service.NewsFehler as exc:
            raise ToolFehler(str(exc)) from exc
        gehalten = _kappe_news(nachrichten)
        antwort: dict[str, Any] = {
            "modus": "symbol",
            "symbol": kuerzel,
            "anzahl": len(gehalten),
            "news": gehalten,
            "hinweis": "Quelle: Yahoo-Finance-RSS (45-min-Cache); den "
            "Volltext einer Nachricht liest web_lesen mit der URL.",
        }
        if isin is not None:
            antwort["isin"] = isin
        if not gehalten:
            antwort["hinweis"] = (
                f"Keine aktuellen Nachrichten für {kuerzel} gefunden — "
                "Symbol prüfen oder modus \u201esuche\u201c mit Firmennamen probieren."
            )
        rohliste = nachrichten

    elif args.modus == "markt":
        try:
            nachrichten = await news_service.markt_news(
                args.markt_kategorie, max_items=args.max_items
            )
        except news_service.NewsFehler as exc:
            raise ToolFehler(str(exc)) from exc
        gehalten = _kappe_news(nachrichten)
        antwort = {
            "modus": "markt",
            "kategorie": args.markt_kategorie,
            "anzahl": len(gehalten),
            "news": gehalten,
            "hinweis": "Quelle: FinanzNachrichten-RSS (45-min-Cache); den "
            "Volltext einer Nachricht liest web_lesen mit der URL.",
        }
        rohliste = nachrichten

    else:  # modus == "suche"
        abfrage = (args.abfrage or "").strip()
        if not abfrage:
            raise ToolFehler(
                'modus "suche" benötigt eine "abfrage" — Schlagwort bzw. '
                'Frage (z. B. "EZB Zinsentscheidung").'
            )
        timelimit = (
            _ZEITRAUM_NACH_TIMELIMIT[args.zeitraum]
            if args.zeitraum is not None
            else None
        )
        try:
            treffer = await websuche.suche(
                abfrage,
                max_results=args.max_items,
                timelimit=timelimit,
                kategorie="news",
            )
        except websuche.WebsucheFehler as exc:
            raise ToolFehler(str(exc)) from exc
        gehalten = _kappe_news(treffer)
        antwort = {
            "modus": "suche",
            "abfrage": abfrage,
            "anzahl": len(gehalten),
            "news": gehalten,
            "hinweis": "Quelle: News-Websuche (45-min-Cache); den Volltext "
            "eines Treffers liest web_lesen mit der URL.",
        }
        rohliste = treffer

    if len(gehalten) < len(rohliste):
        antwort["gekuerzt"] = True
        antwort["hinweis"] = (
            f"{len(rohliste) - len(gehalten)} weitere Nachrichten wurden zur "
            "Kontext-Schonung gekürzt. " + antwort["hinweis"]
        )
    return antwort


# ---------------------------------------------------------------------------
# Registrierung (T-F6/E083 — Muster daten.py: _registrieren + Modul-Call)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="web_suche",
            description=(
                "Durchsucht das Web (DuckDuckGo, mit Cache und eigenem "
                "Tempo-Limit) nach aktuellen Informationen — Quartalszahlen, "
                'Meldungen, Hintergründe; optional auf tag/woche/monat/jahr '
                'eingeschränkt. Mit thema="news" echte Nachrichtensuche '
                "(Treffer dann mit datum und quelle) — für Börsennews ist "
                "aber news_suche die erste Wahl. Liefert je Treffer "
                "titel/url/snippet; den Seiteninhalt liest web_lesen. "
                'Beispiel: {"query": "DAX Quartalszahlen", "thema": "news", '
                '"max_results": 5}.'
            ),
            parameters=schema_von(WebSucheArgs),
            handler=web_suche,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="news_suche",
            description=(
                "Sucht aktuelle Börsen-News in drei Modi: zu einem Symbol "
                "(Yahoo-Finance-RSS; ISIN oder Kürzel wie SAP.DE/AAPL), je "
                "Marktkategorie aktien/markt/adhoc (FinanzNachrichten-RSS) "
                "oder frei nach Schlagwort (News-Websuche). Liefert je "
                "Nachricht titel/datum/quelle/url/kurz (kompakt, max. 20); "
                "den Volltext liest web_lesen mit der URL. "
                'Beispiel: {"modus": "symbol", "symbol": "SAP.DE", '
                '"max_items": 5}.'
            ),
            parameters=schema_von(NewsSucheArgs),
            handler=news_suche,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="web_lesen",
            description=(
                "Liest eine konkrete Webseite (http(s)-URL, z. B. aus "
                "web_suche) und extrahiert den Artikeltext als Plain-Text "
                "(HTML-only, max. ~6000 Zeichen; localhost/privates Netz sind "
                "aus Sicherheitsgründen gesperrt). "
                'Beispiel: {"url": "https://www.example.com/artikel"}.'
            ),
            parameters=schema_von(WebLesenArgs),
            handler=web_lesen,
        )
    )


_registrieren()
