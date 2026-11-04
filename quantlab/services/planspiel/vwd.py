"""VWD-Marktdaten-Snippets (Welle H3, E095): Fundamentaldaten/Analysten der
Planspiel-Börse als HTML-Fragmente laden und in Daten wandeln.

Die Seite bezieht Fundamental- und Marktdaten von
``business-customer.vwd.com/planspiel/snippet.htn?sektion=…&lang=de&i={idNotation}``
(RECHERCHE H.3 — kein Auth nötig, Antwort ist ein HTML-Fragment; ``i`` ist die
idNotation aus ``getInstrumentWithQuotes`` → ``quoteList[0]``). Die Parser sind
1:1 aus der Referenz-Implementierung ``planspiel_data_tools``/``app/psb/
marktdaten_vwd.py`` übernommen (``tabellen_parsen``/``als_dict``/``_zahl``/
``performance_parsen``/``analysten_parsen``) und bewusst tolerant, weil die
Snippets Präsentations-HTML sind, kein Vertrag — nicht-numerische Werte
(z. B. „226,68 Mrd. €“) bleiben Strings.

Fehler folgen dem Fachfehler-Stil des planspiel-Pakets
(:class:`~quantlab.services.planspiel.fehler.PlanspielFehler`, E069-Codes):
Netzpannen → ``netzwerk``, 429/503 → ``rate_limit``, sonstige Nicht-2xx →
``vertrag_geaendert``. Eine unbekannte Sektion ist ein **Aufrufer-Fehler**
(kein Fachfall) → ``ValueError`` deutsch, Muster ``psb_repo.upsert``.
"""

from __future__ import annotations

import html
import re

import httpx

from quantlab.services.planspiel.fehler import PlanspielFehler

#: URL-Vertrag der VWD-Snippets (RECHERCHE H.3, Referenz ``endpunkte.VWD_SNIPPET``).
VWD_SNIPPET_URL = "https://business-customer.vwd.com/planspiel/snippet.htn"

#: Geschlossene Sektions-Menge der Snippet-API (RECHERCHE H.3).
VWD_SEKTIONEN: tuple[str, ...] = (
    "getKennzahlen",
    "getPerformance",
    "getKursdaten",
    "getAnalystenempfehlungen",
    "getUnternehmensprofil",
)

#: Request-Timeout je Snippet-Abruf (Referenz: 15 s).
TIMEOUT_S = 15.0

#: Kappung des Unternehmensprofil-Texts (Referenz ``markt._profil_text``).
PROFIL_MAX_ZEICHEN = 2000

#: Injektionspunkt für Tests (httpx.MockTransport, E013: Suite bleibt netzfrei) —
#: Muster ``screener._TRANSPORT``; ``None`` = echter Netzabruf.
_TRANSPORT: httpx.BaseTransport | None = None

_ZELLE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_ZEILE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    """HTML-Zellen-Fragment → Klartext (Tags weg, Entities dekodiert, NBSP → Leerzeichen)."""
    return html.unescape(_TAG.sub("", fragment)).replace("\xa0", " ").strip()


def _zahl(text: str) -> float | str:
    """„1.234,56 €“ → 1234.56; nicht-numerisch bleibt String (Referenz ``_zahl``)."""
    gereinigt = text.replace("€", "").replace("%", "").replace("−", "-").strip()
    if not gereinigt or gereinigt in ("--", "-", "k.A.", "n/a"):
        return text.strip()
    norm = gereinigt.replace(".", "").replace(",", ".")
    try:
        return float(norm)
    except ValueError:
        return text.strip()


def snippet_holen(
    sektion: str,
    id_notation: str,
    *,
    sprache: str = "de",
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = TIMEOUT_S,
) -> str:
    """Lädt EIN Snippet-Fragment (HTML) — URL-Vertrag RECHERCHE H.3.

    Parameter 1:1 wie die Referenz: ``sektion``/``lang=de``/``i={idNotation}``/
    ``ajax=1``. Unbekannte Sektion → ``ValueError`` (Aufrufer-Fehler, kein
    Fachfall); Netzpanne → ``netzwerk``, 429/503 → ``rate_limit``, sonstiger
    Nicht-2xx → ``vertrag_geaendert`` (E069-Codes, deutscher Fachfehler-Stil).
    """
    if sektion not in VWD_SEKTIONEN:
        raise ValueError(
            f"Unbekannte VWD-Sektion „{sektion}“ — erlaubt sind: "
            f"{', '.join(VWD_SEKTIONEN)}."
        )
    nutze_transport = transport if transport is not None else _TRANSPORT
    try:
        with httpx.Client(transport=nutze_transport, timeout=timeout_s) as client:
            antwort = client.get(
                VWD_SNIPPET_URL,
                params={
                    "sektion": sektion,
                    "lang": sprache,
                    "i": id_notation,
                    "ajax": "1",
                },
            )
    except httpx.HTTPError as exc:
        raise PlanspielFehler(
            "netzwerk",
            "Die VWD-Marktdaten (Snippet) sind gerade nicht erreichbar "
            f"(Netzwerkfehler: {type(exc).__name__}). Bitte Verbindung prüfen "
            "und später erneut versuchen.",
        ) from exc
    if antwort.status_code in (429, 503):
        raise PlanspielFehler(
            "rate_limit",
            "Die VWD-Marktdaten bitten gerade um Entlastung (HTTP "
            f"{antwort.status_code}) — bitte kurz warten und erneut versuchen.",
        )
    if not 200 <= antwort.status_code < 300:
        raise PlanspielFehler(
            "vertrag_geaendert",
            f"Unerwartete HTTP-Antwort der VWD-Marktdaten bei „{sektion}“ "
            f"(Status {antwort.status_code}).",
        )
    return antwort.text


def tabellen_parsen(html_text: str) -> list[list[dict]]:
    """Alle Tabellen des Snippets → Liste von Zeilen-Listen ``{'label','wert'}``.

    Kopfzeilen („Period“/„Periode“) und Zeilen mit weniger als zwei
    verbleibenden Zellen werden übersprungen (tolerant — Präsentations-HTML).
    """
    tabellen: list[list[dict]] = []
    for tab in re.findall(r"<table[^>]*>(.*?)</table>", html_text, re.S | re.I):
        zeilen: list[dict] = []
        for z in _ZEILE.finditer(tab):
            zellen = [_text(c) for c in _ZELLE.findall(z.group(1))]
            zellen = [z for z in zellen if z != ""]
            if len(zellen) >= 2 and zellen[0] not in ("Period", "Periode"):
                zeilen.append({"label": zellen[0], "wert": _zahl(zellen[1])})
        if zeilen:
            tabellen.append(zeilen)
    return tabellen


def als_dict(html_text: str) -> dict:
    """Flacht alle Tabellen zu einem ``label → wert``-Dict zusammen (Referenz)."""
    erg: dict = {}
    for tab in tabellen_parsen(html_text):
        for zeile in tab:
            erg.setdefault(zeile["label"], zeile["wert"])
    return erg


def performance_parsen(html_text: str) -> dict[str, dict]:
    """Spezialfall Performance: Zeiträume mit Kurs- UND Prozent-Spalte.

    Originaltabelle: Zeitraum | Kurs (€) | % →
    ``{'1 Monat': {'kurs': 186.0, 'prozent': -0.96}}`` (Referenz-Muster).
    """
    erg: dict[str, dict] = {}
    for tab in re.findall(r"<table[^>]*>(.*?)</table>", html_text, re.S | re.I):
        for z in _ZEILE.finditer(tab):
            zellen = [_text(c) for c in _ZELLE.findall(z.group(1))]
            zellen = [z for z in zellen if z != ""]
            if len(zellen) < 3 or zellen[0] in ("Period", "Periode", "Zeitraum"):
                continue
            erg[zellen[0]] = {"kurs": _zahl(zellen[1]), "prozent": _zahl(zellen[2])}
    return erg


def analysten_parsen(html_text: str) -> dict:
    """Spezialfall Analysten (Referenz): kauf/neutral/verkauf-Zähler plus
    gewichtete Empfehlung in % und Kursziel in €.

    „Gewichtete Empfehlung“ erscheint im Snippet zweimal: erst als %-Score,
    dann als Kursziel in € — die Reihenfolge entscheidet die Zuordnung.
    Fehlt ein Teil, bleibt der Schlüssel weg (tolerant).
    """
    text = html.unescape(_TAG.sub(" ", html_text))
    erg: dict = {}
    for m in re.finditer(r"Buy:\s*(\d+)", text):
        erg["kauf"] = int(m.group(1))
    for m in re.finditer(r"Neutral:\s*(\d+)", text):
        erg["neutral"] = int(m.group(1))
    for m in re.finditer(r"Sell:\s*(\d+)", text):
        erg["verkauf"] = int(m.group(1))
    treffer = re.findall(r"Gewichtete Empfehlung\s*([0-9][0-9.,]*)\s*(%|&euro;|€)", text)
    for wert, einheit in treffer:
        if einheit == "%":
            erg["empfehlung_prozent"] = _zahl(wert + "%")
        else:
            erg["kursziel"] = _zahl(wert + "€")
    return erg


def profil_text(html_text: str) -> str:
    """Unternehmensprofil-Snippet → Plaintext, auf 2000 Zeichen gekappt
    (Referenz ``markt._profil_text`` — Profil-Kappung)."""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"\s+", " ", text)
    return text.replace("Unternehmensprofil", "", 1).strip()[:PROFIL_MAX_ZEICHEN]


def kennzahlen_laden(
    id_notation: str,
    *,
    sprache: str = "de",
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """Lädt und parst ALLE Sektionen je idNotation (Pipeline des frisch-Refreshs).

    Antwort-Schlüssel sind die Sektions-Namen; ``getKennzahlen``/``getKursdaten``
    als ``label→wert``-Dict, ``getPerformance`` als ``zeitraum→{kurs, prozent}``,
    ``getAnalystenempfehlungen`` als kauf/neutral/verkauf/empfehlung_prozent/
    kursziel, ``getUnternehmensprofil`` als ``{'text': …}`` (gekappt). Ein
    :class:`PlanspielFehler` aus :func:`snippet_holen` bricht ab — bereits
    geladene Sektionen werden dann nicht teilgeschrieben (Upsert macht der
    Aufrufer erst nach vollständigem Lauf).
    """
    erg: dict = {}
    for sektion in VWD_SEKTIONEN:
        html_text = snippet_holen(
            sektion, id_notation, sprache=sprache, transport=transport
        )
        if sektion in ("getKennzahlen", "getKursdaten"):
            erg[sektion] = als_dict(html_text)
        elif sektion == "getPerformance":
            erg[sektion] = performance_parsen(html_text)
        elif sektion == "getAnalystenempfehlungen":
            erg[sektion] = analysten_parsen(html_text)
        else:  # getUnternehmensprofil — letzte Sektion der geschlossenen Menge
            erg[sektion] = {"text": profil_text(html_text)}
    return erg
