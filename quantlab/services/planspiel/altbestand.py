"""Altbestand-Import planspiel.db → QuantLab (Welle H6, E098).

Einmal-Import des SQLite-Altbestands der Referenz-App (``planspiel_data_tools
… /data/planspiel.db``, Schema ``app/daten/db.py`` Z. 19–69) in QuantLab:

1. ``instrumente`` → :func:`psb_repo.upsert` — die Upsert-Semantik der
   H1-Funktion gilt unverändert: die ISIN läuft in den UPPER-Kanon, die
   **Gruppen-Union** fügt die Alt-Tabs (Spalte ``tabs``, kommasepariert)
   hinzu und entfernt **nichts** — bestehende QuantLab-Gruppen (z. B. aus
   einem früheren Universum-Import) bleiben also geschützt; die übrigen
   Felder folgen COALESCE (``None`` überschreibt nicht) und
   ``yahoo_kuerzel`` wird bewusst als ``None`` geliefert, damit ein
   gepflegter Wert (E093-Brücke) keinesfalls verloren geht. Konflikt-
   Verhalten daher: Altbestand ergänzt, aber überschreibt nie.
2. ``historie`` + ``kurse`` → Parquet 1d je ISIN (UPPER-ISIN als QuantLab-
   Symbol — E093/E094-Konsistenz: Byteworx und Bulk-Backfill adressieren
   dieselben Keys). **Bestand gewinnt:** existierende Balken im Store
   werden nie überschrieben, nur fehlende Timestamps werden angefügt
   (vor jedem Schreiben wird der Bestand gelesen und die Quelle darauf
   gefiltert — :func:`parquet_store.upsert_bars` selbst wäre
   last-write-wins, diese Schicht hält die Quelle deshalb vorher zurück).
   Die Quelle-Marke ``bar_meta.source = 'psb_alt'`` wird NUR gesetzt, wenn
   der ISIN-Bestand vorher leer war (Muster ``downloads``: der UPDATE-Zweig
   der Meta-Pflicht erhält eine bestehende Herkunft; hier bewusst auch der
   Import). Zeit: die Quelle führt lokale Börsenzeit („YYYY-MM-DD
   HH:MM:SS“); für den Tages-Balken zählt das Kalenderdatum — Stempel ist
   Mitternacht UTC (E006-Store-Invariante, konsistent zum Byteworx-Bein).
3. ``kennzahlen`` → :func:`psb_repo.kennzahlen_speichern` NUR für (ISIN,
   Sektion)-Paare, die im QuantLab-Cache fehlen (E095-Bestand gewinnt —
   frisch geladene VWD-Daten werden vom Altbestand nicht überstempelt).

Korruptions-Resilienz (E098): die ``historie``-Tabelle des echten Bestands
ist TEILS beschädigt („database disk image is malformed“ beim Voll-Scan,
Einzel-Queries meist intakt). Deshalb wird NUR sie per ISIN gelesen (je
eigenem ``try/except``). Schlägt eine Einzel-Lesung mit
``sqlite3.DatabaseError`` fehl, läuft EINMALIG ein Rettungsscan über
rowid-Fenster (``WHERE rowid > ? AND rowid <= ?`` — „.recover“-artig:
Fenster auf beschädigten Seiten fallen einzeln aus, intakte Fenster liefern
weiterhin Punkte, die ISIN gilt als **gerettet**). Ein REINDEX wird bewusst
NICHT versucht: er ist ein Schreibvorgang — die Quelle ist strikt
read-only (URI ``?mode=ro``) und darf niemals angetastet werden
(DoD H6); rowid-Fenster sind rein lesend und erreichen genau dasselbe.
ISINs ohne verwertbare Punkte landen gekappt im Fehlerprotokoll; der
Import läuft weiter.

Die Funktion ist synchron und netzfrei (nur SQLite + lokaler Parquet-Store);
der Router führt sie via ``asyncio.to_thread`` aus.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab import db
from quantlab.storage import parquet_store
from quantlab.storage.repos import psb_repo

#: ``bar_meta.source``-Marke der importierten Balken (E098: „quelle ``psb_alt``“).
QUELLE_MARKE = "psb_alt"

#: Ziel-Intervall des Altbestands (Tages-Balken, DoD H6).
INTERVALL = "1d"

#: Erwartete Tabellen im Altbestand (Referenz-Schema; ``einstellungen``/
#: ``historie_laeufe`` werden bewusst ignoriert — kein QuantLab-Gegenstück).
_ERWARTETE_TABELLEN = frozenset({"instrumente", "kurse", "historie", "kennzahlen"})

#: Cap der je-ISIN-Fehlerliste im Report (DoD H6: „gekappt auf z. B. 20“).
_FEHLER_CAP = 20

#: Fenstergröße des rowid-Rettungsscans (Punkte je Lesung).
_RETTUNGS_FENSTER = 1000

#: Fortschritts-Hook: ``(stufe, index_1basiert, gesamt)`` — Stufen sind
#: ``instrumente | historie | kurse | kennzahlen``; ein wirfender Hook
#: bricht den Import ab (Vertrag wie ``universum.Fortschritt``).
Fortschritt = Callable[[str, int, int], None]


class AltbestandFehler(Exception):
    """Fachlicher Fehler des Altbestand-Imports (deutsch; Router → 400)."""

    def __init__(
        self, code: str, message: str, hint: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


# ---------------------------------------------------------------------------
# Quelle öffnen + Schema-Grobecheck
# ---------------------------------------------------------------------------


def _pfad_pruefen(pfad: Any) -> Path:
    """Argument → existierende Datei (deutsche Fachfehler, Router-Vertrag 400)."""
    roh = str(pfad or "").strip()
    if not roh:
        raise AltbestandFehler(
            "ungueltiger_pfad",
            "Der Pfad zur planspiel.db fehlt oder ist leer.",
            hint='Body-Format: {"pfad": "/pfad/zu/planspiel.db"}',
        )
    datei = Path(roh).expanduser()
    if not datei.is_file():
        raise AltbestandFehler(
            "datei_fehlt",
            f'Die Datei „{roh}“ existiert nicht (oder ist ein Verzeichnis).',
            hint="Bitte den vollständigen Pfad zur planspiel.db angeben.",
        )
    return datei.resolve()


def _oeffne_readonly(datei: Path) -> sqlite3.Connection:
    """Öffnet die Quelle STRICT read-only (URI ``?mode=ro``) — nie schreibend.

    ``as_uri()`` prozent-kodiert Sonderzeichen im Pfad; ``mode=ro`` verweigert
    SQLite JEDE Schreiboperation auf der Quelldatei (inkl. REINDEX — deshalb
    läuft der Korruptions-Rettungsversuch lesend über rowid-Fenster, s.
    Modul-Docstring).
    """
    try:
        quelle = sqlite3.connect(f"{datei.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise AltbestandFehler(
            "sqlite_fehler",
            f"Die Quelldatei „{datei}“ konnte nicht geöffnet werden: {exc}",
        ) from exc
    quelle.row_factory = sqlite3.Row
    return quelle


def _pruefe_schema(quelle: sqlite3.Connection) -> None:
    """Schema-Grobecheck: alle erwarteten Tabellen vorhanden (sonst Fachfehler)."""
    try:
        zeilen = quelle.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise AltbestandFehler(
            "keine_sqlite_datei",
            f"Die Datei ist keine lesbare SQLite-Datenbank: {exc}",
            hint="Bitte eine planspiel.db der Referenz-App angeben.",
        ) from exc
    vorhanden = {str(zeile["name"]) for zeile in zeilen}
    fehlen = sorted(_ERWARTETE_TABELLEN - vorhanden)
    if fehlen:
        raise AltbestandFehler(
            "schema_unvollstaendig",
            "In der Quelldatei fehlen die Tabellen: " + ", ".join(fehlen) + ".",
            hint=(
                "Erwartet wird der Altbestand der Referenz-App mit den Tabellen "
                "instrumente, kurse, historie und kennzahlen."
            ),
        )


def _lese(quelle: sqlite3.Connection, tabelle: str, sql: str) -> list:
    """Tab-Lesung der intakten Tabellen — Struktur-/Bildfehler als Fachfehler.

    (Die ``historie``-Lesung läuft bewusst NICHT hierüber, sondern per ISIN
    mit Rettungspfad — s. Modul-Docstring.)
    """
    try:
        return quelle.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        raise AltbestandFehler(
            "schema_unvollstaendig",
            f"Die Tabelle „{tabelle}“ hat ein unerwartetes Spaltenschema: {exc}",
        ) from exc
    except sqlite3.DatabaseError as exc:
        raise AltbestandFehler(
            "quelle_korrupt",
            f"Die Tabelle „{tabelle}“ ist beschädigt und nicht lesbar: {exc}",
            hint="Der Import bricht hier bewusst ab — für „historie“ gilt das "
            "nicht: sie wird per ISIN mit Rettungspfad gelesen.",
        ) from exc


# ---------------------------------------------------------------------------
# Zeit- und Balken-Helfer
# ---------------------------------------------------------------------------


def _tag_us(text: Any) -> int | None:
    """Quell-Zeitstempel → Mitternacht-UTC-Mikrosekunden (oder ``None``).

    ANNAHME: Die Quelle führt lokale Börsenzeit (Berlin) im Format
    „YYYY-MM-DD HH:MM:SS“ (verifiziert an der echten planspiel.db; ISO mit
    „T“/Offset wird tolerant akzeptiert). Für den Tages-Balken zählt das
    Kalenderdatum — der Stempel ist Mitternacht UTC (E006; konsistent zum
    Byteworx-Bein, dessen Tages-/Wochen-Stempel equally auf den UTC-Tag
    gefloort werden).
    """
    roh = str(text or "").strip()
    if not roh:
        return None
    try:
        moment = datetime.strptime(roh, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            moment = datetime.fromisoformat(roh)
        except ValueError:
            return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    mitternacht = datetime(moment.year, moment.month, moment.day, tzinfo=UTC)
    return int(mitternacht.timestamp()) * 1_000_000


def _punkt_zu_balken(zeit: Any, kurs: Any, volumen: float) -> dict[str, float | int]:
    """Ein Quellpunkt → Close-only-Balken (o = h = l = c — E094/E096-Konvention:
    die Quelle liefert nur Kurse, keine Hoch/Tiefs; nichts wird erfunden)."""
    return {"ts": zeit, "open": kurs, "high": kurs, "low": kurs, "close": kurs, "volume": volumen}


def _punkte_zu_tagesbalken(punkte: list[dict[str, Any]]) -> dict[int, dict[str, float | int]]:
    """Kurspunkte → EIN Balken je Tag: der LETZTE Kurs des Tages gewinnt.

    ANNAHME: Mehrere Messpunkte je Tag (Recorder-Schnappschüsse bzw. Intraday-
    Punkte) verdichten zum Tagesschluss; Kollisionen Altbestand-kurse vs.
    historie regelt die Phasen-Reihenfolge (historie zuerst → autoritative
    Byteworx-Schlusshistorie gewinnt den Tag, ``kurse`` füllt nur fehlende
    Tage — „Bestand gewinnt“ gilt damit innerhalb der Quelle genauso).
    """
    je_tag: dict[int, tuple[str, float, float]] = {}  # tag_us → (zeit, kurs, volumen)
    for punkt in punkte:
        tag = _tag_us(punkt.get("zeitstempel"))
        kurs = punkt.get("kurs")
        if tag is None or kurs is None:
            continue  # defekter Punkt — einzeln weg (E072: kein stiller Müll)
        zeit_text = str(punkt.get("zeitstempel") or "")
        volumen = float(punkt.get("volumen") or 0.0)
        bekannt = je_tag.get(tag)
        if bekannt is None or zeit_text >= bekannt[0]:  # String-Vergleich = Zeitordnung
            je_tag[tag] = (zeit_text, float(kurs), volumen)
    return {
        tag: _punkt_zu_balken(tag, kurs, volumen)
        for tag, (_zeit, kurs, volumen) in je_tag.items()
    }


def _quelle_setzen(symbol: str) -> None:
    """Setzt ``bar_meta.source = 'psb_alt'`` (NUR bei vorher leerem Bestand).

    Bewusster SQL-Zeilen-Write statt Store-/Repo-Änderung (Muster
    ``downloads._bar_meta_source_setzen``/``recorder._quelle_setzen``): Der
    Import läuft nicht über den Download-Job-Manager. Die Aufrufer entscheiden
    über die Leer-Bedingung — ein bestehender Bestand behält seine Herkunft
    (DoD H6: „sonst bestehende Quelle erhalten“).
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE bar_meta SET source = ? WHERE symbol = ? AND interval = ?",
            (QUELLE_MARKE, symbol, INTERVALL),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _schreibe_isin_bars(
    isin: str,
    balken: dict[int, dict[str, float | int]],
    bericht: dict[str, Any],
    feld_angehaengt: str,
    kontext: str,
) -> None:
    """Bestand-gewinnt-Append je ISIN: nur Timestamps, die der Store fehlt.

    Vorher leerer Bestand → Quelle-Marke ``psb_alt``; sonst bleibt die
    bestehende ``bar_meta.source`` unberührt. Fehlschläge je ISIN (z. B.
    symbol-ungültige Alt-Zeilen) landen gekappt im Protokoll — der Import
    läuft weiter.
    """
    if not balken:
        return
    symbol = isin.strip().upper()
    try:
        bestand = parquet_store.read_bars(symbol, INTERVALL)
    except Exception as exc:  # noqa: BLE001 — je-ISIN-Resilienz (DoD H6)
        _protokoll_notieren(bericht, f"{kontext} {symbol}: Bestand unlesbar ({exc})")
        return
    existierende = set(bestand["ts"].tolist()) if not bestand.empty else set[int]()
    neue = [balken[tag] for tag in sorted(balken) if tag not in existierende]
    if not neue:
        return
    bestand_war_leer = bestand.empty
    try:
        parquet_store.upsert_bars(symbol, INTERVALL, pd.DataFrame(neue))
    except Exception as exc:  # noqa: BLE001 — je-ISIN-Resilienz (DoD H6)
        _protokoll_notieren(bericht, f"{kontext} {symbol}: Balken verworfen ({exc})")
        return
    if bestand_war_leer:
        _quelle_setzen(symbol)
    bericht[feld_angehaengt] += len(neue)


def _protokoll_notieren(bericht: dict[str, Any], text: str) -> None:
    """Sonstige je-ISIN-Vermerke (Schreib-/Lesefehler), gekappt auf _FEHLER_CAP."""
    if len(bericht["fehler"]) < _FEHLER_CAP:
        bericht["fehler"].append(text)
    else:
        bericht["fehler_weitere"] += 1


def _fehler_notieren(bericht: dict[str, Any], text: str) -> None:
    """Korruptions-Fehlerprotokoll je ISIN, gekappt auf _FEHLER_CAP (DoD H6)."""
    if len(bericht["historie_fehler"]) < _FEHLER_CAP:
        bericht["historie_fehler"].append(text)
    else:
        bericht["historie_fehler_weitere"] += 1


# ---------------------------------------------------------------------------
# Phase 1: instrumente
# ---------------------------------------------------------------------------


def _text(wert: Any) -> str | None:
    """Quellwert → getrimmter Text oder ``None`` (leere Strings sind „fehlt“)."""
    if wert is None:
        return None
    text = str(wert).strip()
    return text or None


def _importiere_instrumente(
    quelle: sqlite3.Connection,
    bericht: dict[str, Any],
    fortschritt: Fortschritt | None,
) -> None:
    """``instrumente`` → ``psb_repo.upsert`` je (Zeile, Tab) — Union/COALESCE.

    Upsert-Semantik bei Konflikten (geprüft an der H1-Funktion): Gruppen
    werden nur VEREINT (bestehende QuantLab-Gruppen bleiben), ``None``-Felder
    überschreiben nicht, ``yahoo_kuerzel`` bleibt bewusst ``None`` (gepflegte
    Brücken-Werte gehen nicht verloren). ``instrumente_upsertet`` zählt die
    Upsert-Aufrufe (je Tab einer — Mehrfach-Zugehörigkeit zählt je, wie der
    Universum-Import seine „zeilen“).
    """
    zeilen = _lese(
        quelle,
        "instrumente",
        "SELECT isin, wkn, name, instrument_type, branch, tabs, id_external, "
        "iso_currency FROM instrumente",
    )
    for index, zeile in enumerate(zeilen, start=1):
        if fortschritt is not None:
            fortschritt("instrumente", index, len(zeilen))
        isin = str(zeile["isin"] or "").strip().upper()
        if not isin:
            continue  # zeilenlose Alt-Reste (kein PK-Verstoß in der Quelle erwartet)
        bericht["instrumente_gefunden"] += 1
        for gruppe in str(zeile["tabs"] or "").split(","):
            gruppe = gruppe.strip()
            if not gruppe:
                continue
            psb_repo.upsert(
                isin=isin,
                wkn=_text(zeile["wkn"]),
                name=_text(zeile["name"]),
                typ=_text(zeile["instrument_type"]),
                branchen=_text(zeile["branch"]),
                gruppe=gruppe,
                id_external=_text(zeile["id_external"]),
                waehrung=_text(zeile["iso_currency"]),
                yahoo_kuerzel=None,  # COALESCE: gepflegte E093-Brücke bleibt
            )
            bericht["instrumente_upsertet"] += 1


# ---------------------------------------------------------------------------
# Phase 2: historie (per ISIN — Korruptions-Resilienz)
# ---------------------------------------------------------------------------


def _historie_kandidaten(quelle: sqlite3.Connection, bericht: dict[str, Any]) -> list[str]:
    """ISIN-Liste für die per-ISIN-Historie-Lesung (korruptionsfest).

    Erster Weg ist ``SELECT DISTINCT isin FROM historie``; scheitert der
    Voll-Scan an der Korruption (beim echten Bestand der Fall), werden die
    Kandidaten aus den intakten Tabellen ``instrumente`` + ``kurse``
    ermittelt (die Historie wurde nur für Universum-Instrumente gefüllt —
    verifiziert an der echten DB) und der Umweg protokolliert.
    """
    kandidaten: set[str] = set()
    try:
        zeilen = quelle.execute("SELECT DISTINCT isin FROM historie").fetchall()
    except sqlite3.DatabaseError:
        _fehler_notieren(
            bericht,
            "historie: ISIN-Enumeration per Vollscan unmöglich (Datenbank "
            "beschädigt) — Kandidaten werden aus instrumente/kurse ermittelt.",
        )
        for zeile in _lese(quelle, "instrumente", "SELECT isin FROM instrumente"):
            isin = str(zeile["isin"] or "").strip().upper()
            if isin:
                kandidaten.add(isin)
        for zeile in _lese(quelle, "kurse", "SELECT DISTINCT isin FROM kurse"):
            isin = str(zeile["isin"] or "").strip().upper()
            if isin:
                kandidaten.add(isin)
    else:
        for zeile in zeilen:
            isin = str(zeile["isin"] or "").strip().upper()
            if isin:
                kandidaten.add(isin)
    return sorted(kandidaten)


def _historie_lesen_direkt(quelle: sqlite3.Connection, isin: str) -> list[dict[str, Any]]:
    """Einzel-Lesung je ISIN (der Normalweg — beim echten Bestand meist intakt)."""
    zeilen = quelle.execute(
        "SELECT zeitstempel, kurs FROM historie WHERE isin = ?", (isin,)
    ).fetchall()
    return [{"zeitstempel": z["zeitstempel"], "kurs": z["kurs"]} for z in zeilen]


def _historie_rettung(
    quelle: sqlite3.Connection,
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    """EINMALIGER Rettungsscan: historie über rowid-Fenster (rein lesend).

    „.recover“-artig: Fenster, deren Seiten beschädigt sind, fallen einzeln
    aus (``DatabaseError`` je Fenster); intakte Fenster liefern weiterhin
    Punkte. Rückgabe: ``(punkte_je_isin, fenster_verloren)``. Bewusst KEIN
    REINDEX: der wäre schreibend — die Quelle ist strikt read-only (DoD H6).
    """
    punkte_je_isin: dict[str, list[dict[str, Any]]] = {}
    try:
        zeile = quelle.execute("SELECT MAX(rowid) FROM historie").fetchone()
    except sqlite3.DatabaseError:
        return punkte_je_isin, True
    max_rowid = int(zeile[0] or 0)  # Row ist indexierbar (row_factory)
    verloren = False
    unter = 0
    while unter < max_rowid:
        ober = min(unter + _RETTUNGS_FENSTER, max_rowid)
        try:
            zeilen = quelle.execute(
                "SELECT isin, zeitstempel, kurs FROM historie "
                "WHERE rowid > ? AND rowid <= ?",
                (unter, ober),
            ).fetchall()
        except sqlite3.DatabaseError:
            verloren = True  # beschädigtes Fenster — Rest weiter versuchen
        else:
            for zeile in zeilen:
                isin = str(zeile["isin"] or "").strip().upper()
                if not isin:
                    continue
                punkte_je_isin.setdefault(isin, []).append(
                    {"zeitstempel": zeile["zeitstempel"], "kurs": zeile["kurs"]}
                )
        unter = ober
    return punkte_je_isin, verloren


def _importiere_historie(
    quelle: sqlite3.Connection,
    bericht: dict[str, Any],
    fortschritt: Fortschritt | None,
) -> None:
    """``historie`` → Parquet 1d, per ISIN mit Rettungspfad (E098).

    Direktlesung je ISIN; bei ``DatabaseError`` („disk image malformed“)
    einmalig der rowid-Rettungsscan — Punkte daraus ⇒ ISIN „gerettet“, keine
    Punkte ⇒ „fehlgeschlagen“ + Protokolleintrag. Der Import läuft immer
    weiter (DoD H6).
    """
    kandidaten = _historie_kandidaten(quelle, bericht)
    bericht["historie_isins"] = len(kandidaten)
    rettung: tuple[dict[str, list[dict[str, Any]]], bool] | None = None
    for index, isin in enumerate(kandidaten, start=1):
        if fortschritt is not None:
            fortschritt("historie", index, len(kandidaten))
        try:
            punkte = _historie_lesen_direkt(quelle, isin)
        except sqlite3.DatabaseError:
            if rettung is None:
                rettung = _historie_rettung(quelle)
            punkte = rettung[0].get(isin, [])
            if punkte:
                bericht["historie_gerettet"] += 1
                if rettung[1]:
                    _fehler_notieren(
                        bericht,
                        f"{isin}: historie teilweise gerettet — rowid-Fenster "
                        "auf beschädigten Seiten blieben verloren.",
                    )
            else:
                bericht["historie_fehlgeschlagen"] += 1
                _fehler_notieren(
                    bericht,
                    f"{isin}: historie unlesbar (Datenbank beschädigt) — ISIN "
                    "übersprungen.",
                )
        bericht["historie_punkte"] += len(punkte)
        balken = _punkte_zu_tagesbalken(punkte)
        _schreibe_isin_bars(isin, balken, bericht, "historie_angehaengt", "historie")


# ---------------------------------------------------------------------------
# Phase 3: kurse (Recorder-Schnappschüsse → Tages-Balken)
# ---------------------------------------------------------------------------


def _importiere_kurse(
    quelle: sqlite3.Connection,
    bericht: dict[str, Any],
    fortschritt: Fortschritt | None,
) -> None:
    """``kurse`` → Parquet 1d: je (ISIN, Tag) der letzte Kurs (Modul-Docstring).

    Läuft NACH der historie-Phase: kollidieren Schnappschuss und Historie auf
    denselben Tag, hat die autoritative Schlusshistorie den Balken bereits
    geschrieben und der Schnappschuss wird als „bereits vorhanden“ übersprungen
    (Bestand gewinnt, ANNAHME im Modul-Docstring).
    """
    zeilen = _lese(
        quelle,
        "kurse",
        "SELECT isin, zeitstempel, kurs, volumen FROM kurse ORDER BY isin, zeitstempel",
    )
    bericht["kurse_quelle"] = len(zeilen)
    je_isin: dict[str, dict[int, dict[str, float | int]]] = {}
    for zeile in zeilen:
        isin = str(zeile["isin"] or "").strip().upper()
        tag = _tag_us(zeile["zeitstempel"])
        kurs = zeile["kurs"]
        if not isin or tag is None or kurs is None:
            bericht["kurse_uebersprungen"] += 1  # defekte Zeile (Zeit/ISIN/Kurs)
            continue
        bericht["kurse_gelesen"] += 1
        volumen = float(zeile["volumen"] or 0.0)
        # ORDER BY zeitstempel ⇒ spätere Zeile überschreibt: letzter Kurs gewinnt.
        je_isin.setdefault(isin, {})[tag] = _punkt_zu_balken(tag, float(kurs), volumen)
    for index, (isin, balken) in enumerate(sorted(je_isin.items()), start=1):
        if fortschritt is not None:
            fortschritt("kurse", index, len(je_isin))
        _schreibe_isin_bars(isin, balken, bericht, "kurse_angehaengt", "kurse")


# ---------------------------------------------------------------------------
# Phase 4: kennzahlen (E095-Bestand gewinnt)
# ---------------------------------------------------------------------------


def _importiere_kennzahlen(
    quelle: sqlite3.Connection,
    bericht: dict[str, Any],
    fortschritt: Fortschritt | None,
) -> None:
    """``kennzahlen`` → ``psb_repo.kennzahlen_speichern`` NUR für fehlende Paare."""
    zeilen = _lese(quelle, "kennzahlen", "SELECT isin, sektion, daten FROM kennzahlen")
    bericht["kennzahlen_gefunden"] = len(zeilen)
    cache: dict[str, set[str]] = {}
    for index, zeile in enumerate(zeilen, start=1):
        if fortschritt is not None:
            fortschritt("kennzahlen", index, len(zeilen))
        isin = str(zeile["isin"] or "").strip().upper()
        sektion = str(zeile["sektion"] or "").strip()
        if not isin or not sektion:
            bericht["kennzahlen_uebersprungen"] += 1
            continue
        vorhanden = cache.get(isin)
        if vorhanden is None:
            vorhanden = set(psb_repo.kennzahlen_lesen(isin))
            cache[isin] = vorhanden
        if sektion in vorhanden:
            bericht["kennzahlen_uebersprungen"] += 1  # E095-Bestand gewinnt
            continue
        try:
            daten = json.loads(zeile["daten"])
        except (TypeError, ValueError):
            bericht["kennzahlen_uebersprungen"] += 1  # defektes Alt-JSON — kein stiller Müll
            continue
        if not isinstance(daten, dict):
            bericht["kennzahlen_uebersprungen"] += 1  # Repo-Vertrag verlangt Dict
            continue
        psb_repo.kennzahlen_speichern(isin, sektion, daten)
        vorhanden.add(sektion)
        bericht["kennzahlen_uebernommen"] += 1


# ---------------------------------------------------------------------------
# Einstieg
# ---------------------------------------------------------------------------


def importiere_altbestand(
    pfad: Any, fortschritt: Fortschritt | None = None
) -> dict[str, Any]:
    """Importiert den Altbestand idempotent — Counts-Report als Ergebnis (E098).

    Öffnet die Quelldatei strikt read-only, prüft das Schema (fehlende
    Tabellen → :class:`AltbestandFehler`), importiert instrumente/historie/
    kurse/kennzahlen gemäß Modul-Docstring (Bestand gewinnt) und liefert den
    Counts-Report inkl. ``dauer_s``. Ein zweiter Lauf endet bei gleichen
    Endcounts (angehängt = 0, kennzahlen übersprungen) — idempotent.
    """
    start = time.monotonic()
    datei = _pfad_pruefen(pfad)
    quelle = _oeffne_readonly(datei)
    try:
        _pruefe_schema(quelle)
        bericht: dict[str, Any] = {
            "instrumente_gefunden": 0,
            "instrumente_upsertet": 0,
            "kurse_quelle": 0,
            "kurse_gelesen": 0,
            "kurse_angehaengt": 0,
            "kurse_uebersprungen": 0,
            "historie_isins": 0,
            "historie_punkte": 0,
            "historie_angehaengt": 0,
            "historie_gerettet": 0,
            "historie_fehlgeschlagen": 0,
            "historie_fehler": [],  # gekappt auf _FEHLER_CAP Einträge
            "historie_fehler_weitere": 0,
            "kennzahlen_gefunden": 0,
            "kennzahlen_uebernommen": 0,
            "kennzahlen_uebersprungen": 0,
            "fehler": [],  # sonstige je-ISIN-Vermerke (Schreib-/Lesefehler), gekappt
            "fehler_weitere": 0,
        }
        # Phasen-Reihenfolge dokumentiert: historie VOR kurse (autoritative
        # Schlusshistorie gewinnt den Tag — ANNAHME im Modul-Docstring).
        _importiere_instrumente(quelle, bericht, fortschritt)
        _importiere_historie(quelle, bericht, fortschritt)
        _importiere_kurse(quelle, bericht, fortschritt)
        _importiere_kennzahlen(quelle, bericht, fortschritt)
    finally:
        quelle.close()
    bericht["dauer_s"] = round(time.monotonic() - start, 3)
    return bericht
