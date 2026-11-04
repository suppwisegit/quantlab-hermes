"""PSB-Instrumente-Repository (Welle H1, E093): lokaler Spiegel des Planspiel-
Universums (Tabelle ``psb_instrumente``, Migration 7) — nur DB-Logik.

``gruppen`` hält die **Vereinigung** aller Tab-Zugehörigkeiten (Muster:
``instrument_upsert`` der Referenz ``planspiel_data_tools`` — ein Instrument
steht häufig in mehreren Marktgruppen, z. B. DAX + Selection Europe); ein
Upsert fügt seine Gruppe hinzu, entfernt aber nie (Union). Die übrigen Felder
folgen dem COALESCE-Muster von :func:`symbols_repo.upsert_symbol` (``None``
überschreibt nicht) — der Universum-Import liefert je Zeile stets den
Vollsatz, Alt-Zeilen bleiben durch Teil-Updates gefahrlos pflegbar.

Welle H3 (E095, additiv): :func:`instrument` liest eine einzelne Zeile
(idNotation-Brücke: ``id_external`` → ``getInstrumentWithQuotes``), und
:func:`kennzahlen_speichern`/:func:`kennzahlen_lesen` pflegen den VWD-
Kennzahlen-Cache (Tabelle ``psb_kennzahlen``, Migration 8) — je (ISIN,
Sektion) die **geparste** Snippet-Sektion als JSON (Roundtrip über
``json.dumps``/``json.loads``); ISIN im UPPER-Kanon (H-1-Fix-Konvention).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from quantlab import db


@dataclass(frozen=True)
class PsbInstrument:
    """Ein Eintrag des Planspiel-Universums (Zeile aus ``psb_instrumente``)."""

    isin: str
    wkn: str | None
    name: str | None
    typ: str | None
    branchen: str | None
    #: Vereinigung der Tab-Zugehörigkeit (sortiert, kommasepariert — '' = leer).
    gruppen: str
    id_external: str | None
    waehrung: str | None
    #: E093-Brücke zur symbols-Registry (``None`` = ISIN lokal unaufgelöst).
    yahoo_kuerzel: str | None
    aktualisiert_am: str

    def gruppen_liste(self) -> list[str]:
        """``gruppen`` als Liste (leere Zeile → leere Liste)."""
        return [g for g in self.gruppen.split(",") if g]


def _zu_instrument(row: sqlite3.Row) -> PsbInstrument:
    return PsbInstrument(
        isin=row["isin"],
        wkn=row["wkn"],
        name=row["name"],
        typ=row["typ"],
        branchen=row["branchen"],
        gruppen=row["gruppen"],
        id_external=row["id_external"],
        waehrung=row["waehrung"],
        yahoo_kuerzel=row["yahoo_kuerzel"],
        aktualisiert_am=row["aktualisiert_am"],
    )


def upsert(
    *,
    isin: str,
    wkn: str | None = None,
    name: str | None = None,
    typ: str | None = None,
    branchen: str | None = None,
    gruppe: str | None = None,
    id_external: str | None = None,
    waehrung: str | None = None,
    yahoo_kuerzel: str | None = None,
) -> PsbInstrument:
    """Legt das Instrument an oder aktualisiert es (idempotent, Gruppen-Union).

    Die ISIN wird trimm+groß-normalisiert (ISIN-Kanon). ``gruppe`` (der
    ``key`` aus ``getTabListKeyValue``) wird der bestehenden Zugehörigkeit
    **hinzugefügt** (Union, sortiert kommasepariert — Referenz-Muster
    ``instrument_upsert``); ``None``-Felder überschreiben bestehende Werte
    nicht (COALESCE). ``aktualisiert_am`` wird bei jeder Berührung erneuert
    (:func:`db.utc_now_iso`, E006). Ohne ISIN → ``ValueError`` (Aufrufer-
    Fehler, kein Fachfall).
    """
    geprueft = (isin or "").strip().upper()
    if not geprueft:
        raise ValueError("psb_repo.upsert ohne ISIN ist nicht möglich.")
    neue_gruppe = (gruppe or "").strip()
    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db) — Read-modify-write der
        # Gruppen-Union braucht den Schreib-Lock über beide Statements.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT gruppen FROM psb_instrumente WHERE isin = ?", (geprueft,)
        ).fetchone()
        alt = set((row["gruppen"] if row else "").split(",")) - {""}
        if neue_gruppe:
            alt.add(neue_gruppe)
        conn.execute(
            "INSERT INTO psb_instrumente(isin, wkn, name, typ, branchen, gruppen, "
            "id_external, waehrung, yahoo_kuerzel, aktualisiert_am) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(isin) DO UPDATE SET "
            "wkn = COALESCE(excluded.wkn, psb_instrumente.wkn), "
            "name = COALESCE(excluded.name, psb_instrumente.name), "
            "typ = COALESCE(excluded.typ, psb_instrumente.typ), "
            "branchen = COALESCE(excluded.branchen, psb_instrumente.branchen), "
            "gruppen = excluded.gruppen, "
            "id_external = COALESCE(excluded.id_external, psb_instrumente.id_external), "
            "waehrung = COALESCE(excluded.waehrung, psb_instrumente.waehrung), "
            "yahoo_kuerzel = COALESCE(excluded.yahoo_kuerzel, psb_instrumente.yahoo_kuerzel), "
            "aktualisiert_am = excluded.aktualisiert_am",
            (
                geprueft,
                wkn,
                name,
                typ,
                branchen,
                ",".join(sorted(alt)),
                id_external,
                waehrung,
                yahoo_kuerzel,
                db.utc_now_iso(),
            ),
        )
        zeile = conn.execute(
            "SELECT * FROM psb_instrumente WHERE isin = ?", (geprueft,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert zeile is not None  # frisch eingefügt/aktualisiert — immer lesbar
    return _zu_instrument(zeile)


def liste(gruppe: str | None = None) -> list[PsbInstrument]:
    """Alle Instrumente, nach Name sortiert; mit ``gruppe`` nur deren Tab.

    Der Gruppen-Filter matcht exakt gegen die kommaseparierte Zugehörigkeit
    (``instr`` statt ``LIKE`` — kein Wildcard-Kontext, und ein Teil-String-
    Treffer wie „DAX“ innerhalb von „MDAX“ wird durch die Komma-Grenzen
    ausgeschlossen). Unbekannte Gruppe → leere Liste (tolerant, E072-Stil).
    """
    filter_gruppe = (gruppe or "").strip()
    conn = db.connect()
    try:
        if filter_gruppe:
            rows = conn.execute(
                "SELECT * FROM psb_instrumente "
                "WHERE instr(',' || gruppen || ',', ?) > 0 ORDER BY name, isin",
                (f",{filter_gruppe},",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM psb_instrumente ORDER BY name, isin"
            ).fetchall()
    finally:
        conn.close()
    return [_zu_instrument(row) for row in rows]


def gruppen() -> list[str]:
    """Alle bekannten Gruppen (Vereinigung über alle Zeilen), sortiert."""
    conn = db.connect()
    try:
        rows = conn.execute("SELECT gruppen FROM psb_instrumente").fetchall()
    finally:
        conn.close()
    mengen: set[str] = set()
    for row in rows:
        mengen.update(g for g in row["gruppen"].split(",") if g)
    return sorted(mengen)


def instrument(isin: str) -> PsbInstrument | None:
    """Eine Universum-Zeile je ISIN (UPPER-Kanon) — ``None`` wenn unbekannt.

    Lese-Funktion der idNotation-Brücke (H3/E095): ``id_external`` der Zeile
    ist der Schlüssel für ``PlanspielClient.instrument_mit_kursen`` →
    ``quoteList[0].idNotation`` (Referenz ``markt.kennzahlen``).
    """
    geprueft = (isin or "").strip().upper()
    if not geprueft:
        return None
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM psb_instrumente WHERE isin = ?", (geprueft,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_instrument(row)


def kennzahlen_speichern(isin: str, sektion: str, daten: dict[str, Any]) -> str:
    """Schreibt/aktualisiert eine geparste Sektion (idempotenter Upsert).

    ``daten`` wird als JSON serialisiert (Roundtrip-Vertrag: :func:
    `kennzahlen_lesen` liefert das Dict strukturell identisch zurück). Die
    ISIN wird trimm+groß-normalisiert (UPPER-Kanon); ``aktualisiert_am``
    wird bei jeder Berührung erneuert (:func:`db.utc_now_iso`, E006). Liefert
    den gespeicherten Zeitstempel (Antwort-Vertrag „aktualisiert“). Nicht-
    serialisierbare ``daten`` sind ein Aufrufer-Fehler → ``TypeError`` (der
    Parser liefert nur Dicts/Liste/Skalare).
    """
    geprueft = (isin or "").strip().upper()
    if not geprueft:
        raise ValueError("psb_repo.kennzahlen_speichern ohne ISIN ist nicht möglich.")
    if not sektion or not sektion.strip():
        raise ValueError("psb_repo.kennzahlen_speichern ohne Sektion ist nicht möglich.")
    jetzt = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO psb_kennzahlen(isin, sektion, daten, aktualisiert_am) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(isin, sektion) DO UPDATE SET "
            "daten = excluded.daten, aktualisiert_am = excluded.aktualisiert_am",
            (geprueft, sektion.strip(), json.dumps(daten, ensure_ascii=False), jetzt),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return jetzt


def kennzahlen_lesen(isin: str, sektion: str | None = None) -> dict[str, dict[str, Any]]:
    """Gecachte Sektionen je ISIN (UPPER-Kanon) — ``{sektion: daten}``.

    Mit ``sektion`` nur diese eine (unbekannte/nie geladene → leeres Dict,
    tolerant). JSON-Roundtrip: die Werte sind strukturell identisch mit dem
    :func:`kennzahlen_speichern`-Argument. Kein Cache → leeres Dict (der
    Aufrufer entscheidet über den frisch-Abruf).
    """
    geprueft = (isin or "").strip().upper()
    if not geprueft:
        return {}
    filter_sektion = (sektion or "").strip() or None
    conn = db.connect()
    try:
        if filter_sektion is not None:
            rows = conn.execute(
                "SELECT sektion, daten FROM psb_kennzahlen "
                "WHERE isin = ? AND sektion = ?",
                (geprueft, filter_sektion),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT sektion, daten FROM psb_kennzahlen WHERE isin = ?",
                (geprueft,),
            ).fetchall()
    finally:
        conn.close()
    return {row["sektion"]: json.loads(row["daten"]) for row in rows}


def kennzahlen_aktualisiert(isin: str) -> dict[str, str]:
    """Zeitstempel je gecachter Sektion (Antwort-Vertrag „aktualisiert“, E006)."""
    geprueft = (isin or "").strip().upper()
    if not geprueft:
        return {}
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT sektion, aktualisiert_am FROM psb_kennzahlen WHERE isin = ?",
            (geprueft,),
        ).fetchall()
    finally:
        conn.close()
    return {row["sektion"]: row["aktualisiert_am"] for row in rows}


def zaehler() -> int:
    """Anzahl gespeicherter Universum-Instrumente."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT COUNT(*) FROM psb_instrumente").fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])
