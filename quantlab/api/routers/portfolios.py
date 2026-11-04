"""Portfolio-Endpunkte (§C.4, T23): Depot-CRUD, Ledger, Positionen,
Performance (TWR/XIRR), CSV-Import/-Export.

Schichten (T18/T23-Teilung): Depot-CRUD und Ledger-Persistenz liegen im
``portfolios_repo`` (T18, inkl. Struktur-Validierung: bekannter Typ,
existierendes Depot, Transfer-Gegen-Depot ≠ selbst). **Fachliche**
Feld-Validierung je Buchungstyp (buy → quantity+price usw., §C.4) macht
dieser Router über Pydantic-Discriminated-Union — Verstöße laufen über den
zentralen Validierungs-Handler als 422 mit deutscher Meldung. ``ValueError``
aus dem Repo (z. B. Transfer-Regeln) wird ebenfalls auf 422 abgebildet,
nie 500.

Bewertung/Performance berechnet :mod:`quantlab.services.portfolio` auf
Daily-Closes (raw, E031) — Fehler: ``ProviderError`` → ApiError-Vertrag,
``ValueError`` → 422 (R14: definierte deutsche Meldungen).

CSV-Format (strikt, Roundtrip mit ``/export``):

    ts,type,symbol,quantity,price,amount,fees,taxes,currency,
    transfer_portfolio_id,note

Leere Felder = ``null``; Zahlen mit Dezimalpunkt; ``ts`` ISO-8601 (naive
Angaben gelten als UTC). Injektionsschutz (REV-T23 Fix 5): Felder, die mit
``=``, ``+``, ``-`` oder ``@`` beginnen, trägt der Export mit führendem
``'``; der Import entfernt genau diesen Präfix wieder (ein führendes ``'``
nur, wenn das zweite Zeichen in der Gefahrmenge steht). Der Import ist
**atomar**: jede Zeile wird vorab validiert (Typ-Regeln wie oben,
Zeilennummer aus der Datei), bei Fehlern wird **nichts** eingefügt und alle
Zeilenfehler stehen in der deutschen 422-Meldung.

SSE: Änderungen publizieren ``portfolio_changed`` über den globalen Bus
(``events.publizieren``). ANNAHME (T23): Der Event-Name steht nicht in §C.5
(dort liegen nur T21–T33-Events) — das Frontend ignoriert unbekannte
Event-Namen weiterhin (C.5-Konvention); T27 kann ihn für die
Cache-Invalidierung nutzen.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field, StringConstraints, field_validator

from quantlab import events
from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.services import portfolio as portfolio_service
from quantlab.services.downloads import iso_zeitpunkt_zu_us
from quantlab.storage.repos import portfolios_repo, projects_repo

router = APIRouter(tags=["portfolios"])

#: Buchungstypen des Ledgers (Schema-CHECK §D.2 — ohne out/in-Auflösung).
TX_TYPEN: tuple[str, ...] = (
    "buy",
    "sell",
    "dividend",
    "cash_in",
    "cash_out",
    "transfer",
)

#: CSV-Spalten des Import-/Export-Formats (T23, dokumentiert — Roundtrip-Stift).
CSV_SPALTEN: tuple[str, ...] = (
    "ts",
    "type",
    "symbol",
    "quantity",
    "price",
    "amount",
    "fees",
    "taxes",
    "currency",
    "transfer_portfolio_id",
    "note",
)

#: Maximal in der 422-Meldung gezeigte CSV-Zeilenfehler (Rest als Summe).
_MAX_CSV_FEHLER = 10

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Waehrung = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=12)]


# ---------------------------------------------------------------------------
# Request-Schemas — fachliche Feld-Regeln je Buchungstyp (§C.4)
# ---------------------------------------------------------------------------


def _pruefe_ts(wert: str) -> str:
    """ISO-8601-Validierung + Normalisierung auf UTC (naive = UTC)."""
    try:
        moment = datetime.fromisoformat(str(wert).strip())
    except ValueError as exc:
        raise ValueError(
            f"Ungültiger Buchungszeitpunkt „{wert}“ — ISO-8601 erwartet "
            "(z. B. 2024-06-01T15:30:00+00:00)."
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


IsoTs = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _BuchungBasis(BaseModel):
    """Gemeinsame Felder jeder Buchung (Gebühren/Steuern eigen, E031)."""

    ts: IsoTs
    fees: float = Field(default=0.0, ge=0)
    taxes: float = Field(default=0.0, ge=0)
    currency: Waehrung | None = None
    note: str | None = Field(default=None, max_length=2000)

    @field_validator("ts")
    @classmethod
    def _ts_normalisieren(cls, wert: str) -> str:
        return _pruefe_ts(wert)


class KaufVerkaufRequest(_BuchungBasis):
    """buy/sell: Pflichtfelder quantity > 0 und price > 0 (§C.4)."""

    type: Literal["buy", "sell"]
    symbol: NonEmptyText
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)


class DividendeRequest(_BuchungBasis):
    """dividend: symbol + amount > 0 (quantity/price je Stück nur informativ)."""

    type: Literal["dividend"]
    symbol: NonEmptyText
    amount: float = Field(gt=0)


class CashRequest(_BuchungBasis):
    """cash_in/cash_out: amount > 0, kein Symbol (Ledger-Semantik §D.2)."""

    type: Literal["cash_in", "cash_out"]
    amount: float = Field(gt=0)


class TransferRequest(_BuchungBasis):
    """transfer: Cash-Umbuchung mit Gegen-Depot (≠ selbst, §C.4).

    ANNAHME (T23): v2 umbucht nur Cash (``amount``); Wertpapier-Umbuchungen
    sind v3-Kandidat — das Schema ließe sie zu, die Engine nicht.
    """

    type: Literal["transfer"]
    amount: float = Field(gt=0)
    transfer_portfolio_id: int


BuchungRequest = Annotated[
    KaufVerkaufRequest | DividendeRequest | CashRequest | TransferRequest,
    Field(discriminator="type"),
]


class BuchungPatchRequest(BaseModel):
    """``PATCH …/transactions/{tx_id}`` — Merge je Feld (``None`` = unverändert).

    Das **Ergebnis** (bestehende Buchung + Patch) muss wieder der Typ-Regel
    genügen — der Router validiert es wie eine Neuanlage (``model_validate``
    über die Discriminated-Union).
    """

    ts: IsoTs | None = None
    type: Literal["buy", "sell", "dividend", "cash_in", "cash_out", "transfer"] | None = None
    symbol: NonEmptyText | None = None
    quantity: float | None = Field(default=None, gt=0)
    price: float | None = Field(default=None, gt=0)
    amount: float | None = Field(default=None, gt=0)
    fees: float | None = Field(default=None, ge=0)
    taxes: float | None = Field(default=None, ge=0)
    currency: Waehrung | None = None
    transfer_portfolio_id: int | None = None
    note: str | None = Field(default=None, max_length=2000)


class PortfolioCreateRequest(BaseModel):
    """Body von ``POST /api/portfolios`` (project_id optional, E025)."""

    name: NonEmptyText = Field(max_length=200)
    base_currency: Waehrung = "USD"
    notes: str = Field(default="", max_length=10_000)
    project_id: int | None = None


class PortfolioUpdateRequest(BaseModel):
    """Body von ``PUT /api/portfolios/{id}`` — Merge, alle Felder optional."""

    name: NonEmptyText | None = Field(default=None, max_length=200)
    base_currency: Waehrung | None = None
    notes: str | None = Field(default=None, max_length=10_000)


# ---------------------------------------------------------------------------
# Mapper + 404-/Validierungs-Helfer
# ---------------------------------------------------------------------------


def _depot_als_dict(depot: portfolios_repo.Portfolio) -> dict[str, Any]:
    return {
        "id": depot.id,
        "project_id": depot.project_id,
        "name": depot.name,
        "base_currency": depot.base_currency,
        "notes": depot.notes,
        "created_at": depot.created_at,
        "updated_at": depot.updated_at,
    }


def _tx_als_dict(tx: portfolios_repo.Transaction) -> dict[str, Any]:
    return {
        "id": tx.id,
        "portfolio_id": tx.portfolio_id,
        "ts": tx.ts,
        "type": tx.type,
        "symbol": tx.symbol,
        "quantity": tx.quantity,
        "price": tx.price,
        "amount": tx.amount,
        "fees": tx.fees,
        "taxes": tx.taxes,
        "currency": tx.currency,
        "transfer_portfolio_id": tx.transfer_portfolio_id,
        "note": tx.note,
        "created_at": tx.created_at,
    }


def _depot_or_404(portfolio_id: int) -> portfolios_repo.Portfolio:
    depot = portfolios_repo.get_portfolio(portfolio_id)
    if depot is None:
        raise ApiError(
            404,
            "depot_nicht_gefunden",
            f"Depot mit ID {portfolio_id} nicht gefunden.",
        )
    return depot


def _tx_or_404(portfolio_id: int, tx_id: int) -> portfolios_repo.Transaction:
    tx = portfolios_repo.get_transaction(tx_id)
    if tx is None or tx.portfolio_id != portfolio_id:
        raise ApiError(
            404,
            "transaktion_nicht_gefunden",
            f"Buchung mit ID {tx_id} wurde im Depot {portfolio_id} nicht gefunden.",
        )
    return tx


def _symbol_validieren(symbol: str) -> str:
    """Yahoo-Kürzel validieren (T3-Regeln) und in den ApiError-Vertrag mappen."""
    try:
        return validate_symbol(symbol)
    except ProviderError as exc:
        raise exc.to_api_error() from exc


def _repo_wertefehler(exc: ValueError, code: str = "ungueltige_buchung") -> ApiError:
    """Repo-/Service-``ValueError`` → 422 mit deutscher Meldung (nie 500)."""
    return ApiError(422, code, str(exc))


def _publiziere_geaendert(depot_id: int, grund: str) -> None:
    """``portfolio_changed`` auf den globalen SSE-Bus (siehe Modul-ANNAHME)."""
    events.publizieren("portfolio_changed", {"portfolioId": depot_id, "grund": grund})


def _buchung_einfuegen(
    portfolio_id: int, felder: dict[str, Any]
) -> portfolios_repo.Transaction:
    """Legt eine fachlich validierte Buchung an (Repo-ValueError → 422)."""
    try:
        return portfolios_repo.add_transaction(portfolio_id, **felder)
    except ValueError as exc:
        raise _repo_wertefehler(exc) from exc


def _union_modell(felder: dict[str, Any]) -> (
    KaufVerkaufRequest | DividendeRequest | CashRequest | TransferRequest
):
    """Validiert ein Buchungs-Feldbündel über die Typ-Discriminated-Union.

    Wirft ``ApiError`` 422 (deutsch, erste Regelverletzung) statt der rohen
    Pydantic-ValidationError — für PATCH-Merge und CSV-Import gleichermaßen.
    """
    typ = str(felder.get("type", ""))
    modell_klasse = (
        KaufVerkaufRequest
        if typ in ("buy", "sell")
        else DividendeRequest
        if typ == "dividend"
        else TransferRequest
        if typ == "transfer"
        else CashRequest
        if typ in ("cash_in", "cash_out")
        else None
    )
    if modell_klasse is None:
        raise ApiError(422, "ungueltige_buchung", f"Unbekannter Buchungstyp „{typ}“.")
    try:
        return modell_klasse.model_validate(felder)
    except Exception as exc:  # pydantic.ValidationError → fachliche 422
        detail = exc
        if hasattr(exc, "errors"):
            fehlerliste = exc.errors()
            if fehlerliste:
                detail = fehlerliste[0].get("msg", exc)
        raise ApiError(
            422,
            "ungueltige_buchung",
            f"Die Buchung verletzt die Feldregeln für Typ „{typ}“: {detail}",
        ) from exc


def _validiere_buchungsfelder(
    portfolio_id: int, felder: dict[str, Any]
) -> dict[str, Any]:
    """Fachliche Endprüfung vor dem Einfügen: Symbol + Transfer-Selbstbezug.

    Discriminated-Union hat Typ-Pflichtfelder geprüft; hier bleiben die
    Regeln, die den Pfad-Parameter bzw. die Gegen-Depot-Existenz brauchen.
    Rückgabe: bereinigte Repo-Argumente (Symbol normalisiert, Nones gefällt).
    """
    if felder.get("symbol") is not None:
        felder["symbol"] = _symbol_validieren(str(felder["symbol"]))
    if felder.get("type") == "transfer":
        gegen = felder.get("transfer_portfolio_id")
        if gegen is not None and int(gegen) == int(portfolio_id):
            raise ApiError(
                422,
                "ungueltige_buchung",
                "Eine Umbuchung darf nicht auf dasselbe Depot gehen.",
            )
        if gegen is not None and portfolios_repo.get_portfolio(int(gegen)) is None:
            raise ApiError(
                422,
                "ungueltige_buchung",
                f"Gegen-Depot {int(gegen)} existiert nicht.",
            )
    return felder


# ---------------------------------------------------------------------------
# Depot-CRUD
# ---------------------------------------------------------------------------


@router.get("/portfolios")
def list_portfolios(project_id: int | None = None) -> dict[str, Any]:
    """Depots (nach Name sortiert); ``project_id`` filtert projektgebunden."""
    depot_liste = portfolios_repo.list_portfolios(project_id=project_id)
    return {"portfolios": [_depot_als_dict(d) for d in depot_liste]}


@router.post("/portfolios", status_code=201)
def create_portfolio(payload: PortfolioCreateRequest) -> dict[str, Any]:
    """Legt ein Depot an (global oder projektgebunden, E025)."""
    if payload.project_id is not None and projects_repo.get_project(payload.project_id) is None:
        raise ApiError(
            422,
            "projekt_nicht_gefunden",
            f"Projekt mit ID {payload.project_id} existiert nicht — "
            "Depot kann nicht gebunden werden.",
        )
    try:
        depot = portfolios_repo.create_portfolio(
            payload.name,
            project_id=payload.project_id,
            base_currency=payload.base_currency.upper(),
            notes=payload.notes,
        )
    except ValueError as exc:
        raise _repo_wertefehler(exc) from exc
    _publiziere_geaendert(depot.id, "depot_angelegt")
    return _depot_als_dict(depot)


@router.get("/portfolios/{portfolio_id}")
def get_portfolio(portfolio_id: int) -> dict[str, Any]:
    """Ein Depot anhand seiner ID (404 wenn unbekannt)."""
    return _depot_als_dict(_depot_or_404(portfolio_id))


@router.put("/portfolios/{portfolio_id}")
def update_portfolio(portfolio_id: int, payload: PortfolioUpdateRequest) -> dict[str, Any]:
    """Aktualisiert Name/Basiswährung/Notizen (Merge) + ``updated_at``."""
    _depot_or_404(portfolio_id)
    base_currency = (
        payload.base_currency.upper() if payload.base_currency is not None else None
    )
    try:
        aktualisiert = portfolios_repo.update_portfolio(
            portfolio_id,
            name=payload.name,
            base_currency=base_currency,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise _repo_wertefehler(exc) from exc
    assert aktualisiert is not None  # Existenz oben geprüft
    _publiziere_geaendert(portfolio_id, "depot_geaendert")
    return _depot_als_dict(aktualisiert)


@router.delete("/portfolios/{portfolio_id}", status_code=204)
def delete_portfolio(portfolio_id: int) -> Response:
    """Löscht das Depot samt Ledger und Transfer-Gegenbezügen (T18-Repo)."""
    _depot_or_404(portfolio_id)
    portfolios_repo.delete_portfolio(portfolio_id)
    _publiziere_geaendert(portfolio_id, "depot_geloescht")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Transaktions-Ledger (fachliche Validierung je Typ, §C.4)
# ---------------------------------------------------------------------------


@router.get("/portfolios/{portfolio_id}/transactions")
def list_transactions(portfolio_id: int) -> dict[str, Any]:
    """Ledger des Depots, chronologisch."""
    _depot_or_404(portfolio_id)
    return {
        "transactions": [_tx_als_dict(t) for t in portfolios_repo.list_transactions(portfolio_id)]
    }


@router.post("/portfolios/{portfolio_id}/transactions", status_code=201)
def create_transaction(portfolio_id: int, payload: BuchungRequest) -> dict[str, Any]:
    """Bucht eine Transaktion (Validierung je Typ → 422 bei Regelverstoß)."""
    _depot_or_404(portfolio_id)
    felder = _validiere_buchungsfelder(portfolio_id, payload.model_dump())
    tx = _buchung_einfuegen(portfolio_id, felder)
    _publiziere_geaendert(portfolio_id, "transaktion_angelegt")
    return _tx_als_dict(tx)


@router.patch("/portfolios/{portfolio_id}/transactions/{tx_id}")
def update_transaction(
    portfolio_id: int, tx_id: int, payload: BuchungPatchRequest
) -> dict[str, Any]:
    """Aktualisiert Felder einer Buchung (Merge; Ergebnis wird neu validiert).

    ``None``-Felder bedeuten „unverändert“; da das Repo keine Felder auf
    ``NULL`` setzen kann (T18-Vertrag), bleibt dafür Löschen + Neuanlegen.
    Der **Typ** selbst ist nicht patchbar (REV-T23 Fix 3): ein Wechsel
    (z. B. buy → dividend) würde die Feldsemantik umbiegen → 422 mit dem
    Hinweis „Transaktion löschen und neu anlegen“.
    """
    tx = _tx_or_404(portfolio_id, tx_id)
    patch = {
        schluessel: wert
        for schluessel, wert in payload.model_dump(exclude_unset=True).items()
        if wert is not None
    }
    if patch.get("type") is not None and patch["type"] != tx.type:
        raise ApiError(
            422,
            "typ_wechsel_unterstuetzt",
            f"Der Typ einer Buchung kann nicht geändert werden "
            f"(bestehend: {tx.type}, gewünscht: {patch['type']}).",
            hint="Transaktion löschen und neu anlegen.",
        )
    aktuell = {
        "ts": tx.ts,
        "type": tx.type,
        "symbol": tx.symbol,
        "quantity": tx.quantity,
        "price": tx.price,
        "amount": tx.amount,
        "fees": tx.fees,
        "taxes": tx.taxes,
        "currency": tx.currency,
        "transfer_portfolio_id": tx.transfer_portfolio_id,
        "note": tx.note,
    }
    aktuell.update(patch)
    modell = _union_modell(aktuell)  # wirft 422 bei Regelverstoß des Ergebnisses
    felder = _validiere_buchungsfelder(portfolio_id, modell.model_dump())
    try:
        aktualisiert = portfolios_repo.update_transaction(tx_id, **felder)
    except ValueError as exc:
        raise _repo_wertefehler(exc) from exc
    assert aktualisiert is not None
    _publiziere_geaendert(portfolio_id, "transaktion_geaendert")
    return _tx_als_dict(aktualisiert)


@router.delete("/portfolios/{portfolio_id}/transactions/{tx_id}", status_code=204)
def delete_transaction(portfolio_id: int, tx_id: int) -> Response:
    """Löscht eine Buchung aus dem Ledger."""
    _tx_or_404(portfolio_id, tx_id)
    portfolios_repo.delete_transaction(tx_id)
    _publiziere_geaendert(portfolio_id, "transaktion_geloescht")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Positionen (Ø-EK Default, FIFO Option — E031)
# ---------------------------------------------------------------------------


@router.get("/portfolios/{portfolio_id}/positions")
def get_positions(portfolio_id: int, method: str = "avg_cost") -> dict[str, Any]:
    """Positionen + realisiert/unrealisiert P&L + Dividenden (E031)."""
    depot = _depot_or_404(portfolio_id)
    try:
        return portfolio_service.positionen_report(depot, method)
    except ValueError as exc:
        raise _repo_wertefehler(exc, code="ungueltige_methode") from exc


# ---------------------------------------------------------------------------
# Performance (TWR/XIRR/Equity/Kennzahlen/Benchmark, §C.4)
# ---------------------------------------------------------------------------


@router.get("/portfolios/{portfolio_id}/performance")
def get_performance(
    portfolio_id: int,
    start: str | None = None,
    end: str | None = None,
    benchmark: str | None = None,
    method: str = "all",
) -> dict[str, Any]:
    """TWR/XIRR + Equity-Kurve + Kennzahlen (§F) + optionaler Benchmark.

    ``start`` inklusive / ``end`` inklusive (ISO-Datum genügt); ``benchmark``
    gegen ein **lokal vorhandenes** Symbol (fehlende Daten → 404, §T23).

    XIRR-Semantik (REV-T23 Fix 1): XIRR misst **ab dem ersten Cashflow** —
    Buchungen vor ``start`` gehören zur Geldfluss-Historie (kein unterer
    Schnitt), Buchungen **nach ``end``** flüssen nicht mehr ein.
    """
    depot = _depot_or_404(portfolio_id)
    if benchmark is not None and benchmark.strip():
        benchmark = _symbol_validieren(benchmark.strip())
    else:
        benchmark = None
    try:
        start_us = iso_zeitpunkt_zu_us(start) if start is not None else None
        end_us_exklusiv = iso_zeitpunkt_zu_us(end) if end is not None else None
        end_us = (
            end_us_exklusiv + 86_400_000_000 if end_us_exklusiv is not None else None
        )  # „end“ inklusiv: bis Mitternacht des Folgetags bewerten
        return portfolio_service.performance_report(
            depot, start_us, end_us, benchmark, method
        )
    except ProviderError as exc:
        raise exc.to_api_error() from exc
    except ValueError as exc:
        raise _repo_wertefehler(exc, code="ungueltige_methode") from exc


# ---------------------------------------------------------------------------
# CSV-Import/-Export (dokumentiertes Format, Roundtrip-stabil)
# ---------------------------------------------------------------------------


#: CSV-Injektionsschutz (REV-T23 Fix 5): Felder, die mit einem dieser Zeichen
#: beginnen, werden beim Export mit führendem ``'`` präfixiert (OWASP-Regel);
#: der Import entfernt genau diesen bekannten Präfix wieder.
_CSV_GEFAHR_ZEICHEN: tuple[str, ...] = ("=", "+", "-", "@")


def _csv_feld(wert: Any) -> str:
    """Wert → CSV-Zellen-Text (``None``/NaN → leer; Floats mit Punkt).

    Injektionsschutz (REV-T23 Fix 5): beginn das Feld mit ``=``, ``+``, ``-``
    oder ``@``, wird ein führender Apostroph vorangestellt — Tabellenkalku-
    lationen werten solche Zellen sonst als Formel aus.
    """
    if wert is None:
        return ""
    if isinstance(wert, float) and wert != wert:  # NaN-Sicherheit
        return ""
    text = repr(wert) if isinstance(wert, float) else str(wert)
    if text[:1] in _CSV_GEFAHR_ZEICHEN:
        return "'" + text
    return text


def _csv_praefix_weg(zelle: str) -> str:
    """Entfernt den Export-Schutzpräfix: ein führendes ``'`` **nur** wenn das
    zweite Zeichen in der Gefahrmenge steht — sonst ist ein echtes führendes
    Hochkomma gemeint und bleibt unangetastet (Roundtrip-Stabilität).
    """
    if len(zelle) >= 2 and zelle[0] == "'" and zelle[1] in _CSV_GEFAHR_ZEICHEN:
        return zelle[1:]
    return zelle


@router.get("/portfolios/{portfolio_id}/export")
def export_csv(portfolio_id: int) -> Response:
    """Exportiert das Ledger als CSV (Format: Modulkopf / CSV_SPALTEN)."""
    _depot_or_404(portfolio_id)
    puffer = io.StringIO()
    schreiber = csv.writer(puffer, lineterminator="\n")
    schreiber.writerow(CSV_SPALTEN)
    for tx in portfolios_repo.list_transactions(portfolio_id):
        schreiber.writerow(
            [_csv_feld(wert) for wert in (_tx_als_dict(tx)[spalte] for spalte in CSV_SPALTEN)]
        )
    return Response(
        content=puffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="portfolio-{portfolio_id}.csv"'
        },
    )


def _parse_csv_zahl(text: str) -> float | None:
    """CSV-Zelle → float (Dezimalpunkt, leer = ``None``); Fehler wirft ValueError."""
    wert = text.strip()
    if not wert:
        return None
    return float(wert.replace("\u00a0", ""))  # geschützte Leerzeichen tolerieren


@router.post("/portfolios/{portfolio_id}/import")
async def import_csv(portfolio_id: int, request: Request) -> dict[str, Any]:
    """Importiert Buchungen aus CSV — atomar, Fehler je Zeile mit Zeilennummer.

    Body: roher CSV-Text (UTF-8, BOM wird toleriert). Jede Datenzeile wird
    gegen die Typ-Regeln (§C.4) validiert; schlägt eine Zeile fehl, wird
    **nichts** eingefügt und alle Zeilenfehler (mit Zeilennummer) stehen in
    der 422-Meldung. Transfer-Zeilen brauchen ein existierendes Gegen-Depot.
    Nicht-UTF-8-Bytes → definierte 422 ``csv_encoding`` (REV-T23 Fix 2).
    Zeilen, deren Felder vom Export mit ``'`` präfixiert wurden (Injektions-
    schutz), bekommen diesen Präfix hier wieder entfernt (Fix 5).
    """
    _depot_or_404(portfolio_id)
    try:
        roh = (await request.body()).decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        raise ApiError(
            422,
            "csv_encoding",
            "Die CSV-Datei ist nicht in UTF-8 kodiert und konnte nicht gelesen werden.",
            hint="UTF-8 erwartet (BOM toleriert).",
        ) from None
    leser = csv.reader(io.StringIO(roh))
    try:
        kopf = next(leser)
    except StopIteration:
        raise ApiError(422, "csv_format", "Die CSV-Datei ist leer.") from None
    kopf = [spalte.strip() for spalte in kopf]
    if sorted(kopf) != sorted(CSV_SPALTEN):
        raise ApiError(
            422,
            "csv_format",
            "CSV-Kopf passt nicht zum erwarteten Format.",
            hint="Erwartete Spalten: " + ",".join(CSV_SPALTEN) + ".",
        )

    fehler: list[str] = []
    valide: list[dict[str, Any]] = []
    for zellen in leser:
        zeilen_nr = leser.line_num
        if not zellen or all(not zelle.strip() for zelle in zellen):
            continue  # Leerzeilen überspringen
        if len(zellen) != len(CSV_SPALTEN):
            fehler.append(
                f"Zeile {zeilen_nr}: {len(zellen)} Zellen statt {len(CSV_SPALTEN)}"
            )
            continue
        zeile = dict(
            zip(CSV_SPALTEN, (_csv_praefix_weg(zelle) for zelle in zellen), strict=True)
        )
        try:
            typ = zeile["type"].strip()
            if typ not in TX_TYPEN:
                raise ApiError(
                    422, "ungueltige_buchung", f"Unbekannter Buchungstyp „{typ}“."
                )
            felder: dict[str, Any] = {
                "ts": zeile["ts"],
                "type": typ,
                "symbol": zeile["symbol"].strip() or None,
                "quantity": _parse_csv_zahl(zeile["quantity"]),
                "price": _parse_csv_zahl(zeile["price"]),
                "amount": _parse_csv_zahl(zeile["amount"]),
                "fees": _parse_csv_zahl(zeile["fees"]) or 0.0,
                "taxes": _parse_csv_zahl(zeile["taxes"]) or 0.0,
                "currency": zeile["currency"].strip() or None,
                "transfer_portfolio_id": (
                    int(zeile["transfer_portfolio_id"].strip())
                    if zeile["transfer_portfolio_id"].strip()
                    else None
                ),
                "note": zeile["note"].strip() or None,
            }
            modell = _union_modell(felder)
            bereinigt = _validiere_buchungsfelder(portfolio_id, modell.model_dump())
            valide.append((zeilen_nr, bereinigt))
        except ApiError as exc:
            fehler.append(f"Zeile {zeilen_nr}: {exc.message}")
        except Exception as exc:  # float/int/ISO-PARSE-Fehler je Zelle
            fehler.append(f"Zeile {zeilen_nr}: {exc}")

    if fehler:
        gezeigt = fehler[:_MAX_CSV_FEHLER]
        weitere = len(fehler) - _MAX_CSV_FEHLER
        zusatz = f" (+{weitere} weitere)" if weitere > 0 else ""
        raise ApiError(
            422,
            "csv_import_fehler",
            "CSV-Import fehlgeschlagen: " + "; ".join(gezeigt) + zusatz,
            hint="Es wurde keine Zeile importiert.",
        )

    angelegt = [_buchung_einfuegen(portfolio_id, felder) for _, felder in valide]
    _publiziere_geaendert(portfolio_id, f"csv_import:{len(angelegt)}")
    return {"importiert": len(angelegt), "transaktionen": [_tx_als_dict(t) for t in angelegt]}
