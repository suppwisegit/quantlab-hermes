"""Portfolio-Tools (T34, PLAN §T34): Depots, Buchungen, Performance.

Konsumiert ausschließlich die T23-API-Fläche (:mod:`quantlab.storage.repos.
portfolios_repo`, :mod:`quantlab.services.portfolio`) — keine Router-Imports.
Fachliche Feld-Regeln je Buchungstyp spiegeln den §C.4-Vertrag des
Portfolios-Routers (buy/sell → symbol+quantity+price; dividend → symbol+amount;
cash → amount ohne Symbol; transfer → amount + Gegen-Depot ≠ selbst); das Repo
prüft die strukturelle Integrität zusätzlich (bekannter Typ, existierende
Depots) mit ``ValueError`` → :class:`ToolFehler`.

**Rückfrage-Vertrag (§T34):** ``record_transaction`` bucht erst, wenn der
Nutzer die Buchung im Chat klar beauftragt hat — das erzwingt der System-Prompt
plus Tool-Beschreibung (das Tool selbst kann Nutzer-Intent nicht prüfen). Das
Ergebnis enthält eine Zusammenfassung, die der Agent dem Nutzer im Text
bestätigt. Änderungen publizieren ``portfolio_changed`` (Muster Router), damit
die UI live aktualisiert.

Depots sind optional projektgebunden (``project_id``): ohne Angabe gilt die
E021-Projektbindung (:func:`quantlab.agent.tools.kontext.aktives_projekt`).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantlab import events
from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, daten_fehler, pruefe_args, schema_von
from quantlab.agent.tools.kontext import aktives_projekt
from quantlab.providers.base import ProviderError
from quantlab.services import downloads
from quantlab.services import portfolio as portfolio_service
from quantlab.storage.repos import portfolios_repo

#: Maximale Punkte des Equity-Samples im Tool-Ergebnis (Kontext-Schonung; die
#: Vollkurve bleibt über die Projekt-/Performance-UI sichtbar).
_EQUITY_SAMPLE = 60


def _projekt_id(args_project_id: int | None) -> int | None:
    """Projektbindung: Argument schlägt Run-Kontext (E021)."""
    if args_project_id is not None:
        return args_project_id
    return aktives_projekt()


def _depot_als_dict(depot: portfolios_repo.Portfolio) -> dict[str, Any]:
    return {
        "id": depot.id,
        "name": depot.name,
        "base_currency": depot.base_currency,
        "project_id": depot.project_id,
        "notes": depot.notes,
    }


# ---------------------------------------------------------------------------
# manage_portfolios (list/create)
# ---------------------------------------------------------------------------


class ManagePortfoliosArgs(BaseModel):
    """Argumente von ``manage_portfolios``."""

    action: Literal["list", "create"] = Field(
        description='Aktion: "list" (Depots auflisten) oder "create" (neues Depot, braucht name).'
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description='Depot-Name (nur create), z. B. "Dividendenhelden".',
    )
    base_currency: str = Field(
        default="USD",
        min_length=3,
        max_length=12,
        description='Basiswährung als ISO-Kürzel (nur create), Default "USD".',
    )
    notes: str | None = Field(
        default=None, max_length=10_000, description="Optionale Notiz (nur create)."
    )
    project_id: int | None = Field(
        default=None,
        ge=1,
        description="Projektbindung (Default: das im Chat gewählte Projekt).",
    )


async def manage_portfolios(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Depots auflisten oder ein neues (projektgebundenes) Depot anlegen."""
    args = pruefe_args(ManagePortfoliosArgs, roh_args, "manage_portfolios")

    def _ausfuehren() -> dict[str, Any]:
        if args.action == "list":
            depots = [
                _depot_als_dict(depot) for depot in portfolios_repo.list_portfolios()
            ]
            return {
                "action": "list",
                "anzahl": len(depots),
                "depots": depots,
                "hinweis": "Buchungen/Performance je Depot via record_transaction "
                "bzw. get_portfolio_performance mit der Depot-ID.",
            }
        if args.name is None or not args.name.strip():
            raise ToolFehler('„create“ benötigt einen nicht-leeren "name".')
        try:
            depot = portfolios_repo.create_portfolio(
                args.name,
                project_id=_projekt_id(args.project_id),
                base_currency=args.base_currency,
                notes=args.notes or "",
            )
        except ValueError as exc:
            raise ToolFehler(str(exc)) from exc
        return {
            "action": "create",
            "depot": _depot_als_dict(depot),
            "hinweis": "Erste Buchung via record_transaction (z. B. cash_in als "
            "Startkapital oder ein Kauf).",
        }

    return await asyncio.to_thread(_ausfuehren)


# ---------------------------------------------------------------------------
# record_transaction (Rückfrage-Vertrag)
# ---------------------------------------------------------------------------


class RecordTransactionArgs(BaseModel):
    """Argumente von ``record_transaction`` (Feldregeln je Typ, §C.4-Spiegel)."""

    portfolio_id: int = Field(ge=1, description="Depot-ID aus manage_portfolios, z. B. 1.")
    type: Literal["buy", "sell", "dividend", "cash_in", "cash_out", "transfer"] = Field(
        description="Buchungstyp des Ledgers."
    )
    ts: str | None = Field(
        default=None,
        description=(
            "Buchungszeitpunkt ISO-8601, z. B. \"2024-06-01T15:30:00+00:00\" "
            "(naive Angaben gelten als UTC; leer = jetzt)."
        ),
    )
    symbol: str | None = Field(
        default=None,
        min_length=1,
        max_length=24,
        description='Symbol (buy/sell/dividend), z. B. "AAPL".',
    )
    quantity: float | None = Field(
        default=None, gt=0, description="Stückzahl > 0 (nur buy/sell)."
    )
    price: float | None = Field(
        default=None, gt=0, description="Kurs je Stück > 0 (nur buy/sell)."
    )
    amount: float | None = Field(
        default=None, gt=0,
        description="Betrag > 0 (dividend/cash_in/cash_out/transfer).",
    )
    fees: float = Field(default=0.0, ge=0, description="Gebühren, Default 0.")
    taxes: float = Field(default=0.0, ge=0, description="Steuern, Default 0.")
    currency: str | None = Field(
        default=None, min_length=3, max_length=12,
        description='Quellwährung der Position, z. B. "EUR" (optional).',
    )
    transfer_portfolio_id: int | None = Field(
        default=None, ge=1,
        description="Gegen-Depot (nur transfer, ≠ portfolio_id).",
    )
    note: str | None = Field(
        default=None, max_length=2000, description="Optionale Buchungsnotiz."
    )


def _typ_regeln_pruefen(args: RecordTransactionArgs) -> None:
    """Fachliche Feld-Regeln je Buchungstyp (Spiegel des §C.4-Router-Vertrags)."""
    if args.type in ("buy", "sell"):
        if not args.symbol:
            raise ToolFehler(f'„{args.type}“ benötigt ein "symbol".')
        if args.quantity is None or args.price is None:
            raise ToolFehler(
                f'„{args.type}“ benötigt "quantity" > 0 und "price" > 0 '
                '(letzten Kurs via get_quote ermitteln).'
            )
        return
    if args.type == "dividend":
        if not args.symbol:
            raise ToolFehler('„dividend“ benötigt ein "symbol".')
        if args.amount is None:
            raise ToolFehler('„dividend“ benötigt einen "amount" > 0 (Gesamtbetrag).')
        return
    if args.type in ("cash_in", "cash_out"):
        if args.amount is None:
            raise ToolFehler(f'„{args.type}“ benötigt einen "amount" > 0.')
        if args.symbol:
            raise ToolFehler(
                f'„{args.type}“ ist eine Cash-Buchung und nimmt kein "symbol".'
            )
        return
    # transfer (ANNAHME T23: v2 umbucht nur Cash)
    if args.amount is None:
        raise ToolFehler('„transfer“ benötigt einen "amount" > 0.')
    if args.transfer_portfolio_id is None:
        raise ToolFehler('„transfer“ benötigt ein "transfer_portfolio_id" (Gegen-Depot).')
    if args.transfer_portfolio_id == args.portfolio_id:
        raise ToolFehler('„transfer“ darf nicht auf dasselbe Depot gehen.')


def _ts_normalisieren(ts: str | None) -> str:
    """ISO-8601 validieren und auf UTC normalisieren (naive = UTC; leer = jetzt)."""
    if ts is None or not ts.strip():
        return datetime.now(UTC).isoformat()
    try:
        moment = datetime.fromisoformat(ts.strip())
    except ValueError as exc:
        raise ToolFehler(
            f'Ungültiger Buchungszeitpunkt „{ts}“ — ISO-8601 erwartet '
            '(z. B. 2024-06-01T15:30:00+00:00).'
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


def _zusammenfassung(args: RecordTransactionArgs, ts: str) -> str:
    """Deutsche Ein-Zeilen-Zusammenfassung (Bestätigungstext des Agenten)."""
    typ = args.type
    if typ in ("buy", "sell"):
        kern = f"{args.quantity:g} × {args.symbol} @ {args.price:g}"
    elif typ == "dividend":
        kern = f"Dividende {args.symbol}: {args.amount:g}"
    elif typ == "transfer":
        kern = f"Umbuchung {args.amount:g} → Depot {args.transfer_portfolio_id}"
    else:
        kern = f"{args.amount:g} Cash"
    zusatz = []
    if args.fees:
        zusatz.append(f"Gebühren {args.fees:g}")
    if args.taxes:
        zusatz.append(f"Steuern {args.taxes:g}")
    if args.note:
        zusatz.append(f"Notiz: {args.note}")
    anhang = f" ({'; '.join(zusatz)})" if zusatz else ""
    return f"{kern}{anhang} — gebucht zum {ts}."


async def record_transaction(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Bucht eine Transaktion ins Depot-Ledger (nur bei klarer Nutzeranweisung)."""
    args = pruefe_args(RecordTransactionArgs, roh_args, "record_transaction")
    _typ_regeln_pruefen(args)
    ts = _ts_normalisieren(args.ts)

    def _buchen() -> dict[str, Any]:
        depot = portfolios_repo.get_portfolio(args.portfolio_id)
        if depot is None:
            raise ToolFehler(f"Depot {args.portfolio_id} existiert nicht.")
        try:
            tx = portfolios_repo.add_transaction(
                args.portfolio_id,
                ts=ts,
                type=args.type,
                symbol=args.symbol,
                quantity=args.quantity,
                price=args.price,
                amount=args.amount,
                fees=args.fees,
                taxes=args.taxes,
                currency=args.currency,
                transfer_portfolio_id=args.transfer_portfolio_id,
                note=args.note,
            )
        except ValueError as exc:
            raise ToolFehler(str(exc)) from exc
        # Live-Update der Depot-UI (Muster Portfolio-Router, ANNAHME Event-Name).
        events.publizieren(
            "portfolio_changed", {"portfolioId": args.portfolio_id, "grund": "buchung"}
        )
        return {
            "status": "gebucht",
            "transaction_id": tx.id,
            "portfolio_id": args.portfolio_id,
            "portfolio_name": depot.name,
            "typ": args.type,
            "zusammenfassung": _zusammenfassung(args, ts),
            "hinweis": (
                "Die Buchung liegt im Depot-Ledger und erscheint live in der UI — "
                "bestätige sie dem Nutzer ausdrücklich in deiner Antwort."
            ),
        }

    return await asyncio.to_thread(_buchen)


# ---------------------------------------------------------------------------
# get_portfolio_performance
# ---------------------------------------------------------------------------


class GetPortfolioPerformanceArgs(BaseModel):
    """Argumente von ``get_portfolio_performance``."""

    portfolio_id: int = Field(ge=1, description="Depot-ID aus manage_portfolios, z. B. 1.")
    method: Literal["twr", "xirr", "all"] = Field(
        default="all",
        description='Kennzahlen: "twr" (zeitgewichtet + Equity), "xirr" '
        '(geldflussgewichtet) oder "all" (Default).',
    )
    start: str | None = Field(
        default=None, description='Fensterbeginn inklusive, z. B. "2024-01-01" (leer = ab Anfang).'
    )
    end: str | None = Field(
        default=None, description='Fensterende exklusiv, z. B. "2025-01-01" (leer = bis aktuell).'
    )
    benchmark: str | None = Field(
        default=None,
        description='Optionales Benchmark-Symbol (lokal gespeichert), z. B. "^GSPC".',
    )


async def get_portfolio_performance(roh_args: dict[str, Any]) -> dict[str, Any]:
    """TWR/XIRR + Kennzahlen + Equity-Sample eines Depots (T23-Rechenkern)."""
    args = pruefe_args(GetPortfolioPerformanceArgs, roh_args, "get_portfolio_performance")

    def _lesen() -> dict[str, Any]:
        depot = portfolios_repo.get_portfolio(args.portfolio_id)
        if depot is None:
            raise ToolFehler(f"Depot {args.portfolio_id} existiert nicht.")
        start_us = downloads.iso_zeitpunkt_zu_us(args.start) if args.start else None
        end_us = downloads.iso_zeitpunkt_zu_us(args.end) if args.end else None
        report = portfolio_service.performance_report(
            depot,
            start_us,
            end_us,
            benchmark=args.benchmark,
            method=args.method,
        )
        equity = report.get("equity") or []
        schritt = max(1, len(equity) // _EQUITY_SAMPLE)
        return {
            "portfolio_id": depot.id,
            "name": depot.name,
            "basiswaehrung": depot.base_currency,
            "method": report.get("method"),
            "bewertungstage": report.get("bewertungstage"),
            "twr": report.get("twr"),
            "xirr": report.get("xirr"),
            "xirr_hinweis": report.get("xirr_hinweis"),
            "kennzahlen": report.get("kennzahlen"),
            "cash": report.get("cash"),
            "benchmark": report.get("benchmark"),
            "warnungen": report.get("warnungen", []),
            "equity_sample": equity[::schritt],
            "hinweis": "TWR/XIRR als Dezimalzahl (0.12 = 12 %); Equity-Kurve "
            "gesampelt — Details in der Depot-Ansicht.",
        }

    try:
        return await asyncio.to_thread(_lesen)
    except ProviderError as exc:
        raise daten_fehler(exc) from exc
    except ValueError as exc:
        # Definierte numerische Randfälle des T23-Kerns (R14) — z. B. XIRR
        # ohne Vorzeichenwechsel bei method="xirr".
        raise ToolFehler(str(exc)) from exc


# ---------------------------------------------------------------------------
# Registrierung (T34)
# ---------------------------------------------------------------------------


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="manage_portfolios",
            description=(
                "Verwaltet Depots: auflisten (action=\"list\") oder anlegen "
                "(action=\"create\", name=…, base_currency=…, Default-Währung USD). "
                "Ohne project_id wird das im Chat gewählte Projekt gebunden. "
                'Beispiel: {"action": "create", "name": "Dividendenhelden"}.'
            ),
            parameters=schema_von(ManagePortfoliosArgs),
            handler=manage_portfolios,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="record_transaction",
            description=(
                "Bucht eine Transaktion ins Depot-Ledger: buy/sell (symbol, quantity, "
                "price), dividend (symbol, amount), cash_in/cash_out (amount) oder "
                "transfer (amount, transfer_portfolio_id). Wird SOFORT ausgeführt "
                "(keine Bestätigungsstufe im Tool). NUR ausführen, wenn der "
                "Nutzer die Buchung klar beauftragt hat; gib die zurückgegebene "
                "Zusammenfassung in deiner Antwort ausdrücklich wieder. "
                'Beispiel: {"portfolio_id": 1, "type": "buy", "symbol": "AAPL", '
                '"quantity": 10, "price": 232.1}.'
            ),
            parameters=schema_von(RecordTransactionArgs),
            handler=record_transaction,
        )
    )
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="get_portfolio_performance",
            description=(
                "Depot-Performance: TWR und/oder XIRR (Dezimalzahlen), Kennzahlen "
                "(Vol, Sharpe, MaxDD, …), Cash, Equity-Sample, optionales Benchmark "
                "und Warnungen (z. B. Währungsmix). "
                'Beispiel: {"portfolio_id": 1, "method": "all", "benchmark": "^GSPC"}.'
            ),
            parameters=schema_von(GetPortfolioPerformanceArgs),
            handler=get_portfolio_performance,
        )
    )


_registrieren()
