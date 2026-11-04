"""Portfolios-Repository: ``portfolios`` + ``portfolio_transactions`` (v2 §D.2).

Fachliche Grundlage E025: Ein Depot ist ein Transaktions-Ledger; Positionen und
Performance werden abgeleitet (T23). Depots sind optional projektgebunden
(``project_id = NULL`` = global). Buchungstypen mit Gebühren/Steuern als eigene
Felder; ``transfer`` bucht auf ein Gegen-Depot.

Das Repo ist bewusst dünn (nur DB-Schicht): TWR/XIRR/Positionen rechnet
``quant/portfolio_perf.py`` (T23), CSV/Validierung je Buchungstyp macht der
Router. Strukturelle Integrität (bekannter Typ, Transfer-Regeln, existierendes
Depot) sichert das Repo mit ``ValueError`` ab — SQLite-CHECKs fangen den Rest.

Muster: :mod:`quantlab.db` — eine Verbindung pro Operation, ``BEGIN IMMEDIATE``,
``ROLLBACK`` nur bei offener Transaktion.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from quantlab import db

#: Zulässige Buchungstypen (CHECK-Constraint §D.2, E025).
TX_TYPES: frozenset[str] = frozenset(
    {"buy", "sell", "dividend", "cash_in", "cash_out", "transfer"}
)


@dataclass(frozen=True)
class Portfolio:
    """Zeile aus ``portfolios`` (``project_id`` NULL = global)."""

    id: int
    project_id: int | None
    name: str
    base_currency: str
    notes: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Transaction:
    """Zeile aus ``portfolio_transactions`` (Buchungs-/Ledger-Eintrag, E025)."""

    id: int
    portfolio_id: int
    ts: str
    type: str  # buy | sell | dividend | cash_in | cash_out | transfer
    symbol: str | None
    quantity: float | None
    price: float | None
    amount: float | None
    fees: float
    taxes: float
    currency: str | None
    transfer_portfolio_id: int | None
    note: str | None
    created_at: str


def _zu_portfolio(row: sqlite3.Row) -> Portfolio:
    project_raw = row["project_id"]
    return Portfolio(
        id=int(row["id"]),
        project_id=None if project_raw is None else int(project_raw),
        name=str(row["name"]),
        base_currency=str(row["base_currency"]),
        notes=str(row["notes"] or ""),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _zu_transaction(row: sqlite3.Row) -> Transaction:
    return Transaction(
        id=int(row["id"]),
        portfolio_id=int(row["portfolio_id"]),
        ts=str(row["ts"]),
        type=str(row["type"]),
        symbol=row["symbol"],
        quantity=None if row["quantity"] is None else float(row["quantity"]),
        price=None if row["price"] is None else float(row["price"]),
        amount=None if row["amount"] is None else float(row["amount"]),
        fees=float(row["fees"] or 0.0),
        taxes=float(row["taxes"] or 0.0),
        currency=row["currency"],
        transfer_portfolio_id=(
            None if row["transfer_portfolio_id"] is None else int(row["transfer_portfolio_id"])
        ),
        note=row["note"],
        created_at=str(row["created_at"]),
    )


# ---------------------------------------------------------------------------
# Depots (CRUD)
# ---------------------------------------------------------------------------


def create_portfolio(
    name: str,
    *,
    project_id: int | None = None,
    base_currency: str = "USD",
    notes: str = "",
) -> Portfolio:
    """Legt ein Depot an (global oder projektgebunden) und liefert es zurück.

    ``ValueError`` bei leerem Namen.
    """
    if not name.strip():
        raise ValueError("Depot-Name darf nicht leer sein.")
    now = db.utc_now_iso()
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT INTO portfolios(project_id, name, base_currency, notes, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, name.strip(), base_currency, notes, now, now),
        )
        row = conn.execute(
            "SELECT * FROM portfolios WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_portfolio(row)


def get_portfolio(portfolio_id: int) -> Portfolio | None:
    """Ein Depot anhand seiner ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM portfolios WHERE id = ?", (int(portfolio_id),)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_portfolio(row)


def list_portfolios(*, project_id: int | None = None) -> list[Portfolio]:
    """Depots, nach Name sortiert — ``project_id=None`` liefert alle (auch globale)."""
    query = "SELECT * FROM portfolios"
    args: list[Any] = []
    if project_id is not None:
        query += " WHERE project_id = ?"
        args.append(int(project_id))
    query += " ORDER BY name COLLATE NOCASE, id"
    conn = db.connect()
    try:
        rows = conn.execute(query, args).fetchall()
    finally:
        conn.close()
    return [_zu_portfolio(row) for row in rows]


def update_portfolio(
    portfolio_id: int,
    *,
    name: str | None = None,
    base_currency: str | None = None,
    notes: str | None = None,
) -> Portfolio | None:
    """Aktualisiert Felder schlüsselweise (``None`` = unverändert), stampft ``updated_at``.

    ``None`` wenn das Depot nicht existiert; ``ValueError`` bei leerem Namen.
    """
    if name is not None and not name.strip():
        raise ValueError("Depot-Name darf nicht leer sein.")
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM portfolios WHERE id = ?", (int(portfolio_id),)
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        neu = {
            "name": row["name"] if name is None else name.strip(),
            "base_currency": row["base_currency"] if base_currency is None else base_currency,
            "notes": row["notes"] if notes is None else notes,
        }
        conn.execute(
            "UPDATE portfolios SET name = ?, base_currency = ?, notes = ?, updated_at = ? "
            "WHERE id = ?",
            (neu["name"], neu["base_currency"], neu["notes"], db.utc_now_iso(), int(portfolio_id)),
        )
        updated = conn.execute(
            "SELECT * FROM portfolios WHERE id = ?", (int(portfolio_id),)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_portfolio(updated)


def delete_portfolio(portfolio_id: int) -> bool:
    """Löscht ein Depot samt Ledger; ``True`` wenn es existierte.

    Es werden auch Transaktionen entfernt, die das Depot nur als **Transfer-
    Gegenstelle** referenzieren — es dürfen keine dangling Verweise bleiben
    (keine FOREIGN KEYs im Schema §D.2, der Löschpfad läuft explizit).
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM portfolio_transactions WHERE portfolio_id = ? "
            "OR transfer_portfolio_id = ?",
            (int(portfolio_id), int(portfolio_id)),
        )
        cursor = conn.execute(
            "DELETE FROM portfolios WHERE id = ?", (int(portfolio_id),)
        )
        deleted = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return deleted


# ---------------------------------------------------------------------------
# Transaktions-Ledger
# ---------------------------------------------------------------------------


def _pruefe_transaktion(
    portfolio_id: int,
    typ: str,
    transfer_portfolio_id: int | None,
    conn: sqlite3.Connection,
) -> None:
    """Strukturelle Integrität einer Buchung — wirft ``ValueError`` mit deutscher Meldung.

    Fachliche Feld-Validierung je Typ (buy/sell → quantity+price etc.) macht der
    Router (§C.4, T23); hier wird nur gesichert, dass die Zeile der Ledger-
    Semantik genügt: bekannter Typ, existierendes Depot (bei ``transfer`` auch
    Gegen-Depot), Transfer-Regeln.
    """
    if typ not in TX_TYPES:
        raise ValueError(f"Unbekannter Buchungstyp: {typ!r}")
    vorhanden = conn.execute(
        "SELECT 1 FROM portfolios WHERE id = ?", (int(portfolio_id),)
    ).fetchone()
    if vorhanden is None:
        raise ValueError(f"Depot {int(portfolio_id)} existiert nicht.")
    if typ == "transfer":
        if transfer_portfolio_id is None:
            raise ValueError("Eine Umbuchung braucht ein Gegen-Depot (transfer_portfolio_id).")
        if int(transfer_portfolio_id) == int(portfolio_id):
            raise ValueError("Eine Umbuchung darf nicht auf dasselbe Depot gehen.")
        gegen_depot = conn.execute(
            "SELECT 1 FROM portfolios WHERE id = ?", (int(transfer_portfolio_id),)
        ).fetchone()
        if gegen_depot is None:
            raise ValueError(
                f"Gegen-Depot {int(transfer_portfolio_id)} existiert nicht."
            )


def add_transaction(
    portfolio_id: int,
    *,
    ts: str,
    type: str,
    symbol: str | None = None,
    quantity: float | None = None,
    price: float | None = None,
    amount: float | None = None,
    fees: float = 0.0,
    taxes: float = 0.0,
    currency: str | None = None,
    transfer_portfolio_id: int | None = None,
    note: str | None = None,
) -> Transaction:
    """Fügt eine Ledger-Buchung ein und liefert sie zurück (E025).

    ``ValueError`` bei unbekanntem Typ, fehlendem Depot oder Transfer-Regel-
    Verstoß (kein Gegen-Depot / Selbstbezug / nicht existierendes Gegen-Depot).
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _pruefe_transaktion(portfolio_id, type, transfer_portfolio_id, conn)
        cursor = conn.execute(
            "INSERT INTO portfolio_transactions("
            "portfolio_id, ts, type, symbol, quantity, price, amount, fees, taxes, "
            "currency, transfer_portfolio_id, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(portfolio_id),
                ts,
                type,
                symbol,
                quantity,
                price,
                amount,
                fees,
                taxes,
                currency,
                transfer_portfolio_id,
                note,
                db.utc_now_iso(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM portfolio_transactions WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert row is not None  # frisch eingefügt — immer lesbar
    return _zu_transaction(row)


def get_transaction(transaction_id: int) -> Transaction | None:
    """Eine Buchung anhand ihrer ID — ``None`` wenn unbekannt."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM portfolio_transactions WHERE id = ?", (int(transaction_id),)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else _zu_transaction(row)


def list_transactions(portfolio_id: int) -> list[Transaction]:
    """Ledger eines Depots, chronologisch (``ts`` aufsteigend, ``id`` als Tiebreaker).

    Leer bei unbekanntem Depot (Lese-Pfad bleibt robust; 404 macht der Router
    über :func:`get_portfolio`).
    """
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM portfolio_transactions WHERE portfolio_id = ? ORDER BY ts, id",
            (int(portfolio_id),),
        ).fetchall()
    finally:
        conn.close()
    return [_zu_transaction(row) for row in rows]


def update_transaction(
    transaction_id: int,
    *,
    ts: str | None = None,
    type: str | None = None,
    symbol: str | None = None,
    quantity: float | None = None,
    price: float | None = None,
    amount: float | None = None,
    fees: float | None = None,
    taxes: float | None = None,
    currency: str | None = None,
    transfer_portfolio_id: int | None = None,
    note: str | None = None,
) -> Transaction | None:
    """Aktualisiert Felder schlüsselweise (``None`` = unverändert).

    ``None`` wenn die Buchung nicht existiert. Nullable-Felder (``symbol``,
    ``note``, ``currency``, …) lassen sich per Update nicht auf NULL setzen —
    dafür steht das Neuanlegen der Buchung (Löschen + neu), kein Patch-Fall in
    §C.4 braucht es. Das **Ergebnis** wird erneut gegen die Transfer-Regeln
    geprüft (``ValueError``).
    """
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM portfolio_transactions WHERE id = ?", (int(transaction_id),)
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")  # nichts zu tun — Transaktion sauber schließen
            return None
        neu = {
            "portfolio_id": int(row["portfolio_id"]),
            "ts": row["ts"] if ts is None else ts,
            "type": row["type"] if type is None else type,
            "symbol": row["symbol"] if symbol is None else symbol,
            "quantity": row["quantity"] if quantity is None else quantity,
            "price": row["price"] if price is None else price,
            "amount": row["amount"] if amount is None else amount,
            "fees": row["fees"] if fees is None else fees,
            "taxes": row["taxes"] if taxes is None else taxes,
            "currency": row["currency"] if currency is None else currency,
            "transfer_portfolio_id": (
                row["transfer_portfolio_id"]
                if transfer_portfolio_id is None
                else int(transfer_portfolio_id)
            ),
            "note": row["note"] if note is None else note,
        }
        _pruefe_transaktion(
            neu["portfolio_id"], str(neu["type"]), neu["transfer_portfolio_id"], conn
        )
        conn.execute(
            "UPDATE portfolio_transactions SET ts = ?, type = ?, symbol = ?, quantity = ?, "
            "price = ?, amount = ?, fees = ?, taxes = ?, currency = ?, "
            "transfer_portfolio_id = ?, note = ? WHERE id = ?",
            (
                neu["ts"],
                neu["type"],
                neu["symbol"],
                neu["quantity"],
                neu["price"],
                neu["amount"],
                neu["fees"],
                neu["taxes"],
                neu["currency"],
                neu["transfer_portfolio_id"],
                neu["note"],
                int(transaction_id),
            ),
        )
        updated = conn.execute(
            "SELECT * FROM portfolio_transactions WHERE id = ?", (int(transaction_id),)
        ).fetchone()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    assert updated is not None
    return _zu_transaction(updated)


def delete_transaction(transaction_id: int) -> bool:
    """Löscht eine Buchung; ``True`` wenn sie existierte."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM portfolio_transactions WHERE id = ?", (int(transaction_id),)
        )
        deleted = cursor.rowcount > 0
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return deleted
