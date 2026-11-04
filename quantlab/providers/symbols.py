"""Symbol-Normalisierung und -Validierung (T3, §D.2).

Yahoo-Ticker enthalten Zeichen, die als Dateiname ungeeignet sind bzw. Pfade
verfälschen. Die Normalisierung bildet daher auf einen sicheren Dateinamen ab:

- ``^GSPC``   → ``_GSPC``   (Index)
- ``EURUSD=X``→ ``EURUSD_X`` (Devisenpaar)
- ``ES=F``    → ``ES_F``     (Future)
- ``BRK.B``   → ``BRK.B``    (Klassensuffix bleibt erhalten)

Gültig ist ein Symbol, wenn es nach dem Entfernen von Rand-Whitespace nur
Zeichen aus ``A-Z a-z 0-9 . _ - ^ =`` enthält. Der *sichere Name* besteht
ausschließlich aus der Whitelist ``[A-Za-z0-9_.-]`` (§D.2) — ``^`` und ``=``
werden durch ``_`` ersetzt. Die Registry/Speicherung schlüsselt überall über
den *ursprünglichen* Symbol-Kürzel (Dateiname nur abgeleitet).

ANNAHME: Die Zuordnung ist nicht bijektiv (``^GSPC`` und ``_GSPC`` ergäben
denselben Dateinamen) — Yahoo hat keine Ticker mit führendem Unterstrich, die
Kollision ist praktisch ausgeschlossen.
"""

from __future__ import annotations

import re

from quantlab.providers.base import ProviderError

#: Für den ROH-Symbol-Kürzel erlaubte Zeichen: Whitelist aus §D.2 plus die
#: beiden Yahoo-Sonderzeichen, die beim Normalisieren ersetzt werden.
_SYMBOL_ALLOWED = re.compile(r"^[A-Za-z0-9_.^\-=]+$")

#: Für den SICHEREN Dateinamen erlaubte Zeichen (§D.2) — defensiv geprüft.
_SAFE_ALLOWED = re.compile(r"^[A-Za-z0-9_.\-]+$")

#: Ersetzungstabelle: Index-Präfix und „=“-Suffix werden zu ``_``.
_TO_SAFE = str.maketrans({"^": "_", "=": "_"})

#: Obergrenze gegen pathologisch lange Eingaben (Yahoo-Kürzel sind deutlich kürzer).
MAX_SYMBOL_LENGTH = 24


def validate_symbol(symbol: str) -> str:
    """Prüft den Symbol-Kürzel und liefert ihn getrimmt zurück.

    Fehler (leer, zu lang, unerlaubte Zeichen, kein String) als
    :class:`ProviderError` mit HTTP 422 / Code ``ungueltiges_symbol``.
    Die Schreibweise bleibt erhalten (kein Uppercase-Zwang).
    """
    if not isinstance(symbol, str):
        raise ProviderError(
            "Symbol muss eine Zeichenkette sein.", 422, "ungueltiges_symbol"
        )
    candidate = symbol.strip()
    if not candidate:
        raise ProviderError(
            "Symbol fehlt oder ist leer.", 422, "ungueltiges_symbol"
        )
    if len(candidate) > MAX_SYMBOL_LENGTH:
        raise ProviderError(
            f"Symbol ist zu lang (maximal {MAX_SYMBOL_LENGTH} Zeichen).",
            422,
            "ungueltiges_symbol",
        )
    if not _SYMBOL_ALLOWED.match(candidate):
        raise ProviderError(
            f"Symbol „{candidate}“ enthält ungültige Zeichen.",
            422,
            "ungueltiges_symbol",
            hint="Erlaubt sind Buchstaben, Zahlen sowie . _ - ^ = (Yahoo-Kürzel).",
        )
    return candidate


def safe_filename(symbol: str) -> str:
    """Normalisiert einen Symbol-Kürzel zum sicheren Dateinamen (§D.2).

    Validiert zuerst (siehe :func:`validate_symbol`), ersetzt danach ``^`` und
    ``=`` durch ``_`` — Ergebnis passt exakt in die Whitelist
    ``[A-Za-z0-9_.-]`` und ist ohne Pfad-Sonderzeichen.
    """
    candidate = validate_symbol(symbol)
    safe = candidate.translate(_TO_SAFE)
    if not _SAFE_ALLOWED.match(safe):  # defensiv: durch Konstruktion unerreichbar
        raise ProviderError(
            f"Symbol „{candidate}“ ergibt keinen sicheren Dateinamen.",
            422,
            "ungueltiges_symbol",
        )
    return safe
