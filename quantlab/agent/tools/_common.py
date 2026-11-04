"""Gemeinsame Hilfen aller Agent-Tools (T14): Fehler- und Validierungsvertrag.

Zwei Regeln, beide bindend (§E, §T14):

1. **Fachliche Fehler sind Tool-Ergebnisse, kein Run-Abbruch.** Handler werfen
   für alles Erwartbare (unbekanntes Symbol, fehlende Daten, sinnwidrige
   Argumente, Projekt fehlt) :class:`ToolFehler` — :func:`quantlab.agent.
   registry.dispatch_tool` fängt ihn ab und macht daraus eine ``role:tool``-
   Message mit ``ok=False`` und deutschem Text; der Run läuft weiter.
2. **Args-Validierung via Pydantic.** Die Registry validiert selbst nicht —
   jedes Tool prüft seine Argumente gegen sein Pydantic-Modell
   (:func:`pruefe_args`); Verletzungen werden zu deutschen Tool-Fehlern.

   **Entscheidung (bewusst, dokumentiert):** Extra-Args werden **still
   ignoriert** (Pydantic-Default ``extra="ignore"``). LLMs senden gelegentlich
   harmlose Zusatzfelder (verwechselte Parameternamen, aus dem Beschreibungs-
   text übernommene Beispiele); strikte Ablehnung (``extra="forbid"``) würde
   komplette Runs brechen, ohne fachlich nötig zu sein. Deshalb gibt es auch
   keine Übersetzung für „Extra inputs are not permitted“ — der Fall tritt im
   Tool-Vertrag nie auf.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from quantlab.providers.base import ProviderError


class ToolFehler(Exception):
    """Fachlicher Tool-Fehler → ``ok=False``-Tool-Ergebnis (Run läuft weiter, §E)."""


#: Deutsche Tool-Fehlermeldungen **nach Pydantic-Fehlertyp** (Schlüssel ist
#: ``exc.errors()[i]["type"]``, nicht der msg-String — stabiler über Versionen
#: und Sprachen hinweg). ``{platzhalter}`` werden aus dem Fehler-``ctx`` befüllt.
#: Fehlertypen, die hier fehlen, fallen auf die englische Pydantic-Original-
#: meldung zurück (bewusst akzeptiert: selten, verständlich, fachlich korrekt).
_MSG_DE: dict[str, str] = {
    "missing": "Pflichtfeld fehlt",
    "string_type": "muss ein Text sein",
    "string_too_short": "zu kurz (mindestens {min_length})",
    "string_too_long": "zu lang (maximal {max_length})",
    "too_short": "zu kurz (mindestens {min_length})",
    "too_long": "zu lang (maximal {max_length})",
    "int_type": "muss eine ganze Zahl sein",
    "int_parsing": "muss eine ganze Zahl sein",
    "int_from_float": "muss eine ganze Zahl sein",
    "float_type": "muss eine Zahl sein",
    "float_parsing": "muss eine Zahl sein",
    "bool_type": "erwartet true/false",
    "list_type": "erwartet Liste",
    "dict_type": "erwartet Objekt",
    "enum": "ungültiger Wert (erlaubt: {expected})",
    "literal_error": "ungültiger Wert (erlaubt: {expected})",
    "greater_than_equal": "muss ≥ {ge} sein",
    "less_than_equal": "muss ≤ {le} sein",
}


def _meldung_de(fehler: dict[str, Any]) -> str:
    """Eine Pydantic-Fehlerzeile → deutscher Kurztext (Restfall: englische msg)."""
    typ = fehler.get("type")
    vorlage = _MSG_DE.get(typ) if isinstance(typ, str) else None
    if vorlage is None:
        return str(fehler.get("msg", "ungültige Eingabe"))
    try:
        return vorlage.format(**(fehler.get("ctx") or {}))
    except (KeyError, IndexError):
        # ctx passt nicht zur Vorlage → englische Original-Meldung (Restfall).
        return str(fehler.get("msg", "ungültige Eingabe"))


def pruefe_args[T: BaseModel](
    model_cls: type[T], args: dict[str, Any], tool_name: str
) -> T:
    """Validiert die rohen Tool-Argumente gegen ein Pydantic-Modell.

    Validierungsfehler werden zu :class:`ToolFehler` mit deutscher,
    feldgenauer Meldung (die Registry validiert nicht — §T14: „Pydantic-Prüfung
    in DEINEN Handlern“). Extra-Args werden still ignoriert (siehe Modul-
    Docstring, Regel 2).
    """
    try:
        return model_cls.model_validate(args)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(ort) for ort in fehler['loc']) or '(root)'}: {_meldung_de(fehler)}"
            for fehler in exc.errors()
        )
        raise ToolFehler(f"Ungültige Argumente für „{tool_name}“: {details}") from exc


def daten_fehler(exc: Exception) -> ToolFehler:
    """Mappt einen Daten-Domänen-Fehler (:class:`ProviderError`) auf :class:`ToolFehler`.

    Die Meldungen des Daten-Service sind bereits deutsch (T4) — sie werden
    1:1 übernommen, ergänzt um den Handlungshinweis „Daten herunterladen“,
    wenn das Symbol einfach fehlt (404-Code).
    """
    if isinstance(exc, ProviderError):
        meldung = exc.message
        if exc.code == "symbol_nicht_gefunden":
            meldung = f"{meldung} (Hinweis: download_data kann die Daten zuerst laden.)"
        return ToolFehler(meldung)
    return ToolFehler(str(exc))


def zahl(wert: Any) -> float | None:
    """JSON-sicherer float: ``None`` bleibt ``None``, numpy-Werte werden konvertiert."""
    if wert is None:
        return None
    return round(float(wert), 6)


def _inliniere_refs(knoten: Any, defs: dict[str, Any], offen: frozenset[str]) -> Any:
    """Ersetzt ``{"$ref": "#/$defs/Name"}`` rekursiv durch das Sub-Schema aus ``defs``.

    Geschwister-Schlüssel neben ``$ref`` (z. B. ``description``) bleiben
    erhalten und überschreiben die des Ziel-Schemas. Ein fehlender
    ``$defs``-Eintrag oder ein rekursives Modell (Zyklus) ist ein Programm-
    fehler und wird laut gemeldet, statt ein kaputtes Schema auszuliefern.
    """
    if isinstance(knoten, list):
        return [_inliniere_refs(element, defs, offen) for element in knoten]
    if not isinstance(knoten, dict):
        return knoten
    ref = knoten.get("$ref")
    if not isinstance(ref, str):
        return {
            schluessel: _inliniere_refs(wert, defs, offen)
            for schluessel, wert in knoten.items()
        }
    praefix = "#/$defs/"
    if not ref.startswith(praefix):
        raise ValueError(f"Nicht unterstützter $ref in Tool-Schema: {ref!r}")
    name = ref[len(praefix) :]
    ziel = defs.get(name)
    if ziel is None:
        raise ValueError(f"$defs-Eintrag {name!r} fehlt im Tool-Schema.")
    if name in offen:
        raise ValueError(
            f"Rekursives Modell {name!r} kann nicht inline aufgelöst werden."
        )
    aufgeloest: dict[str, Any] = _inliniere_refs(ziel, defs, offen | {name})
    geschwister = {
        schluessel: _inliniere_refs(wert, defs, offen)
        for schluessel, wert in knoten.items()
        if schluessel != "$ref"
    }
    return {**aufgeloest, **geschwister} if geschwister else aufgeloest


def schema_von(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic-Args-Modell → sauberes OpenAI-Function-Schema (ohne ``title``-Ballast).

    Liefert ``{"type": "object", "properties": …, "required": […]}`` — ``required``
    entfällt, wenn keine Pflichtfelder existieren (leere Listen verwirren nur).

    ``$ref``/``$defs`` werden vorab **rekursiv inline aufgelöst** (Sub-Schemas
    aus ``$defs`` eingebettet, danach verschwinden sie): Das Wire-Format
    z.ai/OpenAI kennt keine lokale Definitionstabelle — ein stehengebliebener
    ``$ref`` würde dem Modell das Sub-Schema verheimlichen (z. B. sähe es bei
    ``create_chart`` nicht, welche Felder ein ``indicators``-Eintrag hat).
    """
    roh = model.model_json_schema()
    defs: dict[str, Any] = roh.get("$defs", {})
    schema: dict[str, Any] = {
        "type": "object",
        "properties": _inliniere_refs(roh.get("properties", {}), defs, frozenset()),
    }
    required = roh.get("required", [])
    if required:
        schema["required"] = required
    return schema
