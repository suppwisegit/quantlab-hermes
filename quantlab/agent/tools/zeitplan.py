"""Zeitplan-Tool des Chat-Agenten (Welle H, E099) — ``verwalte_zeitplaene``.

Der Nutzer sagt im Chat z. B. „Erstelle einen Cron-Job, der alle x Minuten die
neuen Daten lädt, das Depot checkt und neu kauft/verkauft“ — dieses Tool
richtet die Aufgabe **SELBST** ein. Delegation ausschließlich über die
bestehenden Schnittstellen:

- :mod:`quantlab.storage.repos.settings_repo` — Settings-Sektion
  ``agent_cron`` (``aufgaben``/``letzte_laeufe``), schlüsselweiser Merge;
- :func:`quantlab.services.scheduler.agent_cron_einstellungen` — defensiv
  getypter Lese-Stand;
- :func:`quantlab.services.scheduler.run_now("agent_cron")` — der run-now-
  Pfad für ``jetzt_ausfuehren`` (Single-Flight, SSE, letzte-Lauf-Persistenz).

Die Validierung **spiegelt exakt** die Regeln der Einstellungen-API
(``api/routers/settings._agent_cron_aufgaben_fehler``, T-F1/E080): Pflicht-
Feldsatz, ``intervall_min ≥ 5``, Preset-/Modus-Enum, Prompt-Pflicht bei
``preset=custom`` und die Modi-Kette — ``modus=trade`` nur mit
``freigabe_ertelt=true`` plus positivem ``kapital_limit_eur`` und
``max_orders_pro_lauf ≥ 1``; die Freigabe außerhalb von ``trade`` wird
abgelehnt. E080/E081 bleiben intakt: Dieses Tool ändert nur Settings bzw.
löst den bestehenden Tick — die modus-gefilterte Registry des Scheduler-Laufs
baut :func:`scheduler._gefilterte_registry` (Blocklist ``_SCHREIB_TOOLS``,
dort ist dieses Tool selbst eingetragen: ein Cron-Lauf kann sich nicht selbst
umkonfigurieren).
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantlab.agent.registry import GLOBAL_REGISTRY, ToolSpec
from quantlab.agent.tools._common import ToolFehler, pruefe_args, schema_von
from quantlab.storage.repos import settings_repo

#: Preset-Namen (Vertrag der Settings-UI, T-F1) — die Preset-**Texte** hält
#: :data:`quantlab.services.scheduler.AGENT_CRON_PRESETS` als Single Source.
_PRESETS = ("depot_check", "markt_wachposten", "rebalance_vorschlag", "custom")

#: Untergrenze des Intervalls — Spiegel der Router-Validierung
#: (``_AGENT_CRON_MIN_INTERVALL_MIN = 5``, settings.py). Bewusst NICHT 1:
#: Werte < 5 würde die Einstellungen-API mit 400 ablehnen („intervall_min“
#: muss eine Zahl ≥ 5 sein) — das Tool erzeugt keine verbotenen Aufgaben.
_MIN_INTERVALL_MIN = 5

#: Kappung des letzten-Lauf-Kurztexts in der Auflistung (Kontext-Schonung).
_KURZ_MAX_ZEICHEN = 160


class VerwalteZeitplaeneArgs(BaseModel):
    """Argumente von ``verwalte_zeitplaene`` (E099)."""

    aktion: Literal["auflisten", "anlegen", "loeschen", "jetzt_ausfuehren"] = Field(
        description=(
            '"auflisten" (Zeitpläne + letzter Lauf je Aufgabe), "anlegen", '
            '"loeschen" (je mit name) oder "jetzt_ausfuehren" (name → sofortiger '
            "Scheduler-Lauf)."
        ),
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Eindeutiger Aufgabenname (anlegen/loeschen/jetzt_ausfuehren).",
    )
    aufgaben_text: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "Aufgabentext für den Cron-Lauf (Pflicht bei preset=custom): was der "
            "Agent je Lauf tun soll, z. B. neue Daten laden, Depot prüfen, "
            "Handelsvorschläge formulieren."
        ),
    )
    preset: Literal["depot_check", "markt_wachposten", "rebalance_vorschlag", "custom"] = (
        Field(
            default="custom",
            description="Aufgaben-Preset (Text kommt aus dem Scheduler): "
            "depot_check, markt_wachposten, rebalance_vorschlag; custom nutzt "
            "aufgaben_text.",
        )
    )
    intervall_minuten: int | None = Field(
        default=None,
        ge=_MIN_INTERVALL_MIN,
        description="Intervall in Minuten — mindestens 5 (Einstellungen-Regel).",
    )
    modus: Literal["analyse", "vorschlag", "trade"] = Field(
        default="vorschlag",
        description='"analyse" (nur lesen), "vorschlag" (Empfehlungen als Text — '
        "im Wettbewerb empfohlen) oder \"trade\" (nur bei ausdrücklicher "
        "Nutzer-Freigabe, mit Limits).",
    )
    aktiv: bool = Field(default=True, description="Aufgabe aktiviert (Default: ja).")
    nur_handelszeiten: bool = Field(
        default=False,
        description="Nur im Handelsfenster 08–22 Uhr Europe/Berlin ausführen "
        "(Default: nein — für Planspiel-Aufgaben empfohlen).",
    )
    freigabe_ertelt: bool = Field(
        default=False,
        description="Nur true, wenn der Nutzer dem vollautomatischen Handeln "
        "(modus=trade) im Chat ausdrücklich zugestimmt hat.",
    )
    kapital_limit_eur: float | None = Field(
        default=None,
        gt=0,
        description="Pflicht bei modus=trade: hartes Kapital-Limit je Lauf in €.",
    )
    max_orders_pro_lauf: int | None = Field(
        default=None,
        ge=1,
        description="Pflicht bei modus=trade: maximale Anzahl Orders je Lauf.",
    )


def _agent_cron_aufgaben() -> list[dict[str, Any]]:
    """Roh-Aufgabenliste aus der Settings-Sektion (defensiv getypt)."""
    roh = settings_repo.get_section("agent_cron").get("aufgaben")
    if not isinstance(roh, list):
        return []
    return [a for a in roh if isinstance(a, dict)]


def _aufgaben_speichern(aufgaben: list[dict[str, Any]]) -> None:
    """Persistiert die Aufgabenliste — Merge NUR unter dem Key ``aufgaben``;
    ``letzte_laeufe`` (Scheduler-Buchhaltung) bleibt unberührt."""
    settings_repo.update_section("agent_cron", {"aufgaben": aufgaben})


def _finde_aufgabe(name: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Aufgabe per ID oder Name finden — ``None`` + Trefferliste bei Mehrdeutigkeit."""
    gesucht = name.strip()
    aufgaben = _agent_cron_aufgaben()
    for aufgabe in aufgaben:
        if str(aufgabe.get("id") or "") == gesucht:
            return aufgabe, []
    treffer = [
        a for a in aufgaben if str(a.get("name") or "").strip() == gesucht
    ]
    if len(treffer) == 1:
        return treffer[0], []
    return None, treffer


def _validiere_anlegen(args: VerwalteZeitplaeneArgs) -> None:
    """Spiegelt die Router-Validierung (settings.py) — deutsche Meldungen."""
    name = (args.name or "").strip()
    if not name:
        raise ToolFehler(
            "Aufgabe nicht angelegt — „name“ fehlt oder ist leer. Bitte einen "
            "eindeutigen deutschen Namen wählen, z. B. „Depot-Check alle 30 min“."
        )
    for aufgabe in _agent_cron_aufgaben():
        if str(aufgabe.get("name") or "").strip() == name:
            raise ToolFehler(
                f"Aufgabe nicht angelegt — der Name „{name}“ ist bereits vergeben "
                f"(ID {aufgabe.get('id')}). Bitte einen anderen Namen wählen oder "
                "die bestehende Aufgabe erst löschen (aktion=loeschen)."
            )
    # Intervall ≥ 5 erzwingt das Args-Schema (ge=_MIN_INTERVALL_MIN) — exakt die
    # Router-Regel; hier nur die defensive Zweitprüfung (int/bool-Schutz).
    if (
        isinstance(args.intervall_minuten, bool)
        or not isinstance(args.intervall_minuten, int)
        or args.intervall_minuten < _MIN_INTERVALL_MIN
    ):
        raise ToolFehler(
            f"Aufgabe nicht angelegt — „intervall_minuten“ muss eine Zahl ≥ "
            f"{_MIN_INTERVALL_MIN} sein (Einstellungen-Regel für Agent-Zeitpläne)."
        )
    if args.preset == "custom" and not (args.aufgaben_text or "").strip():
        raise ToolFehler(
            "Aufgabe nicht angelegt — Preset „custom“ erfordert einen nicht-leeren "
            "„aufgaben_text“ (was soll der Agent je Lauf tun?). Alternativ ein "
            "fertiges Preset wählen (depot_check, markt_wachposten, "
            "rebalance_vorschlag)."
        )
    if args.modus == "trade":
        if not args.freigabe_ertelt:
            raise ToolFehler(
                "Aufgabe nicht angelegt — Modus „trade“ erfordert "
                "„freigabe_ertelt: true“. Vollautomatisches Handeln richtet der "
                "Agent nur nach ausdrücklicher Nutzer-Freigabe im Chat ein; "
                "für Empfehlungen als Text ist Modus „vorschlag“ der Standard."
            )
        if args.kapital_limit_eur is None or args.kapital_limit_eur <= 0:
            raise ToolFehler(
                "Aufgabe nicht angelegt — Modus „trade“ erfordert ein positives "
                "„kapital_limit_eur“ (hartes Kapital-Limit je Lauf)."
            )
        if args.max_orders_pro_lauf is None or args.max_orders_pro_lauf < 1:
            raise ToolFehler(
                "Aufgabe nicht angelegt — Modus „trade“ erfordert "
                "„max_orders_pro_lauf“ ≥ 1 (Maximalzahl Orders je Lauf)."
            )
    elif args.freigabe_ertelt:
        raise ToolFehler(
            "Aufgabe nicht angelegt — „freigabe_ertelt“ ist nur mit Modus „trade“ "
            "zulässig (Modi-Kette E080/E081)."
        )


def _letzter_lauf_kompakt(aufgabe_id: Any, letzte_laeufe: dict[str, Any]) -> dict[str, Any] | None:
    """Letzter-Lauf-Eintrag einer Aufgabe in Kurzform (ohne run_id-Ballast)."""
    roh = letzte_laeufe.get(aufgabe_id)
    if not isinstance(roh, dict):
        return None
    kurz = str(roh.get("kurz") or "")
    if len(kurz) > _KURZ_MAX_ZEICHEN:
        kurz = kurz[: _KURZ_MAX_ZEICHEN - 1] + "…"
    eintrag: dict[str, Any] = {
        "zeit": roh.get("zeit"),
        "status": roh.get("status"),
        "kurz": kurz,
    }
    if roh.get("hinweis"):
        eintrag["hinweis"] = str(roh["hinweis"])
    return eintrag


async def verwalte_zeitplaene(roh_args: dict[str, Any]) -> dict[str, Any]:
    """Agent-Zeitpläne (agent_cron) auflisten, anlegen, löschen, sofort ausführen."""
    args = pruefe_args(VerwalteZeitplaeneArgs, roh_args, "verwalte_zeitplaene")

    # Lazy-Import (Muster scheduler): scheduler liest agent-Module funktionslokal,
    # das Tool liest den Scheduler funktionslokal — Modul-Ebene bleibt zyklusfrei.
    from quantlab.services import scheduler

    if args.aktion == "auflisten":
        stand = scheduler.agent_cron_einstellungen()
        aufgaben = [
            {
                "id": str(a.get("id") or ""),
                "name": str(a.get("name") or ""),
                "aktiv": bool(a.get("enabled", True)),
                "intervall_min": a.get("intervall_min"),
                "preset": a.get("preset"),
                "modus": str(a.get("modus") or "vorschlag"),
                "nur_handelszeiten": bool(a.get("nur_handelszeiten")),
                "letzter_lauf": _letzter_lauf_kompakt(a.get("id"), stand["letzte_laeufe"]),
            }
            for a in stand["aufgaben"]
        ]
        return {
            "aktion": "auflisten",
            "anzahl": len(aufgaben),
            "aufgaben": aufgaben,
            "hinweis": "Zeitpläne laufen als Agent-Runs im Hintergrund (Scheduler); "
            "letzte Läufe sind auch in den Einstellungen (Agent-Zeitpläne) sichtbar. "
            "Modus-Kette: analyse liest nur, vorschlag empfiehlt als Text, trade "
            "handelt nur mit erteilter Freigabe + Limits.",
        }

    if args.aktion == "anlegen":
        _validiere_anlegen(args)
        name = (args.name or "").strip()
        aufgabe: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "enabled": bool(args.aktiv),
            "intervall_min": int(args.intervall_minuten),
            "preset": args.preset,
            "aufgabe_prompt": (args.aufgaben_text or "").strip(),
            "modus": args.modus,
            "projekt": None,
            "nur_handelszeiten": bool(args.nur_handelszeiten),
            "freigabe_ertelt": bool(args.freigabe_ertelt),
            "kapital_limit_eur": args.kapital_limit_eur or 0,
            "max_orders_pro_lauf": args.max_orders_pro_lauf or 0,
        }
        aufgaben = _agent_cron_aufgaben()
        aufgaben.append(aufgabe)
        _aufgaben_speichern(aufgaben)
        hinweis = (
            "Die Aufgabe greift ohne Neustart — der Scheduler-Tick prüft alle 60 s."
        )
        if args.nur_handelszeiten:
            hinweis += " Ausführung nur im Handelsfenster 08–22 Uhr Europe/Berlin."
        if args.modus == "vorschlag":
            hinweis += (
                " Modus „vorschlag“: der Lauf formuliert Empfehlungen als Text und "
                "platziert keine Orders."
            )
        elif args.modus == "trade":
            hinweis += (
                f" Modus „trade“: Orders nur innerhalb des Limits "
                f"({args.kapital_limit_eur:.2f} € Kapital, max. "
                f"{args.max_orders_pro_lauf} Orders je Lauf)."
            )
        return {
            "aktion": "anlegen",
            "name": name,
            "id": aufgabe["id"],
            "preset": args.preset,
            "modus": args.modus,
            "intervall_min": int(args.intervall_minuten),
            "aktiv": bool(args.aktiv),
            "nur_handelszeiten": bool(args.nur_handelszeiten),
            "hinweis": hinweis,
        }

    if args.aktion == "loeschen":
        if not (args.name or "").strip():
            raise ToolFehler(
                "Aufgabe nicht gelöscht — bitte den Namen der Aufgabe angeben "
                "(oder zuerst aktion=auflisten)."
            )
        aufgabe, mehrfach = _finde_aufgabe(args.name)
        if aufgabe is None:
            if mehrfach:
                namen_ids = ", ".join(
                    f"„{a.get('name')}“ (ID {a.get('id')})" for a in mehrfach
                )
                raise ToolFehler(
                    "Aufgabe nicht gelöscht — der Name ist mehrfach vergeben: "
                    f"{namen_ids}. Bitte über die eindeutige ID (aktion=loeschen, "
                    "name=<ID>) löschen."
                )
            raise ToolFehler(
                f"Keine Zeitplan-Aufgabe mit dem Namen „{args.name}“ gefunden — "
                "bitte zuerst aktion=auflisten ausführen."
            )
        rest = [
            a
            for a in _agent_cron_aufgaben()
            if str(a.get("id") or "") != str(aufgabe.get("id"))
        ]
        _aufgaben_speichern(rest)
        return {
            "aktion": "loeschen",
            "name": str(aufgabe.get("name") or ""),
            "id": str(aufgabe.get("id") or ""),
            "hinweis": "Aufgabe gelöscht — sie feuert nicht mehr; der letzte-Lauf-"
            "Eintrag bleibt in den Einstellungen sichtbar.",
        }

    # aktion == "jetzt_ausfuehren"
    if not (args.name or "").strip():
        raise ToolFehler(
            "Sofort-Ausführung nicht möglich — bitte den Namen der Aufgabe "
            "angeben (oder zuerst aktion=auflisten)."
        )
    aufgabe, mehrfach = _finde_aufgabe(args.name)
    if aufgabe is None:
        if mehrfach:
            namen_ids = ", ".join(
                f"„{a.get('name')}“ (ID {a.get('id')})" for a in mehrfach
            )
            raise ToolFehler(
                "Sofort-Ausführung nicht eindeutig — der Name ist mehrfach "
                f"vergeben: {namen_ids}. Bitte über die eindeutige ID ausführen."
            )
        raise ToolFehler(
            f"Keine Zeitplan-Aufgabe mit dem Namen „{args.name}“ gefunden — "
            "bitte zuerst aktion=auflisten ausführen."
        )
    if not bool(aufgabe.get("enabled", True)):
        raise ToolFehler(
            f"Die Aufgabe „{aufgabe.get('name')}“ ist deaktiviert — bitte sie "
            "erst aktivieren (z. B. neu anlegen mit aktiv=true), bevor sie "
            "sofort ausgeführt wird."
        )
    # Delegation auf den bestehenden run-now-Pfad (E099): erzwingt die Fällig-
    # keit für ALLE aktivierten Aufgaben (Vertrag tick_agent_cron(erzwingen=…));
    # das Ergebnis wird transparent durchgereicht.
    ergebnis = await scheduler.run_now("agent_cron")
    details = dict(ergebnis.get("details") or {})
    return {
        "aktion": "jetzt_ausfuehren",
        "name": str(aufgabe.get("name") or ""),
        "id": str(aufgabe.get("id") or ""),
        "scheduler": {
            "ok": ergebnis.get("ok"),
            "ausgefuehrt": details.get("ausgefuehrt"),
            "abgelehnt": details.get("abgelehnt"),
            "uebersprungen": details.get("uebersprungen"),
            "ausser_handelszeiten": details.get("ausser_handelszeiten"),
        },
        "hinweis": "run-now erzwingt den Lauf aller aktivierten Aufgaben (nicht "
        "nur dieser einen) — Fälligkeits-Intervalle werden ignoriert; das "
        "nur_handelszeiten-Gate (08–22 Uhr Berlin) bleibt bestehen.",
    }


def _registrieren() -> None:
    GLOBAL_REGISTRY.register(
        ToolSpec(
            name="verwalte_zeitplaene",
            description=(
                "Agent-Zeitpläne (Cron-Aufgaben) verwalten: auflisten (inkl. "
                "letztem Lauf), anlegen (Name, Preset oder Aufgabentext, "
                "Intervall ≥ 5 Minuten, Modus analyse/vorschlag/trade, optional "
                "nur Handelszeiten 08–22 Uhr Berlin), löschen oder sofort "
                "ausführen. Modus „trade“ nur bei ausdrücklicher Nutzer-Freigabe "
                "mit Kapital-Limit und Max-Orders; im Wettbewerb ist „vorschlag“ "
                "der Standard. "
                'Beispiel: {"aktion": "anlegen", "name": "Depot-Check alle 30 min", '
                '"aufgaben_text": "Lade die neuen Planspiel-Kurse, prüfe das Depot '
                'und formuliere Handelsvorschläge.", "intervall_minuten": 30, '
                '"modus": "vorschlag", "nur_handelszeiten": true} bzw. '
                '{"aktion": "auflisten"}.'
            ),
            parameters=schema_von(VerwalteZeitplaeneArgs),
            handler=verwalte_zeitplaene,
        )
    )


_registrieren()
