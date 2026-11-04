"""Tests für das Hermes-Skill-Wissen (Welle H, H2) — netzfrei.

Die Regel-Zahlen werden NICHT hart kodiert, sondern aus der Single Source of
Truth ``quantlab.services.planspiel.regeln`` importiert (Gebühren, Limits,
Handelsfenster). Nur rein saisonale Fakten („2026/27“, „25.01.2027“) sind als
String-Asserts fix — Quelle: Task-Konvention H2 / RECHERCHE P.6 (Laufzeit der
Saison 2026/27, erster Handelstag 01.10.2026, Spielende 25.01.2027).
"""

from __future__ import annotations

from quantlab.hermes.skill import HERMES_SKILL
from quantlab.services.planspiel import regeln


def test_skill_ist_nicht_leerer_string() -> None:
    """HERMES_SKILL ist ein str und mit >= 1.500 Zeichen substanziell."""
    assert isinstance(HERMES_SKILL, str)
    assert len(HERMES_SKILL) >= 1_500


def test_skill_identitaet_und_saison() -> None:
    """Identität (Planspiel-Börse, Hermes, Autonomie) und Saison 2026/27."""
    assert "Planspiel-Börse" in HERMES_SKILL
    assert "Hermes" in HERMES_SKILL
    assert "vollautonom" in HERMES_SKILL
    assert "2026/27" in HERMES_SKILL
    assert "25.01.2027" in HERMES_SKILL  # Quelle: Saisonende (RECHERCHE P.6)


def test_skill_regelzahlen_aus_regeln_py() -> None:
    """Kernzahlen stimmen mit den Konstanten aus regeln.py überein."""
    assert f"{regeln.GEBUEHR_PROZENT:.1f}".replace(".", ",") in HERMES_SKILL
    assert f"{regeln.MIN_GEBUEHR_EUR:.0f}" in HERMES_SKILL
    assert f"{regeln.KRYPTO_PROZENT:.2f}".replace(".", ",") in HERMES_SKILL
    assert f"{regeln.MAX_DEPOTGEWICHT * 100:g} %" in HERMES_SKILL
    assert "20 %" in HERMES_SKILL
    assert str(regeln.MAX_ORDERS_JE_TAG) in HERMES_SKILL
    assert f"{regeln.PENNY_GRENZE_EUR:.2f}".replace(".", ",") in HERMES_SKILL
    assert str(regeln.MIN_KAEUFE) in HERMES_SKILL


def test_skill_handelsfenster() -> None:
    """Handelsfenster-Uhrzeiten und Zeitzone aus regeln.py."""
    assert f"{regeln.HANDELSZEIT_START_STUNDE:02d}:00" in HERMES_SKILL
    assert f"{regeln.HANDELSZEIT_ENDE_STUNDE:02d}:00" in HERMES_SKILL
    assert regeln.HANDELSZEIT_ZONE in HERMES_SKILL


def test_skill_pflichtbegriffe() -> None:
    """Kennwörter: Tool-Namen, Guard-Parameter, Mindestaktivität."""
    for begriff in (
        "bestaetigung_wettbewerb",
        "Mindestaktivität",
        "platziere_order",
        "news_suche",
        "run_screener",
        "planspiel_kennzahlen",
        "get_indicators",
        "standard_depot_id",
        "wettbewerbs_depot_id",
    ):
        assert begriff in HERMES_SKILL, f"„{begriff}“ fehlt im Skill"
