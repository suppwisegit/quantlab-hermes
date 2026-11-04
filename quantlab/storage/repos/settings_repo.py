"""Settings-Repository: getypte Einstellungen (Tabelle ``settings``) + API-Key (E010).

**Einstellungen:** In Sektionen organisiert (``DEFAULT_SETTINGS``: darstellung,
ki, allgemein). Gespeichert werden nur explizit gesetzte Overrides — je Sektion
eine Zeile (``key`` = Sektionsname, ``value_json`` = Objekt). Beim Lesen werden
die Defaults schlüsselweise zugemischt; die Antwort ist immer vollständig und
über die JSON-Typen getypt (Roundtrip int/float/str/list bleibt stabil).
Eine **leere** Liste auf einem Listen-Default gilt dabei als „nicht gesetzt“
und fällt auf den Default zurück (V5/F-c: Alt-Bestände mit gespeicherter
leerer ``markets.indizes``-Liste zeigten sonst einen Leerzustand trotz
Default-Indizes; eigene nicht-leere Listen bleiben exakt erhalten).

**API-Key (z.ai, E010):** Gespeichert wird primär im OS-Keyring (``keyring``,
Service ``QuantLab``, Account ``z_ai_api_key``). Schlägt das fehl (kein
Secret-Service/Daemon, Modul fehlt) oder kein Keyring ist erreichbar, dient die
Fallback-Datei ``data/secrets.json`` mit chmod 0600. Der volle Key wird nur
backend-intern gelesen (:func:`api_key_lesen`, LLM-Client in T12); über die API
geht ausschließlich eine Maske (``••••abc``). Beim Löschen werden beide Quellen
geräumt. Das ``keyring``-Modul wird lazy geladen, damit beide Pfade deterministisch
testbar sind (Monkeypatch auf das keyring-Modul bzw. :func:`_load_keyring`).

**Planspiel-Passwort (Welle P, E071):** gleiche Speicher-Muster, aber mit
EIGENEM Keyring-Account (``planspiel_password``) und EIGENER Fallback-Datei
``data/secrets_planspiel.json`` (0600) — bewusst **nicht** ``secrets.json``
geteilt, weil :func:`_fallback_delete` die Datei vollständig unlinkt und ein
geteilter Pfad beim Löschen des z.ai-Keys das Planspiel-Passwort kollateral
mitlöschen würde.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from quantlab import db
from quantlab.config import get_data_dir, get_secrets_path

KEYRING_SERVICE = "QuantLab"
KEYRING_ACCOUNT = "z_ai_api_key"

#: Welle P (E071): eigener Keyring-Account + eigene Fallback-Datei für das
#: Planspiel-Passwort — siehe Kollateral-Befund im Modulkopf.
PLANSPIEL_KEYRING_ACCOUNT = "planspiel_password"
PLANSPIEL_FALLBACK_DATEINAME = "secrets_planspiel.json"

_QUELLE_KEYRING = "keyring"
_QUELLE_DATEI = "datei"
_MASK_PREFIX = "••••"

#: Defaults je Sektion (T15-Sichten: Darstellung, KI, Allgemein). Neue Keys
#: kommen hier hinein — gespeicherte Overrides mit unbekannten Keys werden
#: beim Lesen ignoriert (robust gegen Schema-Drift).
DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "darstellung": {
        "akzentfarbe": "#d97757",  # entspricht --color-accent (T1-Tokens)
        "font_sans": "Inter",
        "font_mono": "JetBrains Mono",
        "schriftgroesse": 14,
        "effekt_intensitaet": 0.6,
        # Welle F (T-F3): Dark (Default) | Light | System (OS-Präferenz) —
        # der Renderer setzt den aufgelösten Wert als [data-theme] am Root
        # (Light-Palette: tokens.css, RECHERCHE F.8).
        "theme": "dark",
    },
    "ki": {
        "modelle": ["glm-5.3", "glm-5.3-flash", "glm-4.6", "glm-4.5-air"],
        "standard_modell": "glm-5.3-flash",
        "standard_thinking_type": "enabled",  # enabled | disabled
        "standard_reasoning_effort": "high",  # low | high | max (§F)
        "max_tokens": 8192,
        "max_tool_iterationen": 25,
        "historien_tiefe": 50,
    },
    "allgemein": {
        "sprache": "de",
        "periods_per_year": 252,  # Quant-Konvention §F/E007
    },
    # T21/E035: Scheduler-Intervalle/-Flags („Scheduler & Live“). Additive
    # Defaults — gespeicherte Overrides anderer Sektionen bleiben unberührt.
    "scheduler": {
        "data_update_enabled": True,  # Auto-Update der Watchlist-Kurse
        "data_update_interval_h": 6.0,  # E035-Default: alle 6 h
        "fundamentals_enabled": True,  # Fundamentals-Refresh im selben Gerüst
        "fundamentals_refresh_days": 7.0,  # E035-Default: alle 7 Tage je Symbol
        "alerts_check_s": 60,  # E035-Default: Tick-Rhythmus für T33
    },
    # T22/E024: Live-Quotes-Poll (eigene Sektion „live“ gemäß E024) —
    # quotes_poll_s 15–300 s (außerhalb → 422, Router-Validierung).
    # Additiv; gespeicherte Overrides anderer Sektionen bleiben unberührt.
    "live": {
        "quotes_enabled": True,  # E024-Default: Quotes-Poll aktiv
        "quotes_poll_s": 60,  # E024: 15–300 s, Default 60 (= TTL-Default)
    },
    # T24 (Chart v2): Renderer-definierte Chart-Konfiguration ({symbol,
    # interval, compareSymbols, chartType, logScale, rangePreset, adjust}) —
    # das Backend nimmt sie **opak** als JSON entgegen (Struktur gehört dem
    # Renderer), deshalb leeres Default-Objekt + :data:`OPAQUE_SEKTIONEN`.
    "charts": {},
    # T28 (Märkte): konfigurierbare Index-Liste der Übersichts-Karten
    # (``GET /api/markets/overview``, §C.4 „overview über konfigurierte
    # Index-Liste“). Getypter Default (keine opake Sektion nötig — der
    # Schlüssel ``indizes`` ist fest); der Markets-Router liest die Sektion
    # je Aufruf und filtert defekte Einträge defensiv heraus. Additiv —
    # gespeicherte Overrides anderer Sektionen bleiben unberührt.
    "markets": {
        "indizes": ["^GSPC", "^DJI", "^IXIC", "^GDAXI", "^FTSE", "^N225"],
    },
    # Welle P (E070): Planspiel-Börse — Zugangs-E-Mail (das Passwort liegt NIE
    # hier, sondern im Keyring/Fallback, s. ``planspiel_passwort_*``), einmalige
    # Depot-Rollen-Markierung (0 = nicht markiert; Fail-safe Schreib-Sperre in
    # P2b) und Poll-Flags für P6 (Depotwert-Polling, Default aus).
    "planspiel": {
        "email": "",
        "standard_depot_id": 0,
        "wettbewerbs_depot_id": 0,
        "depotwert_poll_aktiv": False,
        "depotwert_poll_min_s": 120,
        # Welle H (E096/H4): PSB-Live-Recorder — bewusst AUS (Default false,
        # der Nutzer aktiviert ihn); Intervall 30–3600 s (Validierung deutsch:
        # services/planspiel/recorder.py, defensives Klemmen im Tick).
        "recorder_aktiv": False,
        "recorder_intervall_s": 60,
    },
    # Welle F (T-F1/E080): Agent-Cron — Aufgabenliste + letzter Lauf je Aufgabe.
    # Aufgaben-Dicts: {id, name, enabled, intervall_min, preset, aufgabe_prompt,
    # modus, projekt, nur_handelszeiten, freigabe_ertelt, kapital_limit_eur,
    # max_orders_pro_lauf} (feine Validierung: routers/settings._regel_fehler);
    # letzte_laeufe: aufgabe-id → {zeit, status, kurz, run_id, hinweis?} — der
    # Scheduler-Tick schreibt sie unter EIGENEM Key (Merge), die Aufgabenliste
    # bleibt unberührt. Additiv; Overrides anderer Sektionen bleiben unberührt.
    "agent_cron": {
        "aufgaben": [],
        "letzte_laeufe": {},
    },
    # Welle F (T-F6/E083): Websuche des Agenten (web_suche/web_lesen) —
    # zentral abschaltbar; beide Tools prüfen den Schalter je Aufruf
    # (deaktiviert → deutscher Fachfehler, kein Netz-Call). Additiv.
    "websuche": {
        "enabled": True,
    },
    # Welle G (T-G2/E088): Zweite Tagesdaten-Quelle leeway.tech. Der Key wird
    # hier als Einstellung geführt (leerer String = nicht konfiguriert →
    # das Bein wird im Provider-Verbund gar nicht gebaut); er wird nie im
    # Klartext geloggt. Additiv; Overrides anderer Sektionen bleiben unberührt.
    "daten": {
        "leeway_api_key": "",
    },
}

#: Sektionen ohne Backend-Schema: Keys/Werte werden ungefiltert (opak)
#: gespeichert und zurückgegeben — die Struktur definiert der Renderer (T24).
#: Alle anderen Sektionen behalten das schlüsselweise Whitelisting gegen
#: ihre Defaults (robust gegen Schema-Drift, s. ``read_settings``).
OPAQUE_SEKTIONEN: frozenset[str] = frozenset({"charts"})


def _sektion_gemischt(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Defaults + bekannte Overrides zugemischt (nicht-opake Sektionen).

    V5/F-c (Alt-Bestand): Eine **leere** Liste auf einem Listen-Default gilt
    als „nicht gesetzt“ und fällt auf den Default zurück — Alt-Bestände, die
    eine leere ``markets.indizes``-Liste gespeichert hatten, ließen die
    Märkte-Übersicht sonst trotz Default-Indizes leer erscheinen. Eigene
    **nicht-leere** Listen bleiben exakt erhalten; andere Werte (inkl. leerer
    Strings/Objekte) werden unverändert übernommen.
    """
    bekannt: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in defaults:
            continue
        if value == [] and isinstance(defaults[key], list):
            continue  # leere Liste bei Listen-Default = „nicht gesetzt“ (V5/F-c)
        bekannt[key] = value
    return {**defaults, **bekannt}


@dataclass(frozen=True)
class ApiKeyInfo:
    """Maskierter API-Key-Status — die API-Form des Keys (E010)."""

    konfiguriert: bool
    maske: str | None
    quelle: str | None  # "keyring" | "datei" | None


# ---------------------------------------------------------------------------
# Einstellungen (getypt, Defaults + Overrides)
# ---------------------------------------------------------------------------


def read_settings() -> dict[str, dict[str, Any]]:
    """Alle Sektionen: Defaults + gespeicherte Overrides zugemischt (getypt)."""
    conn = db.connect()
    try:
        rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
    finally:
        conn.close()
    stored = {str(row["key"]): json.loads(row["value_json"]) for row in rows}
    result: dict[str, dict[str, Any]] = {}
    for section, defaults in DEFAULT_SETTINGS.items():
        overrides = stored.get(section)
        if not isinstance(overrides, dict):
            overrides = {}
        if section in OPAQUE_SEKTIONEN:
            # Opake Sektion (T24): Renderer-Keys werden nicht gegen Defaults
            # gefiltert — die Struktur gehört dem Aufrufer, nicht dem Backend.
            result[section] = {**defaults, **overrides}
            continue
        result[section] = _sektion_gemischt(defaults, overrides)
    return result


def get_section(section: str) -> dict[str, Any]:
    """Eine Sektion inkl. Defaults (``ValueError`` bei unbekannter Sektion)."""
    if section not in DEFAULT_SETTINGS:
        raise ValueError(f"Unbekannte Einstellungs-Sektion: {section}")
    return read_settings()[section]


def update_section(section: str, values: dict[str, Any]) -> dict[str, Any]:
    """Merkt Overrides schlüsselweise (Merge) und liefert die volle Sektion zurück.

    Grobe Verteidigung auf Repo-Ebene: unbekannte Sektion → ``ValueError``,
    unbekannte Keys werden gefiltert. Die feine Validierung (bekannte Keys,
    Werttypen → HTTP 400) macht der Router vor dem Aufruf.
    """
    defaults = DEFAULT_SETTINGS.get(section)
    if defaults is None:
        raise ValueError(f"Unbekannte Einstellungs-Sektion: {section}")
    opaque = section in OPAQUE_SEKTIONEN
    if opaque:
        overrides = dict(values)  # opak (T24): keine Key-Filterung
    else:
        overrides = {key: value for key, value in values.items() if key in defaults}

    conn = db.connect()
    try:
        # BEGIN IMMEDIATE (Muster: quantlab.db): Read-modify-write serialisiert
        # sauber unter Konkurrenz — deferred BEGIN würde beim Upgrade zum
        # Schreib-Lock SQLITE_BUSY_SNAPSHOT werfen (ungefangen → HTTP 500).
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value_json FROM settings WHERE key = ?", (section,)
        ).fetchone()
        try:
            stored: dict[str, Any] = json.loads(row["value_json"]) if row else {}
        except json.JSONDecodeError:
            stored = {}  # defekte Zeile → Overrides neu aufbauen
        if not isinstance(stored, dict):
            stored = {}
        stored.update(overrides)
        conn.execute(
            "INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value_json = excluded.value_json, updated_at = excluded.updated_at",
            (section, json.dumps(stored, ensure_ascii=False, sort_keys=True), db.utc_now_iso()),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:  # kein blindes ROLLBACK — BEGIN könnte selbst geworfen haben
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    if opaque:
        return {**defaults, **stored}  # opak: alles Gespeicherte zurückgeben
    return _sektion_gemischt(defaults, stored)  # gleiche Merge-Regel wie read_settings


# ---------------------------------------------------------------------------
# API-Key (E010): keyring primär, Fallback-Datei 0600, API sieht nur Masken
# ---------------------------------------------------------------------------


def _load_keyring() -> ModuleType | None:
    """Lädt ``keyring`` lazy — ``None``, wenn das Modul fehlt/nicht importierbar ist.

    Lazy statt Top-Level-Import: Der Fallback-Pfad (kein keyring) bleibt auch in
    Testumgebungen deterministisch erreichbar.
    """
    try:
        import keyring
    except Exception:  # noqa: BLE001 — fehlendes/broken keyring ist ein erwarteter Zweig
        return None
    return keyring


def _keyring_save(key: str, *, account: str = KEYRING_ACCOUNT) -> None:
    """Speichert im OS-Keyring; wirft bei jeder Störung (kein Daemon, …)."""
    module = _load_keyring()
    if module is None:
        raise RuntimeError("keyring nicht verfügbar")
    module.set_password(KEYRING_SERVICE, account, key)


def _keyring_load(*, account: str = KEYRING_ACCOUNT) -> str | None:
    """Voller Key aus dem OS-Keyring — ``None`` bei Fehler oder wenn nicht gesetzt."""
    module = _load_keyring()
    if module is None:
        return None
    try:
        return module.get_password(KEYRING_SERVICE, account)
    except Exception:  # noqa: BLE001 — Backends werfen heterogene Fehlertypen
        return None


def _keyring_delete(*, account: str = KEYRING_ACCOUNT) -> None:
    """Entfernt den Key aus dem OS-Keyring (Fehler werden bewusst ignoriert)."""
    module = _load_keyring()
    if module is None:
        return
    try:
        module.delete_password(KEYRING_SERVICE, account)
    except Exception:  # noqa: BLE001 — Key war evtl. nie im Keyring
        return


def _fallback_save(
    key: str, *, account: str = KEYRING_ACCOUNT, path: Path | None = None
) -> None:
    """Schreibt die Fallback-Datei — Rechte 0600 bereits ab Erstellung (E010)."""
    ziel = path if path is not None else get_secrets_path()
    ziel.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps({account: key}, ensure_ascii=False, indent=2)
    # os.open mit Modus 0600: die Datei ist zu keinem Zeitpunkt weiter lesbar;
    # das nachgelagerte chmod stellt die Zusicherung unabhängig von der umask sicher.
    fd = os.open(ziel, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(content)
    os.chmod(ziel, 0o600)


def _fallback_load(*, account: str = KEYRING_ACCOUNT, path: Path | None = None) -> str | None:
    ziel = path if path is not None else get_secrets_path()
    try:
        raw = ziel.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None  # defekte Datei → wie „nicht konfiguriert" behandeln
    value = data.get(account) if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else None


def _fallback_delete(*, path: Path | None = None) -> None:
    """Unlinkt die Fallback-Datei (auch mit Fremd-Inhalt — deshalb E071: eigener Pfad!)."""
    ziel = path if path is not None else get_secrets_path()
    try:
        ziel.unlink()
    except FileNotFoundError:
        pass


def mask_key(key: str) -> str:
    """Maske im Stil ``••••abc`` (nur die letzten 3 Zeichen, E010)."""
    return f"{_MASK_PREFIX}{key[-3:]}" if len(key) >= 6 else _MASK_PREFIX


def api_key_read() -> str | None:
    """Voller Key (keyring vor Fallback) — ausschließlich backend-intern nutzen (E010)."""
    key = _keyring_load()
    if key:
        return key
    return _fallback_load()


def api_key_status() -> ApiKeyInfo:
    """Maskierter Status: konfiguriert? Maske? Quelle (keyring|datei)?"""
    key = _keyring_load()
    if key:
        return ApiKeyInfo(konfiguriert=True, maske=mask_key(key), quelle=_QUELLE_KEYRING)
    key = _fallback_load()
    if key:
        return ApiKeyInfo(konfiguriert=True, maske=mask_key(key), quelle=_QUELLE_DATEI)
    return ApiKeyInfo(konfiguriert=False, maske=None, quelle=None)


def api_key_save(key: str) -> ApiKeyInfo:
    """Speichert den Key (keyring primär; bei Fehlschlag Fallback-Datei 0600).

    Gelingt der Keyring-Weg, wird eine evtl. noch vorhandene Fallback-Datei
    entfernt (Migration älterer Bestände auf den Keyring). Dieses Aufräumen
    läuft in eigenem try/except — ein pathologischer Fehler dort (z. B.
    Dateisystem-Störung) darf den erfolgreichen Keyring-Save nicht kippen.
    """
    try:
        _keyring_save(key)
        try:
            _fallback_delete()
        except OSError:
            pass  # Keyring-Save ist bereits erfolgt; Restbestand schadet nicht
    except Exception:  # noqa: BLE001 — keyring-Fehler (R8) → Fallback-Pfad
        _fallback_save(key)
    return api_key_status()


def api_key_delete() -> ApiKeyInfo:
    """Entfernt den Key aus beiden Quellen (keyring-Fehler werden ignoriert)."""
    _keyring_delete()
    _fallback_delete()
    return api_key_status()


# ---------------------------------------------------------------------------
# Planspiel-Passwort (Welle P, E071): eigener Account + eigene Fallback-Datei
# ---------------------------------------------------------------------------


def _planspiel_fallback_pfad() -> Path:
    """Pfad der EIGENEN Planspiel-Fallback-Datei (E071) — nie ``secrets.json``.

    Hintergrund (Code-Befund ARCH): :func:`_fallback_delete` unlinkt die
    **gesamte** Datei — ein geteilter Pfad würde beim Löschen des z.ai-Keys
    das Planspiel-Passwort kollateral mitlöschen (und umgekehrt).
    """
    return get_data_dir() / PLANSPIEL_FALLBACK_DATEINAME


def planspiel_passwort_lesen() -> str | None:
    """Volles Planspiel-Passwort (keyring vor Fallback) — ausschließlich backend-intern (E071)."""
    passwort = _keyring_load(account=PLANSPIEL_KEYRING_ACCOUNT)
    if passwort:
        return passwort
    return _fallback_load(account=PLANSPIEL_KEYRING_ACCOUNT, path=_planspiel_fallback_pfad())


def planspiel_passwort_status() -> ApiKeyInfo:
    """Maskierter Status ({konfiguriert, maske, quelle}) — die API sieht nie Klartext."""
    passwort = _keyring_load(account=PLANSPIEL_KEYRING_ACCOUNT)
    if passwort:
        return ApiKeyInfo(konfiguriert=True, maske=mask_key(passwort), quelle=_QUELLE_KEYRING)
    passwort = _fallback_load(account=PLANSPIEL_KEYRING_ACCOUNT, path=_planspiel_fallback_pfad())
    if passwort:
        return ApiKeyInfo(konfiguriert=True, maske=mask_key(passwort), quelle=_QUELLE_DATEI)
    return ApiKeyInfo(konfiguriert=False, maske=None, quelle=None)


def planspiel_passwort_save(passwort: str) -> ApiKeyInfo:
    """Speichert das Planspiel-Passwort (keyring primär; bei Fehlschlag eigene Fallback-Datei 0600).

    Spiegelt :func:`api_key_save` — das Aufräumen einer veralteten Fallback-Datei
    nach erfolgreichem Keyring-Save berührt **nur** die Planspiel-Datei.
    """
    try:
        _keyring_save(passwort, account=PLANSPIEL_KEYRING_ACCOUNT)
        try:
            _fallback_delete(path=_planspiel_fallback_pfad())
        except OSError:
            pass  # Keyring-Save ist bereits erfolgt; Restbestand schadet nicht
    except Exception:  # noqa: BLE001 — keyring-Fehler (R8) → eigener Fallback-Pfad
        _fallback_save(
            passwort, account=PLANSPIEL_KEYRING_ACCOUNT, path=_planspiel_fallback_pfad()
        )
    return planspiel_passwort_status()


def planspiel_passwort_loeschen() -> ApiKeyInfo:
    """Räumt NUR das Planspiel-Passwort (beide Quellen) — z.ai-Key bleibt unberührt (E071)."""
    _keyring_delete(account=PLANSPIEL_KEYRING_ACCOUNT)
    _fallback_delete(path=_planspiel_fallback_pfad())
    return planspiel_passwort_status()
