"""Settings-Endpunkte (C.1): GET/PUT ``/api/settings`` + API-Key-Management (E010)
+ Daten-Management (T15: ``data-usage``/``data/purge``).

Die Endpunkte sind bewusst normale ``def`` (nicht ``async``): die SQLite-Aufrufe
der Repos laufen synchron und werden von FastAPI im Threadpool ausgeführt —
der Event-Loop blockiert nicht (Muster: ``quantlab.db``).

Der API-Key verlässt das Backend nie im Klartext: Status-Endpunkte liefern
ausschließlich Maske + Quelle (E010).
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantlab import events
from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.providers.symbols import validate_symbol
from quantlab.storage import parquet_store
from quantlab.storage.repos import settings_repo, symbols_repo, watchlists_repo

router = APIRouter(tags=["settings"])

# z.ai-Keys sind deutlich länger; die Grenze fängt Tippfehler früh ab (T2-Umfang).
_MIN_API_KEY_LENGTH = 20

# Geschlossene Wertebereiche für Zahlen-Keys (T15#3): Key → (min, max).
_WERTEBEREICHE: dict[str, tuple[int | float, int | float]] = {
    "schriftgroesse": (12, 18),
    "effekt_intensitaet": (0.0, 1.0),
    "max_tokens": (256, 1_000_000),
    "max_tool_iterationen": (1, 100),
    "historien_tiefe": (0, 500),
}

# T22/E024: Wertebereiche mit HTTP **422** (DoD-Vertrag „quotes_poll_s
# außerhalb 15–300 → 422 beim Settings-Put“) — bewusst abweichend vom
# Haus-Status 400 der übrigen Bereichsregeln (_WERTEBEREICHE), weil der Wert
# direkt den Sidecar-Poll-Takt steuert (unmittelbare Verhaltens-Auswirkung).
_BEREICHE_422: dict[str, tuple[int | float, int | float]] = {
    "quotes_poll_s": (15, 300),
    # H8-Politur (REV-H-2-Backlog): Recorder-Intervall — Klemmen 30–3600
    # existieren zusätzlich im Tick (recorder.einstellungen); hier gilt die
    # harte REST-Grenze wie bei quotes_poll_s (Netzfairness, Pacing ≥ 1 s).
    "recorder_intervall_s": (30, 3600),
}

# Modellkatalog-Limits (T15#3).
_MAX_MODELLE = 32
_MAX_MODELL_LAENGE = 64

# Welle G (T-G2/E088): Zweite Tagesdaten-Quelle leeway.tech — der Key ist ein
# freier String (leerer String = nicht konfiguriert); die Grenze fängt nur
# offensichtlichen Müll ab, der Rest prüft die Quelle beim ersten Abruf.
_MAX_LEEWAY_KEY_LAENGE = 512

_AKZENTFARBE_MUSTER = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# T-F3: Erlaubte Theme-Werte („dark“ = Default, „light“, „system“ = OS-Präferenz,
# Auflösung passiert renderer-seitig via prefers-color-scheme).
_THEME_WERTE: frozenset[str] = frozenset({"dark", "light", "system"})

# Welle F (T-F1/E080): Agent-Cron-Aufgaben (Sektion „agent_cron“, Listen-Key
# „aufgaben“). Presets/Modi sind der Vertrag mit dem Renderer (AgentCronSection);
# der Scheduler (services/scheduler.py) hält die Preset-Tasktexte als Single Source.
_AGENT_CRON_PRESETS: frozenset[str] = frozenset(
    {"depot_check", "markt_wachposten", "rebalance_vorschlag", "custom"}
)
_AGENT_CRON_MODI: frozenset[str] = frozenset({"analyse", "vorschlag", "trade"})
_AGENT_CRON_MIN_INTERVALL_MIN = 5
_AGENT_CRON_PFLICHTFELDER: tuple[str, ...] = ("id", "name", "intervall_min", "preset")
_AGENT_CRON_OPTIONALE_FELDER: tuple[str, ...] = (
    "enabled",
    "aufgabe_prompt",
    "modus",
    "projekt",
    "nur_handelszeiten",
    "freigabe_ertelt",
    "kapital_limit_eur",
    "max_orders_pro_lauf",
)


class ApiKeyStatusResponse(BaseModel):
    """Maskierter API-Key-Status — nie der volle Key (E010)."""

    konfiguriert: bool
    maske: str | None
    quelle: str | None


def _type_matches(value: Any, default: Any) -> bool:
    """Grobe Typprüfung gegen den Default-Typ (int akzeptiert kein bool etc.)."""
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, int | float) and not isinstance(value, bool)
    if isinstance(default, list):
        return isinstance(value, list)
    if isinstance(default, str):
        return isinstance(value, str)
    return type(value) is type(default)


def _zahl(wert: int | float) -> str:
    """Grenzwert ohne überflüssige Nachkommastellen („14“, „0.6“) für Meldungen."""
    return str(int(wert)) if float(wert).is_integer() else str(wert)


def _regel_fehler(key: str, value: Any) -> str | None:
    """Bereichs-/Elementregeln (T15#3): deutsche Meldung oder ``None`` bei Gültigkeit.

    Läuft nach :func:`_type_matches` — Zahlen-/Listenwerte liegen hier bereits
    im erwarteten Typ vor.
    """
    if key in _WERTEBEREICHE:
        minimum, maximum = _WERTEBEREICHE[key]
        if not minimum <= value <= maximum:
            return (
                f"Ungültiger Wert für „{key}“: außerhalb des erlaubten Bereichs "
                f"{_zahl(minimum)}–{_zahl(maximum)}."
            )
        return None
    if key == "modelle":
        if len(value) > _MAX_MODELLE:
            return (
                f"Ungültiger Wert für „modelle“: höchstens {_MAX_MODELLE} Einträge "
                f"(aktuell {len(value)})."
            )
        for eintrag in value:
            if not isinstance(eintrag, str) or not eintrag.strip():
                return (
                    "Ungültiger Wert für „modelle“: Einträge müssen nicht-leere "
                    "Zeichenketten sein."
                )
            if len(eintrag) > _MAX_MODELL_LAENGE:
                return (
                    "Ungültiger Wert für „modelle“: ein Eintrag ist länger als "
                    f"{_MAX_MODELL_LAENGE} Zeichen."
                )
        return None
    if key == "akzentfarbe" and _AKZENTFARBE_MUSTER.match(value) is None:
        return (
            "Ungültiger Wert für „akzentfarbe“: Hex-Farbe im Format „#rgb“ oder "
            "„#rrggbb“ erwartet."
        )
    if key == "theme" and value not in _THEME_WERTE:
        return "Ungültiger Wert für „theme“: „dark“, „light“ oder „system“ erwartet."
    if key == "leeway_api_key":
        if value != value.strip():
            return (
                "Ungültiger Wert für „leeway_api_key“: der Key darf kein "
                "führendes/abschließendes Leerzeichen haben."
            )
        if len(value) > _MAX_LEEWAY_KEY_LAENGE:
            return (
                "Ungültiger Wert für „leeway_api_key“: der Key ist länger als "
                f"{_MAX_LEEWAY_KEY_LAENGE} Zeichen."
            )
        return None
    if key == "aufgaben":
        return _agent_cron_aufgaben_fehler(value)
    return None


def _agent_cron_aufgaben_fehler(aufgaben: list[Any]) -> str | None:
    """Elementregeln der Agent-Cron-Aufgabenliste (T-F1/E080, Muster „modelle“).

    Je Aufgabe: Pflichtfelder, unbekannte Felder, Typen, ``intervall_min >= 5``,
    Preset-/Modus-Enum, Prompt-Pflicht bei ``preset=custom`` und konsistente
    Trade-Felder (``modus=trade`` erfordert Freigabe + Kapital-Limit + Max-Orders;
    die Freigabe außerhalb von ``trade`` wird abgelehnt). Deutsche Meldung oder
    ``None`` bei Gültigkeit.
    """
    erlaubt = frozenset(_AGENT_CRON_PFLICHTFELDER) | frozenset(_AGENT_CRON_OPTIONALE_FELDER)
    ids: set[str] = set()
    for position, aufgabe in enumerate(aufgaben, start=1):
        kopf = f"Agent-Cron-Aufgabe {position}"
        if not isinstance(aufgabe, dict):
            return f"Ungültiger Wert für „aufgaben“: „{kopf}“ muss ein Objekt sein."
        for feld in _AGENT_CRON_PFLICHTFELDER:
            if feld not in aufgabe:
                return f"Ungültiger Wert für „aufgaben“: „{kopf}“ — Pflichtfeld „{feld}“ fehlt."
        fremd = sorted(set(aufgabe) - erlaubt)
        if fremd:
            return (
                f"Ungültiger Wert für „aufgaben“: „{kopf}“ kennt die Felder "
                f"{', '.join(fremd)} nicht."
            )
        for feld in ("id", "name"):
            wert = aufgabe[feld]
            if not isinstance(wert, str) or not wert.strip():
                return (
                    f"Ungültiger Wert für „aufgaben“: „{kopf}“ — „{feld}“ muss eine "
                    "nicht-leere Zeichenkette sein."
                )
        if aufgabe["id"] in ids:
            return (
                f"Ungültiger Wert für „aufgaben“: die ID „{aufgabe['id']}“ ist "
                "mehrfach vergeben."
            )
        ids.add(aufgabe["id"])
        intervall = aufgabe["intervall_min"]
        if (
            not isinstance(intervall, int | float)
            or isinstance(intervall, bool)
            or intervall < _AGENT_CRON_MIN_INTERVALL_MIN
        ):
            return (
                f"Ungültiger Wert für „aufgaben“: „{kopf}“ — „intervall_min“ muss eine "
                f"Zahl ≥ {_AGENT_CRON_MIN_INTERVALL_MIN} sein."
            )
        if aufgabe["preset"] not in _AGENT_CRON_PRESETS:
            return (
                f"Ungültiger Wert für „aufgaben“: „{kopf}“ — unbekanntes Preset "
                f"„{aufgabe['preset']}“ (erlaubt: {', '.join(sorted(_AGENT_CRON_PRESETS))})."
            )
        for feld in ("enabled", "nur_handelszeiten", "freigabe_ertelt"):
            if feld in aufgabe and not isinstance(aufgabe[feld], bool):
                return (
                    f"Ungültiger Wert für „aufgaben“: „{kopf}“ — „{feld}“ muss ein "
                    "Boolean sein."
                )
        modus = aufgabe.get("modus", "vorschlag")
        if modus not in _AGENT_CRON_MODI:
            return (
                f"Ungültiger Wert für „aufgaben“: „{kopf}“ — Modus „{modus}“ ist unbekannt "
                f"(erlaubt: {', '.join(sorted(_AGENT_CRON_MODI))})."
            )
        projekt = aufgabe.get("projekt")
        if projekt is not None and (isinstance(projekt, bool) or not isinstance(projekt, int)):
            return (
                f"Ungültiger Wert für „aufgaben“: „{kopf}“ — „projekt“ muss eine "
                "Projekt-ID (Zahl) oder null sein."
            )
        if aufgabe["preset"] == "custom":
            prompt = aufgabe.get("aufgabe_prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return (
                    f"Ungültiger Wert für „aufgaben“: „{kopf}“ — Preset „custom“ "
                    "erfordert einen nicht-leeren „aufgabe_prompt“."
                )
        freigabe = aufgabe.get("freigabe_ertelt", False)
        if modus == "trade":
            if not freigabe:
                return (
                    f"Ungültiger Wert für „aufgaben“: „{kopf}“ — Modus „trade“ erfordert "
                    "„freigabe_ertelt: true“."
                )
            limit = aufgabe.get("kapital_limit_eur")
            if (
                not isinstance(limit, int | float)
                or isinstance(limit, bool)
                or limit <= 0
            ):
                return (
                    f"Ungültiger Wert für „aufgaben“: „{kopf}“ — Modus „trade“ erfordert "
                    "ein positives „kapital_limit_eur“."
                )
            max_orders = aufgabe.get("max_orders_pro_lauf")
            if (
                not isinstance(max_orders, int)
                or isinstance(max_orders, bool)
                or max_orders < 1
            ):
                return (
                    f"Ungültiger Wert für „aufgaben“: „{kopf}“ — Modus „trade“ erfordert "
                    "„max_orders_pro_lauf“ ≥ 1."
                )
        elif freigabe:
            return (
                f"Ungültiger Wert für „aufgaben“: „{kopf}“ — „freigabe_ertelt“ ist nur "
                "mit Modus „trade“ zulässig."
            )
    return None


def _standard_modell_fehler(values: dict[str, Any]) -> str | None:
    """„standard_modell“ muss im effektiven Modellkatalog liegen (T15#3).

    Effektiv heißt: Overrides aus dem Payload, sonst der gespeicherte Stand —
    so kann der Katalog nicht ohne gültiges Standard-Modell zurückbleiben.
    """
    gespeichert = settings_repo.read_settings().get("ki", {})
    modelle = values.get("modelle")
    if not isinstance(modelle, list):
        modelle = gespeichert.get("modelle")
    standard = values.get("standard_modell")
    if not isinstance(standard, str):
        standard = gespeichert.get("standard_modell")
    if isinstance(modelle, list) and isinstance(standard, str) and standard not in modelle:
        return (
            f"Ungültiger Wert für „standard_modell“: „{standard}“ ist nicht im "
            "Modellkatalog enthalten."
        )
    return None


def _validate_settings_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Prüft Sektionen/Keys, Werttypen und Bereichs-/Elementregeln (→ 400) und
    normalisiert Werte.

    Entscheidung: JSON-Ints werden bei float-Defaults zu float normalisiert
    (``effekt_intensitaet: 1`` → ``1.0``) — verhindert Typdrift im Speicher.
    """
    normalized: dict[str, dict[str, Any]] = {}
    for section, values in payload.items():
        defaults = settings_repo.DEFAULT_SETTINGS.get(section)
        if defaults is None:
            raise ApiError(
                400,
                "ungueltige_einstellungen",
                f"Unbekannte Einstellungs-Sektion: „{section}“.",
                hint=f"Bekannte Sektionen: {', '.join(sorted(settings_repo.DEFAULT_SETTINGS))}.",
            )
        if not isinstance(values, dict):
            raise ApiError(
                400,
                "ungueltige_einstellungen",
                f"Sektion „{section}“ muss ein Objekt mit Schlüssel-Wert-Paaren sein.",
            )
        if section in settings_repo.OPAQUE_SEKTIONEN:
            # Opake Sektion (T24, Chart-Konfiguration): Struktur definiert der
            # Renderer — Keys/Werte werden ungefiltert als JSON durchgereicht
            # (Sektion selbst ist oben gegen DEFAULT_SETTINGS geprüft).
            normalized[section] = dict(values)
            continue
        valid: dict[str, Any] = {}
        for key, value in values.items():
            if key not in defaults:
                raise ApiError(
                    400,
                    "ungueltige_einstellungen",
                    f"Unbekannter Schlüssel „{key}“ in Sektion „{section}“.",
                    hint=f"Bekannte Schlüssel: {', '.join(sorted(defaults))}.",
                )
            if not _type_matches(value, defaults[key]):
                raise ApiError(
                    400,
                    "ungueltige_einstellungen",
                    f"Ungültiger Wert für „{section}.{key}“: Typ passt nicht "
                    f"(erwartet: {type(defaults[key]).__name__}).",
                )
            if key in _BEREICHE_422:
                minimum, maximum = _BEREICHE_422[key]
                if not minimum <= value <= maximum:
                    raise ApiError(
                        422,
                        "ungueltige_einstellungen",
                        f"Ungültiger Wert für „{key}“: außerhalb des erlaubten "
                        f"Bereichs {_zahl(minimum)}–{_zahl(maximum)}.",
                    )
            regel = _regel_fehler(key, value)
            if regel is not None:
                raise ApiError(400, "ungueltige_einstellungen", regel)
            valid[key] = float(value) if isinstance(defaults[key], float) else value
        if section == "ki" and ("modelle" in values or "standard_modell" in values):
            fehler = _standard_modell_fehler(values)
            if fehler is not None:
                raise ApiError(400, "ungueltige_einstellungen", fehler)
        normalized[section] = valid
    return normalized


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    """Alle Einstellungs-Sektionen inkl. Defaults (darstellung, ki, allgemein)."""
    return settings_repo.read_settings()


@router.put("/settings")
def put_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Aktualisiert Sektionen schlüsselweise (Merge, Teil-Updates erlaubt)."""
    sections = _validate_settings_payload(payload)
    for section, values in sections.items():
        settings_repo.update_section(section, values)
    return settings_repo.read_settings()


@router.get("/settings/api-key/status", response_model=ApiKeyStatusResponse)
def get_api_key_status() -> ApiKeyStatusResponse:
    """Ist ein z.ai-API-Key konfiguriert? — ausschließlich Masken-Status (E010)."""
    info = settings_repo.api_key_status()
    return ApiKeyStatusResponse(
        konfiguriert=info.konfiguriert, maske=info.maske, quelle=info.quelle
    )


@router.put("/settings/api-key", response_model=ApiKeyStatusResponse)
def put_api_key(payload: dict[str, Any]) -> ApiKeyStatusResponse:
    """Speichert den z.ai-API-Key (keyring, Fallback-Datei 0600 — E010)."""
    key = payload.get("api_key")
    if not isinstance(key, str) or not key.strip():
        raise ApiError(
            400,
            "ungueltiger_api_key",
            "Der API-Key fehlt oder ist leer.",
            hint='Body-Format: {"api_key": "…"}',
        )
    key = key.strip()
    if len(key) < _MIN_API_KEY_LENGTH:
        raise ApiError(
            400,
            "ungueltiger_api_key",
            f"Der API-Key ist zu kurz (mindestens {_MIN_API_KEY_LENGTH} Zeichen).",
        )
    info = settings_repo.api_key_save(key)
    return ApiKeyStatusResponse(
        konfiguriert=info.konfiguriert, maske=info.maske, quelle=info.quelle
    )


@router.delete("/settings/api-key", response_model=ApiKeyStatusResponse)
def delete_api_key() -> ApiKeyStatusResponse:
    """Entfernt den API-Key aus keyring und Fallback-Datei (E010)."""
    info = settings_repo.api_key_delete()
    return ApiKeyStatusResponse(
        konfiguriert=info.konfiguriert, maske=info.maske, quelle=info.quelle
    )


# ---------------------------------------------------------------------------
# Daten-Management (T15, C.1): Verbrauch je Symbol/Intervall + Purge
# ---------------------------------------------------------------------------


class PurgeRequest(BaseModel):
    """Body von ``POST /api/settings/data/purge`` (§C.1/T15)."""

    symbol: str = Field(min_length=1)


@router.get("/settings/data-usage")
def get_data_usage() -> dict[str, Any]:
    """Speicherverbrauch je (Symbol, Intervall): Parquet-Größe + bar_meta-Abdeckung.

    ``size_bytes`` kommt aus dem Dateisystem (``stat()``), ``row_count``/
    ``first_ts``/``last_ts`` aus ``bar_meta`` (int64 µs UTC, E006). Fehlt das
    Parquet-File trotz Meta-Zeile (oder verschwindet es unter dem Meta —
    ``stat()`` wirft ``OSError``), wird die Zeile mit ``size_bytes = 0``
    geführt (sichtbar statt still verschwunden). ``summe`` aggregiert über
    alle Einträge (``dateien`` zählt nur tatsächlich vorhandene Files).
    """
    eintraege: list[dict[str, Any]] = []
    summe = {"size_bytes": 0, "row_count": 0, "dateien": 0}
    for meta in parquet_store.list_bar_meta():
        try:
            path = parquet_store.parquet_path(meta.symbol, meta.interval)
        except ProviderError:
            # Defekte Registry-Zeile (sollte nicht vorkommen) — überspringen,
            # statt die gesamte Verbrauchsansicht mit 422 zu kippen.
            continue
        vorhanden = path.is_file()
        groesse = 0
        if vorhanden:
            try:
                groesse = path.stat().st_size
            except OSError:
                # Datei zwischen is_file() und stat() verschwunden → 0 führen,
                # statt die Verbrauchsansicht mit 500 zu kippen.
                vorhanden = False
        eintraege.append(
            {
                "symbol": meta.symbol,
                "interval": meta.interval,
                "size_bytes": groesse,
                "row_count": meta.row_count,
                "first_ts": meta.first_ts,
                "last_ts": meta.last_ts,
            }
        )
        summe["size_bytes"] += groesse
        summe["row_count"] += meta.row_count
        if vorhanden:
            summe["dateien"] += 1
    return {"eintraege": eintraege, "summe": summe}


def _watchlist_entfernen(symbol: str) -> bool:
    """Entfernt das Symbol aus allen Watchlists (``True``, wenn vorhanden; E033).

    Purge kennt keine konkrete Liste — in v1 war die Watchlist global, in v2
    entfernt der Aufruf das Symbol aus jeder Liste (dangling-Zeilen vermeiden).
    """
    return watchlists_repo.remove_symbol_from_all(symbol) > 0


@router.post("/settings/data/purge")
def purge_symbol_data(
    payload: PurgeRequest, registry: bool = False
) -> dict[str, Any]:
    """Löscht ALLE gespeicherten Daten des Symbols (über alle Intervalle).

    Semantik (T15): Parquet-Files + ``bar_meta``-Zeilen + Watchlist-Eintrag
    werden immer entfernt; der ``symbols``-Registry-Eintrag (Name, Währung, …)
    nur mit ``?registry=true`` — Default ist „Daten ja, Registry-Eintrag
    bleibt“. Der Aufruf ist idempotent: nicht vorhandene Daten liefern 200
    mit Nullen (kein 404), ungültige Symbol-Kürzel hingegen 422.
    """
    try:
        geprueft = validate_symbol(payload.symbol)
    except ProviderError as exc:
        raise exc.to_api_error() from exc
    geloeschte_dateien = parquet_store.delete_symbol_data(geprueft)
    watchlist_geloescht = _watchlist_entfernen(geprueft)
    registry_geloescht = symbols_repo.delete_symbol(geprueft) if registry else False
    # Frontend verwirft alle Caches des Symbols (interval "*" = alle Intervalle).
    events.data_changed(geprueft, "*", events.REASON_GELOESCHT)
    return {
        "symbol": geprueft,
        "geloeschte_dateien": geloeschte_dateien,
        "watchlist_geloescht": watchlist_geloescht,
        "registry_geloescht": registry_geloescht,
    }
