"""Screener-Endpunkte (§C.4/§T32, E027/E033): Regeln ausführen, Presets,
S&P-500-Bulk-Import (opt-in).

- ``POST /api/screener/run`` — Regeln (AND-verknüpft) über das lokale
  Universum (E027: ausschließlich Parquet/SQLite, synchron); Fachliche
  Validierung (unbekanntes Feld/Op, sinnwidrige Werte) macht
  :mod:`quantlab.services.screener` → 422 deutsch.
- ``GET · POST /api/screener/presets`` · ``GET · PUT · DELETE
  /api/screener/presets/{id}`` — Preset-CRUD über das T18-Repo
  (:mod:`quantlab.storage.repos.screener_repo`).
- ``POST /api/screener/universe/import-sp500`` — **opt-in** Bulk-Import
  (E027): Wikipedia-Liste laden/parsen → Yahoo-Kürzel normalisieren →
  Dedup gegen Registry → Suche-Validierung → Symbole anlegen → **genau
  einen** 1d-Download-Job (FIFO-Schleuse + Provider-Pacing, E035).
  Bestätigungs-Pflicht: der Body muss ``bestaetigt: true`` und die
  erwartete Symbol-Anzahl (``anzahl``) mitliefern — weicht sie vom
  geparsten Stand ab, wird mit 422 abgelehnt (Schutzwert gegen
  versehentlichen Massenimport bei Strukturänderungen).

Pacing-Details des Bulk-Imports: der Download-Job läuft über den bestehenden
Job-Manager (ein Symbol nach dem anderen, prozessweites Yahoo-Pacing 0,75 s,
Circuit-Breaker, Stooq-Fallback) — der Import selbst startet additionally die
Suche-Validierung durch denselben Limiter, kappt sie aber auf
:data:`SUCHE_VALIDIERUNG_MAX` neue Symbole pro Request, damit die Antwortzeit
des Endpunkts begrenzt bleibt; darüber hinaus werden Kürzel nur
format-validiert übernommen (der Download-Job isoliert Fehlschläge je Symbol
in der Fehlerzusammenfassung). ANNAHME (dokumentiert): Klassen-Aktien stehen
bei Wikipedia mit Punkt („BRK.B“) und bei Yahoo mit Bindestrich („BRK-B“) —
:func:`screener.normalisiere_yahoo_kuerzel`.

Konventionen wie ``routers/data``: Sync-Handler im Threadpool für
DB-/Parquet-Pfade, ``async`` für den Import (Wikipedia-Abruf und Job-Anlage
laufen via ``asyncio.to_thread``/Loop); ProviderError → ApiError 1:1
(``to_api_error``), Meldungen deutsch.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field, StringConstraints

from quantlab.api.routers import data as data_router
from quantlab.errors import ApiError
from quantlab.providers.base import ProviderError
from quantlab.services import downloads
from quantlab.services import screener as screener_service
from quantlab.storage import parquet_store
from quantlab.storage.repos import screener_repo, symbols_repo

router = APIRouter(tags=["screener"])

#: Textfeld-Regel (Muster projects-/strategies-Router).
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: Obergrenze der Suche-Validierung pro Bulk-Request (Antwortzeit-Schutz;
#: s. Modul-Docstring). Reihenfolge: Wikipedia-Listenreihenfolge.
SUCHE_VALIDIERUNG_MAX = 25

#: Zeitraum des Bulk-Download-Jobs (deckt perf_1y/ann_vol/SMA-50 des Screeners
#: reichlich ab und begrenzt die Yahoo-Last je Symbol; Pacing liegt beim
#: Provider der Job-Pipeline, E035).
BULK_PERIODE = "2y"


# ---------------------------------------------------------------------------
# Request-Schemas
# ---------------------------------------------------------------------------


class RegelRequest(BaseModel):
    """Eine Regel ``{feld, op, wert}`` (§T32; ``between`` → ``[min, max]``)."""

    feld: str
    op: str
    wert: float | list[float]


class ScreenerRunRequest(BaseModel):
    """Body von ``POST /api/screener/run`` — Regeln + Sortierung."""

    regeln: list[RegelRequest] = Field(min_length=1)
    sort_feld: str | None = None
    richtung: str = "ab"
    limit: int | None = Field(default=None, ge=1, le=5000)


class PresetCreateRequest(BaseModel):
    """Body von ``POST /api/screener/presets``."""

    name: NonEmptyText = Field(max_length=200)
    regeln: list[RegelRequest] = Field(min_length=1)


class PresetUpdateRequest(BaseModel):
    """Body von ``PUT /api/screener/presets/{id}`` (``None`` = unverändert)."""

    name: NonEmptyText | None = Field(default=None, max_length=200)
    regeln: list[RegelRequest] | None = None


class Sp500ImportRequest(BaseModel):
    """Body von ``POST /api/screener/universe/import-sp500`` (Bestätigung).

    ``anzahl`` = vom Nutzer bestätigte Symbolanzahl (muss dem geparsten Stand
    entsprechen); ``limit`` kürzt den Import auf die ersten N **neuen**
    Symbole (Smoke/Teilimport, z. B. ~20 statt aller ~500).
    """

    anzahl: int = Field(ge=1)
    bestaetigt: bool = False
    limit: int | None = Field(default=None, ge=1)


# ---------------------------------------------------------------------------
# Helfer
# ---------------------------------------------------------------------------


def _regeln_roh(regeln: list[RegelRequest]) -> list[dict[str, Any]]:
    """Pydantic-Regeln → plain ``{feld, op, wert}``-AST (Repo-/Service-Vertrag)."""
    return [{"feld": r.feld, "op": r.op, "wert": r.wert} for r in regeln]


def _preset_als_dict(preset: screener_repo.ScreenerPreset) -> dict[str, Any]:
    return {
        "id": preset.id,
        "name": preset.name,
        "regeln": preset.rules,
        "created_at": preset.created_at,
        "updated_at": preset.updated_at,
    }


def _preset_or_404(preset_id: int) -> screener_repo.ScreenerPreset:
    preset = screener_repo.get_preset(preset_id)
    if preset is None:
        raise ApiError(
            404,
            "preset_nicht_gefunden",
            f"Kein Screener-Preset mit der ID {preset_id} gefunden.",
        )
    return preset


def _symbol_bekannt(symbol: str) -> bool:
    """Dedup-Basis (E027): Registry-Eintrag oder 1d-Bestand existiert bereits."""
    if symbols_repo.get_symbol(symbol) is not None:
        return True
    return any(meta.interval == "1d" for meta in parquet_store.list_bar_meta(symbol))


# ---------------------------------------------------------------------------
# POST /api/screener/run (§T32)
# ---------------------------------------------------------------------------


@router.post("/screener/run")
@data_router._ohne_provider_fehler
def run_screener(payload: ScreenerRunRequest) -> dict[str, Any]:
    """Führt die Regeln über das lokale Universum aus (synchron, nur lokal).

    Antwort: ``{universum, count, ergebnisse: [{symbol, name, werte}, …]}`` —
    Details und Feld-/Werte-Konventionen: :func:`screener.screen`.
    """
    return screener_service.screen(
        _regeln_roh(payload.regeln),
        sort_feld=payload.sort_feld,
        richtung=payload.richtung,
        limit=payload.limit,
    )


# ---------------------------------------------------------------------------
# Preset-CRUD (§T32, screener_repo aus T18)
# ---------------------------------------------------------------------------


@router.get("/screener/presets")
def list_presets() -> dict[str, Any]:
    """Alle gespeicherten Presets, nach Name sortiert."""
    return {
        "presets": [_preset_als_dict(preset) for preset in screener_repo.list_presets()]
    }


@router.post("/screener/presets", status_code=201)
def create_preset(payload: PresetCreateRequest) -> dict[str, Any]:
    """Legt ein Preset an (Regeln unvalidiert persistiert — Ausführung prüft)."""
    try:
        preset = screener_repo.create_preset(payload.name, _regeln_roh(payload.regeln))
    except ValueError as exc:
        raise ApiError(422, "preset_name", str(exc)) from exc
    return _preset_als_dict(preset)


@router.get("/screener/presets/{preset_id}")
def get_preset(preset_id: int) -> dict[str, Any]:
    """Ein Preset (Regeln als AST)."""
    return _preset_als_dict(_preset_or_404(preset_id))


@router.put("/screener/presets/{preset_id}")
def update_preset(preset_id: int, payload: PresetUpdateRequest) -> dict[str, Any]:
    """Aktualisiert Name und/oder Regeln (``None`` = unverändert)."""
    _preset_or_404(preset_id)
    roh = _regeln_roh(payload.regeln) if payload.regeln is not None else None
    try:
        aktualisiert = screener_repo.update_preset(
            preset_id, name=payload.name, rules=roh
        )
    except ValueError as exc:
        raise ApiError(422, "preset_name", str(exc)) from exc
    assert aktualisiert is not None  # Existenz oben geprüft
    return _preset_als_dict(aktualisiert)


@router.delete("/screener/presets/{preset_id}", status_code=204)
def delete_preset(preset_id: int) -> Response:
    """Löscht das Preset."""
    _preset_or_404(preset_id)
    screener_repo.delete_preset(preset_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# POST /api/screener/universe/import-sp500 (E027, opt-in)
# ---------------------------------------------------------------------------


@router.post("/screener/universe/import-sp500")
@data_router._ohne_provider_fehler
async def import_sp500(payload: Sp500ImportRequest) -> dict[str, Any]:
    """Opt-in Bulk-Import der S&P-500-Liste → **genau ein** Download-Job.

    Pipeline (E027): Wikipedia laden/parsen (502 bei Struktur-/Netzpanne) →
    Bestätigung prüfen (``bestaetigt`` + ``anzahl``) → Kürzel normalisieren →
    Dedup gegen Registry/Bestand → Suche-Validierung (gekappt, s.
    :data:`SUCHE_VALIDIERUNG_MAX`) → Symbole anlegen → ein 1d-Download-Job
    (``period="2y"``, Pacing/FIFO über den Job-Manager, E035).

    Antwort: ``{job_id, gesamt, bekannt, angelegt, entfallen, geladen}``.
    Neue Symbole landen bewusst **nicht** in der Watchlist (anders als
    ``POST /api/data/symbols``) — ein Massenimport soll die Default-Liste
    nicht fluten.
    """
    if not payload.bestaetigt:
        raise ApiError(
            422,
            "import_nicht_bestaetigt",
            "Der S&P-500-Bulk-Import muss ausdrücklich bestätigt werden "
            "(„bestaetigt“: true) — er legt bis zu ~500 Symbole an und startet "
            "einen langen Download-Job.",
        )
    liste = await asyncio.to_thread(screener_service.lade_sp500_liste)
    if len(liste) != payload.anzahl:
        raise ApiError(
            422,
            "bestaetigung_unpassend",
            "Die bestätigte Symbol-Anzahl stimmt nicht mit der geladenen "
            f"Liste überein (bestätigt: {payload.anzahl}, geladen: {len(liste)}).",
            hint="Die Anzahl im Dialog muss dem aktuellen Listenstand "
            "entsprechen — bitte den Import erneut bestätigen.",
        )

    kandidaten = [
        screener_service.normalisiere_yahoo_kuerzel(str(eintrag["symbol"]))
        for eintrag in liste
    ]
    # M3: Keys ebenfalls Yahoo-normalisieren — die Lookups unten laufen mit den
    # normalisierten Kürzeln (Klassen-Aktien „BRK.B“ → „BRK-B“); mit den
    # Wikipedia-Keys als Schlüssel ging der Name jenseits des Suche-Caps verloren.
    namen = {
        screener_service.normalisiere_yahoo_kuerzel(
            str(eintrag["symbol"])
        ): eintrag["name"]
        for eintrag in liste
    }
    bekannte: list[str] = []
    neue: list[str] = []
    for kuerzel in kandidaten:
        if await asyncio.to_thread(_symbol_bekannt, kuerzel):
            bekannte.append(kuerzel)
        elif kuerzel not in neue:
            neue.append(kuerzel)
    if payload.limit is not None:
        neue = neue[: payload.limit]

    # Suche-Validierung nur für den gekappten Kopf der neuen Symbole (E027);
    # der Rest wird format-validiert übernommen — der Download-Job isoliert
    # Fehlschläge je Symbol (Fehlerzusammenfassung), Rate-Limit bleibt geschont.
    angelegt = 0
    entfallen = 0
    zu_pruefen, ohne_suche = neue[:SUCHE_VALIDIERUNG_MAX], neue[SUCHE_VALIDIERUNG_MAX:]
    uebernommen: list[str] = []
    for kuerzel in zu_pruefen:
        try:
            info = await asyncio.to_thread(
                data_router._symbol_ueber_suche_validieren, kuerzel
            )
        except ProviderError:
            entfallen += 1  # Yahoo kennt das Kürzel nicht → nicht importieren
            continue
        await asyncio.to_thread(
            symbols_repo.upsert_symbol,
            kuerzel,
            name=info.name,
            asset_type=info.asset_type,
            exchange_tz=info.exchange_tz,
            currency=info.currency,
        )
        uebernommen.append(kuerzel)
        angelegt += 1
    for kuerzel in ohne_suche:
        await asyncio.to_thread(
            symbols_repo.upsert_symbol,
            kuerzel,
            name=namen.get(kuerzel) or None,
            asset_type="equity",
            exchange_tz=None,
            currency=None,
        )
        uebernommen.append(kuerzel)
        angelegt += 1

    # Genau EIN Download-Job für alle übernommenen Symbole (E027); ist nichts
    # Neues übrig (alles bekannt/entfallen), entfällt der Job bewusst — ein
    # leerer Download wäre nur Fehlermeldung ohne Nutzen.
    if not uebernommen:
        return {
            "job_id": None,
            "gesamt": len(liste),
            "bekannt": len(bekannte),
            "angelegt": angelegt,
            "entfallen": entfallen,
            "geladen": 0,
        }
    job = await downloads.submit_job(uebernommen, "1d", period=BULK_PERIODE)
    return {
        "job_id": job.job_id,
        "gesamt": len(liste),
        "bekannt": len(bekannte),
        "angelegt": angelegt,
        "entfallen": entfallen,
        "geladen": len(uebernommen),
    }
