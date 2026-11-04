"""Python-Sandbox für User-Strategien (T30, E030) — dokumentierter Unfallschutz.

**Bedrohungsmodell (E030, ehrlich):** einzelner lokaler Nutzer mit eigenem
Code — kein Multi-Tenant. Die Sandbox ist ein **Unfallschutz** (Endlosschleife,
Memory-Bombe, versehentlicher Dateizugriff), **kein Sicherheitsversprechen**:
AST-Verbotslisten sind dokumentiert umgehbar (Dunder-Ketten, Attribut-Walking,
Traceback-Globals — RECHERCHE.md Block 5); entschlossene Escape-Versuche wären
Selbstschädigung. OS-Isolation (Container/seccomp) bleibt dokumentierte
spätere Option.

**Mechanik pro Ausführung:**

1. **AST-Vorprüfung** (:func:`pruefe_code`): Verbotsliste gegen verbotene
   Imports (``os, sys, subprocess, socket, shutil`` + ``ctypes/importlib/
   signal/resource/builtins``), Datei-/Meta-Aufrufe (``open/exec/eval/
   compile``) und Dunder-Attributzugriff (``__…__``) — jede Ablehnung **mit
   Zeilennummer** (deutsche Meldung, Grundlage für den Lint-Gutter im Editor).
2. **Frischer Subprocess** (``sys.executable -c <Runner>``, E030): das Kind
   lädt die OHLCV-Daten als **Arrow-IPC-Datei** aus einem TMP-Unterordner im
   Datenverzeichnis, setzt **direkt vor der Gastcode-Ausführung** unwiderruflich
   ``RLIMIT_AS``/``RLIMIT_CPU``/``RLIMIT_FSIZE`` (soft = hard, Default
   512 MB / 10 s / 16 MB) und armiert ein ``SIGALRM``-Wall-Clock-Budget.
   Die Limits werden nach dem Laden von pyarrow/pandas, aber **vor dem Exec**
   gesetzt — RLIMIT_AS zählt virtuellen Adressraum und pyarrow/jemalloc
   reservieren an sich deutlich mehr virtuellen als residenten Speicher;
   geschützt werden soll ausschließlich der Gastcode.
3. **Wall-Clock von außen:** zusätzlich zum Kind-Alarm kappt der Parent nach
   ``STARTUP_GRACE_S + timeout_s + 2`` per ``communicate(timeout=…)+kill()``
   (E030-Backstop, falls das Kind hängt, bevor es den Alarm armieren kann).
4. **Strukturiertes JSON-Ergebnis** (``result.json`` im TMP-Ordner):
   ``ok / error_kind / meldung (de) / gewichte / verstoesse /
   traceback_zeilen [{zeile, text}]`` — Grundlage für Editor-Lint und
   Testlauf-Zusammenfassung.

**Strategie-Vertrag (E026):** der Gast definiert
``generate_weights(data, **params)`` und erhält den OHLCV-DataFrame (Spalten
``ts, open, high, low, close, volume``); zurück kommt ein DataFrame mit
**genau einer Gewichtsspalte** (oder eine Series), Zeile ``i`` = Zielgewicht
aus Daten ≤ ``i``. Die Sandbox prüft nur die **Form** (Spaltenzahl,
Zeilenzahl) — Wertepolitik (z. B. long/flat 0/1, Summe ≤ 1) bleibt Sache der
Backtest-Engine (T29 erlaubt dort negative Gewichte/Short).

Bekannte Grenze (dokumentiert): im PyInstaller-pakettierten Sidecar ist
``sys.executable`` das Sidecar-Binary, kein Python-Interpreter — die Sandbox
funktioniert dort erst, wenn T35 einen Interpreter mitliefert oder sie dort
deaktiviert (dev/uv-Betrieb ist der Referenzpfad, §T30 „explizites
sys.executable“).
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quantlab.config import get_data_dir

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Konstanten (E030-Defaults)
# ---------------------------------------------------------------------------

#: Speicherlimit (RLIMIT_AS) als **Gast-Budget** in MB über der Adressraum-
#: Basis des Kindes — Default laut §T30. ANNAHME (empirisch belegt, 2026-09-14):
#: pyarrow/jemalloc reservieren beim Import ~0,7–1,9 GB *virtuell*; ein absolutes
#: 512-MB-RLIMIT_AS wäre beim Umgebungsaufbau schon überschritten. Das Limit
#: wird daher als Budget **oberhalb** der gemessenen Basis gesetzt (Linux
#: ``/proc/self/status``) — semantisch: „der Gast darf 512 MB eigenes Wachstum“.
DEFAULT_MEMORY_MB = 512

#: CPU-Zeitlimit (RLIMIT_CPU, soft = hard) in Sekunden — Default laut §T30.
#: ANNAHME (empirisch, 2026-09-14): das Budget läuft ab **Kindprozessstart**
#: und umfasst damit auch den pyarrow/pandas-Import (~0,85 s CPU) — für den
#: Gastcode bleibt entsprechend weniger. Mit soft == hard endet eine
#: Überschreitung per SIGKILL (−9); die Klassifikation differenziert in
#: :func:`_klassifiziere` über die verbrauchte Wall-Clock-Zeit.
DEFAULT_CPU_S = 10.0

#: Dateigrößenlimit (RLIMIT_FSIZE, soft = hard) in MB — Default laut §T30.
DEFAULT_FSIZE_MB = 16

#: Wall-Clock-Budget für den Gastcode in Sekunden (Kind-armiert via SIGALRM).
#: ANNAHME: 30 s sind für OHLCV-Strategien großzügig; das CPU-Limit (10 s)
#: schneidet rechenlastige Endlosschleifen meist früher.
DEFAULT_TIMEOUT_S = 30.0

#: Gnadenfrist für Prozessstart + pyarrow/pandas-Import + Datenladen, bevor
#: der harte Parent-Kill (E030-Backstop) zuschlägt.
STARTUP_GRACE_S = 20.0

#: Obergrenze für Strategie-Code (Zeichen) — Schutz vor Pathologien.
MAX_CODE_CHARS = 200_000

#: Verbotsliste (E030): ``os, sys, subprocess, socket, shutil`` + dokumentierte
#: Erweiterungen — ``ctypes`` (Native-Calls), ``importlib`` (Import-Umwege),
#: ``signal`` (könnte den Sandbox-Alarm abwürgen), ``resource`` (könnte die
#: eigenen Rlimits anheben wollen), ``builtins`` (M3: Import umgeht die
#: Namensbindung der Aufruf-Liste — ``getattr(builtins, "open")`` lieferte
#: ohne Sperre echtes ``open``).
VERBOTENE_MODULE = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "ctypes",
        "importlib",
        "signal",
        "resource",
        "builtins",
    }
)

#: Verbotene Aufrufe (E030): Dateizugriff + dynamische Ausführung.
VERBOTENE_AUFRUFE = frozenset({"open", "exec", "eval", "compile"})

#: Dunder-Muster (E030): Attribut-/Namenszugriff auf ``__…__``.
_DUNDER_MUSTER = re.compile(r"^__.+__$")

#: Vertrags-Funktionsname (E026).
VERTRAGS_FUNKTION = "generate_weights"


# ---------------------------------------------------------------------------
# AST-Vorprüfung (Verbotsliste mit Zeilennummern)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeVerstoss:
    """Ein AST-Verstoß mit Zeilennummer (deutsche Meldung für den Editor)."""

    zeile: int
    text: str  # Quelltext der Zeile (getrimmt)
    regel: str  # stabiler Maschinentyp (import_verboten, aufruf_verboten, …)
    meldung: str  # deutsche Erklärung

    def als_dict(self) -> dict[str, Any]:
        return {
            "zeile": self.zeile,
            "text": self.text,
            "regel": self.regel,
            "meldung": self.meldung,
        }


def _zeilentext(code: str, zeile: int) -> str:
    zeilen = code.splitlines()
    if 1 <= zeile <= len(zeilen):
        return zeilen[zeile - 1].strip()
    return ""


def pruefe_code(code: str) -> list[CodeVerstoss]:
    """Prüft Strategie-Code statisch gegen die Verbotsliste (E030).

    Liefert alle Verstöße (inkl. Syntaxfehler und fehlender Vertrags-Funktion)
    **mit Zeilennummer** — ohne Ausführung. Ein leerer Ergebnis-Liste heißt
    nicht „sicher“, sondern „kein Listenverstoß“ (siehe Modul-Docstring).
    """
    if not code.strip():
        return [
            CodeVerstoss(
                zeile=1,
                text="",
                regel="leer",
                meldung="Der Strategie-Code ist leer.",
            )
        ]

    try:
        baum = ast.parse(code)
    except SyntaxError as exc:
        zeile = exc.lineno or 1
        return [
            CodeVerstoss(
                zeile=zeile,
                text=_zeilentext(code, zeile),
                regel="syntax",
                meldung=f"Syntaxfehler: {exc.msg}.",
            )
        ]

    verstoesse: list[CodeVerstoss] = []
    zeilen = code.splitlines()
    vertrags_definiert = False

    for knoten in ast.walk(baum):
        if isinstance(knoten, ast.FunctionDef | ast.AsyncFunctionDef):
            if knoten.name == VERTRAGS_FUNKTION:
                vertrags_definiert = True
        elif isinstance(knoten, ast.Assign):
            for ziel in knoten.targets:
                if isinstance(ziel, ast.Name) and ziel.id == VERTRAGS_FUNKTION:
                    vertrags_definiert = True

        if isinstance(knoten, ast.Import):
            for alias in knoten.names:
                wurzel = alias.name.split(".")[0]
                if wurzel in VERBOTENE_MODULE:
                    verstoesse.append(
                        CodeVerstoss(
                            zeile=knoten.lineno,
                            text=_zeilentext(code, knoten.lineno),
                            regel="import_verboten",
                            meldung=(
                                f"Der Import von „{alias.name}“ ist in Strategien "
                                f"nicht erlaubt (Verbotsliste: os, sys, subprocess, "
                                f"socket, shutil u. a.)."
                            ),
                        )
                    )

        elif isinstance(knoten, ast.ImportFrom):
            wurzel = (knoten.module or "").split(".")[0]
            if wurzel in VERBOTENE_MODULE:
                verstoesse.append(
                    CodeVerstoss(
                        zeile=knoten.lineno,
                        text=_zeilentext(code, knoten.lineno),
                        regel="import_verboten",
                        meldung=(
                            f"Der Import aus „{knoten.module}“ ist in Strategien "
                            f"nicht erlaubt (Verbotsliste: os, sys, subprocess, "
                            f"socket, shutil u. a.)."
                        ),
                    )
                )

        elif isinstance(knoten, ast.Call):
            func = knoten.func
            if isinstance(func, ast.Name) and func.id in VERBOTENE_AUFRUFE:
                verstoesse.append(
                    CodeVerstoss(
                        zeile=knoten.lineno,
                        text=_zeilentext(code, knoten.lineno),
                        regel="aufruf_verboten",
                        meldung=(
                            f"„{func.id}(...)“ ist in Strategien nicht erlaubt "
                            f"(kein Dateizugriff, keine dynamische Ausführung)."
                        ),
                    )
                )
            elif isinstance(func, ast.Attribute) and func.attr in VERBOTENE_AUFRUFE:
                verstoesse.append(
                    CodeVerstoss(
                        zeile=knoten.lineno,
                        text=_zeilentext(code, knoten.lineno),
                        regel="aufruf_verboten",
                        meldung=(
                            f"„….{func.attr}(...)“ ist in Strategien nicht erlaubt "
                            f"(kein Dateizugriff, keine dynamische Ausführung)."
                        ),
                    )
                )

        elif isinstance(knoten, ast.Attribute):
            if _DUNDER_MUSTER.match(knoten.attr):
                verstoesse.append(
                    CodeVerstoss(
                        zeile=knoten.lineno,
                        text=_zeilentext(code, knoten.lineno),
                        regel="dunder_attribut",
                        meldung=(
                            f"Attributzugriff „{knoten.attr}“ ist in Strategien "
                            f"nicht erlaubt (Dunder-Attribute sind der klassische "
                            f"Sandbox-Escape-Pfad)."
                        ),
                    )
                )

        elif isinstance(knoten, ast.Name):
            if _DUNDER_MUSTER.match(knoten.id):
                verstoesse.append(
                    CodeVerstoss(
                        zeile=knoten.lineno,
                        text=_zeilentext(code, knoten.lineno),
                        regel="dunder_name",
                        meldung=(
                            f"Der Name „{knoten.id}“ ist in Strategien nicht "
                            f"erlaubt (Dunder-Namen sind der klassische "
                            f"Sandbox-Escape-Pfad)."
                        ),
                    )
                )

    if not vertrags_definiert:
        zeile = 1
        for ausdruck in baum.body:
            zeile = getattr(ausdruck, "lineno", 1)
            break
        meldung = (
            f"Die Strategie muss eine Funktion "
            f"„{VERTRAGS_FUNKTION}(data, **params)“ definieren, die die "
            f"Gewichte zurückgibt (siehe Vorlage)."
        )
        verstoesse.append(
            CodeVerstoss(
                zeile=zeile,
                text=_zeilentext(code, zeile) if zeilen else "",
                regel="vertrag",
                meldung=meldung,
            )
        )

    return sorted(verstoesse, key=lambda v: (v.zeile, v.regel))


def pruefe_code_als_dict(code: str) -> list[dict[str, Any]]:
    """:func:`pruefe_code` als serialisierbare Dicts (API/JSON)."""
    return [verusst.als_dict() for verusst in pruefe_code(code)]


# ---------------------------------------------------------------------------
# Strategie-Referenz „user:<id>@<version>“ (E026) — Resolver für T29/T31
# ---------------------------------------------------------------------------


def parse_user_strategy_ref(ref: str) -> tuple[int, int]:
    """Parst ``user:<id>@<version>`` → ``(strategy_id, version)``.

    Alles andere → ``ApiError`` 422 mit deutscher Meldung (Vertrag für den
    Backtest-Router: ``strategy: "user:1@2"``).
    """
    from quantlab.errors import ApiError

    match = re.fullmatch(r"user:(\d+)@(\d+)", ref.strip())
    if match is None:
        raise ApiError(
            422,
            "ungueltige_strategie_referenz",
            f"Ungültige User-Strategie-Referenz „{ref}“.",
            hint="Format: user:<id>@<version>, z. B. user:1@2.",
        )
    return int(match.group(1)), int(match.group(2))


def lade_user_strategie(ref: str) -> tuple[Any, Any]:
    """Löst ``user:<id>@<version>`` über das strategies_repo auf (E026).

    Liefert ``(Strategy, StrategyVersion)``; unbekannte Strategie/Version →
    ``ApiError`` 404, ungültiges Format → 422. Genau die Schnittstelle, die
    ``POST /api/backtest/runs`` (T29/T31) für ``strategy="user:<id>@<version>"``
    konsumieren kann.
    """
    from quantlab.errors import ApiError
    from quantlab.storage.repos import strategies_repo

    strategy_id, version = parse_user_strategy_ref(ref)
    strategy = strategies_repo.get_strategy(strategy_id)
    if strategy is None:
        raise ApiError(
            404,
            "strategie_nicht_gefunden",
            f"User-Strategie mit ID {strategy_id} nicht gefunden.",
        )
    strategy_version = strategies_repo.get_version(strategy_id, version)
    if strategy_version is None:
        raise ApiError(
            404,
            "strategie_version_nicht_gefunden",
            f"Version {version} der User-Strategie „{strategy.name}“ (ID {strategy_id}) "
            f"existiert nicht.",
        )
    return strategy, strategy_version


# ---------------------------------------------------------------------------
# Vertrags-Normalisierung (Kind- und Parent-Seite nutzbar, ohne pandas-Import
# beim Modulladen — pandas wird lazy importiert)
# ---------------------------------------------------------------------------


def normalisiere_gewichte(daten: pd.DataFrame, roh: Any) -> list[float | None]:
    """Validiert die Strategie-Ausgabe gegen den Vertrags (E026) → Gewichtliste.

    Erlaubt: DataFrame mit **genau einer** Spalte oder pandas-Series; Zeilenzahl
    = Balkenzahl; nicht-numerische Werte → Fehler. ``NaN`` (Warm-up) → ``None``
    (= flat). Keine Wertepolitik (0/1, Summen) — das prüft die Engine.
    """
    import pandas as pd

    if isinstance(roh, pd.Series):
        spalte = roh
    elif isinstance(roh, pd.DataFrame):
        if roh.shape[1] != 1:
            raise ValueError(
                f"Das Gewichte-DataFrame muss genau eine Spalte haben "
                f"(v1: ein Symbol) — erhalten: {roh.shape[1]}."
            )
        spalte = roh.iloc[:, 0]
    else:
        raise ValueError(
            "Die Strategie muss ein pandas-DataFrame mit genau einer "
            "Gewichtsspalte oder eine pandas-Series zurückgeben — erhalten: "
            f"{type(roh).__name__}."
        )

    if len(spalte) != len(daten):
        raise ValueError(
            f"Die Gewichte haben {len(spalte)} Zeilen, die Kursdaten haben "
            f"{len(daten)} — die Zeilenzahl muss übereinstimmen."
        )

    try:
        numerisch = pd.to_numeric(spalte, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Die Gewichtsspalte muss numerisch sein (Zahlen; NaN = flat)."
        ) from exc

    return [None if pd.isna(wert) else float(wert) for wert in numerisch]


# ---------------------------------------------------------------------------
# Subprocess-Ausführung (Kind-Programm + Parent-Orchestrierung)
# ---------------------------------------------------------------------------

#: Das Kindprogramm (E030): Arrow laden → AST-erneut prüfen → Rlimits setzen
#: (soft = hard, vor dem Exec) → SIGALRM-Budget → Gastcode ausführen →
#: result.json schreiben. Bewusst in-process eingebettet, damit services/
#: nur :mod:`sandbox.py` umfasst (§T30-Dateiliste).
_RUNNER_PROGRAMM = r'''
"""QuantLab-Sandbox-Kindprozess (T30/E030) — intern, nicht für Nutzer bestimmt."""
import errno
import json
import os
import resource
import signal
import sys
import traceback


class _ZeitAbgelaufen(Exception):
    """Intern: vom SIGALRM-Handler geworfen (Wall-Clock des Gastcodes)."""


def _alarm_handler(signum, frame):
    raise _ZeitAbgelaufen()


def _adressraum_basis_bytes():
    """Aktueller virtueller Adressraum des Kindes (Linux /proc) — 0 wenn unbekannt.

    Basis für das RLIMIT_AS-Gast-Budget: pyarrow/jemalloc reservieren beim
    Import weit mehr *virtuellen* Speicher, als der Gast je nutzen wird — ein
    absolutes Limit würde schon den Umgebungsaufbau treffen (siehe
    DEFAULT_MEMORY_MB-ANNAHME in quantlab/services/sandbox.py).
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for zeile in status:
                if zeile.startswith("VmSize:"):
                    return int(zeile.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _schreibe_ergebnis(pfad, ergebnis):
    try:
        with open(pfad, "w", encoding="utf-8") as datei:
            json.dump(ergebnis, datei, ensure_ascii=False)
    except OSError:
        pass  # Parent klassifiziert über Rückgabecode/stderr


def _gast_frames(fehler, code):
    """Traceback-Frames des Gastcodes (<strategie>) als [{zeile, text}].

    Der Zeilentext wird aus dem Code-String gelesen — ``linecache`` kann bei
    per ``compile`` erzeugtem Pseudo-File ``<strategie>`` keine Quelle laden.
    """
    zeilen = code.splitlines()
    frames = []
    for frame in traceback.extract_tb(fehler.__traceback__):
        if frame.filename == "<strategie>":
            nr = frame.lineno or 0
            text = ""
            if 1 <= nr <= len(zeilen):
                text = zeilen[nr - 1].strip()
            frames.append({"zeile": nr, "text": text})
    return frames


def _main():
    verzeichnis = os.environ["QUANTLAB_SANDBOX_DIR"]
    limits = json.loads(os.environ["QUANTLAB_SANDBOX_LIMITS"])
    ergebnis_pfad = os.path.join(verzeichnis, "result.json")

    ergebnis = {
        "ok": False,
        "error_kind": "intern",
        "meldung": "Interner Sandbox-Fehler: Das Kind hat kein Ergebnis geschrieben.",
        "traceback_zeilen": [],
    }

    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with ipc.open_file(os.path.join(verzeichnis, "data.arrow")) as leser:
            daten = leser.read_all().to_pandas()

        with open(os.path.join(verzeichnis, "code.py"), encoding="utf-8") as datei:
            code = datei.read()
        with open(os.path.join(verzeichnis, "params.json"), encoding="utf-8") as datei:
            params = json.load(datei)

        from quantlab.services import sandbox

        verstoesse = sandbox.pruefe_code_als_dict(code)
        if verstoesse:
            ergebnis = {
                "ok": False,
                "error_kind": "verbotsliste",
                "meldung": (
                    "Der Strategie-Code enthält verbotene Anweisungen "
                    f"({len(verstoesse)} Verstoß/Verstöße — siehe Liste)."
                ),
                "verstoesse": verstoesse,
                "traceback_zeilen": [],
            }
        else:
            signal.signal(signal.SIGALRM, _alarm_handler)
            basis = _adressraum_basis_bytes()
            budget = int(limits["memory_bytes"])
            as_grenze = (basis + budget) if basis > 0 else budget
            resource.setrlimit(resource.RLIMIT_AS, (as_grenze, as_grenze))
            cpu = int(limits["cpu_s"])
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            fsize = int(limits["fsize_bytes"])
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            signal.alarm(int(limits["alarm_s"]))
            try:
                namespace = {"__name__": "quantlab_strategie"}
                # Bewusster Gast-Exec (E030) — hinter Verbotsliste + Rlimits.
                exec(compile(code, "<strategie>", "exec"), namespace)  # noqa: S102
                funktion = namespace.get("generate_weights")
                if not callable(funktion):
                    ergebnis = {
                        "ok": False,
                        "error_kind": "vertrag",
                        "meldung": (
                            "Die Strategie muss eine aufrufbare Funktion "
                            "„generate_weights(data, **params)“ definieren "
                            "(siehe Vorlage)."
                        ),
                        "traceback_zeilen": [],
                    }
                else:
                    roh = funktion(daten, **params)
                    try:
                        gewichte = sandbox.normalisiere_gewichte(daten, roh)
                    except ValueError as vertrag_fehler:
                        ergebnis = {
                            "ok": False,
                            "error_kind": "vertrag",
                            "meldung": str(vertrag_fehler),
                            "traceback_zeilen": [],
                        }
                    else:
                        ergebnis = {
                            "ok": True,
                            "error_kind": None,
                            "meldung": "",
                            "gewichte": gewichte,
                            "traceback_zeilen": [],
                        }
            except _ZeitAbgelaufen:
                ergebnis = {
                    "ok": False,
                    "error_kind": "timeout",
                    "meldung": (
                        "Zeitlimit überschritten: Die Strategie lief länger als "
                        f"{float(limits['timeout_s']):.0f} s (Verdacht: "
                        "Endlosschleife) und wurde abgebrochen."
                    ),
                    "traceback_zeilen": [],
                }
            except MemoryError:
                ergebnis = {
                    "ok": False,
                    "error_kind": "speicher",
                    "meldung": (
                        "Speicherlimit überschritten: Die Strategie benötigte "
                        f"mehr als {int(limits['memory_bytes']) // (1024 * 1024)} MB "
                        "(RLIMIT_AS) und wurde gestoppt."
                    ),
                    "traceback_zeilen": [],
                }
            except BaseException as fehler:
                if isinstance(fehler, OSError) and fehler.errno == errno.EFBIG:
                    # M3 (Fund F3): CPython ignoriert SIGXFSZ (SIG_IGN) — das
                    # RLIMIT_FSIZE-Limit äußert sich im Gast als OSError
                    # [Errno 27] statt −25; hier explizit mappen, damit die
                    # deutsche Meldung den echten Grund nennt.
                    ergebnis = {
                        "ok": False,
                        "error_kind": "dateigroesse",
                        "meldung": (
                            "Dateigrößenlimit überschritten: Die Strategie "
                            "versuchte, mehr als "
                            f"{int(limits['fsize_bytes']) // (1024 * 1024)} MB "
                            "zu schreiben (RLIMIT_FSIZE — "
                            f"{type(fehler).__name__}: {fehler})."
                        ),
                        "traceback_zeilen": _gast_frames(fehler, code),
                    }
                else:
                    meldung = (
                        f"Laufzeitfehler in der Strategie: {type(fehler).__name__}: {fehler}"
                    )
                    ergebnis = {
                        "ok": False,
                        "error_kind": "laufzeit",
                        "meldung": meldung,
                        "traceback_zeilen": _gast_frames(fehler, code),
                    }
            finally:
                signal.alarm(0)
    except MemoryError:
        ergebnis = {
            "ok": False,
            "error_kind": "speicher",
            "meldung": (
                "Speicherlimit überschritten (RLIMIT_AS) — bereits beim Laden "
                "der Umgebung/Daten."
            ),
            "traceback_zeilen": [],
        }
    except BaseException as fehler:
        traceback.print_exc()
        ergebnis = {
            "ok": False,
            "error_kind": "intern",
            "meldung": f"Interner Sandbox-Fehler: {type(fehler).__name__}: {fehler}",
            "traceback_zeilen": [],
        }

    _schreibe_ergebnis(ergebnis_pfad, ergebnis)
    os._exit(0)


_main()
'''


@dataclass(frozen=True)
class SandboxErgebnis:
    """Strukturiertes Ergebnis einer Sandbox-Ausführung (E030)."""

    ok: bool
    # verbotsliste|syntax|timeout|speicher|cpu|dateigroesse|vertrag|laufzeit|intern
    error_kind: str | None
    meldung: str  # deutsch; bei ok leer
    dauer_s: float
    n_balken: int
    gewichte: list[float | None] = field(default_factory=list)
    verstoesse: list[dict[str, Any]] = field(default_factory=list)
    traceback_zeilen: list[dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    def als_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_kind": self.error_kind,
            "meldung": self.meldung,
            "dauer_s": self.dauer_s,
            "n_balken": self.n_balken,
            "gewichte": self.gewichte,
            "verstoesse": self.verstoesse,
            "traceback_zeilen": self.traceback_zeilen,
        }


def _backend_wurzel() -> Path:
    """Repo-``backend/``-Verzeichnis (PYTHONPATH für den Kind-Import von quantlab)."""
    return Path(__file__).resolve().parents[2]


def _schreibe_arrow(daten: pd.DataFrame, pfad: Path) -> None:
    """OHLCV-DataFrame als Arrow-IPC-Datei (E030) — Index bewusst verworfen."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    tabelle = pa.Table.from_pandas(daten.reset_index(drop=True), preserve_index=False)
    with ipc.new_file(pfad, tabelle.schema) as schreiber:
        schreiber.write_table(tabelle)


def _lese_ergebnis(pfad: Path) -> dict[str, Any] | None:
    try:
        with pfad.open(encoding="utf-8") as datei:
            inhalt = json.load(datei)
    except (OSError, json.JSONDecodeError):
        return None
    return inhalt if isinstance(inhalt, dict) else None


def fuehre_strategie_aus(
    code: str,
    daten: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cpu_s: float = DEFAULT_CPU_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    fsize_mb: int = DEFAULT_FSIZE_MB,
    grace_s: float = STARTUP_GRACE_S,
) -> SandboxErgebnis:
    """Führt Strategie-Code gegen ``daten`` in der Sandbox aus (E030).

    Ablauf und Grenzen siehe Modul-Docstring. ``timeout_s`` ist das
    Wall-Clock-Budget des **Gastcodes** (Kind-SIGALRM); der Parent killt
    spätestens nach ``grace_s + timeout_s + 2``. Kleine Budgets (Tests)
    sind ausdrücklich vorgesehen (§T30 Risiko-Notiz).
    """
    gestartet = time.monotonic()
    params = params or {}

    verstoesse = pruefe_code_als_dict(code)
    if verstoesse:
        return SandboxErgebnis(
            ok=False,
            error_kind="verbotsliste",
            meldung=(
                "Der Strategie-Code enthält verbotene Anweisungen "
                f"({len(verstoesse)} Verstoß/Verstöße — siehe Liste)."
            ),
            dauer_s=time.monotonic() - gestartet,
            n_balken=len(daten),
            verstoesse=verstoesse,
        )

    lauf_dir = get_data_dir() / "sandbox" / f"lauf-{uuid.uuid4().hex}"
    lauf_dir.mkdir(parents=True, exist_ok=True)
    try:
        (lauf_dir / "code.py").write_text(code, encoding="utf-8")
        (lauf_dir / "params.json").write_text(
            json.dumps(params, ensure_ascii=False), encoding="utf-8"
        )
        _schreibe_arrow(daten, lauf_dir / "data.arrow")

        limits = {
            "cpu_s": max(1, math.ceil(cpu_s)),
            "memory_bytes": max(1, int(memory_mb)) * 1024 * 1024,
            "fsize_bytes": max(1, int(fsize_mb)) * 1024 * 1024,
            "alarm_s": max(1, math.ceil(timeout_s)),
            "timeout_s": float(timeout_s),
        }
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_backend_wurzel()) + os.pathsep + env.get("PYTHONPATH", "")
        env["QUANTLAB_SANDBOX_DIR"] = str(lauf_dir)
        env["QUANTLAB_SANDBOX_LIMITS"] = json.dumps(limits)

        prozess = subprocess.Popen(  # noqa: S603 — sys.executable + eingebettetes Programm
            [sys.executable, "-B", "-c", _RUNNER_PROGRAMM],
            cwd=str(lauf_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        abgebrochen = False
        try:
            stdout, stderr = prozess.communicate(timeout=grace_s + limits["alarm_s"] + 2.0)
        except subprocess.TimeoutExpired:
            abgebrochen = True
            prozess.kill()
            stdout, stderr = prozess.communicate()

        kind_ergebnis = _lese_ergebnis(lauf_dir / "result.json")
        dauer = time.monotonic() - gestartet
        return _klassifiziere(
            kind_ergebnis,
            abgebrochen=abgebrochen,
            rueckgabe=prozess.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            dauer=dauer,
            n_balken=len(daten),
            limits=limits,
        )
    finally:
        shutil.rmtree(lauf_dir, ignore_errors=True)


def _klassifiziere(
    kind_ergebnis: dict[str, Any] | None,
    *,
    abgebrochen: bool,
    rueckgabe: int | None,
    stdout: str,
    stderr: str,
    dauer: float,
    n_balken: int,
    limits: dict[str, Any],
) -> SandboxErgebnis:
    """Kind-Ausgang → strukturiertes :class:`SandboxErgebnis` (deutsche Meldungen)."""
    basis = {
        "dauer_s": dauer,
        "n_balken": n_balken,
        "stdout": stdout,
        "stderr": stderr,
    }

    if kind_ergebnis is not None:
        return SandboxErgebnis(
            ok=bool(kind_ergebnis.get("ok")),
            error_kind=kind_ergebnis.get("error_kind"),
            meldung=str(kind_ergebnis.get("meldung", "")),
            dauer_s=dauer,
            n_balken=n_balken,
            gewichte=list(kind_ergebnis.get("gewichte") or []),
            verstoesse=list(kind_ergebnis.get("verstoesse") or []),
            traceback_zeilen=list(kind_ergebnis.get("traceback_zeilen") or []),
            stdout=stdout,
            stderr=stderr,
        )

    if abgebrochen or rueckgabe == -9:
        # M3: −9 (SIGKILL) differenzieren — RLIMIT_CPU mit soft == hard tötet
        # das Kind per SIGKILL, **deutlich vor** der Wall-Clock-Frist; der
        # Parent-Kill (Backstop) bzw. ein Gast-Ende erst nahe am Budget ist
        # ein echtes Timeout. Schwelle 90 %: Start-Overhead (pyarrow-Import)
        # geht in die Dauer ein, daher nicht < timeout_s, sondern < 90 %.
        cpu_kill = (
            rueckgabe == -9
            and not abgebrochen
            and dauer < float(limits["timeout_s"]) * 0.9
        )
        if cpu_kill:
            return SandboxErgebnis(
                ok=False,
                error_kind="cpu",
                meldung=(
                    "CPU-Zeitlimit überschritten: Die Strategie verbrauchte mehr als "
                    f"{int(limits['cpu_s'])} s Rechenzeit (RLIMIT_CPU) und wurde gestoppt."
                ),
                **basis,  # type: ignore[arg-type]
            )
        return SandboxErgebnis(
            ok=False,
            error_kind="timeout",
            meldung=(
                "Zeitlimit überschritten: Die Strategie lief länger als "
                f"{float(limits['timeout_s']):.0f} s (Verdacht: Endlosschleife) "
                "und wurde abgebrochen."
            ),
            **basis,  # type: ignore[arg-type]
        )
    if rueckgabe == -24:  # SIGXCPU — RLIMIT_CPU (soft = hard) greift
        return SandboxErgebnis(
            ok=False,
            error_kind="cpu",
            meldung=(
                "CPU-Zeitlimit überschritten: Die Strategie verbrauchte mehr als "
                f"{int(limits['cpu_s'])} s Rechenzeit (RLIMIT_CPU) und wurde gestoppt."
            ),
            **basis,  # type: ignore[arg-type]
        )
    if rueckgabe == -25:  # SIGXFSZ — RLIMIT_FSIZE greift
        return SandboxErgebnis(
            ok=False,
            error_kind="dateigroesse",
            meldung=(
                "Dateigrößenlimit überschritten: Die Strategie versuchte, mehr "
                f"als {int(limits['fsize_bytes']) // (1024 * 1024)} MB zu schreiben "
                "(RLIMIT_FSIZE)."
            ),
            **basis,  # type: ignore[arg-type]
        )
    if "MemoryError" in stderr:
        return SandboxErgebnis(
            ok=False,
            error_kind="speicher",
            meldung=(
                "Speicherlimit überschritten: Die Strategie benötigte mehr als "
                f"{int(limits['memory_bytes']) // (1024 * 1024)} MB (RLIMIT_AS) "
                "und wurde gestoppt."
            ),
            **basis,  # type: ignore[arg-type]
        )
    schwanz = (stderr or "").strip().splitlines()[-5:]
    details = "\n".join(schwanz)
    return SandboxErgebnis(
        ok=False,
        error_kind="intern",
        meldung=(
            "Interner Sandbox-Fehler "
            f"(Rückgabecode {rueckgabe})."
            + (f"\n{details}" if details else "")
        ),
        **basis,  # type: ignore[arg-type]
    )
