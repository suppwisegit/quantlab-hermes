# QuantLab Hermes — Pi-Deployment

MCP-Server (`python -m quantlab.hermes.mcp_server`, stdio) für **Hermes**, den
autonomen Trading-Agenten der Planspiel-Börse auf Raspberry Pi 4 (4 GB).
Stil und Aufbau angelehnt an `mobile/README.md`. Fachlicher Hintergrund:
`.agent-team/PLAN.md` (Welle H, E-H01–E-H05).

---

## Überblick

```
Raspberry Pi 4
├── Hermes-Agent (eigener LLM, MCP-Host, startet den Server als stdio-Child)
│   └── stdio-MCP → python -m quantlab.hermes.mcp_server
│       ├── 20 Tools (Planspiel lesend + platziere_order, Daten,
│       │   Screener/Indikatoren/Metriken/News) — quantlab/hermes/registry.py
│       └── Instructions = HERMES_SKILL (Spielregeln/Playbook/Autonomie)
│           — kommt automatisch mit dem MCP-Handshake
└── Datenhaltung: SQLite (QUANTLAB_BAR_STORE=sqlite) in QUANTLAB_DATA_DIR
```

- **Tools:** das Profil `HERMES_TOOL_NAMEN` (registry.py) umfasst 20 Tools —
  Planspiel (`planspiel_status` … `platziere_order`), Daten (`get_market_data`,
  `download_data`, `get_quote`, `get_fundamentals`, `search_symbols`) und
  Analyse (`get_indicators`, `run_screener`, `compute_metrics`, `news_suche`,
  `web_lesen`). Backtest/Portfolios/Alerts/Subagent sind bewusst draußen
  (`VERBOTENE_TOOL_NAMEN`, registry.py).
- **Skill:** `HERMES_SKILL` (skill.py) wird als `instructions` im
  initialize-Handshake ausgeliefert (mcp_server.py, `baue_server`) — der
  Hermes-Host muss nichts extra konfigurieren.
- **Wire-Protokoll:** stdout ist JSON-RPC-Transport; der Server loggt
  ausschließlich nach stderr (mcp_server.py, `_konfiguriere_logging`).
  Niemals eigene Ausgaben auf stdout des Prozesses umlenken.
- **Datenhaltung auf dem Pi:** `QUANTLAB_BAR_STORE=sqlite` (Dispatcher in
  `storage/parquet_store.py`, Welle M2b) — kein pyarrow nötig; die SQLite-DB
  `quantlab.db` liegt in `QUANTLAB_DATA_DIR` (`config.py`, `get_database_path`).
- **Optionaler Recorder-Tick:** Env-Gate `QUANTLAB_HERMES_RECORDER` (Welle H4,
  Intervall `QUANTLAB_HERMES_RECORDER_INTERVALL_S`) zeichnet bei Bedarf
  Live-1m-Bars im Server-Prozess auf.

---

## Installation auf Raspberry Pi 4

### System-Voraussetzungen

| Werkzeug | Version | Hinweis |
|---|---|---|
| Raspberry Pi OS (64-bit, Bookworm) | — | ARM64-Wheels für pandas/numpy/scipy vorhanden |
| Python | ≥ 3.13 | `pyproject.toml`: `requires-python = ">=3.13"` |
| uv | aktuell | `curl -LsSf https://astral.sh/uv/install.sh | sh` |
| git | — | Repo holen |
| ~2 GB freier Platz | — | venv inkl. pandas/numpy/scipy/sklearn |

### Schritte

```bash
# 1. uv installieren (falls nicht vorhanden)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env        # oder Shell neu öffnen

# 2. Repo bringen und Python 3.13 bereitstellen
git clone <repo-url> ~/QuantLab
cd ~/QuantLab/backend
uv python install 3.13         # falls System-Python < 3.13

# 3. Abhängigkeiten installieren
uv sync
```

**Abhängigkeiten (`backend/pyproject.toml`):** reguläre Dependencies sind u. a.
`fastapi`, `uvicorn[standard]`, `yfinance>=0.2.54`, `httpx`, `pandas>=2.2`,
`numpy>=2.2`, `scipy`, `statsmodels`, `scikit-learn`, `pyarrow>=19`, `openai`,
`keyring`, `ddgs`, `trafilatura`, `mcp>=1.26`. **yfinance ist eine reguläre
Dependency** — kein optionaler Schritt. `uv sync` installiert außerdem die
Default-Group `dev` (pytest, pytest-asyncio, httpx, ruff); wer sie nicht will:
`uv sync --no-dev`.

**Pi-Hinweis:** `scipy`/`scikit-learn`/`statsmodels`/`pyarrow` werden von
`uv sync` mitinstalliert (reguläre Deps), sind aber im **Hermes-Pfad nicht
Pflicht**: das Profil nutzt vom Quant-Modul nur `compute_metrics`
(numpy+pandas), und mit `QUANTLAB_BAR_STORE=sqlite` wird pyarrow nie geladen
(lazy Import, `storage/parquet_store.py`). Der SQLite-Bar-Store ist auf dem Pi
empfohlen (weniger RAM/Platz, kein Parquet-File-Zoo).

Kurzer Smoke-Test nach der Installation:

```bash
cd ~/QuantLab/backend
uv run python -c "from quantlab.hermes.registry import hermes_specs; print(len(hermes_specs()))"
# → 20
```

(Achtung: `hermes_specs()` ohne vorheriges `registriere_hermes_tools()` liefert
die Tools nur, wenn die Tool-Module schon importiert wurden — der Import von
`quantlab.hermes.registry` erledigt das. Der MCP-Server ruft das Pruning in
`main()` selbst.)

---

## Einrichtung Planspiel-Börse (Checkliste)

Alle Schritte headless ohne UI, mit `uv run python` im Verzeichnis
`~/QuantLab/backend`. Die Env `QUANTLAB_DATA_DIR` sollte dabei bereits gesetzt
sein (identisch zur späteren Server-Umgebung), damit Settings/DB/Fallback-Datei
am richtigen Ort landen:

```bash
export QUANTLAB_DATA_DIR=/home/pi/quantlab-daten
export QUANTLAB_BAR_STORE=sqlite
```

1. **Datenverzeichnis + Migrationen einmalig anlegen** (der MCP-Server tut
   dies in `main()` selbst — hier für die Einrichtung):

   ```bash
   uv run python -c "
   from quantlab.config import ensure_directories
   from quantlab.migrations import run_migrations
   ensure_directories(); run_migrations()
   print('ok')"
   ```

2. **E-Mail hinterlegen** — Settings-Sektion `planspiel`, Key `email`
   (`services/planspiel/__init__.py`, `anmeldeinfos_aus_settings`; Defaults in
   `storage/repos/settings_repo.py`):

   ```bash
   uv run python -c "
   from quantlab.storage.repos import settings_repo
   settings_repo.update_section('planspiel', {'email': 'team@example.org'})
   print(settings_repo.get_section('planspiel')['email'])"
   ```

3. **Passwort hinterlegen** — Keyring vor Fallback-Datei
   (`settings_repo.planspiel_passwort_save`):

   ```bash
   uv run python -c "
   from quantlab.storage.repos import settings_repo
   settings_repo.planspiel_passwort_save('GEHEIM')"
   ```

   Ohne funktionierenden Keyring (kein Secret-Service auf dem Pi) schreibt der
   Aufruf automatisch die Fallback-Datei
   `$QUANTLAB_DATA_DIR/secrets_planspiel.json` mit chmod **0600**
   (`settings_repo._fallback_save` mit `os.open(…, 0o600)`; Pfad über
   `_planspiel_fallback_pfad()` → `get_data_dir() / "secrets_planspiel.json"`).
   Manuell anlegen geht genauso (gleicher Dateiname/Format), danach:

   ```bash
   chmod 600 /home/pi/quantlab-daten/secrets_planspiel.json
   ```

   Dateiformat: `{ "planspiel_password": "GEHEIM" }` (Keyring-Account
   `planspiel_password`, Service `QuantLab` — bewusst getrennt vom z.ai-Key
   in `secrets.json`).

4. **Verbindungstest** gegen die echte Börse (Login; der Router-Endpunkt
   `POST /planspiel/verbindung_testen` ruft dieselbe Kette):

   ```bash
   uv run python -c "
   from quantlab.services.planspiel import hole_client
   user = hole_client().authentifizieren()
   print(user.get('username') or user.get('email'))
   print([(d.get('id'), d.get('depotName')) for d in user.get('userPortfolios') or []])"
   ```

   Ausgabe: Benutzername + Liste der Depots mit IDs (für Schritt 5).

5. **Depot-Rollen markieren** — Keys `standard_depot_id` (Trainingsdepot) und
   `wettbewerbs_depot_id` (Wettbewerbsdepot), Default `0` = nicht markiert
   (Defaults in `settings_repo.py`, Sektion `planspiel`):

   ```bash
   uv run python -c "
   from quantlab.storage.repos import settings_repo
   settings_repo.update_section('planspiel', {'standard_depot_id': 1234, 'wettbewerbs_depot_id': 5678})
   print(settings_repo.get_section('planspiel'))"
   ```

   Ohne `standard_depot_id` greift der Lesefallback „erstes Depot aus
   getUser" (`api/routers/planspiel.py`, `_loese_depot_id`) — für den
   Dauerbetrieb sollten beide Rollen explizit gesetzt sein.

6. **Universum importieren** (Marktgruppen → Tabelle `psb_instrumente`;
   `services/planspiel/universum.py`, `importiere_universum`):

   ```bash
   uv run python -c "
   from quantlab.agent.tools.daten import loese_isin_zu_kuerzel
   from quantlab.services.planspiel import hole_client
   from quantlab.services.planspiel.universum import importiere_universum
   print(importiere_universum(hole_client(), kuerzel_aufloeser=loese_isin_zu_kuerzel))"
   ```

   Der Client pacingt selbst (≥ 1 s je Request) — der Lauf dauert real ~20 s
   (Modulkopf universum.py).

7. **Byteworx-5Y-Backfill** (Tageshistorie fürs gesamte Universum;
   `services/planspiel/backfill.py`):

   ```bash
   uv run python -c "
   from quantlab.providers.byteworx import ByteworxProvider
   from quantlab.services.planspiel import backfill
   print(backfill.fuehre_backfill_aus(ByteworxProvider()))"
   ```

   Hinweis Laufzeit: streng serial mit ≥ 0,5 s Pacing je ISIN
   (`PACING_ANNAHME_S`) — bei ~800 ISINs mindestens ~7 Minuten **allein an
   Pacing**, Netzzeit kommt on top (Modulkopf backfill.py). Der Lauf ist
   **resumabel** (Fortschritt je ISIN in `psb_backfill_fortschritt`); ein
   Abbruch verliert nur die gerade laufende ISIN, Fehlversuche werden je ISIN
   bis 3 gezählt und dann übersprungen.

---

## Hermes-Anbindung (mcpServers)

Beispiel für die MCP-Host-Konfiguration des Hermes-Agenten:

```json
{
  "mcpServers": {
    "quantlab": {
      "command": "python",
      "args": ["-m", "quantlab.hermes.mcp_server"],
      "cwd": "/home/pi/QuantLab/backend",
      "env": {
        "QUANTLAB_DATA_DIR": "/home/pi/quantlab-daten",
        "QUANTLAB_BAR_STORE": "sqlite",
        "QUANTLAB_HERMES_RECORDER": "1",
        "QUANTLAB_HERMES_RECORDER_INTERVALL_S": "60"
      }
    }
  }
}
```

- `command`/`args`: der Server-Start ist `python -m quantlab.hermes.mcp_server`
  (mcp_server.py, `main`); `cwd` auf `backend/` (Package-Root, venv aktiv oder
  `command": "uv", "args": ["run", "python", "-m", …]` als Variante).
- `QUANTLAB_HERMES_RECORDER` weglassen oder leeren, wenn kein Live-Recorder
  gewünscht ist (Welle H4).
- **Skill braucht keine Konfiguration:** `HERMES_SKILL` kommt automatisch als
  `instructions` mit dem initialize-Handshake (mcp_server.py).

---

## Dauerbetrieb (systemd)

Der MCP-Server ist ein stdio-**Child** des Hermes-Hosts — er braucht **keine
eigene Unit**. Die Unit verwaltet den Hermes-Host-Prozess (VORLAGE —
ExecStart an den echten Host-Befehl anpassen), z. B.
`/etc/systemd/system/hermes.service`:

```ini
[Unit]
Description=Hermes Trading-Agent (MCP-Host, Planspiel-Börse 2026/27)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/QuantLab/backend
Environment=QUANTLAB_DATA_DIR=/home/pi/quantlab-daten
Environment=QUANTLAB_BAR_STORE=sqlite
Environment=QUANTLAB_HERMES_RECORDER=1
Environment=QUANTLAB_HERMES_RECORDER_INTERVALL_S=60
# VORLAGE: ExecStart auf den echten Hermes-Host zeigen
ExecStart=/home/pi/QuantLab/backend/.venv/bin/python -m quantlab.hermes.mcp_server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hermes
journalctl -u hermes -f          # Logs (stderr des Hosts + Server-Logs)
```

**Log-Hinweis:** stdout des Server-Prozesses ist das MCP-Wire-Protokoll —
in Logs/Journal nur stderr auswerten (mcp_server.py loggt via
`logging.basicConfig(stream=sys.stderr)`).

---

## Betriebsregeln (Saison 2026/27)

Alle Zahlen aus `services/planspiel/regeln.py` (Single Source of Truth; der
Skill interpoliert sie automatisch — skill.py):

| Regel | Wert | Konstante |
|---|---|---|
| Saison | erster Handelstag 01.10.2026, Spielende 25.01.2027 | Skill/skill.py |
| Gebühren | 0,3 % vom Kurswert, min. 15 € je Order | `GEBUEHR_PROZENT`, `MIN_GEBUEHR_EUR` |
| Krypto (nur Trainingsdepot) | 0,35 %, min. 0,01 € | `KRYPTO_PROZENT`, `KRYPTO_MIN_EUR` |
| Order-Storno | 2,50 € | `STORNO_EUR` |
| Depotgewicht je Wertpapier | max. 20 % (inkl. offener Käufe) | `MAX_DEPOTGEWICHT` |
| Orders je Tag und Wertpapier | max. 5 (abgerechnete) | `MAX_ORDERS_JE_TAG` |
| Penny-Sperre | Kauf unter 1,00 € nicht möglich (Verkauf erlaubt) | `PENNY_GRENZE_EUR` |
| Mindestaktivität | ≥ 3 abgerechnete Käufe je Kalendermonat, sonst Ausschluss wegen Passivität | `MIN_KAEUFE` |
| Handelsfenster | 08:00–22:00 Uhr Europe/Berlin | `HANDELSZEIT_START_STUNDE`, `HANDELSZEIT_ENDE_STUNDE`, `HANDELSZEIT_ZONE` |

- **Wochenende:** Handel ruht Wochenende/Feiertage (Skill). Orders außerhalb
  des Fensters bleiben platziert und werden zur Marktöffnung ausgeführt
  (regeln.py, Konstanten-Kommentar) — d. h. spät freitags platzierte Orders
  laufen erst zur nächsten Öffnung an.
- **Guards:** `platziere_order` prüft Kapital und Regeln vorab
  (`regeln.pruefe_order_vorab`: Penny-Sperre, 20-%-Gewicht, 5-Orders-Limit,
  Mindestaktivität) — **nie umgehen**; eine abgelehnte Order ist eine
  Information (Skill, Autonomie-Regeln). Wettbewerbsdepot nur mit
  `bestaetigung_wettbewerb=true` und gesetzter `wettbewerbs_depot_id`.
- Am Saisonende (25.01.2027) die Handelsaktivität einstellen (Skill).

---

## Fehlersuche

Fachfehler des Planspiel-Clients (`services/planspiel/fehler.py`,
`STATUS_NACH_CODE`) — der Code ist stabil und maschinenlesbar:

| Code | HTTP | Bedeutung / typische Ursache |
|---|---|---|
| `nicht_konfiguriert` | 409 | E-Mail/Passwort nicht hinterlegt (Checkliste Schritt 2/3) |
| `login_fehlgeschlagen` | 401 | authenticate mit falschen Credentials |
| `nicht_eingeloggt` | 401 | auch nach dem einen automatischen Re-Login kein Zugriff — Sitzung abgelauben/invalid |
| `vertrag_geaendert` | 502 | Antwort-Format der Börse weicht ab (API-Änderung) |
| `netzwerk` | 503 | Timeout/ConnectError — Börse nicht erreichbar |
| `rate_limit` | 429 | HTTP 429/503 der Börse — Client-Pacing erzwingt Entlastung |
| `order_abgelehnt` | 422 | `validationErrorCode` der placeOrder-Antwort (z. B. Regel-/Kapitalverstoß, s. Betriebsregeln) |
| `order_fehler` | 422 | sonstiger 4xx auf placeOrder |
| `protokoll_verletzt` | 500 | interner Whitelist-Guard: verbotener Endpunkt angefragt — darf im Betrieb nicht auftreten (Bug-Indikator) |

Typische Erste-Hilfe:

- `nicht_konfiguriert` → Verbindungstest-Schnipsel (Schritt 4) laufen lassen;
  Ausgabe von `anmeldeinfos_aus_settings()` prüfen (liefert `None`, wenn E-Mail
  ODER Passwort fehlt).
- `login_fehlgeschlagen` → Passwort neu setzen (Schritt 3; überschreibt
  Keyring UND Fallback-Datei).
- `rate_limit` → laufen lassen; der Client pacingt selbst (≥ 1 s je Request).
- Server startet nicht / Tools fehlen → `journalctl -u hermes`, stderr auf
  Import-/Migrationsfehler prüfen; Smoke-Test aus der Installation (20 Tools).

---

## Bewusst nicht / Grenzen

- Kein Backtest/PCA/VaR, keine Portfolio-Verwaltung, keine Subagent-Tools im
  Profil (`VERBOTENE_TOOL_NAMEN`, registry.py) — Pi-Ressourcen & Scope.
- Keine Order-Storno-/Depot-Reset-Tools — gesperrt (PLAN Welle H).
- Nur stdio-Transport; ein eigener HTTP/SSE-MCP-Transport ist nicht vorgesehen.
- Das LLM/die Entscheidung liegt beim Hermes-Host — der Server liefert nur
  Tools + Skill.
