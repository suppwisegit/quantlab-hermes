# quantlab-hermes

MCP-Toolset für einen autonomen Trading-Agenten (»Hermes«) auf einem Raspberry Pi,
der datenbasiert und dauerhaft bei der [Planspiel-Börse](https://planspiel-boerse.de) handelt.

Enthält **nur den Agenten-Teil von QuantLab** — kein Electron-Frontend, keine Mobile-App:

- `quantlab/hermes/mcp_server.py` — stdio-MCP-Server (`python -m quantlab.hermes.mcp_server`), 20 Tools + Skill-Instructions im Handshake
- `quantlab/hermes/registry.py` — kuratiertes Tool-Profil (Planspiel komplett inkl. `platziere_order` mit allen Guards, Kursdaten/Fundamentals, Screener, Indikatoren, `compute_metrics`, News)
- `quantlab/hermes/skill.py` — das Skill-Wissen (Spielregeln Saison 2026/27, Trading-Playbook, Autonomie-Regeln)
- `quantlab/hermes/recorder_tick.py` — optionaler Live-Kurs-Recorder per Env-Gate
- `DETAILS.md` — vollständige Deployment-Doku (Einrichtungs-Checkliste, Credentials, Universum-Import, Backfill, systemd, Fehlersuche)

## Schnellstart auf dem Raspberry Pi 4

```bash
# 1) uv installieren (falls nicht vorhanden)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Repo klonen und Abhängigkeiten installieren
git clone git@github.com:<dein-name>/quantlab-hermes.git
cd quantlab-hermes
uv sync            # installiert Python 3.13 + Deps (kein pyarrow/scipy/sklearn nötig)

# 3) Smoke-Test
uv run python -c "from quantlab.hermes.registry import HERMES_TOOL_NAMEN; print(len(HERMES_TOOL_NAMEN))"
# → 20
```

Dann die Einrichtungs-Checkliste in `DETAILS.md` (Abschnitt »Einrichtung Planspiel-Börse«) abarbeiten.

## Hermes-Anbindung (mcpServers)

```json
{
  "mcpServers": {
    "quantlab": {
      "command": "uv",
      "args": ["run", "python", "-m", "quantlab.hermes.mcp_server"],
      "cwd": "/home/pi/quantlab-hermes",
      "env": {
        "QUANTLAB_DATA_DIR": "/home/pi/quantlab-daten",
        "QUANTLAB_BAR_STORE": "sqlite",
        "QUANTLAB_HERMES_RECORDER": "1"
      }
    }
  }
}
```

## Tests

```bash
uv run pytest tests/ -q     # 23 Tests, netzfrei
```

## Herkunft

Extrahiert aus dem QuantLab-Monorepo (Welle H, Stand 2026-11-03, Review BESTANDEN —
Suite 2031 passed / 0 failed). Upstream: QuantLab Desktop-Repo. Bewusst hier enthalten:
der komplette `quantlab`-Python-Pfad, den das Tool-Profil importiert (Services, Storage,
Provider) — alles andere (Electron, mobile/, APK, Daten) wurde weggelassen.
