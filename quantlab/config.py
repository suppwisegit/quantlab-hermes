"""Pfad- und Konfigurationslogik des Sidecars.

Datenverzeichnis-Auflösung (in dieser Reihenfolge):
1. ``QUANTLAB_DATA_DIR`` (Env-Override; die Electron-Shell setzt ihn immer,
   siehe ``src/main/paths.ts``)
2. ``<Repo-Root>/data`` im Dev-Checkout (erkennbar am ``quantlab``-Paket)
3. XDG-Datenverzeichnis (``$XDG_DATA_HOME`` bzw. ``~/.local/share``) → ``QuantLab``
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "QuantLab"


def _default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "backend" / "quantlab" / "config.py").is_file():
        return repo_root / "data"
    xdg_data_home_env = os.environ.get("XDG_DATA_HOME", "").strip()
    xdg_data_home = (
        Path(xdg_data_home_env) if xdg_data_home_env else Path.home() / ".local" / "share"
    )
    return xdg_data_home / "QuantLab"


def get_data_dir() -> Path:
    """Aufgelöstes Datenverzeichnis (Override hat Vorrang)."""
    override = os.environ.get("QUANTLAB_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_data_dir()


def get_database_path() -> Path:
    """Pfad der SQLite-Datenbank (T2)."""
    return get_data_dir() / "quantlab.db"


def get_market_dir() -> Path:
    """Wurzel des Parquet-Speichers: ``market/<interval>/<SYMBOL>.parquet`` (D.2)."""
    return get_data_dir() / "market"


def get_secrets_path() -> Path:
    """Pfad der Fallback-Datei für den z.ai-API-Key (E010, chmod 0600)."""
    return get_data_dir() / "secrets.json"


def ensure_directories() -> Path:
    """Legt Datenverzeichnis + ``market/`` an (idempotent) und liefert das Datenverzeichnis."""
    market_dir = get_market_dir()
    market_dir.mkdir(parents=True, exist_ok=True)
    return market_dir.parent
