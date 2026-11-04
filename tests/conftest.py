"""Gemeinsame Fixtures: isoliertes Datenverzeichnis (tmp_path) + FastAPI-TestClient.

Alle Backend-Tests sind netzfrei (E013); spätere Fixtures (Fake-Provider,
Fake-LLM) kommen hier dazu.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from fake_provider import FakeDownloadProvider


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isoliertes Datenverzeichnis via QUANTLAB_DATA_DIR-Override."""
    target = tmp_path / "daten"
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(target))
    return target


@pytest.fixture()
def client(data_dir: Path) -> Iterator[TestClient]:
    """TestClient mit gesetztem Datenverzeichnis-Override (App erst im Fixture erzeugen)."""
    from quantlab.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# T7 (Projekt-System): additive Fixtures — Bewahrung des Bestands (T1/T2-Vertrag)
# ---------------------------------------------------------------------------


@pytest.fixture()
def t7_project(client: TestClient) -> dict:
    """Legt ein Projekt über die API an — Basis für Task-/Notiz-/Kontext-/Artefakt-Tests."""
    response = client.post(
        "/api/projects", json={"name": "T7-Testprojekt", "description": "Fixture-Projekt"}
    )
    assert response.status_code == 201, response.text
    project: dict = response.json()
    return project


# ---------------------------------------------------------------------------
# T8 (Quant-Engine): deterministische Symbol-Universe im Parquet-Store
# ---------------------------------------------------------------------------

#: (seed, Tage, Start-Offset in Tagen) je Test-Symbol — AAA/BBB/BENCH volle
#: Historie, CCC startet 150 Tage später (Alignment-Schnittmenge), KURZ hat
#: nur 2 Balken („zu wenig Daten“-Fall).
_T8_UNIVERSE: dict[str, tuple[int, int, int]] = {
    "AAA": (101, 400, 0),
    "BBB": (202, 400, 0),
    "CCC": (303, 250, 150),
    "KURZ": (404, 2, 0),
    "BENCH": (505, 400, 0),
}


@pytest.fixture()
def t8_quant_universe(client: TestClient) -> dict[str, int]:
    """Seedet deterministische Kurspfade (1d) in den Store; liefert #Balken je Symbol.

    Nutzt den ``client`` (Lifespan → Migrationen) und schreibt über
    ``parquet_store.upsert_bars`` — netzfrei, aus ``numpy``-Seeds reproduzierbar.
    """
    from datetime import UTC, datetime, timedelta

    import numpy as np
    import pandas as pd

    from quantlab.storage import parquet_store

    seeded: dict[str, int] = {}
    for symbol, (seed, days, offset) in _T8_UNIVERSE.items():
        rng = np.random.default_rng(seed)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, days)))
        start = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=offset)
        timestamps = [
            int((start + timedelta(days=i)).timestamp() * 1_000_000) for i in range(days)
        ]
        opens = np.concatenate(([100.0], closes[:-1]))
        bars = pd.DataFrame(
            {
                "ts": pd.Series(timestamps, dtype="int64"),
                "open": pd.Series(opens, dtype="float64"),
                "high": pd.Series(np.maximum(opens, closes) + 1.0, dtype="float64"),
                "low": pd.Series(np.minimum(opens, closes) - 1.0, dtype="float64"),
                "close": pd.Series(closes, dtype="float64"),
                "volume": pd.Series([1000.0] * days, dtype="float64"),
            }
        )
        parquet_store.upsert_bars(symbol, "1d", bars)
        seeded[symbol] = days
    return seeded


# ---------------------------------------------------------------------------
# T10 (Backtesting): synthetischer Marktdaten-Bestand für API-Tests (additiv)
# ---------------------------------------------------------------------------

_US_PER_DAY = 86_400_000_000


def _t10_daily_ts(n: int) -> np.ndarray:
    """n Tages-Bars ab 2023-01-02, Mitternacht UTC (int64 µs, E006)."""
    base = int(datetime(2023, 1, 2, tzinfo=UTC).timestamp()) * 1_000_000
    return base + np.arange(n, dtype="int64") * _US_PER_DAY


def t10_synth_ohlcv(
    n: int = 260, *, seed: int = 42, mu: float = 0.0008, sigma: float = 0.01
) -> pd.DataFrame:
    """Deterministischer Random-Walk mit Aufwärtstrend im kanonischen Schema."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    opens = np.concatenate(([100.0], closes[:-1]))
    return pd.DataFrame(
        {
            "ts": _t10_daily_ts(n),
            "open": pd.Series(opens, dtype="float64"),
            "high": pd.Series(np.maximum(opens, closes) * 1.001, dtype="float64"),
            "low": pd.Series(np.minimum(opens, closes) * 0.999, dtype="float64"),
            "close": pd.Series(closes, dtype="float64"),
            "volume": pd.Series(np.full(n, 1_000.0), dtype="float64"),
        }
    )


@pytest.fixture()
def t10_market(data_dir: Path) -> dict[str, object]:
    """Seedet 260 Tages-Bars für „TREND“ in den Parquet-Store (netzfrei, Seed 42)."""
    from quantlab.storage import parquet_store

    bars = t10_synth_ohlcv(260)
    parquet_store.upsert_bars("TREND", "1d", bars)
    return {"symbol": "TREND", "interval": "1d", "bars": len(bars)}


# ---------------------------------------------------------------------------
# T4 (yfinance-Provider & Download-Pipeline): additive Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def t4_db(data_dir: Path) -> Path:
    """Datenverzeichnis + hochgezogenes Schema (Service-Tests ohne App-Lifespan)."""
    from quantlab.migrations import run_migrations

    run_migrations()
    return data_dir


@pytest.fixture()
def t4_fake_provider(monkeypatch: pytest.MonkeyPatch) -> FakeDownloadProvider:
    """Ersetzt ``downloads.get_provider`` durch den Fake — netzfrei (E013).

    Wichtig für API-Tests: die Endpunkte holen den Provider **zur Laufzeit**
    über :func:`quantlab.services.downloads.get_provider`, daher greift der
    Patch auch für bereits importierte Router.
    """
    from fake_provider import FakeDownloadProvider

    from quantlab.services import downloads

    fake = FakeDownloadProvider()
    monkeypatch.setattr(downloads, "get_provider", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# T18 (v2-Repos): vollständig migriertes Schema für Repo-Unit-Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo_db(data_dir: Path) -> Path:
    """Datenverzeichnis + vollständig migriertes v2-Schema (Repo-Tests ohne App).

    Äquivalent zum ``settings_db``-Muster: Produktion ruft ``run_migrations``
    im Lifespan; Repo-Tests heben das Schema direkt an (Migration 1–3).
    """
    from quantlab.migrations import run_migrations

    run_migrations()
    return data_dir


# ---------------------------------------------------------------------------
# M1 (Mikro-Fix-Runde): Suite-Hygiene — autouse, minimal-additiv
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stooq_status_frisch() -> None:
    """Setzt den prozessglobalen Stooq-Provider-Status vor jedem Test zurück.

    ``stooq_provider`` hält den Zustand (``ok``/``limit``) im Modul — ein
    Test, der ein Tageslimit/eine Blockade erkennt, würde sonst in jeden
    Nachfolger-Test hineinbluten (``/api/data/provider-status`` zeige dann
    „limit“). Rein additiv (M1): keine bestehende Fixture wird verändert.
    """
    from quantlab.providers import stooq_provider

    stooq_provider._zustand_ok()
