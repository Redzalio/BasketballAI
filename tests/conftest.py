"""Shared pytest fixtures for the HoopTracker test suite.

These tests lock in the *pure logic* of the torch-free modules
(detection.shot_logic, stats.consistency, stats.db). They never load a
YOLO/torch model, touch a camera/GPU/network, or write to the real database.

The repo root is put on sys.path here so `from detection ... import` and
`from stats ... import` work no matter where pytest is invoked from.
"""
import sys
from pathlib import Path

import pytest

# --- make the package importable regardless of CWD --------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Database fixture: redirect stats.db to a throwaway DB under tmp_path.
#
# CRITICAL: stats.db keeps the database location in the module global
# `DB_PATH`, and every connection reads it *fresh* at call time
# (see stats/db.py::_conn). So monkeypatching that global, then calling
# init_db(), makes ALL subsequent operations hit the temp file. The real
# data/hooptracker.db is never opened.
#
# (Importing stats.db runs init_db() once against the real path -- that is a
#  harmless `CREATE TABLE IF NOT EXISTS` and is unavoidable on import. Our test
#  *operations* all go through the monkeypatched path below.)
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Yield the stats.db module wired to an isolated temp database."""
    from stats import db

    test_db_path = tmp_path / "hooptracker_test.db"
    real_path = db.DB_PATH

    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()  # build the schema in the temp DB

    # Safety net: confirm we are NOT pointed at the real database.
    assert db.DB_PATH != real_path
    assert db.DB_PATH == test_db_path

    yield db
    # monkeypatch auto-restores db.DB_PATH; tmp_path is cleaned up by pytest.
    # (We intentionally do NOT os.unlink the file -- on Windows the sqlite
    #  handle can linger briefly and pytest's tmp cleanup tolerates it.)
