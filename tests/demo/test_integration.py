"""End to end against live services. Run with: uv run pytest -m network"""

from __future__ import annotations

import pytest

from icechest.demo.credentials import object_store_registry
from icechest.demo.ingest import ingest_batch
from icechest.demo.source import open_archive, select_granules
from icechest.demo.store import ensure_table, open_store

pytestmark = pytest.mark.network


def test_three_real_granules_land_in_one_commit(tmp_path):
    archive = open_archive()
    rows = select_granules(archive, limit=3)
    assert rows.num_rows == 3
    assert all(t is not None for t in rows["proj:transform"].to_pylist())

    repo = open_store(tmp_path)
    ensure_table(repo, archive.schema())
    before = len(list(repo.repo.ancestry(branch="main")))

    result = ingest_batch(repo, rows, registry=object_store_registry())

    # One commit for the whole batch: to_icechunk stages into the session
    # rather than committing it.
    after = list(repo.repo.ancestry(branch="main"))
    assert len(after) == before + 1
    assert after[0].id == result.snapshot_id

    snap = repo.read("main")
    table = snap.table("granules").scan().to_arrow()
    assert set(table["id"].to_pylist()) == set(result.committed)

    granule = result.committed[0]
    band = snap.group[granule]["B04"]
    assert "spatial:transform" in dict(band.attrs)
    assert dict(band.attrs)["proj:code"].startswith("EPSG:")
    levels = band["multiscales"]
    assert len(list(levels)) >= 1


def test_virtual_chunks_read_back_as_real_pixels(tmp_path):
    """The point of the whole exercise: the references resolve to data."""
    archive = open_archive()
    rows = select_granules(archive, limit=1)
    repo = open_store(tmp_path)
    ensure_table(repo, archive.schema())
    result = ingest_batch(repo, rows, registry=object_store_registry())

    granule = result.committed[0]
    array = repo.read("main").group[f"{granule}/B04/multiscales/0"]
    window = array[:16, :16]
    assert window.shape == (16, 16)
