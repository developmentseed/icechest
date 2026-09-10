"""End to end against live services. Run with: uv run pytest -m network

These tests only work from inside AWS ``us-west-2``. LP DAAC's
``lp-prod-protected`` bucket enforces same-region access, so from anywhere else
the Earthdata credentials mint fine and every COG read is then denied. That is
a property of the source data, not of icechest, so a run from the wrong place
skips rather than failing.
"""

from __future__ import annotations

import pytest

from icechest.demo.assets import asset_urls
from icechest.demo.credentials import LPDAAC_REGION, object_store_registry
from icechest.demo.ingest import ingest_batch
from icechest.demo.source import open_archive, select_granules
from icechest.demo.store import ensure_table, open_store
from icechest.demo.virtualize import read_header

pytestmark = pytest.mark.network

OUT_OF_REGION = (
    f"LP DAAC's lp-prod-protected bucket is readable only from AWS "
    f"{LPDAAC_REGION}; the first asset read was denied. The same applies to any "
    f"reader of a store published from here, because s3://lp-prod-protected/... "
    f"is the URL recorded in every virtual reference."
)


def require_asset_access(rows, registry):
    """Skip when the bucket's region lock is what stands in the way.

    Only an access denial on the first asset read is treated that way; anything
    else is a real failure and is re-raised.
    """
    url = next(iter(asset_urls(rows.to_pylist()[0]).values()))
    try:
        read_header(url, registry)
    except Exception as error:
        # Only S3's denial. A missing object, a network fault or an expired
        # token is a real failure and must not be dressed up as a skip.
        text = str(error).lower()
        if "accessdenied" in text or "access denied" in text:
            pytest.skip(f"{OUT_OF_REGION} ({error})")
        raise


def test_three_real_granules_land_in_one_commit(tmp_path):
    archive = open_archive()
    rows = select_granules(archive, limit=3)
    assert rows.num_rows == 3
    assert all(t is not None for t in rows["proj:transform"].to_pylist())

    registry = object_store_registry()
    require_asset_access(rows, registry)

    repo = open_store(tmp_path)
    ensure_table(repo, archive.schema())
    before = len(list(repo.repo.ancestry(branch="main")))

    result = ingest_batch(repo, rows, registry=registry)

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
    registry = object_store_registry()
    require_asset_access(rows, registry)

    repo = open_store(tmp_path)
    ensure_table(repo, archive.schema())
    result = ingest_batch(repo, rows, registry=registry)

    granule = result.committed[0]
    array = repo.read("main").group[f"{granule}/B04/multiscales/0"]
    window = array[:16, :16]
    assert window.shape == (16, 16)
