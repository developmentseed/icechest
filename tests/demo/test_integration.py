"""End to end against live services. Run with: uv run pytest -m network

Parametrised by access mode. ``s3`` is the fast path and works only from inside
AWS ``us-west-2`` -- LP DAAC's ``lp-prod-protected`` bucket enforces same-region
access, so from anywhere else the credentials mint fine and every read is then
denied, and that case skips rather than failing. ``https`` works from anywhere.

Either way the references written are the same relative ``vcc://`` form, so
these two runs differ only in how the bytes are fetched, not in what is stored.
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
    f"LP DAAC's lp-prod-protected bucket is readable over s3 only from AWS "
    f"{LPDAAC_REGION}, and the first asset read was denied. This limits the "
    f"ingest, not the store: references are recorded relative to the container, "
    f"so run with access='https' to build the same store from here."
)


def require_asset_access(rows, registry, access):
    """Skip when the bucket's region lock is what stands in the way.

    Only an access denial on the first asset read is treated that way; anything
    else is a real failure and is re-raised.
    """
    url = next(iter(asset_urls(rows.to_pylist()[0], access=access).values()))
    try:
        read_header(url, registry)
    except Exception as error:
        # Only S3's denial. A missing object, a network fault or an expired
        # token is a real failure and must not be dressed up as a skip.
        text = str(error).lower()
        if "accessdenied" in text or "access denied" in text:
            pytest.skip(f"{OUT_OF_REGION} ({error})")
        raise


@pytest.mark.parametrize("access", ["s3", "https"])
def test_three_real_granules_land_in_one_commit(tmp_path, access):
    archive = open_archive()
    rows = select_granules(archive, limit=3)
    assert rows.num_rows == 3
    assert all(t is not None for t in rows["proj:transform"].to_pylist())

    registry = object_store_registry(access)
    require_asset_access(rows, registry, access)

    repo = open_store(tmp_path, access=access)
    ensure_table(repo, archive.schema())
    before = len(list(repo.repo.ancestry(branch="main")))

    result = ingest_batch(repo, rows, registry=registry, access=access)

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


@pytest.mark.parametrize("access", ["s3", "https"])
def test_virtual_chunks_read_back_as_real_pixels(tmp_path, access):
    """The point of the whole exercise: the references resolve to data."""
    archive = open_archive()
    rows = select_granules(archive, limit=1)
    registry = object_store_registry(access)
    require_asset_access(rows, registry, access)

    repo = open_store(tmp_path, access=access)
    ensure_table(repo, archive.schema())
    result = ingest_batch(repo, rows, registry=registry, access=access)

    granule = result.committed[0]
    array = repo.read("main").group[f"{granule}/B04/multiscales/0"]
    window = array[:16, :16]
    assert window.shape == (16, 16)
