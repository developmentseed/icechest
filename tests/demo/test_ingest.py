"""Batch ingest: one commit, and a failed granule takes nothing with it."""

from __future__ import annotations

import pyarrow as pa
import pytest
import zarr
from pyiceberg.schema import Schema
from pyiceberg.types import IntegerType, ListType, NestedField, StringType

from icechest.demo.ingest import BatchFailed, ingest_batch
from icechest.demo.store import ensure_table, open_store
from icechest.demo.virtualize import GranuleError

SOURCE = Schema(
    NestedField(field_id=1, name="id", field_type=StringType(), required=False),
    NestedField(field_id=2, name="proj:epsg", field_type=IntegerType(), required=False),
    NestedField(
        field_id=3,
        name="proj:shape",
        field_type=ListType(
            element_id=4, element_type=IntegerType(), element_required=False
        ),
        required=False,
    ),
)


def rows(*ids):
    return pa.Table.from_pylist(
        [{"id": i, "proj:epsg": 32620, "proj:shape": [10, 10]} for i in ids],
        schema=SOURCE.as_arrow(),
    )


def fake_writer(failures=()):
    """Stands in for the real writer, staging a small real array per granule."""

    def writer(tx, row, *, registry, bands):
        if row["id"] in failures:
            raise GranuleError(f"{row['id']}: pretend the COG is unreadable")
        group = zarr.open_group(tx.session.store, path=f"/{row['id']}", mode="a")
        group.create_array("B04", shape=(4,), dtype="uint8", chunks=(4,))[:] = 1
        return f"/{row['id']}"

    return writer


def test_batch_lands_in_exactly_one_commit(tmp_path):
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    before = len(list(repo.repo.ancestry(branch="main")))

    result = ingest_batch(repo, rows("g1", "g2"), registry=None, writer=fake_writer())

    after = list(repo.repo.ancestry(branch="main"))
    assert len(after) == before + 1
    assert after[0].id == result.snapshot_id
    assert result.committed == ["g1", "g2"]
    assert result.skipped == {}


def test_rows_and_arrays_arrive_together(tmp_path):
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    ingest_batch(repo, rows("g1", "g2"), registry=None, writer=fake_writer())

    snap = repo.read("main")
    table = snap.table("granules").scan().to_arrow()
    assert set(table["id"].to_pylist()) == {"g1", "g2"}
    assert set(table["array_path"].to_pylist()) == {"/g1", "/g2"}
    assert snap.group["g1"]["B04"][0] == 1


def test_failed_granule_contributes_neither_row_nor_arrays(tmp_path):
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)

    result = ingest_batch(
        repo, rows("g1", "bad", "g3"), registry=None, writer=fake_writer(("bad",))
    )

    assert result.committed == ["g1", "g3"]
    assert "bad" in result.skipped and "unreadable" in result.skipped["bad"]

    snap = repo.read("main")
    ids = snap.table("granules").scan().to_arrow()["id"].to_pylist()
    assert set(ids) == {"g1", "g3"}
    assert "bad" not in list(snap.group)


def test_batch_where_everything_fails_publishes_nothing(tmp_path):
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    before = repo.read("main").snapshot_id

    with pytest.raises(BatchFailed, match="all 2 granules"):
        ingest_batch(
            repo, rows("a", "b"), registry=None, writer=fake_writer(("a", "b"))
        )

    assert repo.read("main").snapshot_id == before


def half_writer(failures=()):
    """Stages arrays and only then fails, the way a real multi-band write can."""

    def writer(tx, row, *, registry, bands):
        group = zarr.open_group(tx.session.store, path=f"/{row['id']}", mode="a")
        group.create_array("B04", shape=(4,), dtype="uint8", chunks=(4,))[:] = 1
        if row["id"] in failures:
            raise GranuleError(f"{row['id']}: failed after staging two bands")
        return f"/{row['id']}"

    return writer


def test_partially_written_granule_leaves_nothing_behind(tmp_path):
    """A granule that fails midway must not publish orphaned arrays: the batch
    commits other granules, so its half-written groups would ride along."""
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)

    result = ingest_batch(
        repo, rows("g1", "half", "g3"), registry=None, writer=half_writer(("half",))
    )

    assert result.committed == ["g1", "g3"]
    assert "half" in result.skipped

    snap = repo.read("main")
    ids = snap.table("granules").scan().to_arrow()["id"].to_pylist()
    assert set(ids) == {"g1", "g3"}
    assert "half" not in list(snap.group)
