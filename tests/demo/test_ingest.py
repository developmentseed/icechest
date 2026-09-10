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


def test_existing_granule_is_skipped_and_its_arrays_survive(tmp_path):
    """Re-ingesting a granule must not touch what an earlier commit published.

    The skip path deletes /{id} to clear a half-written granule. If a granule
    already in the store were allowed into that path, the delete would remove
    arrays a previous batch committed while its row stayed in the table --
    breaking the exact invariant this project exists to hold, via the code
    added to protect it.
    """
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    ingest_batch(repo, rows("g1"), registry=None, writer=fake_writer())

    result = ingest_batch(
        repo, rows("g2", "g1"), registry=None, writer=half_writer(("g1",))
    )

    assert result.committed == ["g2"]
    assert "already in the store" in result.skipped["g1"]

    snap = repo.read("main")
    assert snap.group["g1"]["B04"][0] == 1  # untouched
    ids = snap.table("granules").scan().to_arrow()["id"].to_pylist()
    assert sorted(ids) == ["g1", "g2"]  # and not appended twice


def test_duplicate_id_within_one_batch_keeps_the_first(tmp_path):
    """The second occurrence must not overwrite or delete the first's arrays,
    nor put a second row in the table pointing at the same group."""
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)

    result = ingest_batch(repo, rows("g1", "g1"), registry=None, writer=fake_writer())

    assert result.committed == ["g1"]
    assert "already staged earlier in this batch" in result.skipped["g1"]

    snap = repo.read("main")
    assert snap.group["g1"]["B04"][0] == 1
    assert snap.table("granules").scan().to_arrow()["id"].to_pylist() == ["g1"]


def test_failure_before_any_write_deletes_nothing(tmp_path, monkeypatch):
    """Nothing was staged, so there is nothing to clean up -- and a delete
    issued anyway is the blast radius that made the bug above possible."""
    from icechest.demo import ingest as ingest_module

    deletes = []
    real_sync = ingest_module.sync
    monkeypatch.setattr(
        ingest_module,
        "sync",
        lambda coro: (deletes.append(coro) or real_sync(coro)),
    )

    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    ingest_batch(repo, rows("g1", "bad"), registry=None, writer=fake_writer(("bad",)))
    assert deletes == []

    ingest_batch(repo, rows("h1", "half"), registry=None, writer=half_writer(("half",)))
    assert len(deletes) == 1  # the one that did stage arrays


def test_repeated_failing_id_keeps_every_reason(tmp_path):
    """Two failures under one id: the second reason must not silently replace
    the first, which is the only record of what went wrong."""
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)

    result = ingest_batch(
        repo, rows("g1", "bad", "bad"), registry=None, writer=fake_writer(("bad",))
    )

    assert result.committed == ["g1"]
    assert result.skipped["bad"].count("unreadable") == 2
