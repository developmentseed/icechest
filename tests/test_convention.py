"""Tests for the ``iceberg:`` Zarr convention."""

from __future__ import annotations

import icechunk
import numpy as np
import pyarrow as pa
import pytest
import zarr

from icechest import HybridRepo
from icechest.convention import (
    CONVENTION_UUID,
    CONVENTIONS_KEY,
    TABLES_ATTR,
    TableBinding,
    declare,
    is_declared,
    read_bindings,
)
from tests.test_hybrid import GRANULE_SCHEMA, granules


@pytest.fixture
def repo(tmp_path):
    store = icechunk.local_filesystem_storage(str(tmp_path / "icechunk"))
    return HybridRepo.create(store, warehouse=str(tmp_path / "warehouse"))


def test_convention_is_declared_on_the_group(repo):
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(10,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULE_SCHEMA)

    attrs = dict(repo.read("main").group.attrs)
    cmos = attrs[CONVENTIONS_KEY]
    assert any(c["uuid"] == CONVENTION_UUID for c in cmos)
    assert attrs[TABLES_ATTR]["granules"] == {
        "location": str(repo.warehouse) + "/granules"
    }


def test_store_is_self_describing(repo):
    """A reader finds the table from the Zarr hierarchy alone."""
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(10,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULE_SCHEMA)
    with repo.transaction("main", "ingest") as tx:
        tx.append("granules", granules("g1"))

    snap = repo.read("main")
    assert list(snap.bindings) == ["granules"]
    assert snap.table("granules").scan().to_arrow().num_rows == 1


def test_convention_is_ignorable_by_plain_zarr(repo):
    """The attributes are inert: a Zarr reader that ignores them still works."""
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(10,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULE_SCHEMA)
    with repo.transaction("main", "ingest") as tx:
        tx.group["data"][:] = np.arange(10, dtype="f4")
        tx.append("granules", granules("g1"))

    session = repo.repo.readonly_session("main")
    group = zarr.open_group(session.store, mode="r")
    assert group["data"][3] == 3.0


def test_appends_do_not_touch_group_attributes(repo):
    """The whole point of the indirection: appends must not dirty the Zarr node.

    If they did, concurrent writers would hit an unresolvable
    ZarrMetadataDoubleUpdate on '/' instead of rebasing cleanly.
    """
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(100,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULE_SCHEMA)

    before = dict(repo.read("main").group.attrs)
    for i in range(3):
        with repo.transaction("main", f"append-{i}") as tx:
            tx.append("granules", granules(f"g{i}"))
    after = dict(repo.read("main").group.attrs)

    assert before == after
    # ...while the pointer did move, in commit metadata.
    assert repo.read("main").pointers != {}


def test_declare_is_idempotent():
    group = zarr.open_group(zarr.storage.MemoryStore(), mode="a")
    binding = {"t": TableBinding(location="/w/t")}
    declare(group, binding)
    first = dict(group.attrs)
    declare(group, binding)
    assert dict(group.attrs) == first
    assert is_declared(group)
    assert read_bindings(group)["t"].location == "/w/t"


def test_undeclared_table_is_a_clear_error(repo):
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(10,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULE_SCHEMA)

    with pytest.raises(KeyError, match="not declared"):
        repo.read("main").table("nope")


def test_unknown_arrow_schema_is_accepted(repo):
    """PyArrow schemas convert on the way in, like any PyIceberg catalog."""
    schema = pa.schema([pa.field("x", pa.int64(), nullable=False)])
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(10,), dtype="f4", chunks=(10,))
        tx.create_table("plain", schema)
    assert "plain" in repo.read("main").bindings
