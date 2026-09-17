"""End-to-end tests for the hybrid Icechunk/Iceberg store."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from icechest import read_pointers_at_tag
from tests.helpers import GRANULE_SCHEMA, granules, seed


def test_arrays_and_table_land_in_one_commit(repo):
    seed(repo)
    with repo.transaction("main", "ingest") as tx:
        tx.group["data"][0:10] = np.ones(10, "f4")
        tx.append("granules", granules("g1"))

    snap = repo.read("main")
    assert snap.group["data"][0] == 1.0
    assert snap.table("granules").scan().to_arrow().num_rows == 1
    # One commit carries both: the pointer moved in the same snapshot.
    assert "granules" in snap.pointers


def test_failed_transaction_publishes_nothing(repo):
    seed(repo)
    before = repo.read("main").snapshot_id

    with pytest.raises(RuntimeError):
        with repo.transaction("main", "doomed") as tx:
            tx.group["data"][0:10] = np.ones(10, "f4")
            tx.append("granules", granules("g-never"))
            raise RuntimeError("writer died mid-ingest")

    assert repo.read("main").snapshot_id == before
    assert repo.read("main").table("granules").scan().to_arrow().num_rows == 0


def test_concurrent_appends_both_survive(repo):
    """Ben commits first; Anna replays her table op onto his version."""
    seed(repo)

    ben = repo.transaction("main", "ben")
    anna = repo.transaction("main", "anna")

    ben.group["data"][0:10] = np.ones(10, "f4")
    ben.append("granules", granules("ben-1"))

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.append("granules", granules("anna-1"))

    ben.commit()
    anna.commit()  # conflicts, recovers, lands on top of Ben

    snap = repo.read("main")
    rows = snap.table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"ben-1", "anna-1"}
    assert snap.group["data"][0] == 1.0
    assert snap.group["data"][50] == 2.0


def test_tag_pins_arrays_and_table_version_together(repo):
    seed(repo)
    with repo.transaction("main", "v1") as tx:
        tx.group["data"][0:10] = np.ones(10, "f4")
        tx.append("granules", granules("g1"))
    repo.repo.create_tag("v1", tx.snapshot_id)

    with repo.transaction("main", "v2") as tx2:
        tx2.group["data"][10:20] = np.full(10, 3.0, "f4")
        tx2.append("granules", granules("g2"))

    pinned = repo.read(tag="v1")
    assert pinned.table("granules").scan().to_arrow().num_rows == 1
    assert pinned.group["data"][10] == 0.0

    tip = repo.read("main")
    assert tip.table("granules").scan().to_arrow().num_rows == 2
    assert tip.group["data"][10] == 3.0

    assert read_pointers_at_tag(repo.repo, "v1") == pinned.pointers


def test_gc_safety_properties_are_set(repo):
    seed(repo)
    props = repo.read("main").table("granules").properties
    assert props["gc.enabled"] == "false"
    assert props["write.metadata.delete-after-commit.enabled"] == "false"


def test_unchanged_tables_carry_forward(repo):
    """Every snapshot must resolve every table it declares.

    The convention requires writers to carry unchanged tables forward in each
    commit's pointer map, so a commit touching only one table does not strand
    the others at that snapshot.
    """
    other = pa.schema([pa.field("qa_flag", pa.int32(), nullable=False)])
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(100,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULE_SCHEMA)
        tx.create_table("quality", other)

    with repo.transaction("main", "append to granules only") as tx:
        tx.append("granules", granules("g1"))

    snap = repo.read("main")
    assert set(snap.pointers) == {"granules", "quality"}
    assert snap.table("granules").scan().to_arrow().num_rows == 1
    assert snap.table("quality").scan().to_arrow().num_rows == 0


def test_snapshot_predating_a_table_does_not_declare_it(repo):
    """Declaration and first pointer land in the same commit.

    So a snapshot that predates a table lacks both, and fails at the
    declaration check -- a declared table is always resolvable.
    """
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(10,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULE_SCHEMA)
    early = tx.snapshot_id

    with repo.transaction("main", "add second table") as tx2:
        tx2.create_table("quality", pa.schema([pa.field("qa", pa.int32())]))

    assert "quality" in repo.read("main").pointers
    early_snap = repo.read(snapshot_id=early)
    assert "quality" not in early_snap.bindings
    assert "quality" not in early_snap.pointers
    with pytest.raises(KeyError, match="not declared"):
        early_snap.table("quality")


def test_create_table_forwards_partitioning_and_sort_order(repo):
    """A table's layout is fixed when it is created, so an intent that drops
    the spec on the floor produces an unpartitioned, unsorted table that no
    later call can correct."""
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.table.sorting import SortField, SortOrder
    from pyiceberg.transforms import IdentityTransform, TruncateTransform
    from pyiceberg.types import LongType, NestedField

    schema = Schema(
        NestedField(field_id=1, name="key", field_type=LongType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=1, field_id=1000, transform=TruncateTransform(16), name="block"
        )
    )
    order = SortOrder(SortField(source_id=1, transform=IdentityTransform()))

    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(10,), dtype="f4", chunks=(10,))
        tx.create_table("keyed", schema, partition_spec=spec, sort_order=order)

    table = repo.read("main").table("keyed")
    assert [f.name for f in table.spec().fields] == ["block"]
    assert table.spec().fields[0].transform.width == 16
    assert [f.source_id for f in table.sort_order().fields] == [1]
