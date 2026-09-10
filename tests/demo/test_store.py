"""Store setup: the container is declared before anything writes to it."""

from __future__ import annotations

import pickle

from pyiceberg.schema import Schema
from pyiceberg.table.sorting import SortDirection
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType, TimestamptzType

from icechest.demo.assets import LPDAAC_S3_PREFIX
from icechest.demo.credentials import lpdaac_credentials, virtual_chunk_container
from icechest.demo.store import (
    STAC_HASH_FIELD_ID,
    ensure_table,
    granules_schema,
    open_store,
)

SOURCE = Schema(
    NestedField(field_id=1, name="id", field_type=StringType(), required=False),
    NestedField(
        field_id=2, name="datetime", field_type=TimestamptzType(), required=False
    ),
)


def test_credential_callable_is_picklable():
    """icechunk requires it: refreshable credentials may cross a process."""
    assert pickle.loads(pickle.dumps(lpdaac_credentials)) is lpdaac_credentials


def test_container_covers_the_lpdaac_prefix():
    assert virtual_chunk_container().url_prefix == LPDAAC_S3_PREFIX


def test_granules_schema_is_the_source_plus_our_two_columns():
    schema = granules_schema(SOURCE)
    names = [field.name for field in schema.fields]
    assert names == ["id", "datetime", "array_path", "stac_hash", "stac_hash_block"]


def test_open_store_needs_no_network(tmp_path):
    """Refreshable credentials are lazy, so a store opens offline."""
    repo = open_store(tmp_path)
    assert repo.read("main").pointers == {}


def test_ensure_table_creates_once(tmp_path):
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    first = repo.read("main").snapshot_id
    assert "granules" in repo.read("main").pointers

    ensure_table(repo, SOURCE)
    assert repo.read("main").snapshot_id == first  # no second commit


def test_ensure_table_declares_the_array_path_column(tmp_path):
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    table = repo.read("main").table("granules")
    assert "array_path" in [field.name for field in table.schema().fields]


def test_source_schema_may_not_collide_with_the_array_path_id():
    """The 'well clear of the archive's ids' comment, made self-checking: two
    fields sharing an id is a schema whose columns cannot be told apart."""
    import pytest

    from icechest.demo.store import ARRAY_PATH_FIELD_ID

    colliding = Schema(
        NestedField(
            field_id=ARRAY_PATH_FIELD_ID,
            name="id",
            field_type=StringType(),
            required=False,
        )
    )
    with pytest.raises(ValueError, match="array_path"):
        granules_schema(colliding)


def test_stac_hash_is_a_long():
    """Iceberg has no unsigned integer, and the hash is built to fit int64."""
    schema = granules_schema(SOURCE)
    field = schema.find_field("stac_hash")
    assert isinstance(field.field_type, LongType)
    assert field.field_id == STAC_HASH_FIELD_ID


def test_table_is_sorted_by_the_hash(tmp_path):
    """Declared so compactors and readers know the intended clustering; the
    rows themselves are sorted at append time, since PyIceberg's write path
    does not consult a sort order."""
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)

    table = repo.read("main").table("granules")
    # Iceberg assigns fresh field ids when a table is created and remaps the
    # sort order onto them, so the assertion is "sorted by the hash column",
    # not by whatever number it carried before creation.
    hash_id = table.schema().find_field("stac_hash").field_id
    order = table.sort_order()
    assert len(order.fields) == 1
    sort_field = order.fields[0]
    assert sort_field.source_id == hash_id
    assert sort_field.direction == SortDirection.ASC
    assert isinstance(sort_field.transform, IdentityTransform)


def test_table_is_partitioned_by_the_hash_block(tmp_path):
    """A partition is a block of the hash space, so it stays a contiguous
    spatio-temporal region. Bucket would hash the value again and throw away
    the locality the Morton code exists to provide."""
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)

    table = repo.read("main").table("granules")
    block_id = table.schema().find_field("stac_hash_block").field_id
    spec = table.spec()
    assert len(spec.fields) == 1
    partition_field = spec.fields[0]
    assert partition_field.source_id == block_id
    assert partition_field.name == "stac_hash_block"
    assert isinstance(partition_field.transform, IdentityTransform)
